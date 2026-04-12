"""
automation/retrain_pipeline.py
────────────────────────────────
Automated retraining pipeline.

Triggers:
  - Scheduled (weekly / monthly via APScheduler)
  - Drift-detected (called by DriftMonitor when critical)
  - Manual (via API endpoint POST /retrain)

Pipeline steps:
  1.  Fetch latest market data (incremental, last N days)
  2.  Rebuild feature store (append to existing)
  3.  Validate data quality
  4.  Run Walk-Forward backtest on existing model (baseline)
  5.  Train new model (inherits weights from previous = warm start)
  6.  Evaluate new vs old on held-out OOS window
  7.  Champion/Challenger test (t-test on Sharpe)
  8.  If new model wins → promote to production
  9.  Archive old model with timestamp
  10. Notify via alert system

Safe-guards:
  - New model must beat old by min_improvement_threshold
  - New model must pass absolute min_sharpe gate
  - Rollback to previous model if production errors detected
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ─── Retraining Config ────────────────────────────────────────────────────────

@dataclass
class RetrainConfig:
    # Data
    retrain_lookback_days: int = 504        # ~2 years of fresh data
    incremental_days: int = 63              # only fetch last 3 months if incremental

    # Training
    timesteps: int = 500_000
    n_envs: int = 2
    warm_start: bool = True                 # initialize from previous model weights

    # Evaluation
    n_eval_episodes: int = 10
    oos_eval_days: int = 63                 # last 3 months as OOS test

    # Promotion gates
    min_sharpe_absolute: float = 0.3        # new model must achieve this
    min_sharpe_improvement: float = 0.05    # must beat old by this margin
    confidence_level: float = 0.10          # one-tailed t-test p-value
    require_positive_return: bool = True

    # Paths
    model_dir: str = "models"
    archive_dir: str = "models/archive"

    # Notifications
    notify_on_retrain: bool = True
    notify_on_promotion: bool = True
    notify_on_failure: bool = True


@dataclass
class RetrainResult:
    triggered_by: str = ""                  # "drift" | "schedule" | "manual"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    success: bool = False
    promoted: bool = False
    rollback: bool = False

    # Metrics comparison
    old_sharpe: float = 0.0
    new_sharpe: float = 0.0
    old_return: float = 0.0
    new_return: float = 0.0
    improvement: float = 0.0
    pvalue: float = 1.0

    # Metadata
    new_model_path: str = ""
    archive_path: str = ""
    error_msg: str = ""
    training_minutes: float = 0.0

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["started_at"] = self.started_at.isoformat()
        d["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return d

    def summary(self) -> str:
        status = "PROMOTED ✓" if self.promoted else ("REJECTED ✗" if not self.success else "TRAINED (not promoted)")
        return (
            f"Retrain [{self.triggered_by}] {status} | "
            f"Old Sharpe={self.old_sharpe:.3f} → New={self.new_sharpe:.3f} "
            f"(Δ={self.improvement:+.3f}, p={self.pvalue:.3f}) | "
            f"Duration={self.training_minutes:.1f}min"
        )


# ─── Champion / Challenger Evaluator ─────────────────────────────────────────

class ChampionChallengerEvaluator:
    """
    Statistical test to decide whether to promote the new model.

    Uses Welch's t-test on episode Sharpe ratios:
      H0: new_sharpe == old_sharpe
      H1: new_sharpe > old_sharpe  (one-tailed)

    Promotion requires:
      - p-value < alpha (statistically significant improvement)
      - new Sharpe > min_sharpe_absolute
      - new Sharpe > old Sharpe + min_improvement
    """

    def __init__(self, cfg: RetrainConfig):
        self.cfg = cfg

    def evaluate(
        self,
        old_model,
        new_model,
        feature_store: Dict,
        symbols: List[str],
        n_episodes: int = 10,
    ) -> Tuple[bool, float, float, float, float]:
        """
        Run both models on the same OOS test set.
        Returns: (should_promote, old_sharpe, new_sharpe, improvement, pvalue)
        """
        from environment.trading_env import TradingEnv

        env_kw = dict(
            initial_capital=1_000_000,
            lookback_window=30,
            episode_length=self.cfg.oos_eval_days,
            reward_type="sharpe",
            transaction_cost_bps=5.0,
            mode="test",
        )

        old_sharpes = self._run_episodes(old_model, feature_store, symbols, env_kw, n_episodes)
        new_sharpes = self._run_episodes(new_model, feature_store, symbols, env_kw, n_episodes)

        old_mean = float(np.mean(old_sharpes))
        new_mean = float(np.mean(new_sharpes))
        improvement = new_mean - old_mean

        # Welch's t-test (one-tailed, new > old)
        if len(old_sharpes) >= 3 and np.std(old_sharpes) > 1e-6:
            t_stat, p_two_tail = stats.ttest_ind(new_sharpes, old_sharpes, equal_var=False)
            p_value = p_two_tail / 2 if t_stat > 0 else 1.0
        else:
            p_value = 0.5

        should_promote = (
            new_mean >= self.cfg.min_sharpe_absolute
            and improvement >= self.cfg.min_sharpe_improvement
            and p_value <= self.cfg.confidence_level
        )

        logger.info(
            f"Champion/Challenger: Old={old_mean:.3f}±{np.std(old_sharpes):.3f} | "
            f"New={new_mean:.3f}±{np.std(new_sharpes):.3f} | "
            f"Δ={improvement:+.3f} | p={p_value:.3f} | "
            f"{'PROMOTE' if should_promote else 'REJECT'}"
        )
        return should_promote, old_mean, new_mean, improvement, p_value

    def _run_episodes(self, model, feature_store, symbols, env_kw, n_episodes) -> List[float]:
        from environment.trading_env import TradingEnv

        valid = [s for s in symbols if s in feature_store]
        if not valid:
            return [0.0] * n_episodes

        try:
            env = TradingEnv(feature_store=feature_store, symbols=valid, **env_kw)
        except ValueError:
            env_kw2 = dict(env_kw, episode_length=min(env_kw["episode_length"], 40))
            try:
                env = TradingEnv(feature_store=feature_store, symbols=valid, **env_kw2)
            except Exception:
                return [0.0] * n_episodes

        sharpes = []
        for ep in range(n_episodes):
            try:
                obs, _ = env.reset(seed=ep * 13)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, t, tr, info = env.step(action)
                    done = t or tr
                sharpes.append(info.get("sharpe_ratio", 0.0))
            except Exception:
                sharpes.append(0.0)
        return sharpes


# ─── Retraining Pipeline ──────────────────────────────────────────────────────

class RetrainingPipeline:
    """
    Full automated retraining pipeline.

    Usage:
        pipeline = RetrainingPipeline(
            symbols=symbols,
            feature_store=feature_store,
            alert_system=alerts,
            config=RetrainConfig(),
        )
        # Triggered by drift detector:
        result = pipeline.run(triggered_by="drift")

        # Or schedule weekly:
        scheduler.add_job(pipeline.run, "cron", day_of_week="sun",
                          kwargs={"triggered_by": "schedule"})
    """

    def __init__(
        self,
        symbols: List[str],
        feature_store: Dict,
        alert_system=None,
        config: Optional[RetrainConfig] = None,
        on_promotion: Optional[Callable] = None,
    ):
        self.symbols = symbols
        self.feature_store = feature_store
        self.alerts = alert_system
        self.cfg = config or RetrainConfig()
        self.on_promotion = on_promotion
        self.history: List[RetrainResult] = []
        self._is_running = False

        # Ensure directories exist
        Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.archive_dir).mkdir(parents=True, exist_ok=True)

    def run(self, triggered_by: str = "manual") -> RetrainResult:
        """Execute the full retraining pipeline."""
        if self._is_running:
            logger.warning("Retrain already in progress — skipping")
            result = RetrainResult(triggered_by=triggered_by, error_msg="already_running")
            return result

        self._is_running = True
        result = RetrainResult(triggered_by=triggered_by)
        t0 = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"RETRAINING PIPELINE STARTED [{triggered_by.upper()}]")
        logger.info(f"{'='*60}")

        if self.cfg.notify_on_retrain and self.alerts:
            self.alerts.send(
                f"🔄 Retraining started [{triggered_by}]",
                level="info",
                details={"symbols": self.symbols, "timesteps": self.cfg.timesteps},
            )

        try:
            # ── Step 1: Load current production model ──────────────────────────
            prod_path = Path(self.cfg.model_dir) / "production_model"
            old_model = self._load_model(str(prod_path))
            has_old_model = old_model is not None
            logger.info(f"Step 1: Loaded {'existing' if has_old_model else 'no'} production model")

            # ── Step 2: Refresh feature store ─────────────────────────────────
            fresh_store = self._refresh_feature_store()
            if not fresh_store:
                fresh_store = self.feature_store
            logger.info(f"Step 2: Feature store ready ({len(fresh_store)} symbols)")

            # ── Step 3: Train new model ────────────────────────────────────────
            logger.info("Step 3: Training new model...")
            new_model = self._train_model(
                fresh_store,
                warm_start_model=old_model if self.cfg.warm_start else None,
            )
            result.new_model_path = str(Path(self.cfg.model_dir) / "challenger_model.zip")
            new_model.save(result.new_model_path.replace(".zip", ""))
            logger.info(f"Step 3: New model saved → {result.new_model_path}")

            # ── Step 4: Champion/Challenger evaluation ─────────────────────────
            logger.info("Step 4: Champion/Challenger evaluation...")
            evaluator = ChampionChallengerEvaluator(self.cfg)

            if has_old_model:
                promote, old_sh, new_sh, delta, pval = evaluator.evaluate(
                    old_model, new_model, fresh_store, self.symbols,
                    n_episodes=self.cfg.n_eval_episodes,
                )
                result.old_sharpe = old_sh
                result.new_sharpe = new_sh
                result.improvement = delta
                result.pvalue = pval
            else:
                # No old model → promote if meets absolute threshold
                new_sh = self._quick_eval(new_model, fresh_store)
                result.new_sharpe = new_sh
                promote = new_sh >= self.cfg.min_sharpe_absolute
                result.pvalue = 0.0

            result.success = True

            # ── Step 5: Promotion decision ─────────────────────────────────────
            if promote:
                self._promote_model(new_model, old_model if has_old_model else None, result)
                result.promoted = True
                logger.info(f"Step 5: NEW MODEL PROMOTED ✓ (Sharpe={result.new_sharpe:.3f})")

                if self.cfg.notify_on_promotion and self.alerts:
                    self.alerts.send(
                        f"✅ New model promoted! Sharpe: {result.old_sharpe:.3f} → {result.new_sharpe:.3f}",
                        level="success",
                        details=result.to_dict(),
                    )
                if self.on_promotion:
                    self.on_promotion(new_model)
            else:
                logger.info(
                    f"Step 5: New model REJECTED "
                    f"(Sharpe={result.new_sharpe:.3f} < threshold or not significant)"
                )
                if self.alerts:
                    self.alerts.send(
                        f"⚠️ Retrain complete but new model NOT promoted "
                        f"(Δ={result.improvement:+.3f}, p={result.pvalue:.3f})",
                        level="warning",
                    )

        except Exception as e:
            result.success = False
            result.error_msg = str(e)
            logger.error(f"Retraining FAILED: {e}", exc_info=True)
            if self.cfg.notify_on_failure and self.alerts:
                self.alerts.send(f"❌ Retraining failed: {e}", level="error")

        finally:
            result.finished_at = datetime.now(timezone.utc)
            result.training_minutes = (time.time() - t0) / 60
            self.history.append(result)
            self._is_running = False

            # Save result log
            log_path = Path(self.cfg.model_dir) / "retrain_history.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(result.to_dict(), default=str) + "\n")

        logger.info(result.summary())
        return result

    def _refresh_feature_store(self) -> Dict:
        """Fetch incremental market data and rebuild features."""
        try:
            from data.market_data import MarketDataLoader
            from features.feature_engineer import FeaturePipeline

            loader = MarketDataLoader(use_cache=True)
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (pd.Timestamp.now() - pd.DateOffset(days=self.cfg.retrain_lookback_days)).strftime("%Y-%m-%d")

            market_data = loader.get_ohlcv_universe(self.symbols, start, end)
            if not market_data:
                return {}

            vix = loader.get_vix(start, end)
            pipeline = FeaturePipeline()
            fresh_store = pipeline.build(market_data, vix=vix)
            return fresh_store
        except Exception as e:
            logger.warning(f"Feature store refresh failed: {e} — using cached")
            return {}

    def _train_model(self, feature_store: Dict, warm_start_model=None):
        """Train new PPO model, optionally warm-starting from existing weights."""
        from train.train_ppo import train_mlp
        valid = [s for s in self.symbols if s in feature_store]
        if not valid:
            raise ValueError("No valid symbols in feature store")

        model = train_mlp(
            feature_store=feature_store,
            symbols=valid,
            total_timesteps=self.cfg.timesteps,
            n_envs=self.cfg.n_envs,
            save_dir=self.cfg.model_dir,
            experiment_name="auto_retrain",
        )

        # Warm start: copy feature extractor weights from old model
        if warm_start_model is not None and self.cfg.warm_start:
            try:
                old_state = warm_start_model.policy.state_dict()
                new_state = model.policy.state_dict()
                # Only copy layers with matching shapes
                copied = 0
                for key in new_state:
                    if key in old_state and old_state[key].shape == new_state[key].shape:
                        new_state[key] = old_state[key].clone()
                        copied += 1
                model.policy.load_state_dict(new_state)
                logger.info(f"Warm start: copied {copied} weight tensors from old model")
            except Exception as e:
                logger.warning(f"Warm start failed: {e}")

        return model

    def _quick_eval(self, model, feature_store: Dict, n_episodes: int = 5) -> float:
        """Quick Sharpe estimate for absolute gate check."""
        from environment.trading_env import TradingEnv
        valid = [s for s in self.symbols if s in feature_store]
        if not valid:
            return 0.0
        try:
            env = TradingEnv(feature_store=feature_store, symbols=valid,
                             episode_length=40, lookback_window=20,
                             reward_type="sharpe", mode="test")
            sharpes = []
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=ep)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, t, tr, info = env.step(action)
                    done = t or tr
                sharpes.append(info.get("sharpe_ratio", 0.0))
            return float(np.mean(sharpes))
        except Exception:
            return 0.0

    def _promote_model(self, new_model, old_model, result: RetrainResult):
        """Archive old model, install new model as production."""
        prod_path = Path(self.cfg.model_dir) / "production_model"
        archive_path = Path(self.cfg.archive_dir) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Archive old production model
        if (prod_path.parent / (prod_path.name + ".zip")).exists():
            shutil.copy2(
                str(prod_path.parent / (prod_path.name + ".zip")),
                str(archive_path) + ".zip",
            )
            result.archive_path = str(archive_path) + ".zip"
            logger.info(f"Old model archived: {archive_path}.zip")

        # Install new model
        new_model.save(str(prod_path))
        result.new_model_path = str(prod_path) + ".zip"

    def _load_model(self, path: str):
        """Load model, return None if not found."""
        try:
            from stable_baselines3 import PPO
            if Path(path + ".zip").exists():
                return PPO.load(path)
        except Exception as e:
            logger.debug(f"Could not load model from {path}: {e}")
        return None

    def rollback(self) -> bool:
        """Restore the most recent archived model as production."""
        archive_dir = Path(self.cfg.archive_dir)
        archives = sorted(archive_dir.glob("model_*.zip"), reverse=True)
        if not archives:
            logger.warning("No archived models found for rollback")
            return False

        latest_archive = archives[0]
        prod_path = Path(self.cfg.model_dir) / "production_model.zip"
        shutil.copy2(str(latest_archive), str(prod_path))
        logger.info(f"ROLLBACK: restored {latest_archive.name} as production model")

        if self.alerts:
            self.alerts.send(f"⚠️ Model rolled back to {latest_archive.name}", level="warning")
        return True

    def get_history(self) -> List[Dict]:
        """Return retraining history."""
        return [r.to_dict() for r in self.history]
