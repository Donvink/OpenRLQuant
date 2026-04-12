"""
train/curriculum.py
────────────────────
Curriculum Learning for RL Trading.

Key idea: Don't throw the agent into 2020 crash + 2022 bear market from day 1.
Start with "easy" bull markets, progressively introduce harder regimes.

Stages:
  Stage 1 (calm bull):    low VIX, trending up, few stocks, long episodes
  Stage 2 (normal):       mixed regimes, more stocks
  Stage 3 (volatile):     includes bear markets, VIX spikes
  Stage 4 (full):         all regimes, full universe, realistic costs

Progression trigger: agent must achieve min_sharpe for N consecutive evaluations
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Curriculum Stage Definition ──────────────────────────────────────────────

@dataclass
class CurriculumStage:
    name: str
    description: str

    # Universe constraints
    n_stocks: int = 5                  # how many stocks to trade
    stock_selection: str = "stable"    # "stable" | "volatile" | "all"

    # Episode constraints
    episode_length: int = 63           # trading days per episode
    lookback_window: int = 30

    # Market regime filter
    # Agent only trained on dates matching this VIX range
    vix_min: float = 0.0
    vix_max: float = 100.0
    require_positive_spy_trend: bool = False   # only bull periods

    # Cost model (relaxed early, realistic later)
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 3.0

    # Reward
    reward_type: str = "log_return"    # start simple, add Sharpe complexity later

    # Advancement criteria
    min_sharpe_to_advance: float = 0.5
    min_eval_episodes: int = 10        # must pass N evaluations
    consecutive_passes_needed: int = 3

    # Training budget
    timesteps: int = 500_000


CURRICULUM_STAGES = [
    CurriculumStage(
        name="Stage 1 — Calm Bull",
        description="Low volatility bull market only. 5 stable large-caps. Short episodes.",
        n_stocks=5,
        stock_selection="stable",
        episode_length=63,
        vix_min=0.0, vix_max=20.0,
        require_positive_spy_trend=True,
        transaction_cost_bps=1.0,   # Relaxed costs to let agent learn first
        slippage_bps=1.0,
        reward_type="log_return",
        min_sharpe_to_advance=0.3,
        timesteps=300_000,
    ),
    CurriculumStage(
        name="Stage 2 — Normal Markets",
        description="Mixed regimes (VIX < 25). 8 stocks. Medium episodes.",
        n_stocks=8,
        stock_selection="stable",
        episode_length=126,
        vix_min=0.0, vix_max=25.0,
        transaction_cost_bps=3.0,
        slippage_bps=2.0,
        reward_type="sharpe",
        min_sharpe_to_advance=0.5,
        timesteps=500_000,
    ),
    CurriculumStage(
        name="Stage 3 — Volatile Markets",
        description="All regimes including 2020 crash, 2022 bear. Full cost model.",
        n_stocks=10,
        stock_selection="all",
        episode_length=252,
        vix_min=0.0, vix_max=100.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
        reward_type="sharpe",
        min_sharpe_to_advance=0.8,
        timesteps=1_000_000,
    ),
    CurriculumStage(
        name="Stage 4 — Full Universe",
        description="Production configuration. All stocks, all regimes, full costs.",
        n_stocks=15,
        stock_selection="all",
        episode_length=252,
        vix_min=0.0, vix_max=100.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
        reward_type="sharpe",
        min_sharpe_to_advance=1.0,
        timesteps=2_000_000,
    ),
]


# ─── Curriculum Manager ────────────────────────────────────────────────────────

class CurriculumManager:
    """
    Tracks training progress and decides when to advance stages.

    Usage:
        curriculum = CurriculumManager(stages=CURRICULUM_STAGES)
        env_factory = curriculum.make_env_factory(feature_store, all_symbols, vix)

        while not curriculum.is_complete():
            stage = curriculum.current_stage
            # train model on stage.timesteps
            sharpe = evaluate(model, env_factory)
            curriculum.record_eval(sharpe)
    """

    def __init__(
        self,
        stages: List[CurriculumStage] = None,
        feature_store: Optional[Dict] = None,
        all_symbols: Optional[List[str]] = None,
        vix_series: Optional[pd.Series] = None,
    ):
        self.stages = stages or CURRICULUM_STAGES
        self.feature_store = feature_store
        self.all_symbols = all_symbols or []
        self.vix = vix_series
        self.current_stage_idx = 0
        self.eval_history: List[Dict] = []
        self.consecutive_passes = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    def is_complete(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def record_eval(self, sharpe: float, extra_metrics: Dict = None) -> bool:
        """
        Record an evaluation result.
        Returns True if stage advanced.
        """
        stage = self.current_stage
        passed = sharpe >= stage.min_sharpe_to_advance
        self.eval_history.append({
            "stage": stage.name,
            "sharpe": sharpe,
            "passed": passed,
            **(extra_metrics or {}),
        })

        if passed:
            self.consecutive_passes += 1
            logger.info(
                f"✓ Eval passed: Sharpe={sharpe:.3f} >= {stage.min_sharpe_to_advance:.3f} "
                f"({self.consecutive_passes}/{stage.consecutive_passes_needed})"
            )
        else:
            self.consecutive_passes = 0
            logger.info(f"✗ Eval failed: Sharpe={sharpe:.3f} < {stage.min_sharpe_to_advance:.3f}")

        if self.consecutive_passes >= stage.consecutive_passes_needed:
            return self._advance_stage()
        return False

    def _advance_stage(self) -> bool:
        old_stage = self.current_stage.name
        self.current_stage_idx += 1
        self.consecutive_passes = 0
        if not self.is_complete():
            logger.info(f"\n{'='*50}")
            logger.info(f"CURRICULUM ADVANCE: {old_stage} → {self.current_stage.name}")
            logger.info(f"{'='*50}\n")
            return True
        else:
            logger.info("CURRICULUM COMPLETE: All stages passed!")
            return False

    def select_symbols(self, stage: CurriculumStage) -> List[str]:
        """Select stocks for the current stage."""
        if not self.all_symbols or not self.feature_store:
            return self.all_symbols[:stage.n_stocks]

        if stage.stock_selection == "stable":
            # Prefer low-beta, high-liquidity stocks
            stable = [s for s in self.all_symbols
                      if s in ("AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "WMT", "HD", "XOM")]
            return stable[:stage.n_stocks]
        elif stage.stock_selection == "volatile":
            volatile = [s for s in self.all_symbols
                        if s in ("TSLA", "NVDA", "AMD", "META", "AMZN")]
            return volatile[:stage.n_stocks]
        else:
            return [s for s in self.all_symbols
                    if s in self.feature_store][:stage.n_stocks]

    def filter_dates_by_vix(
        self,
        feature_store: Dict,
        stage: CurriculumStage,
    ) -> Optional[pd.DatetimeIndex]:
        """
        Return date subset where VIX is in [vix_min, vix_max].
        The environment will sample episodes only from this window.
        """
        if self.vix is None:
            return None

        mask = (self.vix >= stage.vix_min) & (self.vix <= stage.vix_max)
        filtered_dates = self.vix[mask].index

        if len(filtered_dates) < 252:
            logger.warning(f"VIX filter leaves only {len(filtered_dates)} days — relaxing")
            return None

        logger.info(
            f"VIX filter [{stage.vix_min}, {stage.vix_max}]: "
            f"{len(filtered_dates)} / {len(self.vix)} days kept"
        )
        return filtered_dates

    def summary(self) -> str:
        lines = ["\nCurriculum Summary:"]
        for i, stage in enumerate(self.stages):
            status = "✓" if i < self.current_stage_idx else ("→" if i == self.current_stage_idx else "·")
            lines.append(f"  {status} [{i+1}/{len(self.stages)}] {stage.name}")
        evals = len(self.eval_history)
        if evals > 0:
            recent = self.eval_history[-5:]
            avg_sharpe = np.mean([e["sharpe"] for e in recent])
            lines.append(f"\n  Recent evals: {evals} total, avg Sharpe={avg_sharpe:.3f}")
        return "\n".join(lines)


# ─── VIX-Adaptive Env Wrapper ─────────────────────────────────────────────────

class RegimeAwareWrapper:
    """
    Wraps TradingEnv to sample episodes only from specific market regimes.
    Used during curriculum training to expose the agent to target regimes.
    """

    def __init__(self, env, allowed_dates: pd.DatetimeIndex):
        self.env = env
        self.allowed_dates = allowed_dates
        # Override train_dates to only allowed regime dates
        common = env.dates[env.dates.isin(allowed_dates)]
        if len(common) >= env.lookback + env.episode_length:
            env.train_dates = common
            logger.info(f"RegimeAwareWrapper: restricted to {len(common)} trading days")

    def __getattr__(self, name):
        return getattr(self.env, name)
