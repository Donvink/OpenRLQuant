"""
train/callbacks.py
───────────────────
Custom SB3 training callbacks.

  EvalCallback_Trading  — evaluates on val set, saves best model, tracks Sharpe
  EarlyStoppingCallback — stops training if no improvement for N evals
  MetricsLoggerCallback — logs rich metrics to TensorBoard + MLflow
  CurriculumCallback    — triggers stage advancement in curriculum learning
  RiskCheckCallback     — monitors training stability (reward collapse, NaN)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


# ─── Trading-Specific Eval Callback ───────────────────────────────────────────

class TradingEvalCallback(BaseCallback):
    """
    Evaluates the agent on the validation environment every N steps.
    Tracks Sharpe ratio (not just mean reward) as the primary metric.
    Saves best model based on Sharpe.

    Usage:
        eval_callback = TradingEvalCallback(
            eval_env=val_env,
            eval_freq=10_000,
            n_eval_episodes=5,
            save_path="models/checkpoints",
        )
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: str = "models/checkpoints",
        best_model_name: str = "best_model",
        deterministic: bool = True,
        verbose: int = 1,
        # Curriculum integration
        curriculum_manager=None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_model_name = best_model_name
        self.deterministic = deterministic
        self.curriculum = curriculum_manager

        self.best_sharpe = -np.inf
        self.eval_results: List[Dict] = []
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.n_calls - self.last_eval_step < self.eval_freq:
            return True

        self.last_eval_step = self.n_calls
        metrics = self._run_evaluation()
        self.eval_results.append({"step": self.num_timesteps, **metrics})

        # Log to TensorBoard
        for k, v in metrics.items():
            self.logger.record(f"eval/{k}", v)
        self.logger.dump(self.num_timesteps)

        sharpe = metrics.get("mean_sharpe", 0.0)

        if self.verbose >= 1:
            logger.info(
                f"Eval @ {self.num_timesteps:,} steps | "
                f"Sharpe={sharpe:.3f} | "
                f"Return={metrics.get('mean_return', 0):+.2%} | "
                f"MaxDD={metrics.get('mean_max_dd', 0):.2%}"
            )

        # Save best model
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            save_path = self.save_path / self.best_model_name
            self.model.save(save_path)
            if self.verbose >= 1:
                logger.info(f"  → New best model saved (Sharpe={sharpe:.3f})")

        # Curriculum advancement
        if self.curriculum is not None:
            advanced = self.curriculum.record_eval(sharpe, metrics)
            if advanced:
                logger.info(self.curriculum.summary())

        return True

    def _run_evaluation(self) -> Dict[str, float]:
        """Run N evaluation episodes and aggregate metrics."""
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_rewards = []

        env = self.eval_env
        if hasattr(env, "envs"):  # VecEnv
            env = env.envs[0]

        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            all_returns.append(info.get("total_return", 0.0))
            all_sharpes.append(info.get("sharpe_ratio", 0.0))
            all_drawdowns.append(info.get("max_drawdown", 0.0))
            all_rewards.append(ep_reward)

        return {
            "mean_return": float(np.mean(all_returns)),
            "std_return": float(np.std(all_returns)),
            "mean_sharpe": float(np.mean(all_sharpes)),
            "mean_max_dd": float(np.mean(all_drawdowns)),
            "mean_ep_reward": float(np.mean(all_rewards)),
            "n_episodes": self.n_eval_episodes,
        }


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStoppingCallback(BaseCallback):
    """
    Stops training if the primary metric doesn't improve for `patience` evals.
    Prevents wasting compute on plateaued training.
    """

    def __init__(
        self,
        monitor: str = "eval/mean_sharpe",
        patience: int = 10,
        min_delta: float = 0.01,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = -np.inf
        self.no_improve_count = 0

    def _on_step(self) -> bool:
        # Read from logger records
        if self.monitor in self.logger.name_to_value:
            value = self.logger.name_to_value[self.monitor]
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.verbose >= 1 and self.no_improve_count > 0:
                    logger.info(
                        f"EarlyStopping: no improvement for {self.no_improve_count}/{self.patience} evals"
                    )

            if self.no_improve_count >= self.patience:
                logger.warning(
                    f"Early stopping triggered: {self.monitor} hasn't improved "
                    f"by {self.min_delta} in {self.patience} evals"
                )
                return False  # Stops training

        return True


# ─── Rich Metrics Logger ──────────────────────────────────────────────────────

class MetricsLoggerCallback(BaseCallback):
    """
    Logs rich training metrics to TensorBoard at each rollout end.

    Captures:
      - Reward statistics (mean, std, min, max)
      - Policy entropy (exploration measure)
      - Value function loss / policy loss
      - Episode length distribution
      - Portfolio-level metrics from info dicts
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_info_buffer: List[Dict] = []
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Collect info from vectorized env
        if hasattr(self.locals, "infos"):
            for info in self.locals.get("infos", []):
                if isinstance(info, dict) and "total_return" in info:
                    self._ep_info_buffer.append(info)

        if self.n_calls % self.log_freq == 0 and self._ep_info_buffer:
            self._flush_metrics()

        return True

    def _flush_metrics(self):
        buf = self._ep_info_buffer[-50:]  # last 50 episodes
        if not buf:
            return

        returns = [ep.get("total_return", 0) for ep in buf]
        sharpes = [ep.get("sharpe_ratio", 0) for ep in buf]
        drawdowns = [ep.get("max_drawdown", 0) for ep in buf]

        self.logger.record("portfolio/mean_return", np.mean(returns))
        self.logger.record("portfolio/mean_sharpe", np.mean(sharpes))
        self.logger.record("portfolio/mean_max_dd", np.mean(drawdowns))
        self.logger.record("portfolio/pct_profitable", np.mean([r > 0 for r in returns]))

        # Training speed
        elapsed = time.time() - self._start_time
        fps = self.num_timesteps / max(elapsed, 1)
        self.logger.record("train/fps", fps)
        self.logger.record("train/elapsed_min", elapsed / 60)

        self._ep_info_buffer.clear()

    def _on_rollout_end(self) -> None:
        self._flush_metrics()


# ─── Checkpoint Callback ──────────────────────────────────────────────────────

class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoint every N steps with timestep in filename.
    Allows resuming from any checkpoint.
    """

    def __init__(
        self,
        save_freq: int = 50_000,
        save_path: str = "models/checkpoints",
        name_prefix: str = "rl_trader",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(path)
            if self.verbose >= 1:
                logger.info(f"Checkpoint saved: {path}.zip")
        return True


# ─── NaN / Stability Monitor ──────────────────────────────────────────────────

class TrainingStabilityCallback(BaseCallback):
    """
    Monitors for training instability: NaN losses, reward collapse, exploding gradients.
    Terminates training early if instability detected (better to catch early).
    """

    def __init__(
        self,
        reward_collapse_threshold: float = -100.0,
        check_freq: int = 500,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.reward_collapse_threshold = reward_collapse_threshold
        self.check_freq = check_freq
        self.recent_rewards: List[float] = []

    def _on_step(self) -> bool:
        # Collect rewards
        if "rewards" in self.locals:
            self.recent_rewards.extend(self.locals["rewards"].tolist())

        if self.n_calls % self.check_freq != 0:
            return True

        # Check for NaN in model parameters
        for name, param in self.model.policy.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error(f"NaN gradient detected in {name} — stopping training")
                    return False
                if param.grad.abs().max() > 100.0:
                    logger.warning(f"Large gradient in {name}: {param.grad.abs().max():.1f}")

        # Check for reward collapse
        if len(self.recent_rewards) >= 100:
            mean_reward = np.mean(self.recent_rewards[-100:])
            if mean_reward < self.reward_collapse_threshold:
                logger.error(
                    f"Reward collapsed to {mean_reward:.3f} — "
                    f"check environment or reduce learning rate"
                )
                return False
            self.recent_rewards = self.recent_rewards[-200:]  # keep window

        return True

    def _on_step_check_import(self):
        pass


# Import torch only in callback that needs it
try:
    import torch
except ImportError:
    pass
