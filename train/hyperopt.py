"""
train/hyperopt.py
──────────────────
Hyperparameter optimization using Optuna (TPE sampler).

Searches over:
  - PPO: lr, n_steps, batch_size, gamma, gae_lambda, clip_range, entropy_coef
  - Network: d_model, n_heads, n_transformer_layers, dropout
  - Environment: reward_type, drawdown_penalty, turnover_penalty
  - Training: n_envs

Usage:
    optimizer = HyperparamOptimizer(feature_store, symbols, n_trials=50)
    best_params = optimizer.run()
    model = optimizer.train_with_best_params(best_params)
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def sample_ppo_params(trial) -> Dict[str, Any]:
    """
    Optuna trial → PPO hyperparameter dict.
    Covers the parameters with highest sensitivity (from literature).
    """
    return {
        # Optimizer
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),

        # PPO-specific
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.01, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),

        # Network (only if using MLP extractor; Transformer params separate)
        "net_arch_pi": trial.suggest_categorical("net_arch_pi", ["[64,64]", "[128,64]", "[256,128]"]),

        # Environment
        "reward_type": trial.suggest_categorical("reward_type", ["log_return", "sharpe", "calmar"]),
        "drawdown_penalty": trial.suggest_float("drawdown_penalty", 0.0, 1.0),
        "turnover_penalty": trial.suggest_float("turnover_penalty", 0.0, 0.05),
    }


def sample_transformer_params(trial) -> Dict[str, Any]:
    """Optuna trial → Transformer architecture params."""
    return {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_transformer_layers": trial.suggest_int("n_transformer_layers", 1, 4),
        "n_cross_asset_layers": trial.suggest_int("n_cross_asset_layers", 0, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }


class HyperparamOptimizer:
    """
    Runs Optuna study to find best hyperparameters.

    Each trial:
      1. Samples parameters
      2. Trains PPO for short_train_steps
      3. Evaluates on val env for n_eval_episodes
      4. Returns mean Sharpe as objective

    Pruning: Optuna prunes unpromising trials early (median pruner).
    """

    def __init__(
        self,
        feature_store: Dict,
        symbols: List[str],
        vix: Optional[Any] = None,
        n_trials: int = 30,
        short_train_steps: int = 100_000,  # short budget per trial
        n_eval_episodes: int = 5,
        n_envs: int = 2,
        study_name: str = "rl_trader_ppo",
        storage: Optional[str] = None,     # e.g. "sqlite:///optuna.db" for persistence
        save_dir: str = "models/hyperopt",
    ):
        self.feature_store = feature_store
        self.symbols = symbols
        self.vix = vix
        self.n_trials = n_trials
        self.short_train_steps = short_train_steps
        self.n_eval_episodes = n_eval_episodes
        self.n_envs = n_envs
        self.study_name = study_name
        self.storage = storage
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _make_env(self, params: Dict, mode: str = "train"):
        """Create env with trial params."""
        from environment.trading_env import TradingEnv
        return TradingEnv(
            feature_store=self.feature_store,
            symbols=self.symbols,
            lookback_window=30,
            episode_length=126,
            reward_type=params.get("reward_type", "sharpe"),
            drawdown_penalty=params.get("drawdown_penalty", 0.5),
            turnover_penalty=params.get("turnover_penalty", 0.01),
            mode=mode,
        )

    def _objective(self, trial) -> float:
        """Single Optuna trial objective."""
        import optuna
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        params = sample_ppo_params(trial)

        try:
            # Build envs
            train_envs = DummyVecEnv([
                lambda: self._make_env(params, "train") for _ in range(self.n_envs)
            ])
            val_env = self._make_env(params, "val")

            # Parse net arch
            import ast
            pi_arch = ast.literal_eval(params.pop("net_arch_pi", "[64,64]"))

            # Build model
            model = PPO(
                "MlpPolicy",
                train_envs,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"] // self.n_envs,
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                policy_kwargs={"net_arch": dict(pi=pi_arch, vf=pi_arch)},
                verbose=0,
            )

            # Short training
            model.learn(total_timesteps=self.short_train_steps)

            # Evaluate
            sharpes = []
            for ep in range(self.n_eval_episodes):
                obs, _ = val_env.reset(seed=ep)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = val_env.step(action)
                    done = term or trunc
                sharpes.append(info.get("sharpe_ratio", 0.0))

            mean_sharpe = float(np.mean(sharpes))

            # Report intermediate value for pruning
            trial.report(mean_sharpe, step=self.short_train_steps)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            train_envs.close()
            logger.info(f"Trial {trial.number}: Sharpe={mean_sharpe:.3f} | params={params}")
            return mean_sharpe

        except Exception as e:
            if "TrialPruned" in str(type(e)):
                raise
            logger.error(f"Trial {trial.number} failed: {e}")
            return -1.0

    def run(self, direction: str = "maximize") -> Dict[str, Any]:
        """
        Run the Optuna study.
        Returns best hyperparameters dict.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("pip install optuna")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        sampler = optuna.samplers.TPESampler(seed=42)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            storage=self.storage,
            load_if_exists=True,
        )

        logger.info(f"Starting Optuna study: {self.n_trials} trials")
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        best = study.best_params
        best_value = study.best_value
        logger.info(f"\nBest trial: Sharpe={best_value:.4f}")
        logger.info(f"Best params: {best}")

        # Save results
        import json
        results = {"best_params": best, "best_sharpe": best_value,
                   "n_trials": len(study.trials)}
        with open(self.save_dir / "best_params.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save importance plot if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            fig.savefig(self.save_dir / "param_importance.png", dpi=100, bbox_inches="tight")
        except Exception:
            pass

        return best

    def print_study_summary(self, study) -> None:
        """Print summary of Optuna study results."""
        trials = study.trials
        completed = [t for t in trials if t.state.name == "COMPLETE"]
        pruned = [t for t in trials if t.state.name == "PRUNED"]
        failed = [t for t in trials if t.state.name == "FAIL"]

        print(f"\n{'='*50}")
        print(f"Optuna Study: {self.study_name}")
        print(f"{'='*50}")
        print(f"  Completed: {len(completed)}")
        print(f"  Pruned:    {len(pruned)}")
        print(f"  Failed:    {len(failed)}")
        if completed:
            values = [t.value for t in completed]
            print(f"  Best Sharpe: {max(values):.4f}")
            print(f"  Mean Sharpe: {np.mean(values):.4f}")
        print(f"{'='*50}\n")
