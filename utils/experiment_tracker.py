"""
utils/experiment_tracker.py
────────────────────────────
Experiment tracking via MLflow.
Logs hyperparameters, metrics, artifacts, and model checkpoints.

Usage:
    tracker = ExperimentTracker("rl_trader_phase2")
    with tracker.start_run("PPO_v1_sharpe_reward"):
        tracker.log_params({"lr": 3e-4, "n_steps": 2048, "reward_type": "sharpe"})
        # ... training loop ...
        tracker.log_metrics({"sharpe": 1.5, "annual_return": 0.25}, step=1000)
        tracker.log_model(model, "ppo_agent")
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Thin wrapper around MLflow with graceful fallback if MLflow not installed.
    In fallback mode, logs to a local CSV file instead.
    """

    def __init__(
        self,
        experiment_name: str = "rl_trader",
        tracking_uri: str = "logs/mlruns",
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self._mlflow = None
        self._run_id = None
        self._fallback_log: Dict = {}

        self._try_init_mlflow()

    def _try_init_mlflow(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            logger.info(f"MLflow tracking: {self.tracking_uri}/{self.experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed — using CSV fallback (pip install mlflow to enable)")

    @contextmanager
    def start_run(self, run_name: str = "", nested: bool = False):
        """Context manager for a training run."""
        if self._mlflow:
            with self._mlflow.start_run(run_name=run_name, nested=nested, tags=self.tags) as run:
                self._run_id = run.info.run_id
                logger.info(f"MLflow run started: {run_name} (id={self._run_id[:8]})")
                yield self
                logger.info(f"MLflow run ended: {run_name}")
        else:
            self._fallback_log = {"run_name": run_name, "params": {}, "metrics": {}}
            yield self
            self._save_fallback()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (strings, ints, floats)."""
        if self._mlflow:
            self._mlflow.log_params(params)
        else:
            self._fallback_log.setdefault("params", {}).update(params)
        logger.debug(f"Params: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log scalar metrics (can be called repeatedly during training)."""
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        else:
            for k, v in metrics.items():
                self._fallback_log.setdefault("metrics", {}).setdefault(k, []).append((step, v))

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.log_metrics({name: value}, step=step)

    def log_dict(self, d: Dict, artifact_name: str = "config.json") -> None:
        """Log a dictionary as a JSON artifact."""
        if self._mlflow:
            import json, tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(d, f, indent=2, default=str)
                tmp = f.name
            self._mlflow.log_artifact(tmp, artifact_name)
            os.unlink(tmp)
        else:
            logger.info(f"Dict artifact '{artifact_name}': {d}")

    def log_figure(self, fig, artifact_name: str) -> None:
        """Log a matplotlib figure."""
        if self._mlflow:
            self._mlflow.log_figure(fig, artifact_name)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, dpi=100, bbox_inches="tight")
                logger.info(f"Figure saved locally: {f.name}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a trained model (SB3 / PyTorch)."""
        if self._mlflow:
            try:
                import mlflow.pytorch
                self._mlflow.pytorch.log_model(model.policy, artifact_path)
            except Exception:
                # Fallback: save SB3 model to zip
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    save_path = os.path.join(tmp, "model")
                    model.save(save_path)
                    self._mlflow.log_artifact(save_path + ".zip", artifact_path)
            logger.info(f"Model logged to MLflow: {artifact_path}")
        else:
            logger.info("MLflow not available — save model manually with model.save(path)")

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self._mlflow:
            self._mlflow.set_tags(tags)

    def _save_fallback(self):
        """Save to CSV when MLflow not available."""
        import json
        log_path = Path(self.tracking_uri) / f"{self._fallback_log.get('run_name', 'run')}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(self._fallback_log, f, indent=2, default=str)
        logger.info(f"Run logged to {log_path}")

    def get_best_run(
        self,
        metric: str = "sharpe_ratio",
        mode: str = "max",
    ) -> Optional[Dict]:
        """Retrieve the best run from the experiment by a given metric."""
        if not self._mlflow:
            logger.warning("MLflow not available for run retrieval")
            return None
        try:
            runs = self._mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
                max_results=1,
            )
            if not runs.empty:
                best = runs.iloc[0].to_dict()
                logger.info(f"Best run: {best.get('tags.mlflow.runName')} | {metric}={best.get(f'metrics.{metric}'):.4f}")
                return best
        except Exception as e:
            logger.error(f"Failed to retrieve best run: {e}")
        return None
