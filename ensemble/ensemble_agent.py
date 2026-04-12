"""
ensemble/ensemble_agent.py
───────────────────────────
Multi-model ensemble for more robust trading decisions.

Strategies:
  WeightedVoting    — weighted average of all model actions
  ConfidenceGating  — only include models above confidence threshold
  UCB1Ensemble      — Upper Confidence Bound: adaptively weights models
                       based on recent performance (multi-armed bandit)
  DisagreementHalt  — halt if models strongly disagree (high variance = uncertainty)

Why ensemble?
  - Reduces variance / overfitting of single model
  - Models trained on different regimes complement each other
  - Disagreement metric = uncertainty signal → reduce position sizing
  - Natural A/B testing: see which model contributes most over time
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Model Wrapper ────────────────────────────────────────────────────────────

@dataclass
class EnsembleMember:
    name: str
    model: object                       # SB3 model with .predict()
    weight: float = 1.0                 # voting weight
    active: bool = True
    recent_sharpes: List[float] = field(default_factory=list)
    n_predictions: int = 0
    last_action: Optional[np.ndarray] = None

    # UCB1 state
    ucb_total_reward: float = 0.0
    ucb_n_pulls: int = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        self.last_action = np.clip(action, 0, 1).astype(np.float64)
        self.n_predictions += 1
        return self.last_action

    def update_performance(self, sharpe: float):
        self.recent_sharpes.append(sharpe)
        if len(self.recent_sharpes) > 21:
            self.recent_sharpes = self.recent_sharpes[-21:]
        self.ucb_total_reward += max(sharpe, 0)
        self.ucb_n_pulls += 1

    @property
    def mean_sharpe(self) -> float:
        if not self.recent_sharpes:
            return 0.0
        return float(np.mean(self.recent_sharpes))

    @property
    def ucb_avg_reward(self) -> float:
        if self.ucb_n_pulls == 0:
            return 0.0
        return self.ucb_total_reward / self.ucb_n_pulls


# ─── Ensemble Strategies ──────────────────────────────────────────────────────

class WeightedVotingEnsemble:
    """
    Weighted average of model actions.
    Weights can be static or dynamically updated by performance.
    """

    def __init__(self, members: List[EnsembleMember], dynamic_weights: bool = True):
        self.members = [m for m in members if m.active]
        self.dynamic = dynamic_weights

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        actions = []
        weights = []

        for m in self.members:
            if not m.active:
                continue
            action = m.predict(obs, deterministic)
            actions.append(action)

            w = m.weight
            if self.dynamic and m.ucb_n_pulls > 0:
                # Soft-max weight based on recent Sharpe
                w = max(m.mean_sharpe, 0.01)
            weights.append(w)

        if not actions:
            n = obs.shape[0] if obs.ndim > 1 else 1
            return np.zeros(n), {}

        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        ensemble_action = np.average(actions, axis=0, weights=weights)

        info = {
            "n_models": len(actions),
            "weights": {m.name: round(float(w), 4) for m, w in zip(self.members, weights)},
            "std_actions": float(np.std(actions, axis=0).mean()),
        }
        return ensemble_action, info


class UCB1Ensemble:
    """
    Upper Confidence Bound ensemble.
    Adaptively selects the best model with exploration bonus.
    At each step, picks the model with highest UCB score:
        UCB = avg_reward + c * sqrt(log(total_pulls) / n_pulls_i)

    This naturally:
    - Exploits consistently good models
    - Explores under-tried models
    - Self-corrects when a model degrades
    """

    def __init__(self, members: List[EnsembleMember], c: float = 1.0):
        self.members = [m for m in members if m.active]
        self.c = c
        self.total_pulls = 0

    def _ucb_score(self, member: EnsembleMember) -> float:
        if member.ucb_n_pulls == 0:
            return float("inf")  # Always try unexplored models first
        exploitation = member.ucb_avg_reward
        exploration = self.c * np.sqrt(np.log(self.total_pulls + 1) / member.ucb_n_pulls)
        return exploitation + exploration

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        if not self.members:
            return np.zeros(1), {}

        scores = [self._ucb_score(m) for m in self.members]
        best_idx = int(np.argmax(scores))
        best_member = self.members[best_idx]

        action = best_member.predict(obs, deterministic)
        self.total_pulls += 1

        info = {
            "selected_model": best_member.name,
            "ucb_scores": {m.name: round(float(s), 4) for m, s in zip(self.members, scores)},
            "n_predictions": best_member.n_predictions,
        }
        return action, info


class DisagreementAwareEnsemble:
    """
    Weighted voting + disagreement-based position scaling.

    When models strongly disagree (high std of actions):
    → Scale down position size (uncertainty = caution)

    When models strongly agree:
    → Full position (high conviction)
    """

    def __init__(
        self,
        members: List[EnsembleMember],
        max_disagreement: float = 0.15,   # if std(actions) > this, scale down
        min_scale: float = 0.3,            # minimum position scale at max disagreement
    ):
        self.members = [m for m in members if m.active]
        self.max_disagreement = max_disagreement
        self.min_scale = min_scale
        self._voting = WeightedVotingEnsemble(members, dynamic_weights=True)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        ensemble_action, info = self._voting.predict(obs, deterministic)
        disagreement = info.get("std_actions", 0.0)

        # Scale down positions proportionally to disagreement
        if disagreement > 0:
            scale = max(
                self.min_scale,
                1.0 - (disagreement / self.max_disagreement) * (1.0 - self.min_scale)
            )
            ensemble_action = ensemble_action * scale
            info["position_scale"] = round(scale, 4)
            info["disagreement"] = round(disagreement, 4)

            if disagreement > self.max_disagreement * 0.8:
                logger.info(f"High model disagreement ({disagreement:.3f}) → scaled to {scale:.0%}")

        return ensemble_action, info

    def update_all(self, sharpe: float):
        for m in self.members:
            m.update_performance(sharpe)


# ─── Ensemble Manager ─────────────────────────────────────────────────────────

class EnsembleManager:
    """
    Manages a pool of models with A/B testing and lifecycle management.

    Lifecycle:
      production   — active, used for trading decisions
      shadow       — running in parallel but NOT trading (observation only)
      archived     — inactive, kept for rollback
      challenger   — newly trained, being evaluated before promotion
    """

    def __init__(self, strategy: str = "disagreement"):
        """
        strategy: "weighted" | "ucb1" | "disagreement"
        """
        self.strategy = strategy
        self._members: Dict[str, EnsembleMember] = {}
        self._ensemble: Optional[object] = None
        self._performance_log: List[Dict] = []

    def add_model(
        self,
        name: str,
        model,
        role: str = "production",   # "production" | "shadow" | "challenger"
        weight: float = 1.0,
    ) -> "EnsembleManager":
        member = EnsembleMember(
            name=name,
            model=model,
            weight=weight,
            active=(role == "production"),
        )
        self._members[name] = member
        logger.info(f"Ensemble: added '{name}' ({role}, weight={weight})")
        self._rebuild_ensemble()
        return self

    def remove_model(self, name: str) -> "EnsembleManager":
        if name in self._members:
            del self._members[name]
            self._rebuild_ensemble()
            logger.info(f"Ensemble: removed '{name}'")
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get ensemble prediction."""
        if self._ensemble is None or not self._active_members:
            raise RuntimeError("No active ensemble members")
        return self._ensemble.predict(obs, deterministic)

    def predict_with_shadows(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get production prediction + shadow model predictions for comparison.
        Used for A/B testing without risking real capital.
        """
        production_action, info = self.predict(obs)

        shadow_actions = {}
        for name, m in self._members.items():
            if not m.active:
                shadow_action = m.predict(obs, deterministic=True)
                shadow_actions[name] = shadow_action

        return production_action, shadow_actions

    def update_performance(self, episode_sharpe: float, model_name: str = None):
        """Update performance tracking for all or specific model."""
        for name, m in self._members.items():
            if model_name is None or name == model_name:
                m.update_performance(episode_sharpe)

        self._performance_log.append({
            "sharpe": episode_sharpe,
            "model": model_name or "ensemble",
            "rankings": {n: m.mean_sharpe for n, m in self._members.items()},
        })

        # Auto-reweight in weighted strategy
        if self.strategy in ("weighted", "disagreement"):
            self._rebalance_weights()

    def _rebalance_weights(self):
        """Softmax reweighting based on recent performance."""
        members = self._active_members
        if not members:
            return

        sharpes = np.array([max(m.mean_sharpe, 0.01) for m in members])
        # Softmax with temperature
        exp_sharpes = np.exp(sharpes / 0.5)
        weights = exp_sharpes / exp_sharpes.sum()

        for m, w in zip(members, weights):
            m.weight = float(w)

    def promote_challenger(self, name: str):
        """Promote a challenger model to production (deactivate others if needed)."""
        if name not in self._members:
            raise ValueError(f"Model '{name}' not in ensemble")
        self._members[name].active = True
        logger.info(f"Ensemble: '{name}' promoted to production")
        self._rebuild_ensemble()

    def get_leaderboard(self) -> List[Dict]:
        """Return models ranked by recent performance."""
        return sorted(
            [{"name": n, "mean_sharpe": m.mean_sharpe,
              "n_predictions": m.n_predictions, "active": m.active}
             for n, m in self._members.items()],
            key=lambda x: -x["mean_sharpe"],
        )

    def _rebuild_ensemble(self):
        active = self._active_members
        if not active:
            self._ensemble = None
            return

        if self.strategy == "ucb1":
            self._ensemble = UCB1Ensemble(active)
        elif self.strategy == "disagreement":
            self._ensemble = DisagreementAwareEnsemble(active)
        else:
            self._ensemble = WeightedVotingEnsemble(active)

    @property
    def _active_members(self) -> List[EnsembleMember]:
        return [m for m in self._members.values() if m.active]
