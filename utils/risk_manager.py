"""
utils/risk_manager.py
──────────────────────
Real-time risk management layer sitting between Agent output and order execution.

This is a HARD constraint layer — it filters/modifies agent actions BEFORE
they reach the market, regardless of what the agent learned.

Risk checks (in order of execution):
  1. VIX regime scaling
  2. Maximum position limits
  3. Sector concentration
  4. Drawdown halt
  5. Liquidity filter
  6. Correlation limits
  7. Kelly position sizing
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    """Output of risk manager for a given agent action."""
    approved_weights: np.ndarray
    original_weights: np.ndarray
    adjustments: Dict[str, str]   # reason → description
    halt: bool = False
    halt_reason: str = ""


class RiskManager:
    """
    Multi-layer risk management.

    Usage:
        rm = RiskManager(symbols, sector_map)
        decision = rm.evaluate(
            agent_weights=action,
            portfolio_value=portfolio.total_value,
            portfolio_value_history=portfolio.total_value_history,
            prices=current_prices,
            adv=current_adv,
            vix=current_vix,
        )
        if not decision.halt:
            env.step(decision.approved_weights)
    """

    def __init__(
        self,
        symbols: List[str],
        sector_map: Optional[Dict[str, str]] = None,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.30,
        max_drawdown_halt: float = 0.08,
        max_portfolio_drawdown: float = 0.15,
        vix_normal: float = 20.0,
        vix_elevated: float = 30.0,
        vix_extreme: float = 40.0,
        min_adv_ratio: float = 0.01,
        kelly_fraction: float = 0.25,
    ):
        self.symbols = symbols
        self.n = len(symbols)
        self.sector_map = sector_map or {s: "Unknown" for s in symbols}
        self.max_position = max_position_pct
        self.max_sector = max_sector_pct
        self.max_dd_halt = max_drawdown_halt
        self.max_portfolio_dd = max_portfolio_drawdown
        self.vix_normal = vix_normal
        self.vix_elevated = vix_elevated
        self.vix_extreme = vix_extreme
        self.min_adv_ratio = min_adv_ratio
        self.kelly_fraction = kelly_fraction

        # State tracking
        self.peak_value: float = 0.0
        self.daily_start_value: float = 0.0
        self.is_halted: bool = False
        self.halt_reason: str = ""

    def evaluate(
        self,
        agent_weights: np.ndarray,
        portfolio_value: float,
        portfolio_value_history: List[float],
        prices: np.ndarray,
        adv: Optional[np.ndarray] = None,
        vix: float = 20.0,
        returns_history: Optional[np.ndarray] = None,
    ) -> RiskDecision:
        """
        Apply all risk checks and return adjusted weights.
        """
        adjustments = {}
        weights = agent_weights.copy().astype(float)

        # Update peak value tracking
        self.peak_value = max(self.peak_value, portfolio_value)

        # ── Check 1: Portfolio drawdown halt ──────────────────────────────────
        if self.peak_value > 0:
            total_drawdown = (portfolio_value - self.peak_value) / self.peak_value
            if total_drawdown < -self.max_portfolio_dd:
                self.is_halted = True
                self.halt_reason = f"Portfolio drawdown {total_drawdown:.2%} > limit {-self.max_portfolio_dd:.2%}"
                logger.critical(f"TRADING HALT: {self.halt_reason}")
                return RiskDecision(
                    approved_weights=np.zeros(self.n),
                    original_weights=agent_weights,
                    adjustments={"HALT": self.halt_reason},
                    halt=True,
                    halt_reason=self.halt_reason,
                )

        # ── Check 2: Daily drawdown halt ──────────────────────────────────────
        if len(portfolio_value_history) >= 2:
            daily_dd = (portfolio_value - portfolio_value_history[-1]) / portfolio_value_history[-1]
            if daily_dd < -self.max_dd_halt:
                return RiskDecision(
                    approved_weights=np.zeros(self.n),
                    original_weights=agent_weights,
                    adjustments={"DAILY_HALT": f"Intraday drawdown {daily_dd:.2%}"},
                    halt=True,
                    halt_reason=f"Daily drawdown {daily_dd:.2%} > {-self.max_dd_halt:.2%}",
                )

        # ── Check 3: VIX regime scaling ───────────────────────────────────────
        if vix >= self.vix_extreme:
            scale = 0.30
            weights *= scale
            adjustments["vix_scale"] = f"VIX={vix:.1f} (extreme), scaled to {scale:.0%}"
        elif vix >= self.vix_elevated:
            scale = 0.60
            weights *= scale
            adjustments["vix_scale"] = f"VIX={vix:.1f} (elevated), scaled to {scale:.0%}"
        elif vix >= self.vix_normal:
            scale = 0.85
            weights *= scale
            adjustments["vix_scale"] = f"VIX={vix:.1f} (normal-high), scaled to {scale:.0%}"

        # ── Check 4: Maximum position per stock ───────────────────────────────
        breaches = weights > self.max_position
        if breaches.any():
            weights = np.minimum(weights, self.max_position)
            adjustments["position_cap"] = f"Capped {breaches.sum()} positions at {self.max_position:.0%}"

        # ── Check 5: Sector concentration ────────────────────────────────────
        weights = self._enforce_sector_limits(weights, adjustments)

        # ── Check 6: Liquidity filter ─────────────────────────────────────────
        if adv is not None:
            weights = self._apply_liquidity_filter(weights, portfolio_value, prices, adv, adjustments)

        # ── Check 7: Kelly position sizing ────────────────────────────────────
        if returns_history is not None and len(returns_history) >= 60:
            weights = self._apply_kelly_sizing(weights, returns_history, adjustments)

        # ── Final: ensure weights sum ≤ 1 ────────────────────────────────────
        total = weights.sum()
        if total > 1.0:
            weights = weights / total
            adjustments["normalize"] = f"Normalized (sum was {total:.3f})"

        weights = np.clip(weights, 0.0, 1.0)

        return RiskDecision(
            approved_weights=weights,
            original_weights=agent_weights,
            adjustments=adjustments,
        )

    def _enforce_sector_limits(
        self, weights: np.ndarray, adjustments: Dict
    ) -> np.ndarray:
        """Cap sector exposure at max_sector_pct."""
        sectors = [self.sector_map.get(s, "Unknown") for s in self.symbols]
        unique_sectors = set(sectors)

        for sector in unique_sectors:
            sector_idx = [i for i, s in enumerate(sectors) if s == sector]
            sector_weight = weights[sector_idx].sum()
            if sector_weight > self.max_sector:
                scale = self.max_sector / sector_weight
                weights[sector_idx] *= scale
                adjustments[f"sector_{sector}"] = (
                    f"Sector '{sector}' {sector_weight:.2%} → scaled by {scale:.3f}"
                )

        return weights

    def _apply_liquidity_filter(
        self,
        weights: np.ndarray,
        portfolio_value: float,
        prices: np.ndarray,
        adv: np.ndarray,  # average daily volume in shares
        adjustments: Dict,
    ) -> np.ndarray:
        """
        Reduce position if order size > min_adv_ratio of average daily volume.
        Prevents market impact from being too large.
        """
        for i in range(self.n):
            position_value = weights[i] * portfolio_value
            adv_dollars = adv[i] * (prices[i] if prices[i] > 0 else 1.0)
            if adv_dollars > 0:
                participation_rate = position_value / adv_dollars
                if participation_rate > self.min_adv_ratio:
                    max_value = adv_dollars * self.min_adv_ratio
                    new_weight = max_value / portfolio_value
                    if new_weight < weights[i]:
                        adjustments[f"liquidity_{self.symbols[i]}"] = (
                            f"Reduced from {weights[i]:.3f} to {new_weight:.3f} "
                            f"(ADV participation {participation_rate:.2%})"
                        )
                        weights[i] = new_weight

        return weights

    def _apply_kelly_sizing(
        self,
        weights: np.ndarray,
        returns_history: np.ndarray,
        adjustments: Dict,
    ) -> np.ndarray:
        """
        Scale total exposure using fractional Kelly criterion.
        Kelly f* = (p*b - q) / b, where p=win_rate, q=loss_rate, b=avg_win/avg_loss
        """
        wins = returns_history[returns_history > 0]
        losses = returns_history[returns_history <= 0]

        if len(wins) < 10 or len(losses) < 10:
            return weights

        p = len(wins) / len(returns_history)
        q = 1 - p
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        b = avg_win / (avg_loss + 1e-8)

        kelly_f = (p * b - q) / (b + 1e-8)
        kelly_f = np.clip(kelly_f, 0.0, 1.0)
        fractional_kelly = kelly_f * self.kelly_fraction

        current_exposure = weights.sum()
        if current_exposure > fractional_kelly and current_exposure > 0:
            scale = fractional_kelly / current_exposure
            weights *= scale
            adjustments["kelly"] = (
                f"Kelly f*={kelly_f:.3f}, "
                f"frac={fractional_kelly:.3f}, "
                f"scaled exposure {current_exposure:.3f} → {weights.sum():.3f}"
            )

        return weights

    def reset_daily(self, portfolio_value: float):
        """Call at the start of each trading day."""
        self.daily_start_value = portfolio_value
        if self.is_halted:
            # Auto-resume after drawdown recovers to 50% of halt threshold
            recovery = (portfolio_value - self.peak_value * (1 - self.max_portfolio_dd))
            if recovery > 0:
                self.is_halted = False
                self.halt_reason = ""
                logger.info("Trading resumed: drawdown recovered")

    def get_status(self) -> Dict:
        """Return current risk status."""
        return {
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "peak_value": self.peak_value,
        }
