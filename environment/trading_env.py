"""
environment/trading_env.py
───────────────────────────
Gymnasium-compatible RL trading environment.

Key design decisions:
  • Continuous action space: portfolio weight per stock ∈ [-1, 1]
    (Phase 1: long-only, so clipped to [0, 1])
  • Observation: stacked feature window + portfolio state
  • Reward: risk-adjusted return (Sharpe-shaped)
  • Episode: one calendar year of trading days
  • Market simulation: realistic cost model (spread + slippage + impact)

Usage:
    env = TradingEnv(feature_store, symbols=["AAPL", "MSFT"], ...)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


# ─── Cost Model ────────────────────────────────────────────────────────────────

class TransactionCostModel:
    """
    Realistic transaction cost model for US equities.
    Components:
      - Commission: fixed bps per trade value
      - Bid-ask spread: estimated from ATR / price
      - Market impact: linear in order size / ADV
    """

    def __init__(
        self,
        commission_bps: float = 5.0,
        spread_bps: float = 3.0,
        impact_factor: float = 0.1,
        min_adv_ratio: float = 0.01,
    ):
        self.commission = commission_bps / 10000
        self.spread = spread_bps / 10000
        self.impact_factor = impact_factor
        self.min_adv_ratio = min_adv_ratio

    def compute_cost(
        self,
        trade_value: float,
        price: float,
        adv: float,  # average daily volume in $
    ) -> float:
        """
        Total cost for a single trade (one-way).
        Returns cost as a fraction of trade_value.
        """
        if trade_value == 0:
            return 0.0

        commission = self.commission
        spread = self.spread / 2  # half-spread per side
        # Square-root market impact (industry standard)
        adv = max(adv, 1e-8)
        size_ratio = abs(trade_value) / adv
        impact = self.impact_factor * np.sqrt(size_ratio)

        total_cost_pct = commission + spread + impact
        return abs(trade_value) * total_cost_pct


# ─── Portfolio State ────────────────────────────────────────────────────────────

class Portfolio:
    """
    Tracks portfolio positions, cash, P&L, and risk metrics.
    All calculations in dollar terms.
    """

    def __init__(self, initial_capital: float, symbols: List[str]):
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.n = len(symbols)
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n)       # shares held
        self.avg_cost = np.zeros(self.n)         # average cost basis per share
        self.total_value_history = [self.initial_capital]
        self.return_history = []
        self.trade_count = 0
        self.total_cost_paid = 0.0

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    @property
    def position_value(self) -> float:
        if not hasattr(self, "_current_prices") or self._current_prices is None:
            return 0.0
        return float(np.sum(self.positions * self._current_prices))

    def update_prices(self, prices: np.ndarray):
        self._current_prices = prices

    def position_weights(self) -> np.ndarray:
        if not hasattr(self, "_current_prices") or self._current_prices is None:
            return np.zeros(self.n)
        tv = self.total_value
        if tv <= 0:
            return np.zeros(self.n)
        values = self.positions * self._current_prices
        return values / tv

    def execute_trade(
        self,
        target_weights: np.ndarray,
        prices: np.ndarray,
        adv: np.ndarray,
        cost_model: TransactionCostModel,
    ) -> float:
        """
        Rebalance to target_weights. Returns total transaction cost paid.
        """
        self._current_prices = prices
        total_value = self.total_value
        target_values = target_weights * total_value
        current_values = self.positions * prices

        total_cost = 0.0

        # Process sells first, then buys (to free up cash)
        deltas = target_values - current_values
        sell_idx = np.where(deltas < 0)[0]
        buy_idx = np.where(deltas > 0)[0]

        for idx in np.concatenate([sell_idx, buy_idx]):
            delta_value = deltas[idx]
            price = prices[idx]
            if price <= 0:
                continue

            shares_delta = delta_value / price
            cost = cost_model.compute_cost(delta_value, price, adv[idx] * price)
            total_cost += cost

            self.positions[idx] += shares_delta
            self.positions[idx] = max(0.0, self.positions[idx])  # no shorts in Phase 1
            self.cash -= (delta_value + np.sign(delta_value) * cost)
            self.trade_count += 1

        self.total_cost_paid += total_cost
        return total_cost

    @property
    def max_drawdown(self) -> float:
        if len(self.total_value_history) < 2:
            return 0.0
        values = np.array(self.total_value_history)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / (peak + 1e-8)
        return float(drawdowns.min())

    @property
    def sharpe_ratio(self) -> float:
        if len(self.return_history) < 5:
            return 0.0
        returns = np.array(self.return_history)
        if returns.std() < 1e-8:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    @property
    def current_return(self) -> float:
        return (self.total_value / self.initial_capital) - 1.0

    def record_step(self):
        value = self.total_value
        prev_value = self.total_value_history[-1]
        step_return = (value - prev_value) / (prev_value + 1e-8)
        self.total_value_history.append(value)
        self.return_history.append(step_return)


# ─── Trading Environment ────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Multi-asset portfolio trading environment.

    Observation space:
        [lookback_window × n_features] + [portfolio_state]
        Flattened to 1D vector.

    Action space:
        Continuous [0, 1]^n_stocks (target portfolio weights)
        Automatically normalized to sum ≤ 1 (remainder in cash).

    Episode lifecycle:
        reset() → randomly sample start date from train set
        step(action) → execute trades, advance one day, return obs/reward
        terminated when: episode_length reached OR bankruptcy
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        feature_store: Dict[str, pd.DataFrame],
        symbols: List[str],
        initial_capital: float = 1_000_000.0,
        lookback_window: int = 60,
        episode_length: int = 252,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 3.0,
        reward_type: str = "sharpe",        # "log_return" | "sharpe" | "calmar"
        drawdown_penalty: float = 0.5,
        turnover_penalty: float = 0.01,
        max_position_pct: float = 0.10,
        mode: str = "train",                 # "train" | "val" | "test"
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.symbols = [s for s in symbols if s in feature_store]
        self.n = len(self.symbols)
        if self.n == 0:
            raise ValueError("No valid symbols found in feature_store")

        self.feature_store = feature_store
        self.initial_capital = initial_capital
        self.lookback = lookback_window
        self.episode_length = episode_length
        self.reward_type = reward_type
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.max_position_pct = max_position_pct
        self.mode = mode
        self.render_mode = render_mode

        self.cost_model = TransactionCostModel(
            commission_bps=transaction_cost_bps,
            spread_bps=slippage_bps,
        )
        self.portfolio = Portfolio(initial_capital, self.symbols)

        # Determine feature dimension from first symbol
        first_df = next(iter(feature_store.values()))
        self.raw_cols = {"open", "high", "low", "close", "volume", "adj_close"}
        self.feature_cols = [c for c in first_df.columns if c not in self.raw_cols]
        self.n_features = len(self.feature_cols)

        # Build aligned date index across all symbols
        self.dates = self._build_date_index()

        # Split dates by mode
        n_dates = len(self.dates)
        self.train_dates = self.dates[:int(n_dates * 0.70)]
        self.val_dates = self.dates[int(n_dates * 0.70):int(n_dates * 0.85)]
        self.test_dates = self.dates[int(n_dates * 0.85):]

        # Observation: [lookback × n_features × n_stocks] + [portfolio_state]
        # Portfolio state: weights(n), cash_ratio, total_return, drawdown, sharpe, vix
        n_portfolio_state = self.n + 5
        obs_dim = self.lookback * self.n_features * self.n + n_portfolio_state
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: target weight per stock, [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

        # State
        self.current_step = 0
        self.episode_start_idx = 0
        self._current_obs = None
        self._prev_weights = np.zeros(self.n)

        logger.info(
            f"TradingEnv: {self.n} stocks | "
            f"obs_dim={obs_dim} | lookback={lookback_window} | "
            f"features={self.n_features} | dates={len(self.dates)}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Sample episode start date
        active_dates = {
            "train": self.train_dates,
            "val": self.val_dates,
            "test": self.test_dates,
        }[self.mode]

        max_start = len(active_dates) - self.episode_length - self.lookback
        if max_start <= 0:
            raise ValueError(f"Not enough dates for mode='{self.mode}'. Reduce episode_length.")

        if self.mode == "train":
            self.episode_start_idx = self.np_random.integers(self.lookback, max_start)
        else:
            # Val/test: evaluate from start (no randomness)
            self.episode_start_idx = self.lookback

        # Map to global date index
        start_date = active_dates[self.episode_start_idx]
        self.global_start = self.dates.get_loc(start_date)
        self.current_step = 0

        self.portfolio.reset()
        self._prev_weights = np.zeros(self.n)

        # Initialize prices so observation works immediately after reset
        initial_prices = self._get_prices(self.global_start)
        self.portfolio.update_prices(initial_prices)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step (one day):
          1. Apply action (target weights) → execute trades
          2. Advance to next day
          3. Compute reward
          4. Return (obs, reward, terminated, truncated, info)
        """
        # ── 1. Get current prices ──────────────────────────────────────────────
        current_prices = self._get_prices(self.global_start + self.current_step)
        adv = self._get_adv(self.global_start + self.current_step)

        self.portfolio.update_prices(current_prices)

        # ── 2. Normalize and constrain action ─────────────────────────────────
        target_weights = self._process_action(action)

        # ── 3. Execute trades ─────────────────────────────────────────────────
        prev_value = self.portfolio.total_value
        prev_weights = self.portfolio.position_weights().copy()

        cost = self.portfolio.execute_trade(
            target_weights, current_prices, adv, self.cost_model
        )

        # ── 4. Advance to next day ────────────────────────────────────────────
        self.current_step += 1
        next_prices = self._get_prices(self.global_start + self.current_step)
        self.portfolio.update_prices(next_prices)
        self.portfolio.record_step()

        # ── 5. Compute reward ─────────────────────────────────────────────────
        reward = self._compute_reward(
            prev_value=prev_value,
            cost=cost,
            prev_weights=prev_weights,
            target_weights=target_weights,
        )

        # ── 6. Check termination ──────────────────────────────────────────────
        terminated = self._check_terminated()
        truncated = self.current_step >= self.episode_length

        obs = self._get_observation()
        info = self._get_info()
        info["cost"] = cost
        info["trade_count"] = self.portfolio.trade_count

        self._prev_weights = target_weights.copy()
        return obs, reward, terminated, truncated, info

    # ── Reward Functions ───────────────────────────────────────────────────────

    def _compute_reward(
        self,
        prev_value: float,
        cost: float,
        prev_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> float:
        portfolio = self.portfolio
        curr_value = portfolio.total_value

        # Log return
        log_ret = np.log(curr_value / (prev_value + 1e-8))

        if self.reward_type == "log_return":
            reward = log_ret

        elif self.reward_type == "sharpe":
            # Smooth Sharpe reward: mean / std of recent returns
            if len(portfolio.return_history) >= 5:
                recent = np.array(portfolio.return_history[-21:])
                std = recent.std()
                reward = recent.mean() / (std + 1e-6)
            else:
                reward = log_ret

        elif self.reward_type == "calmar":
            if len(portfolio.return_history) >= 10:
                ann_ret = np.mean(portfolio.return_history[-63:]) * 252
                max_dd = abs(portfolio.max_drawdown) + 1e-6
                reward = ann_ret / max_dd
            else:
                reward = log_ret
        else:
            reward = log_ret

        # Drawdown penalty
        dd = abs(portfolio.max_drawdown)
        if dd > 0.05:
            reward -= self.drawdown_penalty * (dd - 0.05)

        # Turnover penalty (discourage excessive trading)
        turnover = np.sum(np.abs(target_weights - prev_weights))
        reward -= self.turnover_penalty * turnover

        return float(np.clip(reward, -10.0, 10.0))

    # ── Observation Builder ────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector."""
        step = self.global_start + self.current_step
        window_start = step - self.lookback

        feature_windows = []
        for sym in self.symbols:
            df = self.feature_store[sym]
            dates_in_store = df.index
            # Map global step to symbol's date index
            sym_dates = dates_in_store[
                dates_in_store.isin(self.dates[window_start:step])
            ]
            if len(sym_dates) < self.lookback:
                # Pad with zeros if not enough history
                window = np.zeros((self.lookback, self.n_features))
            else:
                sym_dates = sym_dates[-self.lookback:]
                window = df.loc[sym_dates, self.feature_cols].values[-self.lookback:]
                if len(window) < self.lookback:
                    pad = np.zeros((self.lookback - len(window), self.n_features))
                    window = np.vstack([pad, window])

            window = np.nan_to_num(window, nan=0.0, posinf=5.0, neginf=-5.0)
            feature_windows.append(window.flatten())

        features = np.concatenate(feature_windows)

        # Portfolio state vector
        weights = self.portfolio.position_weights()
        cash_ratio = self.portfolio.cash / (self.portfolio.total_value + 1e-8)
        total_ret = self.portfolio.current_return
        drawdown = self.portfolio.max_drawdown
        sharpe = np.clip(self.portfolio.sharpe_ratio, -5, 5)
        step_pct = self.current_step / self.episode_length  # episode progress

        portfolio_state = np.concatenate([
            weights,
            [cash_ratio, total_ret, drawdown, sharpe, step_pct]
        ])

        obs = np.concatenate([features, portfolio_state]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert raw action to valid portfolio weights.
        - Clip to [0, 1] (long-only)
        - Enforce max position per stock
        - Normalize so weights sum ≤ 1
        """
        weights = np.clip(action, 0.0, self.max_position_pct)
        total = weights.sum()
        if total > 1.0:
            weights = weights / total
        return weights.astype(np.float64)

    def _get_prices(self, global_idx: int) -> np.ndarray:
        date = self.dates[min(global_idx, len(self.dates) - 1)]
        prices = []
        for sym in self.symbols:
            df = self.feature_store[sym]
            if date in df.index:
                prices.append(float(df.loc[date, "close"]))
            else:
                # Use last available price
                avail = df[df.index <= date]
                prices.append(float(avail["close"].iloc[-1]) if not avail.empty else 100.0)
        return np.array(prices)

    def _get_adv(self, global_idx: int, window: int = 20) -> np.ndarray:
        """Average daily volume (dollar) for market impact calculation."""
        date = self.dates[min(global_idx, len(self.dates) - 1)]
        adv = []
        for sym in self.symbols:
            df = self.feature_store[sym]
            avail = df[df.index <= date].tail(window)
            if not avail.empty and "volume" in avail and "close" in avail:
                avg_vol = (avail["volume"] * avail["close"]).mean()
                adv.append(float(avg_vol))
            else:
                adv.append(1e7)  # default $10M ADV
        return np.array(adv)

    def _check_terminated(self) -> bool:
        """Episode ends early on bankruptcy or catastrophic loss."""
        tv = self.portfolio.total_value
        if tv < self.initial_capital * 0.1:
            logger.info("Episode terminated: bankruptcy (<10% capital)")
            return True
        if self.portfolio.max_drawdown < -0.25:
            logger.info("Episode terminated: max drawdown > 25%")
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        return {
            "total_value": self.portfolio.total_value,
            "total_return": self.portfolio.current_return,
            "max_drawdown": self.portfolio.max_drawdown,
            "sharpe_ratio": self.portfolio.sharpe_ratio,
            "step": self.current_step,
            "date": str(self.dates[self.global_start + self.current_step]
                       if self.global_start + self.current_step < len(self.dates) else "end"),
        }

    def _build_date_index(self) -> pd.DatetimeIndex:
        """Common trading dates across all symbols."""
        date_sets = [set(df.index) for df in self.feature_store.values()]
        common = date_sets[0]
        for s in date_sets[1:]:
            common = common.intersection(s)
        return pd.DatetimeIndex(sorted(common))

    # ── Rendering ──────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            p = self.portfolio
            print(
                f"Step {self.current_step:3d} | "
                f"Value: ${p.total_value:>12,.0f} | "
                f"Return: {p.current_return:+.2%} | "
                f"Drawdown: {p.max_drawdown:.2%} | "
                f"Sharpe: {p.sharpe_ratio:.2f}"
            )


# ─── Vectorized Environment Factory ───────────────────────────────────────────

def make_env(feature_store, symbols, mode="train", seed=42, **env_kwargs):
    """Factory function for creating vectorized environments with Ray/SB3."""
    def _init():
        env = TradingEnv(feature_store, symbols, mode=mode, **env_kwargs)
        env.reset(seed=seed)
        return env
    return _init
