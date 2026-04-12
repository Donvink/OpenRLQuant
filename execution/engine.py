"""
execution/engine.py
────────────────────
Real-time trading execution engine.

Responsibilities:
  1. Market open check (NYSE calendar)
  2. Feature computation from live market data
  3. Agent inference → target weights
  4. Risk manager filter
  5. Order submission via broker
  6. Position reconciliation
  7. P&L tracking and logging

Two operation modes:
  LIVE    — runs on market schedule, actual order submission
  BACKTEST — replay historical data, no real orders

Event loop:
  ┌─────────────────────────────────────────────────┐
  │  Market Open                                     │
  │    ↓                                            │
  │  Fetch latest prices + news (every N minutes)   │
  │    ↓                                            │
  │  Build observation vector                        │
  │    ↓                                            │
  │  Agent.predict(obs) → action                    │
  │    ↓                                            │
  │  RiskManager.evaluate(action) → safe_weights    │
  │    ↓                                            │
  │  Broker.submit_target_weights(safe_weights)     │
  │    ↓                                            │
  │  Log P&L, positions, metrics                    │
  │    ↓                                            │
  │  Wait for next cycle (e.g., every 30 min)       │
  └─────────────────────────────────────────────────┘
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from execution.broker import BaseBroker, PaperBroker, Order
from utils.risk_manager import RiskManager

logger = logging.getLogger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

@dataclass
class EngineState:
    """Complete snapshot of engine state at any point in time."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)   # symbol -> qty
    weights: Dict[str, float] = field(default_factory=dict)     # symbol -> weight
    prices: Dict[str, float] = field(default_factory=dict)
    total_return: float = 0.0
    daily_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades_today: int = 0
    is_halted: bool = False
    halt_reason: str = ""
    cycle_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": round(self.portfolio_value, 2),
            "cash": round(self.cash, 2),
            "total_return": round(self.total_return, 4),
            "daily_return": round(self.daily_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "n_trades_today": self.n_trades_today,
            "n_positions": len([v for v in self.positions.values() if abs(v) > 0.001]),
            "is_halted": self.is_halted,
            "cycle_count": self.cycle_count,
        }


# ─── Live Data Feed ────────────────────────────────────────────────────────────

class MarketDataFeed:
    """
    Fetches real-time and recent market data for feature construction.
    Maintains a rolling window of OHLCV bars per symbol.
    """

    def __init__(
        self,
        symbols: List[str],
        lookback_days: int = 120,     # days of history to maintain
        broker: Optional[BaseBroker] = None,
    ):
        self.symbols = symbols
        self.lookback = lookback_days
        self.broker = broker
        self._bars: Dict[str, pd.DataFrame] = {}  # symbol -> OHLCV DataFrame
        self._latest_prices: Dict[str, float] = {}

    def initialize(self, start_date: str = None) -> None:
        """Pre-load historical bars for feature computation warm-up."""
        from data.market_data import MarketDataLoader
        loader = MarketDataLoader(use_cache=True)
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        start = start_date or (pd.Timestamp.now() - pd.DateOffset(days=self.lookback * 1.5)).strftime("%Y-%m-%d")

        logger.info(f"Initializing data feed: {self.symbols} [{start} → {end}]")
        self._bars = loader.get_ohlcv_universe(self.symbols, start, end)
        for sym, df in self._bars.items():
            if not df.empty:
                self._latest_prices[sym] = float(df["close"].iloc[-1])
        logger.info(f"Data feed ready: {len(self._bars)} symbols")

    def update_latest_prices(self) -> Dict[str, float]:
        """Fetch current market prices from broker or fallback."""
        if self.broker and hasattr(self.broker, "get_latest_prices"):
            try:
                prices = self.broker.get_latest_prices(self.symbols)
                if prices:
                    self._latest_prices.update(prices)
                    return self._latest_prices
            except Exception as e:
                logger.warning(f"Live price fetch failed: {e} — using last known prices")

        return self._latest_prices

    def get_bars(self, symbol: str, n: int = 100) -> pd.DataFrame:
        """Get the last N bars for a symbol."""
        df = self._bars.get(symbol, pd.DataFrame())
        return df.tail(n) if not df.empty else df

    def append_today_bar(self, symbol: str, price: float, volume: float = 1e6):
        """Synthesize a partial bar for the current trading day."""
        if symbol not in self._bars or self._bars[symbol].empty:
            return
        today = pd.Timestamp.now().normalize()
        bar = pd.DataFrame({
            "open": [price], "high": [price], "low": [price],
            "close": [price], "volume": [volume],
        }, index=[today])
        bar.index.name = "date"
        self._bars[symbol] = pd.concat([self._bars[symbol], bar]).drop_duplicates()


# ─── Observation Builder ───────────────────────────────────────────────────────

class LiveObservationBuilder:
    """
    Constructs the agent's observation vector from live market data.
    Must match EXACTLY the observation format used during training.
    """

    def __init__(
        self,
        symbols: List[str],
        n_features: int,
        lookback: int,
        feature_pipeline=None,
    ):
        self.symbols = symbols
        self.n_features = n_features
        self.lookback = lookback
        self.pipeline = feature_pipeline

        # Portfolio state buffer
        self._weights = np.zeros(len(symbols))
        self._portfolio_value = 0.0
        self._peak_value = 0.0
        self._return_history: List[float] = []

    def build(
        self,
        data_feed: MarketDataFeed,
        portfolio_value: float,
        positions: Dict[str, float],  # symbol -> qty
        prices: Dict[str, float],
    ) -> np.ndarray:
        """Build flat observation vector for agent inference."""
        # Update portfolio state
        prev_value = self._portfolio_value if self._portfolio_value > 0 else portfolio_value
        if prev_value > 0:
            self._return_history.append((portfolio_value - prev_value) / prev_value)
        self._portfolio_value = portfolio_value
        self._peak_value = max(self._peak_value, portfolio_value)

        # Compute current weights
        total_pos_value = sum(
            positions.get(sym, 0) * prices.get(sym, 0) for sym in self.symbols
        )
        self._weights = np.array([
            positions.get(sym, 0) * prices.get(sym, 0) / (portfolio_value + 1e-8)
            for sym in self.symbols
        ])

        # Build temporal feature windows per stock
        feature_windows = []
        for sym in self.symbols:
            bars = data_feed.get_bars(sym, self.lookback + 60)
            if bars.empty or len(bars) < self.lookback:
                window = np.zeros((self.lookback, self.n_features))
            else:
                if self.pipeline:
                    try:
                        feat_df = self.pipeline.technical.compute(bars.copy())
                        # Select feature columns (no raw OHLCV)
                        raw = {"open","high","low","close","volume","adj_close"}
                        feat_cols = [c for c in feat_df.columns if c not in raw][:self.n_features]
                        feat_df = feat_df[feat_cols].tail(self.lookback)
                        window = feat_df.values
                        if len(window) < self.lookback:
                            pad = np.zeros((self.lookback - len(window), self.n_features))
                            window = np.vstack([pad, window])
                    except Exception:
                        window = np.zeros((self.lookback, self.n_features))
                else:
                    window = np.zeros((self.lookback, self.n_features))

            window = np.nan_to_num(window, nan=0.0, posinf=5.0, neginf=-5.0)
            window = np.clip(window, -10.0, 10.0)
            feature_windows.append(window.flatten())

        temporal_obs = np.concatenate(feature_windows)

        # Portfolio state vector (must match TradingEnv._get_observation)
        cash_ratio = max(0.0, 1.0 - self._weights.sum())
        total_ret = (portfolio_value / (self._peak_value or portfolio_value)) - 1.0
        max_dd = (portfolio_value - self._peak_value) / (self._peak_value + 1e-8) if self._peak_value else 0.0
        sharpe = self._compute_sharpe()
        step_pct = min(len(self._return_history) / 252, 1.0)

        portfolio_state = np.concatenate([
            self._weights,
            [cash_ratio, total_ret, max_dd, sharpe, step_pct],
        ])

        obs = np.concatenate([temporal_obs, portfolio_state]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _compute_sharpe(self) -> float:
        if len(self._return_history) < 5:
            return 0.0
        r = np.array(self._return_history[-63:])
        return float(np.clip(r.mean() / (r.std() + 1e-8) * np.sqrt(252), -5, 5))


# ─── Execution Engine ─────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Main trading execution engine.
    Coordinates: data feed → agent inference → risk check → order execution.
    """

    def __init__(
        self,
        agent,                        # trained SB3 model (.predict method)
        broker: BaseBroker,
        symbols: List[str],
        risk_manager: RiskManager,
        obs_builder: LiveObservationBuilder,
        data_feed: MarketDataFeed,
        cycle_interval_seconds: int = 1800,   # 30 min default
        initial_capital: float = 100_000.0,
        on_state_update: Optional[Callable] = None,  # callback for monitoring
        dry_run: bool = False,        # if True, compute but don't submit orders
    ):
        self.agent = agent
        self.broker = broker
        self.symbols = symbols
        self.risk = risk_manager
        self.obs_builder = obs_builder
        self.data_feed = data_feed
        self.cycle_interval = cycle_interval_seconds
        self.initial_capital = initial_capital
        self.on_state_update = on_state_update
        self.dry_run = dry_run

        self.state = EngineState()
        self._running = False
        self._value_history: List[float] = [initial_capital]
        self._day_start_value: float = initial_capital
        self._order_log: List[Dict] = []
        self._peak_value: float = initial_capital

    def run(self, max_cycles: int = None) -> None:
        """Main synchronous trading loop."""
        logger.info(f"\n{'='*60}")
        logger.info(f"EXECUTION ENGINE STARTED")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Cycle: {self.cycle_interval}s | DryRun: {self.dry_run}")
        logger.info(f"{'='*60}\n")

        self._running = True
        cycle = 0

        while self._running:
            if max_cycles and cycle >= max_cycles:
                logger.info(f"Max cycles ({max_cycles}) reached — stopping")
                break

            self._execute_cycle(cycle)
            cycle += 1
            self.state.cycle_count = cycle

            if self._running and (not max_cycles or cycle < max_cycles):
                logger.info(f"Sleeping {self.cycle_interval}s until next cycle...")
                time.sleep(self.cycle_interval)

        logger.info("Execution engine stopped")

    def _execute_cycle(self, cycle: int) -> EngineState:
        """Run one complete trading cycle."""
        cycle_start = time.time()
        logger.info(f"\n--- Cycle {cycle+1} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        try:
            # 1. Fetch latest prices
            prices = self.data_feed.update_latest_prices()
            if not prices:
                logger.warning("No prices available — skipping cycle")
                return self.state

            # 2. Update broker state
            account = self.broker.get_account()
            positions = self.broker.get_positions()

            portfolio_value = account.equity
            pos_dict = {sym: pos.qty for sym, pos in positions.items()}

            # Update peak and daily tracking
            self._peak_value = max(self._peak_value, portfolio_value)
            if cycle == 0:
                self._day_start_value = portfolio_value

            # 3. Risk daily reset and check
            self.risk.reset_daily(portfolio_value)
            if self.risk.is_halted:
                logger.warning(f"Trading HALTED: {self.risk.halt_reason}")
                self._update_state(account, positions, prices, {}, halted=True)
                return self.state

            # 4. Build observation
            obs = self.obs_builder.build(self.data_feed, portfolio_value, pos_dict, prices)

            # 5. Agent inference
            action, _states = self.agent.predict(obs, deterministic=True)
            action = np.clip(action, 0, 1)

            raw_weights = {sym: float(action[i]) for i, sym in enumerate(self.symbols)}

            # 6. Risk filter
            adv = np.array([prices.get(sym, 1.0) * 1e6 for sym in self.symbols])  # approx ADV
            vix = prices.get("VIX", 20.0)

            risk_decision = self.risk.evaluate(
                agent_weights=action,
                portfolio_value=portfolio_value,
                portfolio_value_history=self._value_history[-10:],
                prices=np.array([prices.get(sym, 100.0) for sym in self.symbols]),
                adv=adv,
                vix=vix,
            )

            if risk_decision.halt:
                logger.warning(f"Risk halt: {risk_decision.halt_reason}")
                self.broker.close_all_positions()
                self._update_state(account, positions, prices, {}, halted=True, halt_reason=risk_decision.halt_reason)
                return self.state

            approved_weights = {sym: float(risk_decision.approved_weights[i])
                                for i, sym in enumerate(self.symbols)}

            logger.info(f"Weights: {', '.join(f'{s}={w:.2%}' for s,w in approved_weights.items() if w > 0.001)}")
            if risk_decision.adjustments:
                logger.info(f"Risk adjustments: {list(risk_decision.adjustments.keys())}")

            # 7. Submit orders
            orders = []
            if not self.dry_run:
                orders = self.broker.submit_target_weights(
                    target_weights=approved_weights,
                    prices=prices,
                    portfolio_value=portfolio_value,
                    min_order_value=500.0,
                )
                logger.info(f"Submitted {len(orders)} orders")

            # 8. Update state
            self._value_history.append(portfolio_value)
            self._update_state(account, positions, prices, approved_weights)
            self._log_cycle(cycle, orders, raw_weights, approved_weights)

            if self.on_state_update:
                self.on_state_update(self.state)

            elapsed = time.time() - cycle_start
            logger.info(f"Cycle complete in {elapsed:.2f}s | "
                        f"Portfolio: ${portfolio_value:,.0f} ({self.state.total_return:+.2%})")

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

        return self.state

    def _update_state(self, account, positions, prices, weights,
                      halted=False, halt_reason=""):
        pv = account.equity
        self.state.portfolio_value = pv
        self.state.cash = account.cash
        self.state.prices = dict(prices)
        self.state.weights = dict(weights)
        self.state.positions = {sym: pos.qty for sym, pos in positions.items()}
        self.state.total_return = (pv - self.initial_capital) / self.initial_capital
        self.state.daily_return = (pv - self._day_start_value) / (self._day_start_value + 1e-8)
        self.state.max_drawdown = (pv - self._peak_value) / (self._peak_value + 1e-8)
        self.state.is_halted = halted
        self.state.halt_reason = halt_reason
        self.state.timestamp = datetime.now(timezone.utc)

    def _log_cycle(self, cycle, orders, raw_weights, approved_weights):
        self._order_log.append({
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": self.state.portfolio_value,
            "total_return": self.state.total_return,
            "n_orders": len(orders),
            "raw_weights": raw_weights,
            "approved_weights": approved_weights,
        })

    def stop(self):
        """Graceful shutdown."""
        logger.info("Engine stopping...")
        self._running = False

    def get_trade_log(self) -> pd.DataFrame:
        """Return full trade log as DataFrame."""
        return pd.DataFrame(self._order_log)

    @classmethod
    def from_config(
        cls,
        model_path: str,
        symbols: List[str],
        broker_mode: str = "paper_local",
        initial_capital: float = 100_000.0,
        feature_store: Optional[Dict] = None,
        dry_run: bool = True,
        **kwargs,
    ) -> "ExecutionEngine":
        """
        Convenient factory: load model + wire up all components.

        Example:
            engine = ExecutionEngine.from_config(
                model_path="models/ppo_mlp_final",
                symbols=["AAPL", "MSFT", "GOOGL"],
                broker_mode="paper_local",
                initial_capital=100_000,
                dry_run=True,  # Set False when ready for real orders
            )
            engine.run(max_cycles=5)
        """
        from stable_baselines3 import PPO
        from execution.broker import create_broker

        logger.info(f"Loading model from {model_path}")
        agent = PPO.load(model_path)

        broker = create_broker(broker_mode, initial_capital=initial_capital, **kwargs)

        risk_mgr = RiskManager(symbols=symbols, max_position_pct=0.10,
                               max_sector_pct=0.30)
        risk_mgr.peak_value = initial_capital

        # Determine obs dimensions from model
        obs_dim = int(np.prod(agent.observation_space.shape))
        n = len(symbols)
        port_dim = n + 5
        lookback = 30
        n_features = max(1, (obs_dim - port_dim) // (n * lookback))

        obs_builder = LiveObservationBuilder(symbols, n_features, lookback)
        data_feed = MarketDataFeed(symbols, lookback_days=120, broker=broker)

        try:
            data_feed.initialize()
        except Exception as e:
            logger.warning(f"Data feed init failed: {e} — will use cached prices")

        return cls(
            agent=agent,
            broker=broker,
            symbols=symbols,
            risk_manager=risk_mgr,
            obs_builder=obs_builder,
            data_feed=data_feed,
            initial_capital=initial_capital,
            dry_run=dry_run,
            **{k: v for k, v in kwargs.items()
               if k in ("cycle_interval_seconds", "on_state_update")},
        )
