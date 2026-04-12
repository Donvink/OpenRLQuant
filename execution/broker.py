"""
execution/broker.py
────────────────────
Unified broker interface with two implementations:

  PaperBroker   — local simulation, instant fills, perfect for testing
  AlpacaBroker  — real Alpaca API (paper or live)

Both implement the same BaseBroker interface so the execution engine
doesn't need to know which mode it's running in.

Usage:
    # Paper trading (no API key needed)
    broker = PaperBroker(initial_capital=100_000)

    # Alpaca paper trading
    broker = AlpacaBroker(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )

    # Same interface for both:
    order = broker.submit_order("AAPL", qty=10, side="buy", order_type="market")
    positions = broker.get_positions()
    account = broker.get_account()
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Types ────────────────────────────────────────────────────────────────

@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: str            # "buy" | "sell"
    order_type: str      # "market" | "limit" | "stop"
    status: str          # "pending" | "filled" | "cancelled" | "rejected"
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def notional_value(self) -> float:
        return self.filled_qty * self.filled_avg_price


@dataclass
class Position:
    symbol: str
    qty: float               # positive=long, negative=short
    avg_entry_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.qty * (self.current_price - self.avg_entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class Account:
    equity: float                  # total portfolio value
    cash: float
    buying_power: float
    day_trade_count: int = 0
    portfolio_value: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0


# ─── Base Interface ────────────────────────────────────────────────────────────

class BaseBroker(ABC):
    """Abstract base class for all broker implementations."""

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Order:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        ...

    @abstractmethod
    def get_account(self) -> Account:
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        ...

    def submit_target_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        min_order_value: float = 500.0,
    ) -> List[Order]:
        """
        Translate target portfolio weights → individual buy/sell orders.
        Handles sells-before-buys ordering to free up cash.
        """
        positions = self.get_positions()
        current_values = {sym: pos.qty * prices.get(sym, pos.current_price)
                         for sym, pos in positions.items()}

        target_values = {sym: w * portfolio_value for sym, w in target_weights.items()}

        orders = []
        deltas = {}  # symbol -> dollar delta

        all_symbols = set(target_values) | set(current_values)
        for sym in all_symbols:
            target = target_values.get(sym, 0.0)
            current = current_values.get(sym, 0.0)
            delta = target - current
            if abs(delta) >= min_order_value:
                deltas[sym] = delta

        # Process sells first (to free cash), then buys
        for sym in sorted(deltas, key=lambda s: deltas[s]):  # sells first (negative)
            delta = deltas[sym]
            price = prices.get(sym, 1.0)
            if price <= 0:
                continue
            qty = abs(delta) / price
            side = "buy" if delta > 0 else "sell"

            try:
                order = self.submit_order(sym, qty, side, "market")
                orders.append(order)
                logger.info(f"Order submitted: {side.upper()} {qty:.2f} {sym} @ ~${price:.2f}")
            except Exception as e:
                logger.error(f"Order failed for {sym}: {e}")

        return orders

    def close_all_positions(self) -> List[Order]:
        """Liquidate all open positions (emergency or end-of-day)."""
        positions = self.get_positions()
        orders = []
        for sym, pos in positions.items():
            if abs(pos.qty) > 0.001:
                side = "sell" if pos.qty > 0 else "buy"
                try:
                    order = self.submit_order(sym, abs(pos.qty), side, "market")
                    orders.append(order)
                    logger.info(f"CLOSE: {side.upper()} {abs(pos.qty):.2f} {sym}")
                except Exception as e:
                    logger.error(f"Failed to close {sym}: {e}")
        return orders


# ─── Paper Broker (local simulation) ──────────────────────────────────────────

class PaperBroker(BaseBroker):
    """
    Local paper trading simulation.
    Instant fills at market price + configurable slippage.
    No API calls needed — perfect for backtesting and CI/CD.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,      # 5bps per trade
        slippage_bps: float = 3.0,        # 3bps adverse slippage
        price_feed: Optional[Dict[str, float]] = None,
    ):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._price_feed = price_feed or {}

    def submit_order(self, symbol, qty, side, order_type="market",
                     limit_price=None, stop_price=None, time_in_force="day") -> Order:
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            status="pending",
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=f"paper_{order_id}",
        )
        self._orders[order_id] = order
        self._fill_order(order)
        return order

    def _fill_order(self, order: Order):
        """Simulate market fill with slippage."""
        price = self._price_feed.get(order.symbol, 100.0)

        # Apply slippage: adverse to the trader
        if order.side == "buy":
            fill_price = price * (1 + self.slippage)
        else:
            fill_price = price * (1 - self.slippage)

        cost = order.qty * fill_price
        commission = cost * self.commission

        if order.side == "buy":
            total_cost = cost + commission
            if total_cost > self.cash:
                # Partial fill
                max_qty = self.cash / (fill_price * (1 + self.commission))
                if max_qty < 0.001:
                    order.status = "rejected"
                    return
                order.qty = max_qty
                cost = max_qty * fill_price
                commission = cost * self.commission
                total_cost = cost + commission

            self.cash -= total_cost
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                new_qty = pos.qty + order.qty
                pos.avg_entry_price = (pos.avg_entry_price * pos.qty + fill_price * order.qty) / new_qty
                pos.qty = new_qty
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol, qty=order.qty, avg_entry_price=fill_price,
                    current_price=fill_price,
                )
        else:  # sell
            pos = self._positions.get(order.symbol)
            sell_qty = min(order.qty, pos.qty if pos else 0.0)
            if sell_qty <= 0.001:
                order.status = "rejected"
                return
            proceeds = sell_qty * fill_price - sell_qty * fill_price * self.commission
            self.cash += proceeds
            if pos:
                pos.qty -= sell_qty
                if pos.qty < 0.001:
                    del self._positions[order.symbol]

        order.status = "filled"
        order.filled_qty = order.qty
        order.filled_avg_price = fill_price
        order.filled_at = datetime.now(timezone.utc)

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def get_account(self) -> Account:
        portfolio_value = self.cash + sum(
            p.qty * self._price_feed.get(p.symbol, p.avg_entry_price)
            for p in self._positions.values()
        )
        return Account(
            equity=portfolio_value,
            cash=self.cash,
            buying_power=self.cash,
            portfolio_value=portfolio_value,
        )

    def get_latest_price(self, symbol: str) -> float:
        return self._price_feed.get(symbol, 0.0)

    def update_prices(self, prices: Dict[str, float]):
        """Update internal price feed and position mark-to-market."""
        self._price_feed.update(prices)
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.current_price = prices[sym]

    def get_pnl_summary(self) -> Dict:
        account = self.get_account()
        return {
            "portfolio_value": account.portfolio_value,
            "cash": self.cash,
            "total_return": (account.portfolio_value - self.initial_capital) / self.initial_capital,
            "unrealized_pnl": sum(p.unrealized_pnl for p in self._positions.values()),
            "n_positions": len(self._positions),
        }


# ─── Alpaca Broker (real API) ──────────────────────────────────────────────────

class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker implementation.

    Supports:
      - Paper trading (paper=True, free account at alpaca.markets)
      - Live trading (paper=False, funded account)
      - Both REST (orders) and WebSocket (real-time quotes) APIs

    Free paper trading: https://alpaca.markets/docs/trading/paper-trading/
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL  = "https://api.alpaca.markets"
    DATA_URL  = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret required. "
                             "Get free paper trading keys at alpaca.markets")

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.base_url = self.PAPER_URL if paper else self.LIVE_URL

        self._client = self._init_client()
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"AlpacaBroker initialized: {mode} trading")

    def _init_client(self):
        try:
            from alpaca.trading.client import TradingClient
            return TradingClient(self.api_key, self.secret_key, paper=self.paper)
        except ImportError:
            raise ImportError("pip install alpaca-py")

    def _headers(self) -> Dict:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}

    def submit_order(self, symbol, qty, side, order_type="market",
                     limit_price=None, stop_price=None, time_in_force="day") -> Order:
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY

            if order_type == "market":
                req = MarketOrderRequest(symbol=symbol, qty=round(qty, 4),
                                         side=side_enum, time_in_force=tif)
            elif order_type == "limit" and limit_price:
                req = LimitOrderRequest(symbol=symbol, qty=round(qty, 4),
                                        side=side_enum, time_in_force=tif,
                                        limit_price=round(limit_price, 2))
            else:
                req = MarketOrderRequest(symbol=symbol, qty=round(qty, 4),
                                         side=side_enum, time_in_force=tif)

            resp = self._client.submit_order(order_data=req)
            return self._alpaca_order_to_order(resp)

        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        try:
            resp = self._client.get_order_by_id(order_id)
            return self._alpaca_order_to_order(resp)
        except Exception:
            return None

    def get_positions(self) -> Dict[str, Position]:
        try:
            alpaca_positions = self._client.get_all_positions()
            return {
                p.symbol: Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price or p.avg_entry_price),
                )
                for p in alpaca_positions
            }
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return {}

    def get_account(self) -> Account:
        try:
            a = self._client.get_account()
            return Account(
                equity=float(a.equity),
                cash=float(a.cash),
                buying_power=float(a.buying_power),
                day_trade_count=int(a.daytrade_count or 0),
                portfolio_value=float(a.portfolio_value),
            )
        except Exception as e:
            logger.error(f"get_account failed: {e}")
            return Account(equity=0, cash=0, buying_power=0)

    def get_latest_price(self, symbol: str) -> float:
        try:
            import httpx
            resp = httpx.get(
                f"{self.DATA_URL}/v2/stocks/{symbol}/quotes/latest",
                headers=self._headers(), timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                quote = data.get("quote", {})
                bid = float(quote.get("bp", 0))
                ask = float(quote.get("ap", 0))
                return (bid + ask) / 2 if bid and ask else 0.0
        except Exception:
            pass
        return 0.0

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Batch price fetch — much more efficient than one-by-one."""
        try:
            import httpx
            resp = httpx.get(
                f"{self.DATA_URL}/v2/stocks/quotes/latest",
                params={"symbols": ",".join(symbols), "feed": "iex"},
                headers=self._headers(), timeout=10,
            )
            if resp.status_code == 200:
                quotes = resp.json().get("quotes", {})
                return {
                    sym: (float(q.get("bp", 0)) + float(q.get("ap", 0))) / 2
                    for sym, q in quotes.items()
                }
        except Exception as e:
            logger.error(f"Batch price fetch failed: {e}")
        return {}

    def is_market_open(self) -> bool:
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception:
            return False

    def get_market_hours(self) -> Dict:
        try:
            clock = self._client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            }
        except Exception:
            return {}

    @staticmethod
    def _alpaca_order_to_order(resp) -> Order:
        return Order(
            id=str(resp.id),
            symbol=resp.symbol,
            qty=float(resp.qty or 0),
            side=str(resp.side).lower().replace("orderside.", ""),
            order_type=str(resp.order_type).lower().replace("ordertype.", ""),
            status=str(resp.status).lower().replace("orderstatus.", ""),
            filled_qty=float(resp.filled_qty or 0),
            filled_avg_price=float(resp.filled_avg_price or 0),
            client_order_id=str(resp.client_order_id or ""),
        )


# ─── Broker Factory ────────────────────────────────────────────────────────────

def create_broker(mode: str = "paper", **kwargs) -> BaseBroker:
    """
    Factory function for creating broker instances.

    mode:
      "paper_local"  — PaperBroker (no API, instant fills)
      "paper_alpaca" — AlpacaBroker(paper=True)
      "live"         — AlpacaBroker(paper=False) ⚠️ REAL MONEY
    """
    if mode == "paper_local":
        return PaperBroker(**kwargs)
    elif mode == "paper_alpaca":
        import os
        return AlpacaBroker(
            api_key=kwargs.get("api_key") or os.getenv("ALPACA_API_KEY", ""),
            secret_key=kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY", ""),
            paper=True,
        )
    elif mode == "live":
        import os
        logger.warning("⚠️  LIVE TRADING MODE — REAL MONEY AT RISK")
        return AlpacaBroker(
            api_key=kwargs.get("api_key") or os.getenv("ALPACA_API_KEY", ""),
            secret_key=kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY", ""),
            paper=False,
        )
    else:
        raise ValueError(f"Unknown broker mode: {mode}. Use 'paper_local', 'paper_alpaca', or 'live'")
