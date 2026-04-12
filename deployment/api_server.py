"""
deployment/api_server.py
─────────────────────────
FastAPI inference + monitoring REST API.

Endpoints:
  GET  /health           — liveness check
  GET  /status           — current engine state (P&L, positions, weights)
  GET  /portfolio        — full portfolio breakdown
  POST /predict          — run agent inference on provided features
  GET  /metrics          — performance metrics (Sharpe, return, DD)
  GET  /drift            — drift detection history
  POST /halt             — emergency halt
  POST /resume           — resume after halt
  GET  /trade-log        — recent trade history
  GET  /positions        — current open positions

Start server:
  uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── Global engine reference (set by run_phase3.py) ───────────────────────────
_engine = None
_drift_monitor = None
_start_time = datetime.now(timezone.utc)


def set_engine(engine, drift_monitor=None):
    global _engine, _drift_monitor
    _engine = engine
    _drift_monitor = drift_monitor


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    observation: List[float]
    deterministic: bool = True


class PredictResponse(BaseModel):
    action: List[float]
    symbols: List[str]
    weights: Dict[str, float]
    timestamp: str


class HaltRequest(BaseModel):
    reason: str = "Manual halt via API"


class StatusResponse(BaseModel):
    status: str
    timestamp: str
    portfolio_value: float
    total_return: float
    daily_return: float
    max_drawdown: float
    n_positions: int
    n_trades_today: int
    is_halted: bool
    halt_reason: str
    cycle_count: int
    uptime_hours: float


# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RL Trader API",
    description="Real-time trading agent inference and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Kubernetes/Docker health probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status", response_model=StatusResponse)
async def status():
    """Current engine state — primary monitoring endpoint."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    state = _engine.state
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds() / 3600

    return StatusResponse(
        status="halted" if state.is_halted else "running",
        timestamp=state.timestamp.isoformat(),
        portfolio_value=state.portfolio_value,
        total_return=state.total_return,
        daily_return=state.daily_return,
        max_drawdown=state.max_drawdown,
        n_positions=len([v for v in state.positions.values() if abs(v) > 0.001]),
        n_trades_today=state.n_trades_today,
        is_halted=state.is_halted,
        halt_reason=state.halt_reason,
        cycle_count=state.cycle_count,
        uptime_hours=round(uptime, 2),
    )


@app.get("/portfolio")
async def portfolio():
    """Full portfolio breakdown: positions, weights, P&L per position."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    state = _engine.state
    broker = _engine.broker

    positions_detail = []
    for sym, qty in state.positions.items():
        if abs(qty) < 0.001:
            continue
        price = state.prices.get(sym, 0.0)
        weight = state.weights.get(sym, 0.0)
        market_val = qty * price
        positions_detail.append({
            "symbol": sym,
            "qty": round(qty, 4),
            "price": round(price, 2),
            "market_value": round(market_val, 2),
            "weight": round(weight, 4),
        })

    return {
        "timestamp": state.timestamp.isoformat(),
        "portfolio_value": round(state.portfolio_value, 2),
        "cash": round(state.cash, 2),
        "invested": round(state.portfolio_value - state.cash, 2),
        "total_return": round(state.total_return, 4),
        "positions": sorted(positions_detail, key=lambda x: -x["market_value"]),
        "target_weights": {k: round(v, 4) for k, v in state.weights.items() if v > 0.001},
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run agent inference on an observation vector.
    Useful for testing and external integration.
    """
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    obs = np.array(req.observation, dtype=np.float32)
    expected_dim = int(np.prod(_engine.agent.observation_space.shape))

    if len(obs) != expected_dim:
        raise HTTPException(400, f"Obs dim mismatch: got {len(obs)}, expected {expected_dim}")

    action, _ = _engine.agent.predict(obs, deterministic=req.deterministic)
    action = np.clip(action, 0, 1).tolist()

    weights = {sym: round(float(action[i]), 4) for i, sym in enumerate(_engine.symbols)}

    return PredictResponse(
        action=action,
        symbols=_engine.symbols,
        weights=weights,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics")
async def metrics():
    """Rolling performance metrics."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    state = _engine.state
    value_history = _engine._value_history

    if len(value_history) < 2:
        return {"error": "Insufficient history"}

    returns = np.diff(value_history) / np.array(value_history[:-1])
    ann_return = float(np.mean(returns) * 252) if len(returns) > 0 else 0.0
    ann_vol = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = ann_return / (ann_vol + 1e-8)

    peak = np.maximum.accumulate(value_history)
    dd = (np.array(value_history) - peak) / peak
    max_dd = float(dd.min())

    return {
        "timestamp": state.timestamp.isoformat(),
        "total_return": round(state.total_return, 4),
        "annual_return": round(ann_return, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "current_drawdown": round(state.max_drawdown, 4),
        "n_cycles": state.cycle_count,
        "portfolio_value": round(state.portfolio_value, 2),
    }


@app.get("/drift")
async def drift():
    """Drift detection history and current status."""
    if _drift_monitor is None:
        return {"status": "drift_monitor_not_initialized"}

    history = _drift_monitor.get_drift_history()
    latest_reports = _drift_monitor.reports[-5:] if _drift_monitor.reports else []

    return {
        "drift_events": len(history),
        "last_drift": history[-1] if history else None,
        "recent_sharpe": latest_reports[-1].recent_sharpe if latest_reports else None,
        "baseline_sharpe": latest_reports[-1].baseline_sharpe if latest_reports else None,
        "history": history[-20:],  # last 20 events
    }


@app.post("/halt")
async def halt(req: HaltRequest):
    """Emergency halt — stops trading and closes all positions."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    logger.warning(f"Manual halt via API: {req.reason}")
    _engine.stop()

    if not _engine.dry_run:
        orders = _engine.broker.close_all_positions()
        return {"status": "halted", "reason": req.reason, "positions_closed": len(orders)}

    return {"status": "halted", "reason": req.reason, "dry_run": True}


@app.post("/resume")
async def resume():
    """Resume trading after a halt (requires manual operator action)."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")
    _engine._running = True
    _engine.state.is_halted = False
    _engine.state.halt_reason = ""
    logger.info("Trading resumed via API")
    return {"status": "resumed"}


@app.get("/trade-log")
async def trade_log(limit: int = 50):
    """Recent trade history."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")
    log = _engine._order_log[-limit:]
    return {"count": len(log), "trades": log}


@app.get("/positions")
async def positions():
    """Current open positions from broker."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    pos = _engine.broker.get_positions()
    account = _engine.broker.get_account()

    return {
        "account": {
            "equity": round(account.equity, 2),
            "cash": round(account.cash, 2),
            "buying_power": round(account.buying_power, 2),
        },
        "positions": [
            {
                "symbol": sym,
                "qty": round(p.qty, 4),
                "avg_cost": round(p.avg_entry_price, 2),
                "current_price": round(p.current_price, 2),
                "market_value": round(p.market_value, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4),
            }
            for sym, p in pos.items()
        ],
    }


@app.get("/risk-status")
async def risk_status():
    """Current risk manager state."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")

    risk = _engine.risk
    return {
        "is_halted": risk.is_halted,
        "halt_reason": risk.halt_reason,
        "peak_value": round(risk.peak_value, 2),
        "max_drawdown_limit": risk.max_portfolio_dd,
        "max_position_pct": risk.max_position,
    }
