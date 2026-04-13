"""
run_trade.py
─────────────
第三步：实盘/模拟执行

Input:   models/           训练好的模型（由 run_train.py 生成）
         data/processed/   特征文件（由 run_data.py 生成）
         .env              ALPACA_API_KEY / ALPACA_SECRET_KEY（alpaca/live 模式需要）
Output:  logs/trade_log.csv   每个 cycle 的下单记录
         logs/engine_state.json  最新引擎状态快照
         monitoring/       Prometheus 指标 + Grafana 面板（自动生成，无需 commit）

Modes:
  --mode paper      本地模拟撮合，无需 API Key，适合初步验证
  --mode alpaca     Alpaca 纸交易，使用真实行情模拟成交（需 .env）
  --mode live       Alpaca 实盘 ⚠️ 真实资金（需 funded 账户）
  --mode backtest   用历史数据跑完整执行栈，验证下单逻辑

Components started:
  1. FastAPI REST server (port 8000)
  2. Prometheus metrics exporter (port 9090)
  3. Trading execution engine (continuous loop)
  4. Drift detection monitor
  5. State persistence (JSON checkpoints)

Usage:
  # 最安全的首次运行 — 本地模拟，dry-run（不真正下单）:
  python run_trade.py --mode paper --dry-run \\
      --model-path models/ppo_mlp_final --n-stocks 5

  # Alpaca 纸交易（需 .env 配置 API Key）:
  python run_trade.py --mode alpaca --model-path models/ppo_mlp_final

  # 历史数据回测执行栈:
  python run_trade.py --mode backtest --start 2024-01-01 --end 2024-12-31

Next:
  python run_autopilot.py --mode paper   # 开启自动化运维
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG, DATA_DIR, LOG_DIR, MODEL_DIR
from utils.helpers import set_seed, setup_logging

logger = logging.getLogger("phase3")


# ═══════════════════════════════════════════════════════════════════════════════
# Engine Setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_engine(args, feature_store, symbols, model):
    """Wire up all Phase 3 components."""
    from execution.broker import create_broker
    from execution.engine import ExecutionEngine, MarketDataFeed, LiveObservationBuilder
    from utils.risk_manager import RiskManager
    from monitoring.drift_detector import DriftMonitor

    # Broker
    broker_mode_map = {"paper": "paper_local", "alpaca": "paper_alpaca", "live": "live"}
    broker = create_broker(
        broker_mode_map.get(args.mode, "paper_local"),
        initial_capital=args.capital,
    )

    # Risk manager
    risk_mgr = RiskManager(
        symbols=symbols,
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_drawdown_halt=0.08,
        max_portfolio_drawdown=0.15,
        kelly_fraction=0.25,
    )
    risk_mgr.peak_value = args.capital

    # Data feed
    data_feed = MarketDataFeed(symbols, lookback_days=120, broker=broker)
    try:
        data_feed.initialize()
    except Exception as e:
        logger.warning(f"Data feed init failed: {e} — building synthetic")
        from train.train_ppo import build_synthetic_feature_store
        if not feature_store:
            feature_store, _ = build_synthetic_feature_store(symbols)

    # Observation builder
    obs_dim = int(np.prod(model.observation_space.shape))
    port_dim = len(symbols) + 5
    lookback = 30
    n_features = max(1, (obs_dim - port_dim) // (len(symbols) * lookback))
    obs_builder = LiveObservationBuilder(symbols, n_features, lookback)

    # Drift monitor
    drift_monitor = DriftMonitor(
        auto_retrain=False,  # Phase 3: alert only; Phase 4: enable auto-retrain
        check_every_n_cycles=10,
    )

    # State update callback
    from monitoring.metrics_exporter import MetricsExporter
    metrics_exporter = MetricsExporter()

    def on_state_update(state):
        # Update Prometheus metrics
        pv = state.portfolio_value
        returns_hist = [(pv - args.capital) / args.capital]
        sharpe = 0.0
        metrics_exporter.update_from_state(state, sharpe=sharpe)

        # Update drift monitor
        if len(returns_hist) >= 2:
            drift_monitor.update(daily_return=returns_hist[-1])

        # Save state checkpoint
        state_path = LOG_DIR / "engine_state.json"
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

    # Start Prometheus server
    metrics_exporter.start_server(port=9090)

    # Save Grafana dashboard
    from monitoring.metrics_exporter import save_grafana_dashboard
    save_grafana_dashboard(str(Path("monitoring/grafana_dashboard.json")))

    engine = ExecutionEngine(
        agent=model,
        broker=broker,
        symbols=symbols,
        risk_manager=risk_mgr,
        obs_builder=obs_builder,
        data_feed=data_feed,
        cycle_interval_seconds=args.cycle_interval,
        initial_capital=args.capital,
        on_state_update=on_state_update,
        dry_run=args.dry_run,
    )

    return engine, drift_monitor, metrics_exporter


# ═══════════════════════════════════════════════════════════════════════════════
# API Server Thread
# ═══════════════════════════════════════════════════════════════════════════════

def start_api_server(engine, drift_monitor, port: int = 8000):
    """Start FastAPI server in a background thread."""
    try:
        import uvicorn
        from deployment.api_server import app, set_engine
        set_engine(engine, drift_monitor)

        def run():
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info(f"API server started: http://localhost:{port}")
        logger.info(f"  Docs: http://localhost:{port}/docs")
        return thread
    except Exception as e:
        logger.warning(f"API server failed to start: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Mode (historical replay through execution engine)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_mode(args, feature_store, symbols, model):
    """
    Replay historical data through the full execution stack.
    Validates that the engine, risk manager, and broker all work correctly
    before going live.
    """
    from execution.broker import PaperBroker
    from execution.engine import ExecutionEngine, MarketDataFeed, LiveObservationBuilder
    from utils.risk_manager import RiskManager
    from environment.backtester import compute_metrics, BuyAndHoldBenchmark

    logger.info("=" * 60)
    logger.info("BACKTEST MODE: Replaying through execution engine")
    logger.info("=" * 60)

    broker = PaperBroker(initial_capital=args.capital)
    risk_mgr = RiskManager(symbols=symbols)
    risk_mgr.peak_value = args.capital

    obs_dim = int(np.prod(model.observation_space.shape))
    port_dim = len(symbols) + 5
    lookback = 30
    n_features = max(1, (obs_dim - port_dim) // (len(symbols) * lookback))
    obs_builder = LiveObservationBuilder(symbols, n_features, lookback)
    data_feed = MarketDataFeed(symbols, lookback_days=120)

    # Use feature store as the data source
    if feature_store:
        for sym, df in feature_store.items():
            if sym in symbols:
                data_feed._bars[sym] = df
                if not df.empty and "close" in df.columns:
                    data_feed._latest_prices[sym] = float(df["close"].iloc[-1])

    portfolio_values = [args.capital]
    dates_available = []

    # Get common dates
    if feature_store:
        date_sets = [set(df.index) for df in feature_store.values() if not df.empty]
        if date_sets:
            common_dates = sorted(set.intersection(*date_sets))
            dates_available = [d for d in common_dates if str(d.date()) >= (args.start or "2019-01-01")]
            dates_available = dates_available[:252]  # max 1 year

    if not dates_available:
        logger.warning("No dates available for backtest — using 50 synthetic cycles")
        dates_available = list(range(50))

    logger.info(f"Simulating {len(dates_available)} trading days")

    for i, date in enumerate(dates_available):
        # Update broker prices from feature store
        prices = {}
        for sym in symbols:
            if feature_store and sym in feature_store:
                df = feature_store[sym]
                if hasattr(date, 'date'):
                    date_rows = df[df.index <= date]
                    if not date_rows.empty:
                        prices[sym] = float(date_rows["close"].iloc[-1])
                else:
                    prices[sym] = float(df["close"].iloc[min(i, len(df)-1)])

        if not prices:
            prices = {sym: 100.0 + i * 0.1 for sym in symbols}

        broker.update_prices(prices)
        data_feed._latest_prices = prices

        # Build observation
        account = broker.get_account()
        pos = {sym: p.qty for sym, p in broker.get_positions().items()}
        obs = obs_builder.build(data_feed, account.equity, pos, prices)

        # Agent inference
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, 0, 1)

        # Risk filter
        risk_dec = risk_mgr.evaluate(
            agent_weights=action,
            portfolio_value=account.equity,
            portfolio_value_history=portfolio_values[-10:],
            prices=np.array([prices.get(s, 100.0) for s in symbols]),
            adv=np.array([1e7] * len(symbols)),
            vix=20.0,
        )

        if not risk_dec.halt:
            weights = {s: float(risk_dec.approved_weights[j]) for j, s in enumerate(symbols)}
            broker.submit_target_weights(weights, prices, account.equity, min_order_value=100.0)

        portfolio_values.append(broker.get_account().equity)

        if i % 20 == 0:
            pv = portfolio_values[-1]
            ret = (pv - args.capital) / args.capital
            logger.info(f"Day {i+1:3d}: Portfolio=${pv:>10,.0f} ({ret:+.2%})")

    # Compute metrics
    pv_series = pd.Series(portfolio_values)
    from environment.backtester import compute_metrics
    report = compute_metrics(pv_series)
    logger.info(report.summary())

    # Save results
    results = {
        "backtest_mode": True,
        "n_days": len(dates_available),
        "initial_capital": args.capital,
        "final_value": float(portfolio_values[-1]),
        "total_return": report.total_return,
        "annual_return": report.annual_return,
        "sharpe_ratio": report.sharpe_ratio,
        "max_drawdown": report.max_drawdown,
        "calmar_ratio": report.calmar_ratio,
    }
    with open(MODEL_DIR / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Backtest results saved to {MODEL_DIR}/backtest_results.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Trader Phase 3 — Live Execution")
    parser.add_argument("--mode", choices=["paper", "alpaca", "live", "backtest"], default="paper")
    parser.add_argument("--model-path", default=None, help="Path to trained SB3 model (.zip)")
    parser.add_argument("--n-stocks", type=int, default=5)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cycle-interval", type=int, default=60,
                        help="Seconds between trading cycles (paper: 60, live: 1800)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Stop after N cycles")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Compute but don't submit orders (default True for safety)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Actually submit orders")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--use-synthetic", action="store_true", default=False)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("INFO", LOG_DIR / "phase3.log")
    set_seed(args.seed)

    logger.info("\n" + "╔" + "═" * 58 + "╗")
    logger.info("║" + "   RL TRADER — PHASE 3: LIVE EXECUTION".center(58) + "║")
    logger.info("║" + f"   Mode: {args.mode} | DryRun: {args.dry_run} | Capital: ${args.capital:,.0f}".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝\n")

    if args.mode == "live" and args.dry_run:
        logger.warning("Live mode but dry_run=True — no real orders will be submitted")
    if args.mode == "live" and not args.dry_run:
        logger.warning("⚠️  LIVE TRADING WITH REAL MONEY — ensure risk limits are correct!")
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm.strip() != "CONFIRM":
            logger.info("Live trading cancelled")
            return

    # ── Symbol selection ───────────────────────────────────────────────────────
    default_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "HD", "META", "AMZN"]
    symbols = args.symbols or default_symbols[:args.n_stocks]
    logger.info(f"Trading universe: {symbols}")

    # ── Feature store ──────────────────────────────────────────────────────────
    feature_store = None
    if args.use_synthetic or not args.model_path:
        from train.train_ppo import build_synthetic_feature_store
        feature_store, _ = build_synthetic_feature_store(symbols, seed=args.seed)
    else:
        try:
            from features.feature_engineer import FeaturePipeline
            feature_store = FeaturePipeline().load(str(DATA_DIR / "processed"), symbols)
        except Exception:
            from train.train_ppo import build_synthetic_feature_store
            feature_store, _ = build_synthetic_feature_store(symbols, seed=args.seed)

    # ── Load or train model ────────────────────────────────────────────────────
    from stable_baselines3 import PPO
    if args.model_path and Path(args.model_path + ".zip").exists():
        logger.info(f"Loading model: {args.model_path}")
        model = PPO.load(args.model_path)
    else:
        logger.info("No model found — training quick MLP baseline...")
        from train.train_ppo import train_mlp
        valid_fs = {s: feature_store[s] for s in symbols if s in feature_store}
        model = train_mlp(
            valid_fs, list(valid_fs.keys()),
            total_timesteps=30_000, n_envs=1,
            save_dir=str(MODEL_DIR), experiment_name="phase3_baseline",
        )

    valid_symbols = [s for s in symbols if s in (feature_store or {})]

    # ── Backtest mode ──────────────────────────────────────────────────────────
    if args.mode == "backtest":
        results = run_backtest_mode(args, feature_store, valid_symbols, model)
        logger.info(f"\n✓ Backtest complete: Sharpe={results['sharpe_ratio']:.3f} | Return={results['total_return']:+.2%}")
        return

    # ── Live / paper trading ───────────────────────────────────────────────────
    engine, drift_monitor, metrics_exporter = setup_engine(args, feature_store, valid_symbols, model)

    # Start API server in background
    api_thread = start_api_server(engine, drift_monitor, port=args.api_port)
    time.sleep(1)  # let server start

    logger.info(f"\n{'='*60}")
    logger.info(f"SYSTEM READY")
    logger.info(f"  API:     http://localhost:{args.api_port}")
    logger.info(f"  Metrics: http://localhost:9090/metrics")
    logger.info(f"  Mode:    {args.mode} | DryRun: {args.dry_run}")
    logger.info(f"  Ctrl+C to stop")
    logger.info(f"{'='*60}\n")

    try:
        engine.run(max_cycles=args.max_cycles)
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
        engine.stop()

    # Save final trade log
    trade_log = engine.get_trade_log()
    if not trade_log.empty:
        trade_log.to_csv(LOG_DIR / "trade_log.csv", index=False)
        logger.info(f"Trade log saved: {len(trade_log)} cycles")

    logger.info("\n✓ Phase 3 shutdown complete")


if __name__ == "__main__":
    main()
