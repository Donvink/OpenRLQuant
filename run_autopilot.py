"""
run_autopilot.py
─────────────────
第四步：全自动运维（自动再训练 + 集成投票 + 告警）

Input:   models/           已有模型（没有则自动训练）
         data/processed/   特征文件（由 run_data.py 生成）
         .env              ALPACA_API_KEY、Slack/Email 告警配置（可选）
Output:  logs/phase4_report.json   运行报告
         models/archive/           历史模型归档

集成所有步骤的组件:
  run_data.py     → 数据管道 + 特征工程
  run_train.py    → PPO 训练 + Curriculum
  run_trade.py    → 实盘执行 + 监控
  run_autopilot.py → 自动再训练 + Ensemble + 调度 + 告警

Modes:
  --mode paper    本地模拟，适合验证全流程
  --mode alpaca   Alpaca 纸交易，接近生产环境
  --mode live     实盘 ⚠️ 真实资金
  --mode demo     快速演示所有 Phase 4 功能（3 个 cycle，不下真实订单）

Usage:
  # 演示所有功能（推荐首次运行）
  python run_autopilot.py --mode demo --max-cycles 3

  # 生产纸交易（需 .env）
  python run_autopilot.py --mode alpaca

  # 本地全自动（无需 API Key）
  python run_autopilot.py --mode paper --use-synthetic
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG, DATA_DIR, LOG_DIR, MODEL_DIR
from utils.helpers import set_seed, setup_logging

logger = logging.getLogger("phase4")


# ═══════════════════════════════════════════════════════════════════════════════
# System Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_system(args) -> Dict:
    """Initialize all Phase 4 components and return component registry."""
    components = {}

    # ── 1. Feature Store ───────────────────────────────────────────────────────
    from train.train_ppo import build_synthetic_feature_store
    symbols = args.symbols or ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM",
                                "JNJ", "XOM", "META", "HD", "AMZN"]
    symbols = symbols[:args.n_stocks]

    if args.use_synthetic or not os.getenv("POLYGON_API_KEY"):
        logger.info("Building synthetic feature store...")
        feature_store, vix = build_synthetic_feature_store(symbols, n_days=1000, seed=args.seed)
    else:
        from features.feature_engineer import FeaturePipeline
        try:
            feature_store = FeaturePipeline().load(str(DATA_DIR / "processed"), symbols)
            vix = None
        except Exception:
            feature_store, vix = build_synthetic_feature_store(symbols, seed=args.seed)

    valid_symbols = [s for s in symbols if s in feature_store]
    components["feature_store"] = feature_store
    components["symbols"] = valid_symbols
    logger.info(f"Feature store: {len(feature_store)} symbols")

    # ── 2. Models (train if needed) ───────────────────────────────────────────
    from stable_baselines3 import PPO
    from train.train_ppo import train_mlp

    def get_or_train_model(name: str, timesteps: int = 20_000) -> object:
        path = MODEL_DIR / f"{name}.zip"
        if path.exists():
            logger.info(f"Loading model: {path}")
            return PPO.load(str(path)[:-4])
        logger.info(f"Training model: {name} ({timesteps} steps)")
        return train_mlp(feature_store, valid_symbols, timesteps, n_envs=1,
                         save_dir=str(MODEL_DIR), experiment_name=name)

    model_a = get_or_train_model("model_a", timesteps=15_000)
    model_b = get_or_train_model("model_b", timesteps=15_000)
    components["model_a"] = model_a
    components["model_b"] = model_b
    logger.info("Models ready")

    # ── 3. Alert System ───────────────────────────────────────────────────────
    from automation.alerts import AlertSystem
    alerts = AlertSystem.from_env()
    components["alerts"] = alerts

    # ── 4. Ensemble ───────────────────────────────────────────────────────────
    from ensemble.ensemble_agent import EnsembleManager
    ensemble = (EnsembleManager(strategy="disagreement")
                .add_model("model_a", model_a, role="production", weight=0.6)
                .add_model("model_b", model_b, role="production", weight=0.4))
    components["ensemble"] = ensemble
    logger.info(f"Ensemble: {len(ensemble._active_members)} active models")

    # ── 5. Drift Monitor ──────────────────────────────────────────────────────
    from monitoring.drift_detector import DriftMonitor

    def on_critical_drift(report):
        logger.critical(f"Drift triggered retraining: {report.summary()}")
        alerts.send(f"🚨 Critical drift — retraining triggered", level="critical",
                    details={"type": report.drift_type, "sharpe": report.recent_sharpe})
        if "retrain_pipeline" in components:
            components["retrain_pipeline"].run(triggered_by="drift")

    drift_monitor = DriftMonitor(
        auto_retrain=False,
        retrain_callback=on_critical_drift,
        check_every_n_cycles=5,
    )
    components["drift_monitor"] = drift_monitor

    # ── 6. Risk Manager ───────────────────────────────────────────────────────
    from utils.risk_manager import RiskManager
    risk_mgr = RiskManager(symbols=valid_symbols, max_position_pct=0.10,
                            max_drawdown_halt=0.08, max_portfolio_drawdown=0.15)
    risk_mgr.peak_value = args.capital
    components["risk_manager"] = risk_mgr

    # ── 7. Broker ─────────────────────────────────────────────────────────────
    from execution.broker import create_broker
    broker_map = {"paper": "paper_local", "alpaca": "paper_alpaca", "live": "live"}
    broker = create_broker(broker_map.get(args.mode, "paper_local"),
                           initial_capital=args.capital)
    components["broker"] = broker

    # ── 8. Execution Engine ───────────────────────────────────────────────────
    from execution.engine import ExecutionEngine, MarketDataFeed, LiveObservationBuilder
    from monitoring.metrics_exporter import MetricsExporter

    data_feed = MarketDataFeed(valid_symbols, lookback_days=120, broker=broker)
    for sym in valid_symbols:
        if sym in feature_store:
            data_feed._bars[sym] = feature_store[sym]
            data_feed._latest_prices[sym] = float(feature_store[sym]["close"].iloc[-1])

    obs_dim = int(np.prod(model_a.observation_space.shape))
    port_dim = len(valid_symbols) + 5
    n_features = max(1, (obs_dim - port_dim) // (len(valid_symbols) * 30))
    obs_builder = LiveObservationBuilder(valid_symbols, n_features, lookback=30)

    metrics_exp = MetricsExporter()
    metrics_exp.start_server(port=9090)

    def on_state_update(state):
        # Update metrics
        metrics_exp.update_from_state(state, sharpe=0.0)
        # Update drift monitor
        if state.daily_return != 0:
            drift_monitor.update(state.daily_return)
        # Ensemble performance tracking
        if state.cycle_count > 0 and state.daily_return != 0:
            ensemble.update_performance(state.daily_return)

    # Create a wrapper that uses ensemble for predictions
    class EnsembleAgentWrapper:
        def __init__(self, ens):
            self.ens = ens
            self.observation_space = model_a.observation_space
            self.action_space = model_a.action_space

        def predict(self, obs, deterministic=True):
            action, info = self.ens.predict(obs, deterministic)
            return action, None

    ensemble_agent = EnsembleAgentWrapper(ensemble)

    engine = ExecutionEngine(
        agent=ensemble_agent,
        broker=broker,
        symbols=valid_symbols,
        risk_manager=risk_mgr,
        obs_builder=obs_builder,
        data_feed=data_feed,
        cycle_interval_seconds=args.cycle_interval,
        initial_capital=args.capital,
        on_state_update=on_state_update,
        dry_run=args.dry_run,
    )
    components["engine"] = engine
    components["metrics_exporter"] = metrics_exp

    # ── 9. Retraining Pipeline ────────────────────────────────────────────────
    from automation.retrain_pipeline import RetrainingPipeline, RetrainConfig

    def on_model_promotion(new_model):
        ensemble.add_model("retrained", new_model, role="production", weight=0.5)
        alerts.send("New model integrated into ensemble", level="success")

    retrain_cfg = RetrainConfig(
        timesteps=15_000,
        n_envs=1,
        min_sharpe_absolute=0.1,
        min_sharpe_improvement=0.02,
        model_dir=str(MODEL_DIR),
        archive_dir=str(MODEL_DIR / "archive"),
    )
    retrain_pipeline = RetrainingPipeline(
        symbols=valid_symbols,
        feature_store=feature_store,
        alert_system=alerts,
        config=retrain_cfg,
        on_promotion=on_model_promotion,
    )
    components["retrain_pipeline"] = retrain_pipeline
    # Wire drift monitor to retraining
    drift_monitor.retrain_callback = lambda r: retrain_pipeline.run("drift")

    # ── 10. Scheduler ─────────────────────────────────────────────────────────
    from automation.scheduler import TradingScheduler
    scheduler = TradingScheduler(
        engine=engine,
        retrain_pipeline=retrain_pipeline,
        alert_system=alerts,
        drift_monitor=drift_monitor,
        ensemble_manager=ensemble,
    )
    components["scheduler"] = scheduler

    # ── 11. API Server ────────────────────────────────────────────────────────
    from deployment.api_server import app, set_engine
    set_engine(engine, drift_monitor)

    def run_api():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.api_port, log_level="warning")

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    components["api_thread"] = api_thread

    return components


# ═══════════════════════════════════════════════════════════════════════════════
# Demo Mode — showcase all Phase 4 features in a short run
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo(args, components: Dict):
    """Run a condensed demo of all Phase 4 features."""
    engine = components["engine"]
    ensemble = components["ensemble"]
    drift_monitor = components["drift_monitor"]
    retrain_pipeline = components["retrain_pipeline"]
    alerts = components["alerts"]
    scheduler = components["scheduler"]

    logger.info("\n" + "="*60)
    logger.info("PHASE 4 DEMO: All components active")
    logger.info("="*60)

    # 1. Execute trading cycles
    logger.info("\n[1/5] Running trading cycles with ensemble...")
    for cycle in range(args.max_cycles or 3):
        state = engine._execute_cycle(cycle)
        leaderboard = ensemble.get_leaderboard()
        logger.info(f"  Cycle {cycle+1}: ${state.portfolio_value:,.0f} | "
                    f"Top model: {leaderboard[0]['name'] if leaderboard else 'N/A'}")
        time.sleep(0.1)

    # 2. Drift detection
    logger.info("\n[2/5] Testing drift detection...")
    for i in range(20):
        ret = -0.005 if i > 12 else 0.001
        drift_monitor.update(ret)
    drift_events = [r for r in drift_monitor.reports if r.drift_detected]
    logger.info(f"  Drift checks: {len(drift_monitor.reports)} | events: {len(drift_events)}")

    # 3. Champion/Challenger evaluation
    logger.info("\n[3/5] Champion/Challenger evaluation...")
    from automation.retrain_pipeline import ChampionChallengerEvaluator, RetrainConfig
    evaluator = ChampionChallengerEvaluator(RetrainConfig(
        oos_eval_days=30, n_eval_episodes=3, min_sharpe_absolute=0.0,
        min_sharpe_improvement=-9.9, confidence_level=0.9))
    promote, old_sh, new_sh, delta, pval = evaluator.evaluate(
        components["model_a"], components["model_b"],
        components["feature_store"], components["symbols"], n_episodes=3)
    logger.info(f"  model_a={old_sh:.3f} vs model_b={new_sh:.3f} | Δ={delta:+.3f} | promote={promote}")

    # 4. Ensemble leaderboard
    logger.info("\n[4/5] Ensemble leaderboard...")
    ensemble.update_performance(1.2, "model_a")
    ensemble.update_performance(0.8, "model_b")
    for row in ensemble.get_leaderboard():
        logger.info(f"  {row['name']}: sharpe={row['mean_sharpe']:.3f} | active={row['active']}")

    # 5. Scheduler status
    logger.info("\n[5/5] Scheduler jobs...")
    for job in scheduler.get_job_status():
        logger.info(f"  {job['name']}: next={job['next_run']}")

    # 6. Alerts summary
    logger.info("\n--- Alert History ---")
    for a in alerts.get_history(last_n=5):
        logger.info(f"  [{a['level'].upper()}] {a['message'][:60]}")

    # Final state
    state = engine.state
    logger.info(f"\n{'='*60}")
    logger.info(f"DEMO COMPLETE")
    logger.info(f"  Portfolio: ${state.portfolio_value:,.0f}")
    logger.info(f"  Return:    {state.total_return:+.2%}")
    logger.info(f"  Cycles:    {state.cycle_count}")
    logger.info(f"  Halted:    {state.is_halted}")
    logger.info(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Trader Phase 4 — Autonomous System")
    parser.add_argument("--mode", choices=["paper", "alpaca", "live", "demo"], default="demo")
    parser.add_argument("--n-stocks", type=int, default=5)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cycle-interval", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--use-synthetic", action="store_true", default=True)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("INFO", LOG_DIR / "phase4.log")
    set_seed(args.seed)

    logger.info("\n╔" + "═"*58 + "╗")
    logger.info("║" + "   RL TRADER — PHASE 4: AUTONOMOUS SYSTEM".center(58) + "║")
    logger.info("║" + f"   Mode: {args.mode} | Capital: ${args.capital:,.0f} | DryRun: {args.dry_run}".center(58) + "║")
    logger.info("╚" + "═"*58 + "╝\n")

    # Bootstrap all components
    components = bootstrap_system(args)
    time.sleep(1)  # let API server start

    logger.info(f"\n✓ All systems initialized")
    logger.info(f"  API:     http://localhost:{args.api_port}/docs")
    logger.info(f"  Metrics: http://localhost:9090/metrics")

    if args.mode == "demo":
        run_demo(args, components)
    else:
        # Start scheduler
        scheduler = components["scheduler"]
        scheduler.start()

        engine = components["engine"]
        alerts = components["alerts"]

        alerts.send(
            f"RL Trader started | Mode={args.mode} | Capital=${args.capital:,.0f}",
            level="info",
        )

        try:
            engine.run(max_cycles=args.max_cycles)
        except KeyboardInterrupt:
            logger.info("\nShutdown requested...")
            engine.stop()
            scheduler.stop()

        alerts.send(
            f"RL Trader stopped | Return={engine.state.total_return:+.2%}",
            level="info",
        )

    # Save system report
    report = {
        "phase": 4,
        "mode": args.mode,
        "symbols": components.get("symbols", []),
        "final_portfolio": getattr(components.get("engine"), "state", None) and
                          components["engine"].state.portfolio_value,
    }
    with open(LOG_DIR / "phase4_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n✓ Phase 4 complete. Report saved to {LOG_DIR}/phase4_report.json")


if __name__ == "__main__":
    main()
