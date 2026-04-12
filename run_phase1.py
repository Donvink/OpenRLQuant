"""
run_phase1.py
──────────────
Phase 1 主运行脚本：数据采集 → 特征工程 → 环境验证 → 基线回测

运行步骤:
  1. 数据下载与缓存（Yahoo Finance，无需API Key）
  2. 特征工程（技术指标 + 宏观因子）
  3. 环境单元测试（随机动作验证环境正确性）
  4. 基线策略回测（Buy & Hold vs 随机策略）
  5. 输出性能报告

Usage:
    # 快速验证（小数据集）
    python run_phase1.py --mode quick

    # 完整数据集
    python run_phase1.py --mode full

    # 指定股票
    python run_phase1.py --symbols AAPL MSFT GOOGL NVDA --start 2020-01-01
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG, DATA_DIR, LOG_DIR
from data.market_data import MarketDataLoader
from features.feature_engineer import FeaturePipeline
from environment.trading_env import TradingEnv
from environment.backtester import (
    BuyAndHoldBenchmark, compute_metrics, compare_strategies, print_comparison_table
)
from utils.risk_manager import RiskManager


# ── Logging setup ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "phase1.log"),
    ],
)
logger = logging.getLogger("phase1")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Data Download
# ═══════════════════════════════════════════════════════════════════════════════

def step1_download_data(symbols: list, start: str, end: str) -> dict:
    """Download and cache OHLCV data for all symbols."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA DOWNLOAD")
    logger.info("=" * 60)

    loader = MarketDataLoader(
        polygon_key=os.getenv("POLYGON_API_KEY", ""),
        finnhub_key=os.getenv("FINNHUB_KEY", ""),
        cache_dir=DATA_DIR / "cache",
        cache_ttl=86400,
        use_cache=True,
    )

    logger.info(f"Downloading {len(symbols)} symbols: {start} → {end}")
    market_data = loader.get_ohlcv_universe(symbols, start, end)

    # Report
    logger.info(f"\n{'Symbol':<8} {'Start':<12} {'End':<12} {'Days':>6} {'AvgVol':>12}")
    logger.info("-" * 55)
    ok_symbols = []
    for sym, df in sorted(market_data.items()):
        if df.empty:
            logger.warning(f"{sym:<8} — NO DATA")
            continue
        avg_vol = df["volume"].mean() if "volume" in df else 0
        logger.info(
            f"{sym:<8} {str(df.index[0].date()):<12} {str(df.index[-1].date()):<12} "
            f"{len(df):>6,} {avg_vol:>12,.0f}"
        )
        ok_symbols.append(sym)

    logger.info(f"\n✓ Downloaded {len(ok_symbols)}/{len(symbols)} symbols successfully")

    # Get VIX
    vix = loader.get_vix(start, end)
    logger.info(f"✓ VIX data: {len(vix)} days (mean={vix.mean():.1f})")

    return market_data, vix, loader


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def step2_build_features(market_data: dict, vix: pd.Series) -> dict:
    """Build full feature store from raw market data."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    pipeline = FeaturePipeline()

    # Note: In Phase 1, we skip news sentiment if no Finnhub key
    # (add news_df=news_df to enable once API key is configured)
    feature_store = pipeline.build(
        market_data=market_data,
        news_df=None,           # Will enable in Phase 2 with real API key
        fundamentals=None,      # Will enable in Phase 2
        vix=vix,
    )

    # Report feature stats
    if feature_store:
        sample_sym = next(iter(feature_store))
        sample_df = feature_store[sample_sym]
        n_features = len([c for c in sample_df.columns if c not in {"open","high","low","close","volume"}])
        logger.info(f"\nFeature store: {len(feature_store)} symbols × {n_features} features")
        logger.info(f"Date range: {sample_df.index[0].date()} → {sample_df.index[-1].date()}")
        logger.info(f"Feature columns (first 20): {list(sample_df.columns[:20])}")

        # Save feature store
        save_path = str(DATA_DIR / "processed")
        pipeline.save(feature_store, save_path)
        logger.info(f"✓ Feature store saved to {save_path}")

    return feature_store


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Environment Validation
# ═══════════════════════════════════════════════════════════════════════════════

def step3_validate_environment(feature_store: dict, symbols: list) -> None:
    """Sanity-check the Gymnasium environment with random actions."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: ENVIRONMENT VALIDATION")
    logger.info("=" * 60)

    # Use symbols that have data
    valid_symbols = [s for s in symbols if s in feature_store][:10]
    if not valid_symbols:
        logger.error("No valid symbols for environment test!")
        return

    env = TradingEnv(
        feature_store=feature_store,
        symbols=valid_symbols,
        initial_capital=1_000_000,
        lookback_window=30,         # shorter for quick test
        episode_length=63,          # 1 quarter
        reward_type="sharpe",
        render_mode="human",
    )

    logger.info(f"Environment created: {env.n} stocks")
    logger.info(f"Observation space: {env.observation_space.shape}")
    logger.info(f"Action space: {env.action_space.shape}")

    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape, f"Obs shape mismatch: {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in observation!"
    logger.info(f"✓ Reset OK | obs shape={obs.shape} | no NaN")

    # Test 5 random episodes
    episode_returns = []
    for ep in range(5):
        obs, _ = env.reset(seed=ep * 100)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        episode_returns.append(info["total_return"])
        logger.info(
            f"Episode {ep+1}: {steps} steps | "
            f"Return={info['total_return']:+.2%} | "
            f"MaxDD={info['max_drawdown']:.2%} | "
            f"TotalReward={total_reward:.4f}"
        )

    logger.info(
        f"\n✓ Environment validation PASSED\n"
        f"  Mean episode return: {np.mean(episode_returns):+.2%}\n"
        f"  Std episode return:  {np.std(episode_returns):.2%}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Baseline Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def step4_baseline_backtest(market_data: dict, feature_store: dict, symbols: list) -> None:
    """Compare baseline strategies as lower bounds to beat."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: BASELINE BACKTEST")
    logger.info("=" * 60)

    valid_symbols = [s for s in symbols if s in market_data and not market_data[s].empty]
    if not valid_symbols:
        logger.error("No valid symbols for backtest!")
        return

    # Build aligned price matrix
    price_frames = {s: market_data[s]["close"] for s in valid_symbols}
    prices = pd.DataFrame(price_frames).dropna()
    logger.info(f"Price matrix: {prices.shape[0]} days × {prices.shape[1]} stocks")

    # ── Strategy 1: Equal-weight Buy & Hold ───────────────────────────────────
    bah = BuyAndHoldBenchmark()
    bah_values = bah.run(prices, initial_capital=1_000_000)

    # ── Strategy 2: SPY Benchmark ─────────────────────────────────────────────
    spy_values = None
    if "SPY" in market_data and not market_data["SPY"].empty:
        spy_prices = market_data["SPY"]["close"]
        spy_aligned = spy_prices.reindex(prices.index).ffill()
        spy_values = 1_000_000 * spy_aligned / spy_aligned.iloc[0]

    # ── Strategy 3: Random RL agent (lower bound) ─────────────────────────────
    valid_for_env = [s for s in valid_symbols if s in feature_store][:8]
    random_values = None
    if len(valid_for_env) >= 2:
        try:
            env = TradingEnv(
                feature_store=feature_store,
                symbols=valid_for_env,
                initial_capital=1_000_000,
                lookback_window=30,
                episode_length=min(252, len(prices) - 60),
                mode="test",
            )
            obs, _ = env.reset(seed=42)
            values = [env.portfolio.total_value]
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                values.append(info["total_value"])
                done = terminated or truncated
            random_values = pd.Series(values)
            logger.info(f"Random agent: {len(values)} steps")
        except Exception as e:
            logger.warning(f"Random agent failed: {e}")

    # ── Compute and compare metrics ────────────────────────────────────────────
    strategies = {"Equal-Weight B&H": bah_values}
    if spy_values is not None:
        strategies["SPY Benchmark"] = spy_values

    spy_ref = spy_values

    for name, pv in strategies.items():
        report = compute_metrics(pv, spy_ref)
        logger.info(report.summary())

    if len(strategies) >= 2:
        comp = compare_strategies(strategies, spy_ref)
        print_comparison_table(comp)

    logger.info("✓ Baseline backtest complete")
    return strategies, spy_values


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Risk Manager Validation
# ═══════════════════════════════════════════════════════════════════════════════

def step5_risk_manager_test(symbols: list) -> None:
    """Validate risk manager logic with synthetic scenarios."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: RISK MANAGER VALIDATION")
    logger.info("=" * 60)

    sector_map = {s: "Technology" if s in ["AAPL","MSFT","GOOGL","NVDA","META","AMZN"] else "Other"
                  for s in symbols}

    rm = RiskManager(
        symbols=symbols[:5],
        sector_map=sector_map,
        max_position_pct=0.10,
        max_sector_pct=0.30,
        vix_elevated=30.0,
    )
    rm.peak_value = 1_000_000

    test_cases = [
        # (description, weights, vix, expected_behavior)
        ("Normal market",     np.array([0.1, 0.1, 0.1, 0.1, 0.1]), 18.0, "pass_through"),
        ("High VIX",          np.array([0.1, 0.1, 0.1, 0.1, 0.1]), 35.0, "scale_down"),
        ("Over-concentration",np.array([0.5, 0.0, 0.0, 0.0, 0.0]), 18.0, "cap_position"),
        ("Sector breach",     np.array([0.1, 0.1, 0.1, 0.0, 0.0]), 18.0, "cap_sector"),
        ("Extreme VIX",       np.array([0.2, 0.2, 0.2, 0.1, 0.1]), 45.0, "heavy_scale"),
    ]

    for desc, weights, vix, expected in test_cases:
        decision = rm.evaluate(
            agent_weights=weights,
            portfolio_value=950_000,
            portfolio_value_history=[1_000_000, 980_000, 960_000, 950_000],
            prices=np.array([150.0, 300.0, 2800.0, 420.0, 250.0]),
            adv=np.array([8e7, 5e7, 2e7, 4e7, 3e7]),
            vix=vix,
        )
        logger.info(
            f"{desc:<25} | VIX={vix:4.0f} | "
            f"In={weights.sum():.3f} → Out={decision.approved_weights.sum():.3f} | "
            f"Adj: {list(decision.adjustments.keys())}"
        )

    logger.info("✓ Risk manager validation complete")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Trader Phase 1")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick=5 stocks 2 years, full=20 stocks 6 years")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    # ── Configuration ──────────────────────────────────────────────────────────
    if args.mode == "quick":
        symbols = args.symbols or ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "SPY"]
        start = args.start or "2021-01-01"
    else:
        symbols = args.symbols or CONFIG.data.universe
        start = args.start or CONFIG.data.train_start

    end = args.end

    logger.info("\n" + "╔" + "═"*58 + "╗")
    logger.info("║" + "   RL TRADING SYSTEM — PHASE 1".center(58) + "║")
    logger.info("║" + f"   Mode: {args.mode} | {len(symbols)} symbols | {start}→{end}".center(58) + "║")
    logger.info("╚" + "═"*58 + "╝\n")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    # Step 1: Data
    if not args.skip_download:
        market_data, vix, loader = step1_download_data(symbols, start, end)
    else:
        from data.market_data import MarketDataLoader
        loader = MarketDataLoader(cache_dir=DATA_DIR/"cache", use_cache=True)
        market_data = loader.get_ohlcv_universe(symbols, start, end)
        vix = loader.get_vix(start, end)

    if not market_data:
        logger.error("No market data downloaded. Check internet connection.")
        return

    # Step 2: Features
    feature_store = step2_build_features(market_data, vix)

    if not feature_store:
        logger.error("Feature store is empty. Check data pipeline.")
        return

    # Step 3: Environment validation
    step3_validate_environment(feature_store, symbols)

    # Step 4: Baseline backtest
    step4_baseline_backtest(market_data, feature_store, symbols)

    # Step 5: Risk manager
    step5_risk_manager_test([s for s in symbols if s != "SPY"])

    logger.info("\n" + "╔" + "═"*58 + "╗")
    logger.info("║" + "   PHASE 1 COMPLETE ✓".center(58) + "║")
    logger.info("║" + "   Ready for Phase 2: PPO Agent Training".center(58) + "║")
    logger.info("╚" + "═"*58 + "╝\n")


if __name__ == "__main__":
    main()
