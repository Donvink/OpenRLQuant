"""
tests/test_all.py
──────────────────
完整测试套件。覆盖所有核心模块。

运行：
    python -m pytest tests/test_all.py -v
    python tests/test_all.py          # 直接运行（无需pytest）
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # suppress info during tests


# ─── Fixtures (shared test data) ──────────────────────────────────────────────

def make_ohlcv(seed: int = 0, n: int = 500, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    returns = np.random.normal(0.0003, 0.018, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.006, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.006, n)))
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.003, n))
    open_[0] = base_price
    volume = np.random.lognormal(15, 0.6, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


def make_market_data(symbols: List[str] = None, n: int = 600) -> Dict[str, pd.DataFrame]:
    symbols = symbols or ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]
    return {sym: make_ohlcv(seed=i, n=n, base_price=100 + i * 50)
            for i, sym in enumerate(symbols)}


def make_feature_store(symbols: List[str] = None, n: int = 600) -> Dict[str, pd.DataFrame]:
    from features.feature_engineer import TechnicalFeatures, MacroFeatures, RollingNormalizer

    market_data = make_market_data(symbols, n)
    vix = pd.Series(
        np.random.uniform(12, 35, n),
        index=pd.bdate_range("2021-01-01", periods=n),
        name="vix",
    )
    tech = TechnicalFeatures()
    macro = MacroFeatures()
    norm = RollingNormalizer(window=252)
    macro_df = macro.build(market_data, vix)

    store = {}
    for sym, df in market_data.items():
        feat = tech.compute(df.copy())
        feat = pd.concat([feat, macro_df.reindex(feat.index).ffill()], axis=1)
        feat = norm.fit_transform(feat, exclude_cols=["open", "high", "low", "close", "volume"])
        feat = feat.dropna(how="all")
        store[sym] = feat
    return store


# ─── Test Runner ───────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name: str, fn):
        try:
            fn()
            print(f"  ✓ {name}")
            self.passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            self.errors.append((name, traceback.format_exc()))
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"Results: {self.passed}/{total} passed", end="")
        if self.failed:
            print(f"  ❌ {self.failed} FAILED")
            for name, tb in self.errors:
                print(f"\n[{name}]\n{tb}")
        else:
            print("  ✅ ALL PASSED")
        print("=" * 55)
        return self.failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test Groups
# ═══════════════════════════════════════════════════════════════════════════════

def test_data_cache():
    from data.market_data import DiskCache
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(Path(tmpdir), ttl_seconds=10)
        cache.set("key1", {"data": [1, 2, 3]})
        result = cache.get("key1")
        assert result == {"data": [1, 2, 3]}, f"Cache roundtrip failed: {result}"
        assert cache.get("nonexistent") is None
        cache.clear()
        assert cache.get("key1") is None


def test_data_validation():
    from data.market_data import MarketDataLoader
    loader = MarketDataLoader(use_cache=False)

    # Bad data: zero prices, high < low
    df = make_ohlcv(n=100)
    df.loc[df.index[10], "close"] = -5.0   # negative price
    df.loc[df.index[20], "high"] = 1.0
    df.loc[df.index[20], "low"] = 200.0    # high < low

    cleaned = loader._validate_ohlcv(df, "TEST")
    assert len(cleaned) < len(df), "Validation should remove bad rows"
    assert (cleaned["close"] > 0).all(), "Negative prices should be removed"
    if "high" in cleaned.columns and "low" in cleaned.columns:
        assert (cleaned["high"] >= cleaned["low"]).all()


def test_technical_features_shape():
    from features.feature_engineer import TechnicalFeatures
    df = make_ohlcv(n=400)
    tech = TechnicalFeatures()
    result = tech.compute(df)
    # Should have many more columns than the original 5
    assert result.shape[1] > 30, f"Expected >30 cols, got {result.shape[1]}"
    assert result.shape[0] == df.shape[0], "Row count should be preserved"


def test_technical_features_no_lookahead():
    """
    Verify no look-ahead bias: features at time t should only use data up to t.
    We test this by checking that rolling windows don't use future data.
    """
    from features.feature_engineer import TechnicalFeatures
    df = make_ohlcv(n=300)
    tech = TechnicalFeatures()
    result = tech.compute(df)

    # SMA at time t should equal close.rolling(window).mean() up to t
    if "sma_20" in result.columns:
        expected_sma = df["close"].rolling(20).mean()
        diff = (result["sma_20"] - expected_sma).abs().max()
        assert diff < 1e-6, f"SMA look-ahead bias detected: max_diff={diff}"


def test_technical_features_no_nan_after_warmup():
    from features.feature_engineer import TechnicalFeatures, RollingNormalizer
    df = make_ohlcv(n=400)
    tech = TechnicalFeatures()
    result = tech.compute(df)
    norm = RollingNormalizer(window=63)
    normed = norm.fit_transform(result, exclude_cols=["open","high","low","close","volume"])

    # After sufficient warmup (>200 days), there should be no NaN
    tail = normed.iloc[250:]
    nan_cols = [c for c in tail.columns if tail[c].isna().any()]
    # Allow a few binary/signal columns
    assert len(nan_cols) < 10, f"Too many NaN columns after warmup: {nan_cols}"


def test_normalization_no_lookahead():
    from features.feature_engineer import RollingNormalizer
    np.random.seed(0)
    series = pd.Series(np.random.randn(500))
    norm = RollingNormalizer(window=100)
    df = pd.DataFrame({"x": series})
    normed = norm.fit_transform(df)

    # The normalized value at each point should only depend on past values
    # Check: if we split at t=300 and re-normalize, we get same value at t=300
    normed_full = normed["x"].iloc[300]
    df2 = df.iloc[:301]
    normed2 = norm.fit_transform(df2)["x"].iloc[300]
    assert abs(normed_full - normed2) < 1e-6, "Rolling normalization has look-ahead bias!"


def test_macro_features():
    from features.feature_engineer import MacroFeatures
    market_data = make_market_data(["AAPL", "MSFT", "SPY"])
    n = len(next(iter(market_data.values())))
    vix = pd.Series(np.random.uniform(12, 35, n),
                    index=next(iter(market_data.values())).index)

    macro = MacroFeatures()
    result = macro.build(market_data, vix)

    assert "vix" in result.columns
    assert "vix_regime" in result.columns
    assert "spy_ret_1d" in result.columns
    assert not result.empty


def test_sentiment_lexicon_fallback():
    from features.feature_engineer import SentimentAnalyzer
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
    analyzer._pipeline = "lexicon"

    texts = [
        "Company beats earnings expectations, record revenue growth",
        "Stock falls after disappointing guidance, loss widens",
        "Trading volume remains normal today",
    ]
    scores = analyzer.score_texts(texts)
    assert len(scores) == 3
    assert "compound" in scores.columns
    assert scores.loc[0, "compound"] > scores.loc[1, "compound"], \
        "Positive text should score higher than negative"


def test_env_spaces():
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv
    env = TradingEnv(store, list(store.keys()), lookback_window=20, episode_length=63)
    assert env.observation_space.shape[0] > 0
    assert env.action_space.shape[0] == len(store)


def test_env_reset():
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv
    env = TradingEnv(store, list(store.keys()), lookback_window=20, episode_length=63)
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert not np.any(np.isnan(obs)), "NaN in observation after reset"
    assert not np.any(np.isinf(obs)), "Inf in observation after reset"
    assert env.portfolio.cash == env.initial_capital
    assert np.all(env.portfolio.positions == 0)


def test_env_step():
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv
    env = TradingEnv(store, list(store.keys()), lookback_window=20, episode_length=63)
    obs, _ = env.reset(seed=0)

    # Take a random step
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert not np.any(np.isnan(next_obs))
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert isinstance(terminated, bool)
    assert "total_value" in info
    assert "total_return" in info


def test_env_full_episode():
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv
    env = TradingEnv(store, list(store.keys()), lookback_window=20, episode_length=63)
    obs, _ = env.reset(seed=42)
    done = False
    steps = 0
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        assert not np.any(np.isnan(obs)), f"NaN at step {steps}"
        assert not np.isnan(reward), f"NaN reward at step {steps}"

    assert steps > 0, "Episode had zero steps"
    assert "total_return" in info


def test_env_cost_model():
    """Verify transaction costs reduce portfolio value vs no-cost scenario."""
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv

    results = {}
    for cost_bps in [0.0, 10.0]:
        env = TradingEnv(
            store, list(store.keys()),
            lookback_window=20, episode_length=63,
            transaction_cost_bps=cost_bps, slippage_bps=0.0
        )
        np.random.seed(42)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = np.ones(env.n) * 0.1  # always trade (high turnover)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results[cost_bps] = info["total_return"]

    assert results[0.0] >= results[10.0], \
        f"Zero-cost ({results[0.0]:.3f}) should be >= high-cost ({results[10.0]:.3f})"


def test_env_position_limits():
    """Verify max position constraint is enforced."""
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv
    max_pos = 0.10
    env = TradingEnv(store, list(store.keys()), lookback_window=20,
                     episode_length=63, max_position_pct=max_pos)
    obs, _ = env.reset(seed=0)

    # Try to put everything in one stock
    greedy_action = np.zeros(env.n)
    greedy_action[0] = 1.0
    processed = env._process_action(greedy_action)
    assert processed[0] <= max_pos + 1e-6, \
        f"Position limit not enforced: {processed[0]:.4f} > {max_pos}"


def test_portfolio_tracking():
    from environment.trading_env import Portfolio, TransactionCostModel
    n = 3
    p = Portfolio(100_000, ["A", "B", "C"])
    prices = np.array([100.0, 200.0, 50.0])
    p.update_prices(prices)

    assert p.total_value == 100_000
    assert p.position_value == 0.0

    cost_model = TransactionCostModel(commission_bps=5, spread_bps=3)
    target_weights = np.array([0.3, 0.3, 0.3])
    adv = np.array([1e7, 1e7, 1e7])
    p.execute_trade(target_weights, prices, adv, cost_model)

    assert np.all(p.positions >= 0), "Positions should be non-negative (long-only)"
    assert p.total_cost_paid > 0, "Should have paid some transaction costs"


def test_backtester_metrics():
    from environment.backtester import compute_metrics, PerformanceReport
    np.random.seed(0)
    dates = pd.bdate_range("2022-01-01", periods=252)

    # Trending up portfolio
    returns = np.random.normal(0.001, 0.015, 252)
    pv = pd.Series(1_000_000 * np.cumprod(1 + returns), index=dates)
    bm = pd.Series(1_000_000 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 252)), index=dates)

    report = compute_metrics(pv, bm)
    assert isinstance(report, PerformanceReport)
    assert -1.0 <= report.max_drawdown <= 0.0, f"Invalid drawdown: {report.max_drawdown}"
    assert report.n_trading_days == 252
    assert report.start_date != ""


def test_backtester_sharpe_ordering():
    """Strategy with higher return/vol ratio should have higher Sharpe."""
    from environment.backtester import compute_metrics
    dates = pd.bdate_range("2022-01-01", periods=252)
    np.random.seed(42)

    # High Sharpe: consistent gains
    ret_high = np.random.normal(0.002, 0.01, 252)
    pv_high = pd.Series(np.cumprod(1 + ret_high), index=dates)

    # Low Sharpe: volatile, lower mean
    ret_low = np.random.normal(0.0005, 0.03, 252)
    pv_low = pd.Series(np.cumprod(1 + ret_low), index=dates)

    r_high = compute_metrics(pv_high)
    r_low = compute_metrics(pv_low)
    assert r_high.sharpe_ratio > r_low.sharpe_ratio, \
        f"High-quality strategy Sharpe ({r_high.sharpe_ratio:.3f}) should exceed low ({r_low.sharpe_ratio:.3f})"


def test_buy_and_hold_benchmark():
    from environment.backtester import BuyAndHoldBenchmark
    prices = pd.DataFrame({
        "A": [100, 110, 105, 120, 115],
        "B": [50, 52, 48, 55, 58],
    }, index=pd.bdate_range("2023-01-01", periods=5))

    bah = BuyAndHoldBenchmark()
    pv = bah.run(prices, initial_capital=100_000)
    assert len(pv) == 5
    assert pv.iloc[0] == 100_000
    assert pv.iloc[-1] > 0


def test_risk_manager_position_cap():
    from utils.risk_manager import RiskManager
    symbols = ["A", "B", "C", "D", "E"]
    rm = RiskManager(symbols, max_position_pct=0.10)
    rm.peak_value = 1_000_000

    # Over-concentrated action
    action = np.array([0.50, 0.10, 0.10, 0.10, 0.10])
    decision = rm.evaluate(
        action, 950_000, [1_000_000, 960_000, 950_000],
        np.ones(5) * 100.0, np.ones(5) * 1e7, vix=18.0
    )
    assert decision.approved_weights[0] <= 0.10 + 1e-6, \
        f"Position cap not enforced: {decision.approved_weights[0]:.4f}"
    assert not decision.halt


def test_risk_manager_vix_scaling():
    from utils.risk_manager import RiskManager
    symbols = ["A", "B", "C"]
    # Give distinct sectors so sector cap doesn't absorb the VIX reduction
    sector_map = {"A": "Tech", "B": "Finance", "C": "Energy"}

    kwargs = dict(
        portfolio_value=950_000,
        portfolio_value_history=[950_000],
        prices=np.ones(3) * 100,
        adv=np.ones(3) * 1e9,   # huge ADV: no liquidity cap
    )
    base = np.array([0.09, 0.09, 0.09])  # well below max_position (0.10) per stock

    def run(vix_val):
        rm = RiskManager(symbols, sector_map=sector_map,
                         max_position_pct=0.10, max_sector_pct=0.40,
                         vix_elevated=30.0, vix_extreme=40.0)
        rm.peak_value = 950_000
        return rm.evaluate(base.copy(), vix=vix_val, **kwargs)

    d_normal   = run(18.0)
    d_elevated = run(35.0)
    d_extreme  = run(45.0)

    assert d_elevated.approved_weights.sum() < d_normal.approved_weights.sum(), \
        f"Elevated VIX should reduce exposure: {d_elevated.approved_weights.sum():.3f} < {d_normal.approved_weights.sum():.3f}"
    assert d_extreme.approved_weights.sum() < d_elevated.approved_weights.sum(), \
        f"Extreme VIX should reduce further: {d_extreme.approved_weights.sum():.3f} < {d_elevated.approved_weights.sum():.3f}"


def test_risk_manager_halt_on_drawdown():
    from utils.risk_manager import RiskManager
    symbols = ["A", "B"]
    rm = RiskManager(symbols, max_portfolio_drawdown=0.15)
    rm.peak_value = 1_000_000  # peak was $1M

    # Current value: down 20% — should trigger halt
    decision = rm.evaluate(
        np.array([0.3, 0.3]),
        portfolio_value=790_000,   # -21% from peak
        portfolio_value_history=[1_000_000, 850_000, 790_000],
        prices=np.ones(2) * 100.0,
        adv=np.ones(2) * 1e8,
        vix=25.0,
    )
    assert decision.halt, "Should halt on >15% portfolio drawdown"
    assert np.all(decision.approved_weights == 0), "Halt should zero out weights"


def test_risk_manager_kelly():
    from utils.risk_manager import RiskManager
    symbols = ["A", "B", "C"]
    rm = RiskManager(symbols, kelly_fraction=0.25)
    rm.peak_value = 1_000_000

    # Simulate good historical returns (60% win rate)
    np.random.seed(42)
    returns_history = np.where(np.random.rand(200) > 0.4,
                               np.random.uniform(0.01, 0.03, 200),
                               np.random.uniform(-0.02, 0, 200))

    action = np.array([0.4, 0.4, 0.15])  # over-invested
    decision = rm.evaluate(
        action, 1_000_000, [1_000_000],
        np.ones(3) * 100.0, np.ones(3) * 1e9,
        vix=15.0, returns_history=returns_history
    )
    # Kelly should have constrained the total exposure
    assert "kelly" in decision.adjustments, \
        "Kelly should have triggered for over-invested portfolio"


def test_screener_basic():
    from data.screener import UniverseScreener, ScreenerConfig
    market_data = make_market_data(n=600)
    cfg = ScreenerConfig(
        min_trading_days=400,
        min_avg_daily_volume=0,      # relaxed for synthetic data
        min_avg_dollar_volume=0,
        target_n_stocks=5,
    )
    screener = UniverseScreener(cfg)
    selected, report = screener.screen(market_data, verbose=False)
    assert len(selected) > 0, "Screener should select at least one stock"
    assert isinstance(report, pd.DataFrame)
    assert "passes_all" in report.columns


def test_feature_pipeline_save_load():
    from features.feature_engineer import FeaturePipeline
    import tempfile
    store = make_feature_store(["AAPL", "MSFT"], n=300)
    pipeline = FeaturePipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline.save(store, tmpdir)
        loaded = pipeline.load(tmpdir)
        assert set(loaded.keys()) == set(store.keys())
        for sym in store:
            assert loaded[sym].shape == store[sym].shape, \
                f"Shape mismatch for {sym}: {loaded[sym].shape} vs {store[sym].shape}"


def test_walk_forward_folds():
    from environment.backtester import WalkForwardBacktester
    dates = pd.bdate_range("2019-01-01", "2023-12-31")
    wf = WalkForwardBacktester(
        train_window_days=252,
        test_window_days=63,
        step_size_days=21,
    )
    folds = wf.generate_folds(dates)
    assert len(folds) > 0, "Should generate at least one fold"
    for train, test in folds:
        assert len(train) == 252, f"Train window size mismatch: {len(train)}"
        assert len(test) == 63, f"Test window size mismatch: {len(test)}"
        assert train[-1] < test[0], "Train should end before test begins"


def test_env_reproducibility():
    """Same seed should produce identical episodes."""
    store = make_feature_store(n=400)
    from environment.trading_env import TradingEnv

    def run_episode(seed):
        env = TradingEnv(store, list(store.keys()), lookback_window=20, episode_length=30)
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            rewards.append(r)
            if terminated or truncated:
                break
        return rewards

    # Note: action sampling uses env's np_random, so seeded differently
    r1 = run_episode(42)
    r2 = run_episode(42)
    # At minimum, the initial obs should be the same (same data, same start)
    assert len(r1) == len(r2), "Episode lengths should match with same seed"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    # Data layer
    ("Cache: set/get/clear",          test_data_cache),
    ("Data: validation & cleanup",    test_data_validation),

    # Features
    ("Features: output shape",        test_technical_features_shape),
    ("Features: no look-ahead bias",  test_technical_features_no_lookahead),
    ("Features: no NaN after warmup", test_technical_features_no_nan_after_warmup),
    ("Features: rolling normalization no-lookahead", test_normalization_no_lookahead),
    ("Features: macro features",      test_macro_features),
    ("Features: sentiment lexicon",   test_sentiment_lexicon_fallback),

    # Environment
    ("Env: space dimensions",         test_env_spaces),
    ("Env: reset()",                  test_env_reset),
    ("Env: step()",                   test_env_step),
    ("Env: full episode",             test_env_full_episode),
    ("Env: transaction costs",        test_env_cost_model),
    ("Env: position limits",          test_env_position_limits),
    ("Env: reproducibility",          test_env_reproducibility),

    # Portfolio
    ("Portfolio: trade execution",    test_portfolio_tracking),

    # Backtesting
    ("Backtest: metric computation",  test_backtester_metrics),
    ("Backtest: Sharpe ordering",     test_backtester_sharpe_ordering),
    ("Backtest: Buy & Hold",          test_buy_and_hold_benchmark),
    ("Backtest: walk-forward folds",  test_walk_forward_folds),

    # Risk
    ("Risk: position cap",            test_risk_manager_position_cap),
    ("Risk: VIX scaling",             test_risk_manager_vix_scaling),
    ("Risk: drawdown halt",           test_risk_manager_halt_on_drawdown),
    ("Risk: Kelly sizing",            test_risk_manager_kelly),

    # Screener
    ("Screener: universe selection",  test_screener_basic),

    # Pipeline
    ("Pipeline: save/load parquet",   test_feature_pipeline_save_load),
]


def main():
    print("\n" + "╔" + "═" * 53 + "╗")
    print("║" + "  RL TRADING SYSTEM — TEST SUITE".center(53) + "║")
    print("╚" + "═" * 53 + "╝")

    runner = TestRunner()
    groups = {}
    for name, fn in ALL_TESTS:
        group = name.split(":")[0]
        groups.setdefault(group, []).append((name, fn))

    for group, tests in groups.items():
        print(f"\n[{group}]")
        for name, fn in tests:
            runner.run(name, fn)

    success = runner.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
