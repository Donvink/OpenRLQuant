"""
Global configuration for RL Trading System - Phase 1
All parameters centralized here for easy tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# ─── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Data Sources ───────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    # API Keys (set via environment variables in production)
    polygon_api_key: str = ""          # https://polygon.io  (free tier: 5 calls/min)
    alpha_vantage_key: str = ""        # https://alphavantage.co
    news_api_key: str = ""             # https://newsapi.org
    finnhub_key: str = ""              # https://finnhub.io  (free tier available)

    # Universe
    # S&P 500 subset for Phase 1 — start small, validate, then scale
    universe: List[str] = field(default_factory=lambda: [
        # Tech
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA",
        # Finance
        "JPM", "BAC", "GS", "MS",
        # Healthcare
        "JNJ", "UNH", "PFE",
        # Energy
        "XOM", "CVX",
        # Consumer
        "WMT", "HD", "MCD",
        # Benchmark ETFs
        "SPY", "QQQ", "IWM", "VIX",
    ])

    # Time range
    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"

    # OHLCV granularity options: "1d", "1h", "5m"
    primary_timeframe: str = "1d"
    intraday_timeframe: str = "1h"   # for microstructure features

    # Cache settings
    cache_ttl_seconds: int = 3600    # 1 hour for intraday, 24h for daily
    use_cache: bool = True


# ─── Feature Engineering ────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    # Price-based technical indicators
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_windows: List[int] = field(default_factory=lambda: [5, 12, 26, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    vwap_window: int = 20

    # Volume indicators
    obv_enabled: bool = True
    cmf_period: int = 20             # Chaikin Money Flow
    mfi_period: int = 14             # Money Flow Index

    # Momentum
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # Return features
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 21, 63])
    log_returns: bool = True

    # Volatility features
    realized_vol_windows: List[int] = field(default_factory=lambda: [5, 21, 63])
    parkinson_vol: bool = True       # High-Low volatility estimator

    # Fundamental features (quarterly)
    fundamental_features: List[str] = field(default_factory=lambda: [
        "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
        "roe", "roa", "debt_to_equity", "current_ratio",
        "revenue_growth_yoy", "earnings_growth_yoy",
        "gross_margin", "operating_margin", "net_margin",
        "free_cash_flow_yield", "dividend_yield",
    ])

    # NLP sentiment
    sentiment_model: str = "ProsusAI/finbert"   # HuggingFace model
    sentiment_lookback_days: int = 3             # aggregate last N days of news
    sentiment_batch_size: int = 32
    max_news_per_day: int = 20                   # top N articles by relevance

    # Macro / market regime features
    macro_features: List[str] = field(default_factory=lambda: [
        "vix", "spy_return", "qqq_return", "iwm_return",
        "dxy",          # US Dollar Index
        "treasury_10y", # 10Y yield
        "credit_spread", # HY - IG spread
    ])

    # Feature normalization: "zscore", "minmax", "robust"
    normalization: str = "zscore"
    zscore_window: int = 252         # rolling 1-year z-score


# ─── Environment ────────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    initial_capital: float = 1_000_000.0   # $1M starting portfolio
    max_position_pct: float = 0.10          # max 10% per single stock
    min_trade_amount: float = 1000.0        # minimum order size ($)
    transaction_cost_bps: float = 5.0       # 5bps per trade (0.05%)
    slippage_bps: float = 3.0              # 3bps slippage model
    market_impact_factor: float = 0.1      # sqrt(size/adv) impact
    short_enabled: bool = False            # Phase 1: long-only
    margin_ratio: float = 1.0             # no leverage in Phase 1

    # Observation window
    lookback_window: int = 60              # 60 trading days of history
    use_portfolio_features: bool = True    # include account state in obs

    # Episode settings
    episode_length: int = 252              # 1 trading year per episode
    reset_on_bankruptcy: bool = True       # end episode if portfolio < 10%

    # Reward shaping
    reward_type: str = "sharpe"           # "log_return", "sharpe", "calmar"
    sharpe_window: int = 21               # rolling window for sharpe calc
    drawdown_penalty: float = 0.5         # weight on max drawdown penalty
    turnover_penalty: float = 0.01        # penalty for excessive trading


# ─── Risk Management ────────────────────────────────────────────────────────────
@dataclass
class RiskConfig:
    # Hard limits — breach triggers immediate action
    max_drawdown_halt: float = 0.08       # halt trading if daily DD > 8%
    max_portfolio_drawdown: float = 0.15  # liquidate if total DD > 15%
    max_single_position: float = 0.10     # 10% per stock
    max_sector_exposure: float = 0.30     # 30% per sector
    min_liquidity_adv_ratio: float = 0.01 # max 1% of avg daily volume

    # VIX regime rules
    vix_normal_threshold: float = 20.0
    vix_elevated_threshold: float = 30.0
    vix_extreme_threshold: float = 40.0
    vix_elevated_position_scale: float = 0.7   # reduce to 70% when VIX > 30
    vix_extreme_position_scale: float = 0.3    # reduce to 30% when VIX > 40

    # Correlation limits
    max_portfolio_correlation: float = 0.7
    correlation_lookback: int = 63        # 3-month rolling correlation

    # Kelly criterion
    kelly_fraction: float = 0.25          # use 25% Kelly (conservative)
    kelly_lookback: int = 126             # 6 months to estimate win rate


# ─── Assembled Config ────────────────────────────────────────────────────────────
class Config:
    data = DataConfig()
    features = FeatureConfig()
    env = EnvConfig()
    risk = RiskConfig()

    @classmethod
    def from_env(cls):
        """Load API keys from environment variables."""
        import os
        cfg = cls()
        cfg.data.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        cfg.data.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        cfg.data.news_api_key = os.getenv("NEWS_API_KEY", "")
        cfg.data.finnhub_key = os.getenv("FINNHUB_KEY", "")
        return cfg


CONFIG = Config()
