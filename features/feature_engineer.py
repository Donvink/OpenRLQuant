"""
features/feature_engineer.py
─────────────────────────────
Complete feature engineering pipeline for the RL trading agent.

Feature groups:
  1. Technical indicators  (price + volume)
  2. Return / volatility   (statistical)
  3. Fundamental ratios    (quarterly, forward-filled)
  4. NLP sentiment         (FinBERT on news headlines)
  5. Macro / regime        (VIX, yield curve, dollar index)
  6. Portfolio state       (current positions, P&L)

All features normalized via rolling z-score to prevent look-ahead bias.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)


# ─── Technical Indicators ──────────────────────────────────────────────────────

class TechnicalFeatures:
    """
    Computes technical indicators for a single symbol's OHLCV DataFrame.
    All computations are vectorized with pandas/numpy — no TA-Lib dependency
    (so this runs anywhere without binary installs).
    """

    def __init__(self, cfg=None):
        from config.settings import CONFIG
        self.cfg = cfg or CONFIG.features

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input:  OHLCV DataFrame (index=DatetimeIndex)
        Output: same DataFrame + all technical feature columns
        """
        df = df.copy()

        df = self._moving_averages(df)
        df = self._momentum(df)
        df = self._macd(df)
        df = self._rsi(df)
        df = self._bollinger_bands(df)
        df = self._atr(df)
        df = self._volume_features(df)
        df = self._return_features(df)
        df = self._volatility_features(df)
        df = self._price_patterns(df)

        return df

    # ── Moving Averages ────────────────────────────────────────────────────────

    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for w in self.cfg.sma_windows:
            df[f"sma_{w}"] = close.rolling(w).mean()
            df[f"sma_{w}_ratio"] = close / df[f"sma_{w}"] - 1  # price relative to MA

        for w in self.cfg.ema_windows:
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
            df[f"ema_{w}_ratio"] = close / df[f"ema_{w}"] - 1

        # Golden/Death cross signals
        df["sma_cross_50_200"] = np.where(
            df.get("sma_50", close) > df.get("sma_200", close), 1.0, -1.0
        )
        return df

    # ── Momentum ───────────────────────────────────────────────────────────────

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for w in self.cfg.momentum_windows:
            df[f"mom_{w}d"] = close / close.shift(w) - 1  # pure price momentum

        # Stochastic oscillator
        k = self.cfg.stoch_k_period
        d = self.cfg.stoch_d_period
        low_k = df["low"].rolling(k).min()
        high_k = df["high"].rolling(k).max()
        df["stoch_k"] = (df["close"] - low_k) / (high_k - low_k + 1e-8) * 100
        df["stoch_d"] = df["stoch_k"].rolling(d).mean()
        df["stoch_signal"] = np.where(df["stoch_k"] > df["stoch_d"], 1.0, -1.0)

        return df

    # ── MACD ───────────────────────────────────────────────────────────────────

    def _macd(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        fast = self.cfg.macd_fast
        slow = self.cfg.macd_slow
        sig = self.cfg.macd_signal

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1.0, -1.0)

        # Normalize MACD by price level
        df["macd_norm"] = df["macd"] / close
        return df

    # ── RSI ────────────────────────────────────────────────────────────────────

    def _rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.rsi_period
        delta = df["close"].diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - 100 / (1 + rs)

        # RSI regime labels
        df["rsi_oversold"] = (df["rsi"] < 30).astype(float)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(float)
        return df

    # ── Bollinger Bands ────────────────────────────────────────────────────────

    def _bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.bollinger_period
        n_std = self.cfg.bollinger_std
        close = df["close"]

        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        bandwidth = (upper - lower) / mid

        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_mid"] = mid
        df["bb_pct"] = (close - lower) / (upper - lower + 1e-8)  # [0,1] within bands
        df["bb_bandwidth"] = bandwidth
        df["bb_squeeze"] = (bandwidth < bandwidth.rolling(125).quantile(0.10)).astype(float)

        return df

    # ── ATR (Average True Range) ───────────────────────────────────────────────

    def _atr(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.atr_period
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        df["atr"] = tr.ewm(com=period - 1, adjust=False).mean()
        df["atr_pct"] = df["atr"] / close   # normalized by price
        return df

    # ── Volume Features ────────────────────────────────────────────────────────

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df["volume"]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Volume momentum
        for w in [5, 10, 20]:
            df[f"vol_ratio_{w}d"] = vol / vol.rolling(w).mean()

        # OBV (On-Balance Volume)
        direction = np.sign(close.diff())
        obv = (direction * vol).cumsum()
        df["obv"] = obv
        df["obv_ema"] = obv.ewm(span=20, adjust=False).mean()
        df["obv_signal"] = (obv > df["obv_ema"]).astype(float)

        # VWAP (rolling approximation)
        typical_price = (high + low + close) / 3
        df["vwap"] = (typical_price * vol).rolling(self.cfg.vwap_window).sum() / \
                     vol.rolling(self.cfg.vwap_window).sum()
        df["vwap_ratio"] = close / df["vwap"] - 1

        # Chaikin Money Flow
        mf_mult = ((close - low) - (high - close)) / (high - low + 1e-8)
        mf_vol = mf_mult * vol
        period = self.cfg.cmf_period
        df["cmf"] = mf_vol.rolling(period).sum() / vol.rolling(period).sum()

        # Money Flow Index
        mfi_period = self.cfg.mfi_period
        typical_price_diff = typical_price.diff()
        pos_flow = (typical_price * vol).where(typical_price_diff > 0, 0).rolling(mfi_period).sum()
        neg_flow = (typical_price * vol).where(typical_price_diff <= 0, 0).rolling(mfi_period).sum()
        df["mfi"] = 100 - 100 / (1 + pos_flow / (neg_flow + 1e-8))

        return df

    # ── Returns ────────────────────────────────────────────────────────────────

    def _return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for w in self.cfg.return_windows:
            if self.cfg.log_returns:
                df[f"log_ret_{w}d"] = np.log(close / close.shift(w))
            else:
                df[f"ret_{w}d"] = close.pct_change(w)

        # Overnight gap
        df["overnight_gap"] = df["open"] / close.shift(1) - 1

        # Intraday range
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"]

        # Close-to-open ratio
        df["close_open_ratio"] = close / df["open"] - 1

        return df

    # ── Volatility ─────────────────────────────────────────────────────────────

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1))

        for w in self.cfg.realized_vol_windows:
            df[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

        # Parkinson volatility (uses High-Low range, more efficient)
        if self.cfg.parkinson_vol:
            log_hl = np.log(df["high"] / df["low"]) ** 2
            df["parkinson_vol_21d"] = np.sqrt(
                log_hl.rolling(21).mean() / (4 * np.log(2)) * 252
            )

        # Volatility ratio (current vs historical)
        df["vol_ratio"] = df.get("realized_vol_5d", log_ret.rolling(5).std()) / \
                          (df.get("realized_vol_63d", log_ret.rolling(63).std()) + 1e-8)

        return df

    # ── Price Patterns ─────────────────────────────────────────────────────────

    def _price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # 52-week high/low distance
        df["dist_52w_high"] = close / high.rolling(252).max() - 1
        df["dist_52w_low"] = close / low.rolling(252).min() - 1

        # Candlestick body and shadow ratios
        body = (close - open_).abs()
        total_range = high - low + 1e-8
        df["body_ratio"] = body / total_range
        df["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / total_range
        df["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / total_range

        # Higher highs / lower lows (trend detection)
        df["higher_high_5d"] = (high > high.shift(1)).rolling(5).sum() / 5
        df["lower_low_5d"] = (low < low.shift(1)).rolling(5).sum() / 5

        return df


# ─── NLP Sentiment (FinBERT) ───────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Uses FinBERT (financial domain BERT) to score news sentiment.
    Model: ProsusAI/finbert — 3-class: positive / negative / neutral

    Lazy-loads the model on first use to avoid startup delay.
    Falls back to a simple lexicon-based scorer if transformers not available.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            import torch
            device = 0 if (self.device == "auto" and torch.cuda.is_available()) else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device,
                max_length=512,
                truncation=True,
                top_k=None,  # return all class scores
            )
            logger.info(f"Loaded FinBERT on {'GPU' if device == 0 else 'CPU'}")
        except ImportError:
            logger.warning("transformers not installed — using lexicon fallback")
            self._pipeline = "lexicon"

    def score_texts(self, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Input:  list of text strings (headlines, summaries)
        Output: DataFrame with columns: positive, negative, neutral, compound
        """
        self._load()

        if not texts:
            return pd.DataFrame(columns=["positive", "negative", "neutral", "compound"])

        if self._pipeline == "lexicon":
            return self._lexicon_score(texts)

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                outputs = self._pipeline(batch)
                for output in outputs:
                    scores = {item["label"].lower(): item["score"] for item in output}
                    results.append({
                        "positive": scores.get("positive", 0.0),
                        "negative": scores.get("negative", 0.0),
                        "neutral": scores.get("neutral", 0.0),
                        "compound": scores.get("positive", 0.0) - scores.get("negative", 0.0),
                    })
            except Exception as e:
                logger.error(f"FinBERT batch failed: {e}")
                results.extend([{"positive": 0, "negative": 0, "neutral": 1, "compound": 0}] * len(batch))

        return pd.DataFrame(results)

    def build_daily_sentiment(
        self,
        news_df: pd.DataFrame,
        lookback_days: int = 3,
        max_articles: int = 20,
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment by (date, symbol).
        Returns DataFrame indexed by date with columns:
          {symbol}_sent_pos, {symbol}_sent_neg, {symbol}_sent_compound, {symbol}_n_articles
        """
        if news_df.empty:
            return pd.DataFrame()

        self._load()
        grouped = news_df.groupby(["date", "symbol"])
        sentiment_rows = []

        for (date, symbol), group in grouped:
            # Take top N most relevant articles
            articles = group.head(max_articles)
            texts = (articles["headline"] + ". " + articles.get("summary", "").fillna("")).tolist()
            scores = self.score_texts(texts)

            sentiment_rows.append({
                "date": date,
                "symbol": symbol,
                "sent_pos": scores["positive"].mean(),
                "sent_neg": scores["negative"].mean(),
                "sent_neutral": scores["neutral"].mean(),
                "sent_compound": scores["compound"].mean(),
                "sent_std": scores["compound"].std(),
                "n_articles": len(texts),
            })

        if not sentiment_rows:
            return pd.DataFrame()

        df = pd.DataFrame(sentiment_rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    # ── Lexicon fallback ───────────────────────────────────────────────────────

    POSITIVE_WORDS = {
        "beat", "record", "growth", "surge", "rally", "upgrade", "exceed",
        "strong", "profit", "gain", "revenue", "rise", "up", "higher", "bullish",
        "positive", "outperform", "buy", "acquisition", "partnership",
    }
    NEGATIVE_WORDS = {
        "miss", "loss", "decline", "fall", "drop", "downgrade", "disappoint",
        "weak", "debt", "lawsuit", "investigation", "cut", "lower", "bearish",
        "negative", "underperform", "sell", "layoff", "warning", "default",
    }

    def _lexicon_score(self, texts: List[str]) -> pd.DataFrame:
        rows = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & self.POSITIVE_WORDS) / (len(words) + 1)
            neg = len(words & self.NEGATIVE_WORDS) / (len(words) + 1)
            total = pos + neg + 1e-8
            rows.append({
                "positive": pos / total,
                "negative": neg / total,
                "neutral": max(0, 1 - pos / total - neg / total),
                "compound": (pos - neg) / total,
            })
        return pd.DataFrame(rows)


# ─── Fundamental Feature Builder ──────────────────────────────────────────────

class FundamentalFeatures:
    """
    Converts quarterly fundamental data to daily features via forward-fill.
    This is the correct approach: use the last available fundamental value
    until the next earnings release (point-in-time safe).
    """

    def merge_with_prices(
        self,
        price_df: pd.DataFrame,
        fund_df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Merge fundamentals (quarterly point-in-time) into daily price DataFrame.
        All values forward-filled — never look-ahead.
        """
        if fund_df.empty:
            return price_df

        sym_fund = fund_df[fund_df.get("symbol", pd.Series()) == symbol].copy() if "symbol" in fund_df else fund_df.copy()
        if "fetch_date" in sym_fund.columns:
            sym_fund["date"] = pd.to_datetime(sym_fund["fetch_date"])
            sym_fund = sym_fund.set_index("date")

        # Select numeric fundamental columns
        num_cols = sym_fund.select_dtypes(include=[np.number]).columns.tolist()

        # Reindex to daily and forward-fill
        daily_fund = sym_fund[num_cols].reindex(
            price_df.index.union(sym_fund.index)
        ).sort_index().ffill()
        daily_fund = daily_fund.reindex(price_df.index)

        for col in num_cols:
            price_df[f"fund_{col}"] = daily_fund[col].values

        return price_df


# ─── Macro Feature Builder ─────────────────────────────────────────────────────

class MacroFeatures:
    """
    Adds market-wide / macro features to each symbol's feature set.
    These act as the market "regime" context for the RL agent.
    """

    def build(
        self,
        market_data: Dict[str, pd.DataFrame],
        vix: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build a daily macro feature DataFrame shared across all symbols.
        Columns: vix, vix_regime, spy_return, qqq_return, market_breadth, etc.
        """
        macro = pd.DataFrame(index=self._get_common_index(market_data))

        # VIX
        if vix is not None:
            macro["vix"] = vix.reindex(macro.index).ffill()
            macro["vix_norm"] = (macro["vix"] - macro["vix"].rolling(252).mean()) / \
                                 (macro["vix"].rolling(252).std() + 1e-8)
            macro["vix_regime"] = pd.cut(
                macro["vix"],
                bins=[0, 15, 25, 35, 1000],
                labels=[0, 1, 2, 3],  # calm, normal, elevated, extreme
            ).astype(float)
            macro["vix_spike"] = (macro["vix"].pct_change() > 0.20).astype(float)

        # SPY as market benchmark
        if "SPY" in market_data:
            spy = market_data["SPY"]["close"]
            spy_ret = np.log(spy / spy.shift(1))
            macro["spy_ret_1d"] = spy_ret.reindex(macro.index)
            macro["spy_ret_5d"] = spy.pct_change(5).reindex(macro.index)
            macro["spy_ret_21d"] = spy.pct_change(21).reindex(macro.index)
            macro["spy_above_200ma"] = (spy > spy.rolling(200).mean()).astype(float).reindex(macro.index)
            macro["market_trend"] = np.sign(macro["spy_ret_21d"])

        # Market breadth: % of universe stocks above 50-day MA
        closes = {sym: df["close"] for sym, df in market_data.items() if "close" in df}
        if closes:
            close_df = pd.DataFrame(closes).reindex(macro.index)
            ma50_df = close_df.rolling(50).mean()
            macro["breadth_50ma"] = (close_df > ma50_df).mean(axis=1)

        # Realized correlation (average pairwise)
        if len(closes) > 1:
            ret_df = pd.DataFrame(closes).pct_change().reindex(macro.index)
            macro["avg_correlation"] = ret_df.rolling(21).corr().groupby(level=0).mean().mean(axis=1)

        return macro.ffill()

    def _get_common_index(self, market_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        indices = [df.index for df in market_data.values() if not df.empty]
        if not indices:
            return pd.DatetimeIndex([])
        combined = indices[0]
        for idx in indices[1:]:
            combined = combined.union(idx)
        return combined


# ─── Normalizer ────────────────────────────────────────────────────────────────

class RollingNormalizer:
    """
    Rolling z-score normalization.
    CRITICAL: uses only PAST data to compute mean/std (no look-ahead bias).
    """

    def __init__(self, window: int = 252, clip_value: float = 5.0):
        self.window = window
        self.clip_value = clip_value

    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize all numeric columns with rolling z-score.
        Binary/categorical columns are excluded automatically.
        """
        exclude = set(exclude_cols or [])
        result = df.copy()

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in exclude:
                continue
            series = df[col]
            # Skip binary-ish columns
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 5 and all(v in {0, 1, 2, 3, -1} for v in unique_vals):
                continue

            rolling_mean = series.rolling(self.window, min_periods=20).mean()
            rolling_std = series.rolling(self.window, min_periods=20).std()
            z = (series - rolling_mean) / (rolling_std + 1e-8)
            result[col] = z.clip(-self.clip_value, self.clip_value)

        return result


# ─── Master Pipeline ───────────────────────────────────────────────────────────

class FeaturePipeline:
    """
    Orchestrates the full feature engineering process.

    Usage:
        pipeline = FeaturePipeline()
        feature_store = pipeline.build(
            market_data=loader.get_ohlcv_universe(symbols, start, end),
            news_df=loader.get_news_sentiment_raw(symbols, start, end),
            fundamentals={sym: loader.get_fundamentals(sym) for sym in symbols},
            vix=loader.get_vix(start, end),
        )
        # Returns: Dict[symbol -> normalized feature DataFrame]
    """

    def __init__(self, cfg=None):
        from config.settings import CONFIG
        self.cfg = cfg or CONFIG.features
        self.technical = TechnicalFeatures(self.cfg)
        self.sentiment = SentimentAnalyzer(self.cfg.sentiment_model)
        self.fundamentals = FundamentalFeatures()
        self.macro = MacroFeatures()
        self.normalizer = RollingNormalizer(self.cfg.zscore_window)

    def build(
        self,
        market_data: Dict[str, pd.DataFrame],
        news_df: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
        vix: Optional[pd.Series] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Full pipeline: raw data → normalized features per symbol.
        Returns dict: symbol → feature DataFrame ready for RL environment.
        """
        logger.info("Building macro features...")
        macro_df = self.macro.build(market_data, vix)

        logger.info("Building sentiment features...")
        sent_df = pd.DataFrame()
        if news_df is not None and not news_df.empty:
            sent_df = self.sentiment.build_daily_sentiment(
                news_df,
                lookback_days=self.cfg.sentiment_lookback_days,
                max_articles=self.cfg.max_news_per_day,
            )

        feature_store = {}
        symbols = [s for s in market_data if s not in ("SPY", "QQQ", "IWM", "^VIX")]

        for sym in symbols:
            df = market_data[sym]
            if df.empty or len(df) < 100:
                logger.warning(f"Skipping {sym}: insufficient data ({len(df)} rows)")
                continue

            logger.info(f"Processing features for {sym}...")

            # 1. Technical features
            df = self.technical.compute(df)

            # 2. Merge macro features
            macro_aligned = macro_df.reindex(df.index).ffill()
            df = pd.concat([df, macro_aligned], axis=1)

            # 3. Merge fundamental features
            if fundamentals and sym in fundamentals:
                df = self.fundamentals.merge_with_prices(df, fundamentals[sym], sym)

            # 4. Merge sentiment features
            if not sent_df.empty and sym in sent_df.get("symbol", pd.Series()).values:
                sym_sent = sent_df[sent_df.get("symbol") == sym].drop(columns=["symbol"], errors="ignore")
                sym_sent = sym_sent.reindex(df.index).ffill().fillna(0)
                df = pd.concat([df, sym_sent], axis=1)

            # 5. Normalize (rolling z-score, no look-ahead)
            raw_cols = ["open", "high", "low", "close", "volume"]
            df = self.normalizer.fit_transform(df, exclude_cols=raw_cols)

            # 6. Drop NaN rows from indicator warmup period
            df = df.dropna(subset=[c for c in df.columns if c not in raw_cols])

            feature_store[sym] = df
            logger.info(f"  {sym}: {len(df)} days × {len(df.columns)} features")

        return feature_store

    def get_feature_names(self, feature_store: Dict[str, pd.DataFrame]) -> List[str]:
        """Return list of feature column names (excluding raw OHLCV)."""
        raw = {"open", "high", "low", "close", "volume", "adj_close"}
        if not feature_store:
            return []
        first = next(iter(feature_store.values()))
        return [c for c in first.columns if c not in raw]

    def save(self, feature_store: Dict[str, pd.DataFrame], path: str) -> None:
        """Save feature store to parquet files."""
        import os
        os.makedirs(path, exist_ok=True)
        for sym, df in feature_store.items():
            df.to_parquet(f"{path}/{sym}.parquet")
        logger.info(f"Saved {len(feature_store)} symbols to {path}")

    def load(self, path: str, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load feature store from parquet files."""
        import glob
        files = glob.glob(f"{path}/*.parquet")
        result = {}
        for f in files:
            sym = Path(f).stem
            if symbols is None or sym in symbols:
                result[sym] = pd.read_parquet(f)
        logger.info(f"Loaded {len(result)} symbols from {path}")
        return result
