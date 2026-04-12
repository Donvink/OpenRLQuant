"""
data/market_data.py
───────────────────
Multi-source market data ingestion with caching and fallback.

Sources (in priority order):
  1. Polygon.io  — OHLCV, trades, quotes (best quality, paid)
  2. Yahoo Finance — OHLCV fallback (free, rate-limited)
  3. Alpha Vantage — fundamentals backup

Usage:
    from data.market_data import MarketDataLoader
    loader = MarketDataLoader()
    df = loader.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")
"""

import time
import logging
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ─── Cache ─────────────────────────────────────────────────────────────────────

class DiskCache:
    """Simple pickle-based disk cache with TTL."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.pkl"

    def get(self, key: str) -> Optional[object]:
        p = self._key_path(key)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.ttl:
            p.unlink()
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def set(self, key: str, value: object) -> None:
        with open(self._key_path(key), "wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.pkl"):
            p.unlink()


# ─── Yahoo Finance Fetcher (free fallback) ─────────────────────────────────────

class YahooFinanceFetcher:
    """
    Fetches OHLCV data via yfinance (no API key needed).
    Adequate for daily data; not suitable for intraday production use.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("Run: pip install yfinance")

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns: open, high, low, close, volume, adj_close
        Index: DatetimeIndex (UTC, date only for daily)
        """
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning(f"No data returned for {symbol} [{start}:{end}]")
            return pd.DataFrame()

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
        })
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "adj_close", "volume"]].copy()

        # Adjust OHLC by split/dividend ratio
        ratio = df["adj_close"] / df["close"]
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * ratio
        df["close"] = df["adj_close"]

        return df.dropna()

    def get_batch_ohlcv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Download multiple tickers at once (yfinance batches internally)."""
        import yfinance as yf
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        result = {}
        for sym in symbols:
            try:
                if len(symbols) == 1:
                    df = raw.copy()
                else:
                    df = raw[sym].copy()
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                df = df.dropna(subset=["close"])
                result[sym] = df
            except Exception as e:
                logger.error(f"Failed to parse {sym}: {e}")
        return result

    def get_info(self, symbol: str) -> Dict:
        """Company metadata and fundamental snapshot."""
        ticker = self.yf.Ticker(symbol)
        return ticker.info or {}

    def get_financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Quarterly financials: income statement, balance sheet, cash flow."""
        ticker = self.yf.Ticker(symbol)
        return {
            "income": ticker.quarterly_financials,
            "balance": ticker.quarterly_balance_sheet,
            "cashflow": ticker.quarterly_cashflow,
        }


# ─── Polygon.io Fetcher (production quality) ───────────────────────────────────

class PolygonFetcher:
    """
    Polygon.io REST API v2.
    Free tier: 5 API calls/minute, 2 years history.
    Paid tiers: unlimited, real-time, options, full tick data.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY env var.")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._last_call = 0.0
        self._min_interval = 12.0  # 5 calls/min on free tier = 12s between calls

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Rate-limited GET request."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        self._last_call = time.time()

        if resp.status_code == 429:
            logger.warning("Rate limited by Polygon, sleeping 60s...")
            time.sleep(60)
            return self._get(endpoint, params)

        resp.raise_for_status()
        return resp.json()

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        timespan: str = "day",   # "minute", "hour", "day", "week"
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV aggregates from Polygon.
        Handles pagination automatically.
        """
        all_results = []
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": 50000,
        }
        url = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"

        while url:
            data = self._get(url, params)
            results = data.get("results", [])
            all_results.extend(results)
            url = data.get("next_url", "").replace(self.BASE_URL, "") or None
            params = {}  # next_url has params embedded

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date").rename(columns={
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap", "n": "trades",
        })
        cols = [c for c in ["open", "high", "low", "close", "volume", "vwap", "trades"] if c in df]
        return df[cols].sort_index()

    def get_last_quote(self, symbol: str) -> Dict:
        """Real-time last quote (bid/ask)."""
        return self._get(f"/v2/last/nbbo/{symbol}")

    def get_ticker_details(self, symbol: str) -> Dict:
        """Company metadata, sector, market cap."""
        return self._get(f"/v3/reference/tickers/{symbol}")


# ─── News Fetcher ──────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    Fetches financial news from multiple sources.
    Primary: Finnhub (free tier: 60 calls/min)
    Fallback: NewsAPI
    """

    FINNHUB_BASE = "https://finnhub.io/api/v1"
    NEWSAPI_BASE = "https://newsapi.org/v2"

    def __init__(self, finnhub_key: str = "", news_api_key: str = ""):
        self.finnhub_key = finnhub_key
        self.news_api_key = news_api_key

    def get_company_news_finnhub(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> List[Dict]:
        """
        Finnhub company news API.
        Returns list of articles with: datetime, headline, summary, source, url, sentiment
        """
        if not self.finnhub_key:
            logger.warning("No Finnhub key, skipping news fetch")
            return []

        url = f"{self.FINNHUB_BASE}/company-news"
        params = {
            "symbol": symbol,
            "from": start,
            "to": end,
            "token": self.finnhub_key,
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []

        articles = resp.json()
        return [
            {
                "symbol": symbol,
                "datetime": datetime.fromtimestamp(a["datetime"]),
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "category": a.get("category", ""),
            }
            for a in articles
            if a.get("headline")
        ]

    def get_market_news(self, category: str = "general") -> List[Dict]:
        """General market news (earnings, macro, Fed, etc.)."""
        if not self.finnhub_key:
            return []
        url = f"{self.FINNHUB_BASE}/news"
        resp = requests.get(url, params={"category": category, "token": self.finnhub_key})
        return resp.json() if resp.ok else []

    def get_sec_filings(self, symbol: str, form_type: str = "10-K,10-Q,8-K") -> List[Dict]:
        """
        SEC filings via Finnhub.
        Useful for: earnings (10-Q), guidance (8-K), annual (10-K).
        """
        if not self.finnhub_key:
            return []
        url = f"{self.FINNHUB_BASE}/stock/filings"
        params = {"symbol": symbol, "form": form_type, "token": self.finnhub_key}
        resp = requests.get(url, params=params, timeout=15)
        return resp.json().get("filings", []) if resp.ok else []

    def build_daily_news_df(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Build a DataFrame of news, indexed by (date, symbol).
        Columns: headline, summary, source, n_articles
        """
        all_articles = []
        for sym in symbols:
            articles = self.get_company_news_finnhub(sym, start, end)
            for a in articles:
                a["date"] = a["datetime"].date()
                all_articles.append(a)
            time.sleep(0.5)  # Respect rate limits

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df["date"] = pd.to_datetime(df["date"])
        return df


# ─── Main DataLoader (orchestrator) ───────────────────────────────────────────

class MarketDataLoader:
    """
    Unified interface for all data sources.
    Handles caching, fallback, and data validation.

    Example:
        loader = MarketDataLoader(polygon_key="...", use_cache=True)
        prices = loader.get_ohlcv_universe(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start="2020-01-01",
            end="2024-01-01",
        )
        # Returns: Dict[symbol -> DataFrame]
    """

    def __init__(
        self,
        polygon_key: str = "",
        finnhub_key: str = "",
        news_api_key: str = "",
        cache_dir: Path = Path("data/cache"),
        cache_ttl: int = 86400,  # 24h for daily data
        use_cache: bool = True,
    ):
        self.yahoo = YahooFinanceFetcher()
        self.polygon = PolygonFetcher(polygon_key) if polygon_key else None
        self.news = NewsFetcher(finnhub_key, news_api_key)
        self.cache = DiskCache(cache_dir, cache_ttl) if use_cache else None

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        source: str = "auto",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a single symbol.
        source: "auto" tries Polygon first, falls back to Yahoo.
        """
        cache_key = f"ohlcv_{symbol}_{start}_{end}_{interval}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached

        df = pd.DataFrame()

        if source in ("auto", "polygon") and self.polygon:
            try:
                timespan = {"1d": "day", "1h": "hour", "5m": "minute"}.get(interval, "day")
                df = self.polygon.get_ohlcv(symbol, start, end, timespan=timespan)
                logger.info(f"Polygon: {symbol} {len(df)} rows")
            except Exception as e:
                logger.warning(f"Polygon failed for {symbol}: {e}, falling back to Yahoo")

        if df.empty:
            try:
                df = self.yahoo.get_ohlcv(symbol, start, end, interval)
                logger.info(f"Yahoo: {symbol} {len(df)} rows")
            except Exception as e:
                logger.error(f"Yahoo also failed for {symbol}: {e}")
                return pd.DataFrame()

        df = self._validate_ohlcv(df, symbol)

        if self.cache and not df.empty:
            self.cache.set(cache_key, df)

        return df

    def get_ohlcv_universe(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for all symbols in universe.
        Uses yfinance batch download when Polygon not available.
        """
        logger.info(f"Fetching {len(symbols)} symbols [{start} → {end}]")

        if self.polygon is None:
            # Batch download via yfinance (much faster)
            raw = self.yahoo.get_batch_ohlcv(symbols, start, end, interval)
            result = {}
            for sym, df in raw.items():
                df = self._validate_ohlcv(df, sym)
                if not df.empty:
                    result[sym] = df
                    if self.cache:
                        self.cache.set(f"ohlcv_{sym}_{start}_{end}_{interval}", df)
            return result

        # Individual Polygon fetches (better quality, slower)
        result = {}
        for i, sym in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Fetching {sym}")
            df = self.get_ohlcv(sym, start, end, interval)
            if not df.empty:
                result[sym] = df

        return result

    def get_aligned_prices(
        self,
        symbols: List[str],
        start: str,
        end: str,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Returns a wide DataFrame: rows=dates, cols=symbols.
        All series aligned to the same trading calendar (NYSE).
        Missing values forward-filled (e.g. halted stocks).
        """
        universe_data = self.get_ohlcv_universe(symbols, start, end)
        frames = {sym: df[price_col] for sym, df in universe_data.items() if price_col in df}
        prices = pd.DataFrame(frames)
        prices = prices.sort_index()

        # Use NYSE trading calendar for index
        prices = self._align_to_trading_calendar(prices, start, end)
        prices = prices.ffill().dropna(how="all")
        return prices

    def get_fundamentals(self, symbol: str) -> pd.DataFrame:
        """
        Return quarterly fundamental ratios as a time-indexed DataFrame.
        Data from yfinance (free) — merge with price data later.
        """
        cache_key = f"fundamentals_{symbol}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        info = self.yahoo.get_info(symbol)
        financials = self.yahoo.get_financials(symbol)

        # Extract key ratios from info snapshot
        snap = {
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "net_margin": info.get("profitMargins"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "earnings_growth_yoy": info.get("earningsGrowth"),
            "free_cash_flow_yield": info.get("freeCashflow"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
            "beta": info.get("beta"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }

        df = pd.DataFrame([snap])
        df["symbol"] = symbol
        df["fetch_date"] = pd.Timestamp.now().date()

        if self.cache:
            self.cache.set(cache_key, df)
        return df

    def get_news_sentiment_raw(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Raw news articles before NLP processing."""
        cache_key = f"news_raw_{'_'.join(sorted(symbols))}_{start}_{end}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        df = self.news.build_daily_news_df(symbols, start, end)

        if self.cache and not df.empty:
            self.cache.set(cache_key, df)
        return df

    def get_vix(self, start: str, end: str) -> pd.Series:
        """VIX index as a Series (key market regime indicator)."""
        df = self.get_ohlcv("^VIX", start, end)
        return df["close"].rename("vix") if not df.empty else pd.Series(name="vix")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Data quality checks and cleanup."""
        if df.empty:
            return df

        original_len = len(df)

        # Remove rows with negative or zero prices
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            df = df[df[col] > 0]

        # Remove rows where high < low (data error)
        if "high" in df.columns and "low" in df.columns:
            df = df[df["high"] >= df["low"]]

        # Remove extreme outliers (price change > 80% in one day — likely split unadjusted)
        if "close" in df.columns and len(df) > 1:
            pct_change = df["close"].pct_change().abs()
            df = df[pct_change < 0.80]

        # Remove duplicate dates
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        removed = original_len - len(df)
        if removed > 0:
            logger.warning(f"{symbol}: removed {removed} bad rows ({original_len} → {len(df)})")

        return df

    def _align_to_trading_calendar(
        self, df: pd.DataFrame, start: str, end: str
    ) -> pd.DataFrame:
        """Reindex to NYSE trading days (approximated as weekdays)."""
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=start, end_date=end)
            trading_days = mcal.date_range(schedule, frequency="1D")
            trading_days = trading_days.normalize().tz_localize(None)
        except ImportError:
            # Fallback: use business days
            trading_days = pd.bdate_range(start=start, end=end, freq="B")

        return df.reindex(trading_days)

    def describe_universe(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        """Summary statistics for all symbols in universe."""
        data = self.get_ohlcv_universe(symbols, start, end)
        rows = []
        for sym, df in data.items():
            if df.empty:
                continue
            returns = df["close"].pct_change().dropna()
            rows.append({
                "symbol": sym,
                "start": df.index[0].date(),
                "end": df.index[-1].date(),
                "n_days": len(df),
                "avg_price": df["close"].mean(),
                "avg_volume": df["volume"].mean(),
                "ann_return": returns.mean() * 252,
                "ann_vol": returns.std() * np.sqrt(252),
                "sharpe": (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                "max_drawdown": (df["close"] / df["close"].cummax() - 1).min(),
            })
        return pd.DataFrame(rows).set_index("symbol").round(4)
