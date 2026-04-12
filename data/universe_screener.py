"""
data/universe_screener.py
──────────────────────────
Stock universe screening and filtering.

Screens the investable universe based on:
  - Liquidity (min average daily volume)
  - Price (exclude penny stocks)
  - Data completeness (min history length)
  - Volatility regime (exclude extreme movers)
  - Sector diversification (cap per sector)

Usage:
    screener = UniverseScreener()
    valid_symbols = screener.screen(
        market_data=loader.get_ohlcv_universe(sp500_symbols, ...),
        fundamentals=fund_dict,
    )
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UniverseScreener:
    """
    Filters a raw universe of stocks down to a clean, investable set.
    All filters are documented so you can tune thresholds.
    """

    def __init__(
        self,
        min_price: float = 5.0,               # exclude penny stocks
        min_avg_daily_volume: float = 1e6,    # min 1M shares/day
        min_avg_dollar_volume: float = 1e7,   # min $10M/day ADV
        min_history_days: int = 252,          # at least 1 year of data
        max_missing_pct: float = 0.05,        # max 5% missing trading days
        min_annual_vol: float = 0.05,         # exclude near-zero vol (ETFs etc.)
        max_annual_vol: float = 2.0,          # exclude extreme movers
        max_per_sector: int = 10,             # sector cap for diversification
        min_market_cap: float = 1e9,          # min $1B market cap (midcap+)
    ):
        self.min_price = min_price
        self.min_adv_shares = min_avg_daily_volume
        self.min_adv_dollars = min_avg_dollar_volume
        self.min_history = min_history_days
        self.max_missing = max_missing_pct
        self.min_vol = min_annual_vol
        self.max_vol = max_annual_vol
        self.max_per_sector = max_per_sector
        self.min_mktcap = min_market_cap

    def screen(
        self,
        market_data: Dict[str, pd.DataFrame],
        fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        reference_index: Optional[pd.DatetimeIndex] = None,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Apply all filters and return (valid_symbols, filter_report).

        filter_report: DataFrame showing why each symbol passed/failed.
        """
        results = []

        for sym, df in market_data.items():
            row = {"symbol": sym, "passed": True, "fail_reason": ""}

            # ── Filter 1: Minimum history ──────────────────────────────────────
            if df.empty or len(df) < self.min_history:
                row.update({"passed": False, "fail_reason": f"Insufficient history ({len(df)} days)"})
                results.append(row)
                continue

            # ── Filter 2: Data completeness ────────────────────────────────────
            if reference_index is not None:
                expected = len(reference_index)
                actual = len(df.index.intersection(reference_index))
                missing_pct = 1 - actual / max(expected, 1)
                row["missing_pct"] = round(missing_pct, 4)
                if missing_pct > self.max_missing:
                    row.update({"passed": False, "fail_reason": f"Too many missing days ({missing_pct:.1%})"})
                    results.append(row)
                    continue
            else:
                row["missing_pct"] = 0.0

            # ── Filter 3: Price filter ─────────────────────────────────────────
            avg_price = df["close"].mean()
            row["avg_price"] = round(avg_price, 2)
            if avg_price < self.min_price:
                row.update({"passed": False, "fail_reason": f"Penny stock (avg=${avg_price:.2f})"})
                results.append(row)
                continue

            # ── Filter 4: Liquidity — share volume ────────────────────────────
            avg_vol_shares = df["volume"].mean() if "volume" in df else 0
            row["avg_vol_shares"] = round(avg_vol_shares, 0)
            if avg_vol_shares < self.min_adv_shares:
                row.update({"passed": False, "fail_reason": f"Illiquid ({avg_vol_shares:,.0f} shares/day)"})
                results.append(row)
                continue

            # ── Filter 5: Liquidity — dollar volume ───────────────────────────
            if "volume" in df and "close" in df:
                avg_dollar_vol = (df["volume"] * df["close"]).mean()
            else:
                avg_dollar_vol = avg_vol_shares * avg_price
            row["avg_dollar_vol"] = round(avg_dollar_vol, 0)
            if avg_dollar_vol < self.min_adv_dollars:
                row.update({"passed": False, "fail_reason": f"Low dollar volume (${avg_dollar_vol/1e6:.1f}M/day)"})
                results.append(row)
                continue

            # ── Filter 6: Volatility bounds ────────────────────────────────────
            log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
            annual_vol = log_ret.std() * np.sqrt(252)
            row["annual_vol"] = round(annual_vol, 4)
            if annual_vol < self.min_vol:
                row.update({"passed": False, "fail_reason": f"Near-zero vol ({annual_vol:.1%})"})
                results.append(row)
                continue
            if annual_vol > self.max_vol:
                row.update({"passed": False, "fail_reason": f"Extreme vol ({annual_vol:.1%})"})
                results.append(row)
                continue

            # ── Filter 7: Basic quality check (no suspicious gaps) ────────────
            daily_returns = df["close"].pct_change().abs()
            extreme_days = (daily_returns > 0.50).sum()
            row["extreme_move_days"] = int(extreme_days)
            if extreme_days > 5:
                row.update({"passed": False, "fail_reason": f"{extreme_days} days with >50% moves (data quality)"})
                results.append(row)
                continue

            # ── Fundamentals filter (optional) ────────────────────────────────
            if fundamentals and sym in fundamentals:
                fund = fundamentals[sym]
                if not fund.empty:
                    mktcap = fund.get("market_cap", pd.Series([None])).iloc[0]
                    if mktcap and mktcap < self.min_mktcap:
                        row.update({"passed": False, "fail_reason": f"Small cap (${mktcap/1e9:.1f}B)"})
                        results.append(row)
                        continue

            # ── Compute useful stats for passing symbols ───────────────────────
            ann_return = log_ret.mean() * 252
            sharpe = ann_return / (annual_vol + 1e-8)
            max_dd = (df["close"] / df["close"].cummax() - 1).min()

            row.update({
                "ann_return": round(ann_return, 4),
                "sharpe": round(sharpe, 3),
                "max_drawdown": round(max_dd, 4),
                "n_days": len(df),
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
            })
            results.append(row)

        # Build report DataFrame
        report_df = pd.DataFrame(results)
        if report_df.empty:
            return [], report_df

        passed = report_df[report_df["passed"]]
        failed = report_df[~report_df["passed"]]

        # ── Sector diversification cap ─────────────────────────────────────────
        if sector_map and not passed.empty:
            passed = passed.copy()
            passed["sector"] = passed["symbol"].map(sector_map).fillna("Unknown")

            # Within each sector, keep top N by dollar volume
            def sector_cap(group):
                return group.nlargest(self.max_per_sector, "avg_dollar_vol")

            passed = passed.groupby("sector", group_keys=False).apply(sector_cap)
            capped_out = set(report_df[report_df["passed"]]["symbol"]) - set(passed["symbol"])
            if capped_out:
                logger.info(f"Sector cap removed {len(capped_out)} symbols: {capped_out}")

        valid_symbols = passed["symbol"].tolist()

        # Logging summary
        logger.info(
            f"\nUniverse screening: {len(market_data)} → {len(valid_symbols)} symbols\n"
            f"  Passed: {len(passed)}\n"
            f"  Failed: {len(failed)}\n"
        )
        if not failed.empty:
            for _, row in failed.iterrows():
                logger.debug(f"  ✗ {row['symbol']:<8} {row['fail_reason']}")

        return valid_symbols, report_df

    def get_sp500_symbols(self) -> List[str]:
        """
        Fetch current S&P 500 constituents from Wikipedia.
        Falls back to a hardcoded subset if network unavailable.
        """
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            sp500_df = tables[0]
            symbols = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500 list: {e}. Using fallback subset.")
            return self._fallback_universe()

    def _fallback_universe(self) -> List[str]:
        """Hardcoded representative S&P 500 subset by sector."""
        return [
            # Technology
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
            "AMD", "INTC", "ORCL", "CRM", "ADBE", "QCOM", "TXN",
            # Financials
            "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP",
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "BMY", "AMGN",
            # Consumer
            "WMT", "HD", "MCD", "COST", "NKE", "SBUX", "TGT",
            # Industrials
            "CAT", "HON", "UPS", "RTX", "DE", "BA", "GE",
            # Energy
            "XOM", "CVX", "COP", "SLB",
            # Utilities / REITs
            "NEE", "DUK", "AMT", "PLD",
            # Communication
            "VZ", "T", "NFLX", "DIS",
            # Materials
            "LIN", "APD", "NEM",
            # Benchmarks
            "SPY", "QQQ", "IWM",
        ]

    def rank_by_quality(
        self, report_df: pd.DataFrame, top_n: int = 30
    ) -> List[str]:
        """
        Rank passing symbols by composite quality score.
        Score = 0.4*sharpe + 0.3*liquidity_score - 0.3*abs(max_drawdown)
        """
        passed = report_df[report_df["passed"]].copy()
        if passed.empty:
            return []

        # Normalize each component to [0, 1]
        def minmax(s):
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng > 0 else s * 0

        if "sharpe" in passed.columns:
            passed["score_sharpe"] = minmax(passed["sharpe"])
        else:
            passed["score_sharpe"] = 0.5

        if "avg_dollar_vol" in passed.columns:
            passed["score_liq"] = minmax(np.log1p(passed["avg_dollar_vol"]))
        else:
            passed["score_liq"] = 0.5

        if "max_drawdown" in passed.columns:
            passed["score_dd"] = minmax(-passed["max_drawdown"])  # less DD = better
        else:
            passed["score_dd"] = 0.5

        passed["quality_score"] = (
            0.4 * passed["score_sharpe"]
            + 0.3 * passed["score_liq"]
            + 0.3 * passed["score_dd"]
        )

        top = passed.nlargest(top_n, "quality_score")
        return top["symbol"].tolist()
