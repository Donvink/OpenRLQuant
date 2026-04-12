"""
data/screener.py
─────────────────
选股引擎：从大盘股票池中筛选出适合RL训练的交易标的。

筛选维度：
  1. 流动性过滤   — 日均成交额、自由流通市值
  2. 价格过滤     — 避免penny stocks
  3. 历史完整性   — 足够的历史数据
  4. 波动率窗口   — 适合交易的波动率区间
  5. 基本面过滤   — 盈利质量、财务健康
  6. 行业分散度   — 确保组合多样性

Universe构建策略：
  - Phase 1: S&P 500 子集（流动性好，数据完整）
  - Phase 2: 扩展至 Russell 1000
  - Phase 3: 加入期权链数据验证流动性
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScreenerConfig:
    # 流动性门槛
    min_avg_daily_volume: float = 1_000_000       # 日均成交量 > 100万股
    min_avg_dollar_volume: float = 10_000_000     # 日均成交额 > 1000万美元
    min_market_cap: float = 2_000_000_000         # 市值 > 20亿（mid-cap以上）

    # 价格门槛
    min_price: float = 5.0                         # 避免 penny stocks
    max_price: float = 10_000.0                    # 避免BRK.A类极高价股

    # 数据完整性
    min_trading_days: int = 500                    # 至少2年完整数据
    max_missing_pct: float = 0.02                  # 最多2%缺失

    # 波动率区间（年化）
    min_annual_vol: float = 0.08                   # 波动率太低：无交易机会
    max_annual_vol: float = 0.80                   # 波动率太高：噪声过多

    # 行业分散
    max_stocks_per_sector: int = 8                 # 每个行业最多入选8只
    target_n_stocks: int = 30                      # 目标股票数量

    # 基本面（可选，需要数据）
    require_positive_earnings: bool = False        # Phase 1 暂不强制
    max_debt_to_equity: float = 5.0


# ─── S&P 500 预设列表（Phase 1 使用，无需API）─────────────────────────────────

SP500_SUBSET = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO",
        "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    ],
    "Communication Services": [
        "NFLX", "DIS", "CMCSA", "T", "VZ",
    ],
    "Industrials": [
        "CAT", "BA", "HON", "UPS", "RTX", "GE", "MMM",
    ],
    "Consumer Staples": [
        "WMT", "PG", "KO", "PEP", "COST", "CL",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB",
    ],
    "Materials": [
        "LIN", "APD", "ECL", "NEM",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "AEP",
    ],
    "Benchmarks": [
        "SPY", "QQQ", "IWM",   # ETFs for macro features
    ],
}

ALL_SYMBOLS = [s for sector_stocks in SP500_SUBSET.values() for s in sector_stocks]
SECTOR_MAP = {s: sector for sector, stocks in SP500_SUBSET.items() for s in stocks}


class UniverseScreener:
    """
    从候选股票池中筛选出适合RL训练的股票集合。

    核心思路：
      RL环境的观测维度随股票数量线性增长，
      所以要精选N只（20-50只）高质量、流动性好、分散化的股票，
      而不是直接用全部500只。
    """

    def __init__(self, cfg: ScreenerConfig = None):
        self.cfg = cfg or ScreenerConfig()

    def screen(
        self,
        market_data: Dict[str, pd.DataFrame],
        fundamentals: Optional[Dict[str, Dict]] = None,
        verbose: bool = True,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        对所有候选股票进行多维度筛选。

        Returns:
            (selected_symbols, screening_report_df)
        """
        results = []

        for sym, df in market_data.items():
            if sym in ("SPY", "QQQ", "IWM"):
                continue  # Benchmark ETFs: always include separately

            row = self._screen_single(sym, df, fundamentals)
            results.append(row)

        if not results:
            return [], pd.DataFrame()

        report = pd.DataFrame(results).set_index("symbol")

        # Apply all filters
        passed = report[report["passes_all"]].copy()

        # Apply sector diversification cap
        selected = self._apply_sector_cap(passed)

        if verbose:
            self._print_report(report, selected)

        return selected, report

    def _screen_single(
        self,
        symbol: str,
        df: pd.DataFrame,
        fundamentals: Optional[Dict] = None,
    ) -> Dict:
        cfg = self.cfg
        row = {"symbol": symbol, "sector": SECTOR_MAP.get(symbol, "Unknown")}
        fails = []

        if df.empty:
            row.update({"passes_all": False, "fail_reasons": "no_data"})
            return row

        # ── Liquidity ──────────────────────────────────────────────────────────
        avg_vol = df["volume"].mean() if "volume" in df else 0
        avg_price = df["close"].mean() if "close" in df else 0
        avg_dollar_vol = avg_vol * avg_price

        row["avg_volume"] = round(avg_vol, 0)
        row["avg_dollar_vol_M"] = round(avg_dollar_vol / 1e6, 1)
        row["avg_price"] = round(avg_price, 2)

        if avg_vol < cfg.min_avg_daily_volume:
            fails.append(f"low_volume({avg_vol:.0f})")
        if avg_dollar_vol < cfg.min_avg_dollar_volume:
            fails.append(f"low_dollar_vol(${avg_dollar_vol/1e6:.1f}M)")

        # ── Price ──────────────────────────────────────────────────────────────
        last_price = df["close"].iloc[-1] if "close" in df else 0
        row["last_price"] = round(last_price, 2)
        if last_price < cfg.min_price:
            fails.append(f"price_too_low({last_price:.2f})")
        if last_price > cfg.max_price:
            fails.append(f"price_too_high({last_price:.2f})")

        # ── Data completeness ─────────────────────────────────────────────────
        n_days = len(df)
        row["n_trading_days"] = n_days
        if n_days < cfg.min_trading_days:
            fails.append(f"insufficient_history({n_days}d)")

        if "close" in df:
            missing_pct = df["close"].isna().mean()
            row["missing_pct"] = round(missing_pct, 4)
            if missing_pct > cfg.max_missing_pct:
                fails.append(f"too_many_missing({missing_pct:.1%})")

        # ── Volatility ────────────────────────────────────────────────────────
        if "close" in df and len(df) > 21:
            log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
            ann_vol = log_ret.std() * np.sqrt(252)
            row["annual_vol"] = round(ann_vol, 4)
            if ann_vol < cfg.min_annual_vol:
                fails.append(f"vol_too_low({ann_vol:.1%})")
            if ann_vol > cfg.max_annual_vol:
                fails.append(f"vol_too_high({ann_vol:.1%})")

            # Sharpe (rough)
            ann_ret = log_ret.mean() * 252
            row["annual_return"] = round(ann_ret, 4)
            row["sharpe_approx"] = round(ann_ret / (ann_vol + 1e-8), 3)

            # Max drawdown
            close = df["close"]
            dd = (close / close.expanding().max() - 1).min()
            row["max_drawdown"] = round(dd, 4)

        # ── Fundamentals (optional) ───────────────────────────────────────────
        if fundamentals and symbol in fundamentals:
            fund = fundamentals[symbol]
            de = fund.get("debt_to_equity", 0) or 0
            row["debt_to_equity"] = round(de, 2)
            if cfg.require_positive_earnings:
                pe = fund.get("pe_ratio", 1) or 1
                if pe <= 0:
                    fails.append("negative_earnings")

        row["passes_all"] = len(fails) == 0
        row["fail_reasons"] = "; ".join(fails) if fails else ""
        return row

    def _apply_sector_cap(self, passed_df: pd.DataFrame) -> List[str]:
        """
        从通过筛选的股票中，按行业分散度选取最终名单。
        在每个行业内，优先选取成交额最大（流动性最好）的股票。
        """
        if passed_df.empty:
            return []

        selected = []
        sector_counts: Dict[str, int] = {}

        # Sort by dollar volume (best liquidity first)
        sorted_df = passed_df.sort_values("avg_dollar_vol_M", ascending=False)

        for sym, row in sorted_df.iterrows():
            if len(selected) >= self.cfg.target_n_stocks:
                break
            sector = row.get("sector", "Unknown")
            count = sector_counts.get(sector, 0)
            if count < self.cfg.max_stocks_per_sector:
                selected.append(sym)
                sector_counts[sector] = count + 1

        return selected

    def _print_report(self, full_report: pd.DataFrame, selected: List[str]) -> None:
        total = len(full_report)
        passed = full_report["passes_all"].sum()

        logger.info(f"\n{'='*60}")
        logger.info(f"UNIVERSE SCREENING REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Candidates: {total} | Passed: {passed} | Selected: {len(selected)}")

        if "sector" in full_report.columns:
            sector_counts = pd.Series(selected).map(SECTOR_MAP).value_counts()
            logger.info("\nSector breakdown:")
            for sector, count in sector_counts.items():
                logger.info(f"  {sector:<30} {count:>3} stocks")

        logger.info(f"\nSelected universe ({len(selected)} stocks):")
        logger.info(f"  {selected}")

        if not full_report.empty:
            failed = full_report[~full_report["passes_all"]]
            if not failed.empty:
                logger.info(f"\nFailed screening ({len(failed)} stocks):")
                for sym, row in failed.iterrows():
                    logger.info(f"  {sym:<8} — {row['fail_reasons']}")


# ─── Quick universe builder ────────────────────────────────────────────────────

def build_phase1_universe(
    market_data: Dict[str, pd.DataFrame],
    target_n: int = 20,
    min_years: float = 2.0,
) -> Tuple[List[str], List[str]]:
    """
    Quick universe selection for Phase 1.
    Returns: (trading_symbols, benchmark_symbols)

    Skips full screening — just validates data quality.
    """
    cfg = ScreenerConfig(
        min_trading_days=int(252 * min_years),
        target_n_stocks=target_n,
        min_avg_daily_volume=500_000,  # relaxed for Phase 1
        min_avg_dollar_volume=5_000_000,
    )
    screener = UniverseScreener(cfg)
    selected, report = screener.screen(market_data, verbose=True)

    benchmarks = [s for s in ["SPY", "QQQ", "IWM"] if s in market_data]

    if not selected:
        # Fallback: use any symbol with enough data
        logger.warning("Screening too strict — using all symbols with sufficient data")
        selected = [
            sym for sym, df in market_data.items()
            if sym not in benchmarks and len(df) >= cfg.min_trading_days
        ]

    logger.info(f"Phase 1 universe: {len(selected)} trading stocks + {len(benchmarks)} benchmarks")
    return selected, benchmarks
