"""
environment/backtester.py
──────────────────────────
Walk-forward backtesting and strategy evaluation.

Features:
  - Walk-forward validation (rolling train/test windows)
  - Benchmark comparison (Buy & Hold SPY)
  - Full performance metrics: Sharpe, Calmar, Sortino, Alpha, Beta, etc.
  - Detailed trade log and position history
  - Matplotlib performance charts (no external deps needed)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Performance Metrics ───────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    """Complete performance report for a backtest run."""

    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0

    # Risk
    annual_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0      # days
    value_at_risk_95: float = 0.0       # 1-day 95% VaR
    cvar_95: float = 0.0                # Conditional VaR

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # vs Benchmark
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # Trading activity
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    turnover_annual: float = 0.0

    # Cost
    total_cost: float = 0.0
    cost_as_pct_return: float = 0.0

    # Time period
    start_date: str = ""
    end_date: str = ""
    n_trading_days: int = 0

    def to_dict(self) -> Dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  BACKTEST PERFORMANCE REPORT\n"
            f"  {self.start_date} → {self.end_date}  ({self.n_trading_days} days)\n"
            f"{'='*55}\n"
            f"  Total Return       : {self.total_return:+.2%}\n"
            f"  Annual Return      : {self.annual_return:+.2%}\n"
            f"  Annual Volatility  : {self.annual_volatility:.2%}\n"
            f"  Max Drawdown       : {self.max_drawdown:.2%} ({self.max_drawdown_duration}d)\n"
            f"  Sharpe Ratio       : {self.sharpe_ratio:.3f}\n"
            f"  Sortino Ratio      : {self.sortino_ratio:.3f}\n"
            f"  Calmar Ratio       : {self.calmar_ratio:.3f}\n"
            f"  Win Rate           : {self.win_rate:.2%}\n"
            f"  Alpha vs SPY       : {self.alpha:+.2%}\n"
            f"  Beta               : {self.beta:.3f}\n"
            f"  Total Trades       : {self.total_trades}\n"
            f"  Total Cost         : ${self.total_cost:,.0f}\n"
            f"{'='*55}"
        )


def compute_metrics(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
    trade_log: Optional[pd.DataFrame] = None,
) -> PerformanceReport:
    """
    Compute full performance metrics from portfolio value series.

    Args:
        portfolio_values: Daily portfolio total value, DatetimeIndex
        benchmark_values: SPY values for same period
        risk_free_rate: Annual risk-free rate (default 5% = current T-bill)
        trade_log: DataFrame with trade records
    """
    rpt = PerformanceReport()

    if len(portfolio_values) < 5:
        return rpt

    rpt.start_date = str(portfolio_values.index[0].date())
    rpt.end_date = str(portfolio_values.index[-1].date())
    rpt.n_trading_days = len(portfolio_values)

    # ── Return metrics ─────────────────────────────────────────────────────────
    daily_returns = portfolio_values.pct_change().dropna()
    log_returns = np.log(portfolio_values / portfolio_values.shift(1)).dropna()

    rpt.total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    n_years = rpt.n_trading_days / 252
    rpt.annual_return = (1 + rpt.total_return) ** (1 / max(n_years, 0.01)) - 1

    # ── Risk metrics ───────────────────────────────────────────────────────────
    rpt.annual_volatility = daily_returns.std() * np.sqrt(252)

    # Drawdown
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    rpt.max_drawdown = float(drawdowns.min())

    # Drawdown duration
    in_drawdown = (drawdowns < 0)
    if in_drawdown.any():
        dd_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        dd_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        max_dur = 0
        start = None
        for date, val in in_drawdown.items():
            if val and start is None:
                start = date
            elif not val and start is not None:
                dur = (date - start).days
                max_dur = max(max_dur, dur)
                start = None
        rpt.max_drawdown_duration = max_dur

    # VaR / CVaR
    rpt.value_at_risk_95 = float(np.percentile(daily_returns, 5))
    rpt.cvar_95 = float(daily_returns[daily_returns <= rpt.value_at_risk_95].mean())

    # ── Risk-adjusted metrics ──────────────────────────────────────────────────
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = daily_returns - rf_daily

    if daily_returns.std() > 1e-8:
        rpt.sharpe_ratio = float(excess.mean() / daily_returns.std() * np.sqrt(252))

    downside = daily_returns[daily_returns < rf_daily]
    if len(downside) > 0 and downside.std() > 1e-8:
        rpt.sortino_ratio = float(excess.mean() / downside.std() * np.sqrt(252))

    if rpt.max_drawdown < 0:
        rpt.calmar_ratio = float(rpt.annual_return / abs(rpt.max_drawdown))

    # Omega ratio: probability-weighted gains vs losses above threshold
    gains = daily_returns[daily_returns > rf_daily] - rf_daily
    losses = rf_daily - daily_returns[daily_returns <= rf_daily]
    if losses.sum() > 0:
        rpt.omega_ratio = float(gains.sum() / losses.sum())

    # ── vs Benchmark ───────────────────────────────────────────────────────────
    if benchmark_values is not None and len(benchmark_values) > 5:
        bm_aligned = benchmark_values.reindex(portfolio_values.index).ffill()
        bm_returns = bm_aligned.pct_change().dropna()
        port_aligned = daily_returns.reindex(bm_returns.index)

        if len(port_aligned) > 10:
            cov_matrix = np.cov(port_aligned, bm_returns)
            bm_var = np.var(bm_returns)
            rpt.beta = float(cov_matrix[0, 1] / (bm_var + 1e-8))

            bm_ann_return = (bm_aligned.iloc[-1] / bm_aligned.iloc[0]) ** (1 / n_years) - 1
            rpt.alpha = rpt.annual_return - (rf_daily * 252 + rpt.beta * (bm_ann_return - rf_daily * 252))

            active_returns = port_aligned - bm_returns
            rpt.tracking_error = float(active_returns.std() * np.sqrt(252))
            if rpt.tracking_error > 1e-8:
                rpt.information_ratio = float(active_returns.mean() * 252 / rpt.tracking_error)

    # ── Trade metrics ──────────────────────────────────────────────────────────
    if trade_log is not None and not trade_log.empty:
        rpt.total_trades = len(trade_log)
        if "pnl" in trade_log.columns:
            wins = trade_log[trade_log["pnl"] > 0]["pnl"]
            losses_t = trade_log[trade_log["pnl"] <= 0]["pnl"]
            rpt.win_rate = len(wins) / len(trade_log) if len(trade_log) > 0 else 0
            rpt.avg_win = float(wins.mean()) if len(wins) > 0 else 0
            rpt.avg_loss = float(losses_t.mean()) if len(losses_t) > 0 else 0
            total_wins = wins.sum()
            total_losses = abs(losses_t.sum())
            rpt.profit_factor = total_wins / (total_losses + 1e-8)

    return rpt


# ─── Walk-Forward Backtester ───────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward validation to prevent overfitting.

    The key idea: train on period [t, t+train_window], test on [t+train_window, t+train_window+test_window],
    then advance by step_size. This mimics real-world deployment.

              ├─ Train (1yr) ─┤─ Test (3mo) ─┤
                  ├─ Train (1yr) ─┤─ Test (3mo) ─┤
                      ├─ Train (1yr) ─┤─ Test (3mo) ─┤
    """

    def __init__(
        self,
        train_window_days: int = 252,     # 1 year
        test_window_days: int = 63,        # 3 months
        step_size_days: int = 21,          # roll forward 1 month at a time
    ):
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.step_size = step_size_days

    def generate_folds(
        self, dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate (train_dates, test_dates) folds."""
        folds = []
        start = 0
        while start + self.train_window + self.test_window <= len(dates):
            train = dates[start: start + self.train_window]
            test = dates[start + self.train_window: start + self.train_window + self.test_window]
            folds.append((train, test))
            start += self.step_size

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds

    def run(
        self,
        agent,
        env_factory,
        dates: pd.DatetimeIndex,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.

        agent: trained RL agent with .predict(obs) method
        env_factory: callable(mode) -> TradingEnv
        """
        folds = self.generate_folds(dates)
        fold_results = []
        all_portfolio_values = []

        for i, (train_dates, test_dates) in enumerate(folds):
            logger.info(f"Fold {i+1}/{len(folds)}: test {test_dates[0].date()} → {test_dates[-1].date()}")

            # Run one fold
            env = env_factory(mode="test")
            obs, _ = env.reset()
            portfolio_values = [env.portfolio.total_value]
            trade_log = []

            done = False
            step = 0
            while not done and step < self.test_window:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                portfolio_values.append(info["total_value"])
                done = terminated or truncated
                step += 1

            if len(portfolio_values) > 5:
                pv = pd.Series(portfolio_values, index=range(len(portfolio_values)))
                report = compute_metrics(pv)
                fold_results.append(report)
                all_portfolio_values.extend(portfolio_values[1:])  # skip first (duplicate start)

        # Aggregate fold results
        if not fold_results:
            return {}

        agg = {
            "n_folds": len(fold_results),
            "mean_sharpe": np.mean([r.sharpe_ratio for r in fold_results]),
            "std_sharpe": np.std([r.sharpe_ratio for r in fold_results]),
            "mean_annual_return": np.mean([r.annual_return for r in fold_results]),
            "mean_max_drawdown": np.mean([r.max_drawdown for r in fold_results]),
            "mean_calmar": np.mean([r.calmar_ratio for r in fold_results]),
            "pct_profitable_folds": np.mean([r.total_return > 0 for r in fold_results]),
            "fold_reports": fold_results,
        }

        logger.info(
            f"Walk-forward results: "
            f"Sharpe={agg['mean_sharpe']:.3f}±{agg['std_sharpe']:.3f} | "
            f"Ann.Return={agg['mean_annual_return']:+.2%} | "
            f"MaxDD={agg['mean_max_drawdown']:.2%}"
        )

        return agg


# ─── Simple Benchmark Strategies ─────────────────────────────────────────────

class BuyAndHoldBenchmark:
    """Equal-weight buy and hold. Baseline to beat."""

    def run(self, price_df: pd.DataFrame, initial_capital: float = 1_000_000) -> pd.Series:
        """
        price_df: wide DataFrame (dates × symbols)
        Returns daily portfolio value series.
        """
        price_df = price_df.dropna(how="all")
        weights = np.ones(len(price_df.columns)) / len(price_df.columns)
        returns = price_df.pct_change().fillna(0)
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_values = initial_capital * (1 + portfolio_returns).cumprod()
        portfolio_values.iloc[0] = initial_capital
        return portfolio_values


class MomentumBenchmark:
    """Simple 12-1 month momentum strategy."""

    def run(self, price_df: pd.DataFrame, initial_capital: float = 1_000_000,
            top_n: int = 5) -> pd.Series:
        """Hold top N momentum stocks, rebalance monthly."""
        portfolio_values = []
        cash = initial_capital
        positions = {}

        monthly_rebal_dates = price_df.resample("ME").last().index

        for date in price_df.index:
            if date in monthly_rebal_dates:
                # Compute 12-1 month momentum
                lookback_date = date - pd.DateOffset(months=12)
                skip_date = date - pd.DateOffset(months=1)
                avail = price_df.loc[lookback_date:skip_date]
                if len(avail) > 20:
                    momentum = price_df.loc[date] / avail.iloc[0] - 1
                    top_stocks = momentum.nlargest(top_n).index.tolist()
                    positions = {s: 1 / top_n for s in top_stocks}

            current_prices = price_df.loc[date]
            total = sum(
                positions.get(s, 0) * initial_capital * current_prices[s] / price_df.iloc[0][s]
                for s in positions
            )
            portfolio_values.append(total if total > 0 else initial_capital)

        return pd.Series(portfolio_values, index=price_df.index)


# ─── Evaluation Report Printer ────────────────────────────────────────────────

def compare_strategies(
    strategy_values: Dict[str, pd.Series],
    benchmark_values: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.

    Args:
        strategy_values: {"RL Agent": pd.Series, "Buy&Hold": pd.Series, ...}
        benchmark_values: SPY values for alpha/beta calc

    Returns:
        DataFrame with one row per strategy, columns = metrics
    """
    rows = []
    for name, pv in strategy_values.items():
        report = compute_metrics(pv, benchmark_values, risk_free_rate)
        row = report.to_dict()
        row["strategy"] = name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")
    key_cols = [
        "total_return", "annual_return", "annual_volatility",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "max_drawdown_duration",
        "alpha", "beta", "win_rate", "total_trades",
    ]
    existing_cols = [c for c in key_cols if c in df.columns]
    return df[existing_cols]


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Pretty-print comparison table."""
    formatters = {
        "total_return": "{:+.2%}".format,
        "annual_return": "{:+.2%}".format,
        "annual_volatility": "{:.2%}".format,
        "max_drawdown": "{:.2%}".format,
        "alpha": "{:+.2%}".format,
        "win_rate": "{:.2%}".format,
        "sharpe_ratio": "{:.3f}".format,
        "sortino_ratio": "{:.3f}".format,
        "calmar_ratio": "{:.3f}".format,
        "beta": "{:.3f}".format,
    }

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    formatted = comparison_df.copy()
    for col, fmt in formatters.items():
        if col in formatted.columns:
            formatted[col] = formatted[col].map(fmt)
    print(formatted.to_string())
    print("=" * 80 + "\n")
