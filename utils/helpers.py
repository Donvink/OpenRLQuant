"""
utils/helpers.py
─────────────────
Shared utilities: logging setup, plotting, reproducibility, timing.
"""

import os
import random
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logging.getLogger(__name__).debug(f"Global seed set to {seed}")


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_str: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
) -> logging.Logger:
    """Configure root logger with console + optional file handler."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("rl_trader")


# ─── Timing ───────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger = logging.getLogger(__name__)
    logger.info(f"⏱ {label}: {elapsed:.3f}s")


# ─── Data Utilities ────────────────────────────────────────────────────────────

def align_dataframes(dfs: Dict[str, pd.DataFrame], freq: str = "B") -> Dict[str, pd.DataFrame]:
    """
    Align multiple DataFrames to the same date index.
    Uses business day frequency by default.
    """
    if not dfs:
        return {}

    # Get union of all indices
    all_dates = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    common_idx = pd.DatetimeIndex(all_dates)

    return {sym: df.reindex(common_idx).ffill() for sym, df in dfs.items()}


def compute_rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlation for portfolio construction.
    Returns average correlation over the last `window` days.
    """
    recent = returns.tail(window)
    return recent.corr()


def detect_regime(
    vix: pd.Series,
    spy_returns: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """
    Simple market regime detection using VIX + trend.

    Returns DataFrame with columns:
      vix_regime: 0=calm, 1=normal, 2=elevated, 3=crisis
      trend:      1=bull, -1=bear, 0=sideways
      regime:     combined label string
    """
    result = pd.DataFrame(index=vix.index)

    # VIX regime
    vix_aligned = vix.reindex(result.index)
    result["vix_regime"] = pd.cut(
        vix_aligned, bins=[0, 15, 25, 35, 1000], labels=[0, 1, 2, 3]
    ).astype(float)

    # Trend: rolling return direction
    spy_aligned = spy_returns.reindex(result.index)
    rolling_ret = spy_aligned.rolling(window).sum()
    result["trend"] = np.where(rolling_ret > 0.05, 1,
                      np.where(rolling_ret < -0.05, -1, 0))

    # Combined regime label
    regime_map = {
        (0, 1): "Calm Bull", (1, 1): "Normal Bull", (2, 1): "Elevated Bull",
        (0, 0): "Calm Sideways", (1, 0): "Normal Sideways", (2, 0): "Elevated Sideways",
        (0, -1): "Calm Bear", (1, -1): "Normal Bear", (2, -1): "Elevated Bear",
        (3, 1): "Crisis / Rally", (3, 0): "Crisis", (3, -1): "Crisis Bear",
    }
    result["regime"] = [
        regime_map.get((int(v), int(t)), "Unknown")
        for v, t in zip(result["vix_regime"].fillna(1), result["trend"])
    ]

    return result


def build_returns_matrix(
    market_data: Dict[str, pd.DataFrame],
    log_returns: bool = True,
    freq: str = "1d",
) -> pd.DataFrame:
    """Wide DataFrame of returns: rows=dates, cols=symbols."""
    frames = {}
    for sym, df in market_data.items():
        if "close" in df.columns and len(df) > 1:
            if log_returns:
                frames[sym] = np.log(df["close"] / df["close"].shift(1))
            else:
                frames[sym] = df["close"].pct_change()

    return pd.DataFrame(frames).dropna(how="all")


# ─── Portfolio Analytics ───────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.05) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    rf_daily = (1 + rf) ** (1 / 252) - 1
    excess = returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / (roll_std + 1e-8)) * np.sqrt(252)


def rolling_drawdown(values: pd.Series) -> pd.Series:
    """Rolling drawdown from peak."""
    peak = values.expanding().max()
    return (values - peak) / peak


def compute_turnover(weights_history: pd.DataFrame) -> pd.Series:
    """
    Compute daily portfolio turnover.
    weights_history: DataFrame with dates as index, symbols as columns.
    """
    return weights_history.diff().abs().sum(axis=1)


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly criterion position sizing.
    fraction=0.25 means 25% Kelly (conservative, industry standard).
    """
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly_f = (win_rate * b - q) / b
    return max(0.0, kelly_f * fraction)


# ─── Plotting (matplotlib) ─────────────────────────────────────────────────────

def plot_portfolio_performance(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "Portfolio Performance",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot cumulative returns, drawdown, and rolling Sharpe.
    Pure matplotlib — no external dependencies.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logging.getLogger(__name__).warning("matplotlib not installed, skipping plot")
        return

    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.08)

    colors = {"port": "#00d4ff", "bench": "#ff6b35", "dd": "#ef4444", "sharpe": "#00ff88"}

    # Normalize to 100
    port_norm = portfolio_values / portfolio_values.iloc[0] * 100
    returns = portfolio_values.pct_change().dropna()

    # ── Panel 1: Cumulative returns ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0d1117")
    ax1.plot(port_norm.index, port_norm.values, color=colors["port"], linewidth=1.5,
             label="RL Strategy", zorder=3)

    if benchmark_values is not None:
        bm_norm = benchmark_values.reindex(portfolio_values.index).ffill()
        bm_norm = bm_norm / bm_norm.iloc[0] * 100
        ax1.plot(bm_norm.index, bm_norm.values, color=colors["bench"], linewidth=1.0,
                 alpha=0.7, linestyle="--", label="Benchmark (SPY)")

    ax1.axhline(100, color="#334155", linewidth=0.8, linestyle=":")
    ax1.fill_between(port_norm.index, 100, port_norm.values,
                     where=port_norm.values >= 100, alpha=0.1, color=colors["port"])
    ax1.fill_between(port_norm.index, 100, port_norm.values,
                     where=port_norm.values < 100, alpha=0.1, color=colors["dd"])

    total_ret = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
    ax1.set_title(f"{title}  |  Total Return: {total_ret:+.2%}", color="white",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155",
               labelcolor="white", fontsize=9)
    ax1.set_ylabel("Indexed (Base=100)", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#64748b")
    ax1.spines[:].set_color("#1e293b")
    ax1.set_xticklabels([])

    # ── Panel 2: Drawdown ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0d1117")
    dd = rolling_drawdown(portfolio_values) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=colors["dd"], alpha=0.6)
    ax2.plot(dd.index, dd.values, color=colors["dd"], linewidth=0.8)
    ax2.axhline(0, color="#334155", linewidth=0.8)
    ax2.set_ylabel("Drawdown %", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#64748b")
    ax2.spines[:].set_color("#1e293b")
    ax2.set_xticklabels([])

    # ── Panel 3: Rolling Sharpe ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#0d1117")
    roll_sharpe = rolling_sharpe(returns, window=63)
    ax3.plot(roll_sharpe.index, roll_sharpe.values, color=colors["sharpe"], linewidth=1.0)
    ax3.axhline(0, color="#334155", linewidth=0.8)
    ax3.axhline(1.0, color=colors["sharpe"], linewidth=0.5, linestyle=":", alpha=0.5)
    ax3.fill_between(roll_sharpe.index, 0, roll_sharpe.values,
                     where=roll_sharpe.values >= 0, alpha=0.2, color=colors["sharpe"])
    ax3.fill_between(roll_sharpe.index, 0, roll_sharpe.values,
                     where=roll_sharpe.values < 0, alpha=0.2, color=colors["dd"])
    ax3.set_ylabel("Rolling Sharpe (63d)", color="#94a3b8", fontsize=9)
    ax3.tick_params(colors="#64748b", axis="x", rotation=20)
    ax3.spines[:].set_color("#1e293b")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logging.getLogger(__name__).info(f"Chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of feature importances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    idx = np.argsort(importance_values)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importance_values[idx]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    colors_bar = ["#00d4ff" if v > 0 else "#ef4444" for v in vals]
    bars = ax.barh(names, vals, color=colors_bar, alpha=0.85)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_color("#1e293b")
    ax.axvline(0, color="#334155", linewidth=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    else:
        plt.show()
    plt.close()


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Return Correlation Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Heatmap of pairwise return correlations."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    corr = returns.corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#ef4444", "#1e293b", "#00d4ff"]
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#94a3b8", fontsize=8)
    ax.set_yticklabels(corr.index, color="#94a3b8", fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.5 else "#64748b")

    plt.colorbar(im, ax=ax, fraction=0.03, label="Correlation")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    else:
        plt.show()
    plt.close()
