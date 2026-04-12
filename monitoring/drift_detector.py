"""
monitoring/drift_detector.py
──────────────────────────────
Model drift detection — triggers retraining when the agent degrades.

Two types of drift monitored:
  1. Performance drift  — rolling Sharpe / return falls below threshold
  2. Distribution drift — feature distributions shift (KS test on inputs)

When drift detected:
  → Alert logged + notification sent
  → Retraining triggered (if auto_retrain=True)
  → Model rolled back to last best checkpoint until retrain completes
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ─── Drift Result ─────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    drift_detected: bool = False
    drift_type: str = ""           # "performance" | "feature" | "both"
    severity: str = ""             # "warning" | "critical"

    # Performance metrics
    recent_sharpe: float = 0.0
    baseline_sharpe: float = 0.0
    recent_return: float = 0.0
    sharpe_degradation: float = 0.0

    # Feature drift metrics
    feature_ks_statistic: float = 0.0
    feature_ks_pvalue: float = 1.0
    drifted_features: List[str] = field(default_factory=list)

    # Action
    action_taken: str = ""

    def summary(self) -> str:
        if not self.drift_detected:
            return f"No drift | Sharpe={self.recent_sharpe:.3f}"
        return (
            f"DRIFT [{self.severity}] {self.drift_type} | "
            f"Sharpe: {self.baseline_sharpe:.3f} → {self.recent_sharpe:.3f} "
            f"({self.sharpe_degradation:+.2%}) | Action: {self.action_taken}"
        )


# ─── Performance Drift Detector ───────────────────────────────────────────────

class PerformanceDriftDetector:
    """
    Monitors rolling performance metrics and detects significant degradation.

    Uses CUSUM (Cumulative Sum) algorithm — sensitive to sustained shifts,
    not single bad days. More reliable than simple threshold crossing.
    """

    def __init__(
        self,
        window_baseline: int = 63,    # 3-month baseline window
        window_recent: int = 21,      # 1-month recent window
        sharpe_warning_threshold: float = 0.3,   # warn if drops by 30%
        sharpe_critical_threshold: float = 0.5,  # critical if drops by 50%
        min_return_warning: float = -0.05,        # warn if return < -5%
        cusum_k: float = 0.5,         # CUSUM slack parameter
        cusum_h: float = 5.0,         # CUSUM decision threshold
    ):
        self.window_baseline = window_baseline
        self.window_recent = window_recent
        self.sharpe_warn = sharpe_warning_threshold
        self.sharpe_critical = sharpe_critical_threshold
        self.min_return_warn = min_return_warning

        # CUSUM state
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

        # Rolling history
        self._daily_returns: deque = deque(maxlen=window_baseline + window_recent)
        self._sharpe_history: List[float] = []

    def update(self, daily_return: float) -> None:
        """Add one day's return to the history."""
        self._daily_returns.append(daily_return)

        # Update CUSUM
        mu = self._get_baseline_mean()
        self._cusum_pos = max(0, self._cusum_pos + daily_return - mu - self.cusum_k)
        self._cusum_neg = max(0, self._cusum_neg - daily_return + mu - self.cusum_k)

    def check(self) -> Tuple[bool, str, float, float]:
        """
        Check for performance drift.
        Returns: (drift_detected, severity, recent_sharpe, baseline_sharpe)
        """
        if len(self._daily_returns) < self.window_recent + 5:
            return False, "", 0.0, 0.0

        returns = list(self._daily_returns)
        baseline_ret = returns[:self.window_baseline]
        recent_ret = returns[-self.window_recent:]

        # Compute rolling Sharpes
        def sharpe(r):
            r = np.array(r)
            if r.std() < 1e-8:
                return 0.0
            return float(r.mean() / r.std() * np.sqrt(252))

        baseline_sharpe = sharpe(baseline_ret) if len(baseline_ret) >= 5 else 0.0
        recent_sharpe = sharpe(recent_ret)

        degradation = (baseline_sharpe - recent_sharpe) / (abs(baseline_sharpe) + 1e-8)
        recent_cumret = float(np.prod(1 + np.array(recent_ret)) - 1)

        # CUSUM alarm
        cusum_alarm = (self._cusum_pos > self.cusum_h or self._cusum_neg > self.cusum_h)

        # Determine severity
        is_critical = (
            degradation > self.sharpe_critical or
            recent_cumret < self.min_return_warn * 2 or
            (cusum_alarm and degradation > 0.3)
        )
        is_warning = (
            degradation > self.sharpe_warn or
            recent_cumret < self.min_return_warn or
            cusum_alarm
        )

        if is_critical:
            return True, "critical", recent_sharpe, baseline_sharpe
        elif is_warning:
            return True, "warning", recent_sharpe, baseline_sharpe
        return False, "", recent_sharpe, baseline_sharpe

    def _get_baseline_mean(self) -> float:
        if len(self._daily_returns) >= self.window_baseline:
            return float(np.mean(list(self._daily_returns)[:self.window_baseline]))
        return 0.0

    def reset_cusum(self):
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0


# ─── Feature Distribution Drift ───────────────────────────────────────────────

class FeatureDriftDetector:
    """
    Detects shifts in input feature distributions using Kolmogorov-Smirnov test.
    Compares reference distribution (training time) vs current distribution.
    """

    def __init__(
        self,
        feature_names: List[str],
        ks_threshold: float = 0.05,    # p-value threshold
        drift_ratio_threshold: float = 0.2,  # fraction of features that must drift
        reference_window: int = 252,
        current_window: int = 21,
    ):
        self.feature_names = feature_names
        self.ks_threshold = ks_threshold
        self.drift_ratio = drift_ratio_threshold
        self.ref_window = reference_window
        self.cur_window = current_window
        self._feature_buffer: Dict[str, deque] = {
            f: deque(maxlen=reference_window + current_window)
            for f in feature_names
        }
        self._reference_set: Optional[Dict[str, np.ndarray]] = None

    def set_reference(self, obs_history: np.ndarray) -> None:
        """Set the training-time reference distribution."""
        if obs_history.shape[1] != len(self.feature_names):
            return
        for i, name in enumerate(self.feature_names):
            self._reference_set = self._reference_set or {}
            self._reference_set[name] = obs_history[:, i]
        logger.info(f"Feature drift reference set: {obs_history.shape[0]} samples")

    def update(self, obs_vector: np.ndarray) -> None:
        """Add one observation to current buffer."""
        n = min(len(obs_vector), len(self.feature_names))
        for i in range(n):
            self._feature_buffer[self.feature_names[i]].append(float(obs_vector[i]))

    def check(self) -> Tuple[bool, float, float, List[str]]:
        """
        Run KS test on all features.
        Returns: (drift_detected, ks_stat, ks_pvalue, drifted_feature_names)
        """
        if self._reference_set is None:
            return False, 0.0, 1.0, []

        min_samples = min(
            min(len(v) for v in self._feature_buffer.values()),
            len(self.feature_names)
        )
        if min_samples < 10:
            return False, 0.0, 1.0, []

        drifted = []
        ks_stats = []
        ks_pvals = []

        for name in self.feature_names:
            ref = self._reference_set.get(name)
            cur = np.array(list(self._feature_buffer[name])[-self.cur_window:])

            if ref is None or len(cur) < 5:
                continue

            ks_stat, ks_pval = stats.ks_2samp(ref, cur)
            ks_stats.append(ks_stat)
            ks_pvals.append(ks_pval)

            if ks_pval < self.ks_threshold:
                drifted.append(name)

        if not ks_stats:
            return False, 0.0, 1.0, []

        mean_stat = float(np.mean(ks_stats))
        min_pval = float(np.min(ks_pvals)) if ks_pvals else 1.0
        drift_fraction = len(drifted) / len(self.feature_names)

        drift_detected = drift_fraction > self.drift_ratio
        return drift_detected, mean_stat, min_pval, drifted[:10]  # top 10


# ─── Unified Drift Monitor ─────────────────────────────────────────────────────

class DriftMonitor:
    """
    Combines performance + feature drift detection.
    Triggers alerts and optional retraining.
    """

    def __init__(
        self,
        feature_names: List[str] = None,
        auto_retrain: bool = False,
        retrain_callback: Optional[Callable] = None,
        alert_callback: Optional[Callable[[DriftReport], None]] = None,
        check_every_n_cycles: int = 5,
    ):
        self.perf_detector = PerformanceDriftDetector()
        self.feat_detector = FeatureDriftDetector(feature_names or [])
        self.auto_retrain = auto_retrain
        self.retrain_callback = retrain_callback
        self.alert_callback = alert_callback
        self.check_every = check_every_n_cycles
        self._cycle = 0
        self.reports: List[DriftReport] = []
        self._retrain_pending = False

    def update(self, daily_return: float, obs: Optional[np.ndarray] = None) -> DriftReport:
        """Update detectors and check for drift."""
        self.perf_detector.update(daily_return)
        if obs is not None:
            self.feat_detector.update(obs)

        self._cycle += 1
        if self._cycle % self.check_every != 0:
            return DriftReport()

        return self._run_check()

    def _run_check(self) -> DriftReport:
        report = DriftReport()

        # Performance check
        p_drift, p_severity, recent_sharpe, baseline_sharpe = self.perf_detector.check()
        report.recent_sharpe = recent_sharpe
        report.baseline_sharpe = baseline_sharpe
        report.sharpe_degradation = (baseline_sharpe - recent_sharpe) / (abs(baseline_sharpe) + 1e-8)

        # Feature check
        f_drift, ks_stat, ks_pval, drifted_feats = self.feat_detector.check()
        report.feature_ks_statistic = ks_stat
        report.feature_ks_pvalue = ks_pval
        report.drifted_features = drifted_feats

        # Combined assessment
        if p_drift or f_drift:
            report.drift_detected = True
            if p_drift and f_drift:
                report.drift_type = "both"
            elif p_drift:
                report.drift_type = "performance"
            else:
                report.drift_type = "feature"

            report.severity = p_severity if p_drift else "warning"

            # Action
            if report.severity == "critical":
                report.action_taken = "retraining_triggered" if self.auto_retrain else "alert_sent"
                logger.critical(f"CRITICAL DRIFT: {report.summary()}")
                if self.auto_retrain and self.retrain_callback and not self._retrain_pending:
                    self._retrain_pending = True
                    self.retrain_callback(report)
            else:
                report.action_taken = "alert_sent"
                logger.warning(f"DRIFT WARNING: {report.summary()}")

            if self.alert_callback:
                self.alert_callback(report)

        self.reports.append(report)
        return report

    def get_drift_history(self) -> List[Dict]:
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "drift_detected": r.drift_detected,
                "drift_type": r.drift_type,
                "severity": r.severity,
                "recent_sharpe": round(r.recent_sharpe, 3),
                "baseline_sharpe": round(r.baseline_sharpe, 3),
                "action": r.action_taken,
            }
            for r in self.reports if r.drift_detected
        ]

    def retrain_complete(self):
        """Call when retraining finishes to reset state."""
        self._retrain_pending = False
        self.perf_detector.reset_cusum()
        logger.info("Drift monitor: retraining complete, CUSUM reset")
