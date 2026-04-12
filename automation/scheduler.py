"""
automation/scheduler.py
────────────────────────
APScheduler-based job scheduler for automated trading operations.

Scheduled jobs:
  daily_report       07:00 ET  — P&L summary, position report
  pre_market_check   09:00 ET  — data quality, model health, risk limits
  post_market_wrap   16:30 ET  — daily wrap, archive state, update features
  weekly_retrain     Sun 22:00 ET — Champion/Challenger retraining
  drift_check        Every 6h  — check for model/market drift
  health_check       Every 5m  — engine heartbeat
  performance_report Mon 07:30 ET — weekly performance summary

Usage:
    scheduler = TradingScheduler(engine, retrain_pipeline, alert_system)
    scheduler.start()
    # ... runs indefinitely in background threads
    scheduler.stop()
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    Central scheduler for all automated trading operations.
    Uses APScheduler's BackgroundScheduler so the main thread stays free.
    """

    def __init__(
        self,
        engine=None,
        retrain_pipeline=None,
        alert_system=None,
        drift_monitor=None,
        ensemble_manager=None,
        timezone: str = "America/New_York",
    ):
        self.engine = engine
        self.retrain = retrain_pipeline
        self.alerts = alert_system
        self.drift = drift_monitor
        self.ensemble = ensemble_manager
        self.tz = timezone

        self._scheduler = BackgroundScheduler(timezone=timezone)
        self._job_results: Dict[str, List] = {}
        self._setup_jobs()

    def _setup_jobs(self):
        """Register all scheduled jobs."""

        # ── Market-day jobs (NYSE hours) ────────────────────────────────────────

        # Pre-market check: 9:00 AM ET every weekday
        self._scheduler.add_job(
            self._pre_market_check,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=self.tz),
            id="pre_market_check",
            name="Pre-Market Health Check",
            replace_existing=True,
        )

        # Post-market wrap: 4:30 PM ET every weekday
        self._scheduler.add_job(
            self._post_market_wrap,
            CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=self.tz),
            id="post_market_wrap",
            name="Post-Market Wrap",
            replace_existing=True,
        )

        # Daily P&L report: 7:00 AM ET
        self._scheduler.add_job(
            self._daily_report,
            CronTrigger(day_of_week="mon-fri", hour=7, minute=0, timezone=self.tz),
            id="daily_report",
            name="Daily P&L Report",
            replace_existing=True,
        )

        # ── Periodic jobs ───────────────────────────────────────────────────────

        # Drift check every 6 hours
        self._scheduler.add_job(
            self._drift_check,
            IntervalTrigger(hours=6),
            id="drift_check",
            name="Drift Detection Check",
            replace_existing=True,
        )

        # Health heartbeat every 5 minutes
        self._scheduler.add_job(
            self._health_check,
            IntervalTrigger(minutes=5),
            id="health_check",
            name="Engine Health Check",
            replace_existing=True,
        )

        # ── Weekly jobs ─────────────────────────────────────────────────────────

        # Weekly retraining: Sunday 10 PM ET
        self._scheduler.add_job(
            self._weekly_retrain,
            CronTrigger(day_of_week="sun", hour=22, minute=0, timezone=self.tz),
            id="weekly_retrain",
            name="Weekly Champion/Challenger Retrain",
            replace_existing=True,
        )

        # Weekly performance report: Monday 7:30 AM ET
        self._scheduler.add_job(
            self._weekly_performance_report,
            CronTrigger(day_of_week="mon", hour=7, minute=30, timezone=self.tz),
            id="weekly_performance_report",
            name="Weekly Performance Report",
            replace_existing=True,
        )

        logger.info(f"Scheduled {len(self._scheduler.get_jobs())} jobs")

    # ── Job Implementations ───────────────────────────────────────────────────

    def _pre_market_check(self):
        """9:00 AM — verify everything is ready for market open."""
        logger.info("=== PRE-MARKET CHECK ===")
        issues = []

        # Check engine
        if self.engine:
            if self.engine.state.is_halted:
                issues.append(f"Trading HALTED: {self.engine.state.halt_reason}")
            if self.engine.state.portfolio_value < self.engine.initial_capital * 0.5:
                issues.append(f"Portfolio down >50%: ${self.engine.state.portfolio_value:,.0f}")

        # Check model
        if self.engine and hasattr(self.engine, "agent"):
            try:
                obs = self.engine.agent.observation_space.sample()
                self.engine.agent.predict(obs)
                logger.info("Model inference: OK")
            except Exception as e:
                issues.append(f"Model inference failed: {e}")

        if issues:
            msg = "Pre-market issues: " + "; ".join(issues)
            if self.alerts:
                self.alerts.send(msg, level="error")
        else:
            logger.info("Pre-market check: ALL CLEAR ✓")
            if self.alerts:
                self.alerts.send("Pre-market check: all systems operational ✓",
                                 level="info", min_level="warning")  # suppress routine info

        self._record_job("pre_market_check", {"issues": issues})

    def _post_market_wrap(self):
        """4:30 PM — wrap up trading day, archive state."""
        logger.info("=== POST-MARKET WRAP ===")
        if not self.engine:
            return

        state = self.engine.state
        daily_ret = state.daily_return
        pv = state.portfolio_value

        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_value": pv,
            "daily_return": daily_ret,
            "total_return": state.total_return,
            "n_trades": state.n_trades_today,
        }

        level = "success" if daily_ret > 0 else "warning" if daily_ret > -0.02 else "error"
        if self.alerts:
            self.alerts.send(
                f"Market close: {daily_ret:+.2%} today | ${pv:,.0f}",
                level=level, details=summary,
            )

        # Update drift monitor with today's return
        if self.drift:
            self.drift.update(daily_return=daily_ret)

        # Update ensemble performance tracking
        if self.ensemble and len(self.engine._value_history) >= 2:
            sharpe = state.daily_return / (np.std(self.engine._value_history[-21:] or [1]) + 1e-8)
            self.ensemble.update_performance(float(sharpe))

        self._record_job("post_market_wrap", summary)

    def _daily_report(self):
        """7:00 AM — send daily P&L summary."""
        if not self.engine or not self.alerts:
            return

        state = self.engine.state
        vh = self.engine._value_history
        returns = np.diff(vh) / np.array(vh[:-1]) if len(vh) > 1 else [0]

        report = {
            "portfolio_value": f"${state.portfolio_value:,.0f}",
            "total_return": f"{state.total_return:+.2%}",
            "cycle_count": state.cycle_count,
            "n_positions": len([v for v in state.positions.values() if abs(v) > 0.001]),
            "rolling_sharpe_21d": round(
                float(np.mean(returns[-21:]) / (np.std(returns[-21:]) + 1e-8) * np.sqrt(252)), 3
            ) if len(returns) >= 5 else "N/A",
        }

        self.alerts.send(
            f"Daily Report — Portfolio: ${state.portfolio_value:,.0f} ({state.total_return:+.2%})",
            level="info", details=report,
        )

    def _drift_check(self):
        """Every 6h — run drift detection checks."""
        if not self.drift:
            return

        reports = [r for r in self.drift.reports[-3:] if r.drift_detected]
        if reports:
            latest = reports[-1]
            logger.warning(f"Drift check: {latest.summary()}")
        else:
            logger.debug("Drift check: no drift detected")

        self._record_job("drift_check", {"drift_events": len(reports)})

    def _health_check(self):
        """Every 5min — engine heartbeat."""
        if not self.engine:
            return

        state = self.engine.state
        if state.is_halted and self.alerts:
            # Re-alert every 30 minutes if still halted
            last_halt_alerts = [
                j for j in self._job_results.get("health_check", [])
                if j.get("halted", False)
            ]
            if len(last_halt_alerts) % 6 == 0:  # every 6 × 5min = 30 min
                self.alerts.send(
                    f"⚠️ Trading still HALTED: {state.halt_reason}",
                    level="critical",
                )

        self._record_job("health_check", {"halted": state.is_halted, "pv": state.portfolio_value})

    def _weekly_retrain(self):
        """Sunday 10pm — champion/challenger retraining."""
        logger.info("=== WEEKLY RETRAIN TRIGGERED ===")
        if not self.retrain:
            logger.info("No retrain pipeline configured")
            return

        try:
            result = self.retrain.run(triggered_by="schedule")
            self._record_job("weekly_retrain", result.to_dict())
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
            if self.alerts:
                self.alerts.send(f"Weekly retrain error: {e}", level="error")

    def _weekly_performance_report(self):
        """Monday 7:30 AM — weekly summary."""
        if not self.engine or not self.alerts:
            return

        vh = self.engine._value_history[-6:]  # last 5 trading days
        if len(vh) < 2:
            return

        weekly_ret = (vh[-1] - vh[0]) / vh[0]
        daily_returns = np.diff(vh) / np.array(vh[:-1])

        report = {
            "week_return": f"{weekly_ret:+.2%}",
            "best_day": f"{max(daily_returns):+.2%}" if len(daily_returns) else "N/A",
            "worst_day": f"{min(daily_returns):+.2%}" if len(daily_returns) else "N/A",
            "portfolio_value": f"${self.engine.state.portfolio_value:,.0f}",
            "total_return": f"{self.engine.state.total_return:+.2%}",
        }

        if self.ensemble:
            leaderboard = self.ensemble.get_leaderboard()
            report["top_model"] = leaderboard[0]["name"] if leaderboard else "N/A"

        level = "success" if weekly_ret > 0.01 else "warning" if weekly_ret > -0.02 else "error"
        self.alerts.send(
            f"Weekly Report: {weekly_ret:+.2%} | ${self.engine.state.portfolio_value:,.0f}",
            level=level, details=report,
        )

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        """Start the scheduler."""
        self._scheduler.start()
        logger.info(f"Scheduler started: {len(self._scheduler.get_jobs())} jobs registered")

    def stop(self):
        """Graceful shutdown."""
        try:
            if self._scheduler.running:
                self._scheduler.shutdown(wait=False)
        except Exception:
            pass
        logger.info("Scheduler stopped")

    def trigger_now(self, job_id: str):
        """Manually trigger a job immediately."""
        job = self._scheduler.get_job(job_id)
        if job:
            job.modify(next_run_time=datetime.now(timezone.utc))
            logger.info(f"Job '{job_id}' triggered manually")
        else:
            logger.warning(f"Job '{job_id}' not found")

    def get_job_status(self) -> List[Dict]:
        """Return status of all scheduled jobs."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(getattr(job, "next_run_time", None) or
                             getattr(job, "next_fire_time", None) or "scheduled"),
                "recent_results": len(self._job_results.get(job.id, [])),
            }
            for job in self._scheduler.get_jobs()
        ]

    def _record_job(self, job_id: str, result: Dict):
        if job_id not in self._job_results:
            self._job_results[job_id] = []
        self._job_results[job_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        })
        # Keep last 100 results per job
        self._job_results[job_id] = self._job_results[job_id][-100:]
