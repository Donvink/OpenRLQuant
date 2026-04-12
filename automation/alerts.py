"""
automation/alerts.py
─────────────────────
Multi-channel alert system.

Channels:
  - Slack webhook (instant, free)
  - Email via SMTP (Gmail, SendGrid, etc.)
  - Generic webhook (custom endpoint)
  - Console log (always active, fallback)

Alert levels:
  info     → green
  warning  → yellow
  error    → red
  success  → blue (promotions, milestones)
  critical → red + @channel ping

Usage:
    alerts = AlertSystem.from_env()   # reads from environment variables
    alerts.send("Model promoted!", level="success",
                details={"sharpe": 1.5, "return": "18%"})

Environment variables:
    SLACK_WEBHOOK_URL   — Slack incoming webhook
    ALERT_EMAIL_TO      — recipient email
    ALERT_EMAIL_FROM    — sender (e.g. your Gmail)
    ALERT_EMAIL_PASS    — app password (not main password)
    ALERT_WEBHOOK_URL   — generic HTTP webhook
"""

import json
import logging
import os
import smtplib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ─── Alert Message ────────────────────────────────────────────────────────────

@dataclass
class Alert:
    message: str
    level: str = "info"               # info | warning | error | success | critical
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "rl_trader"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def emoji(self) -> str:
        return {"info": "ℹ️", "warning": "⚠️", "error": "❌",
                "success": "✅", "critical": "🚨"}.get(self.level, "📢")

    @property
    def color(self) -> str:
        return {"info": "#00d4ff", "warning": "#fbbf24", "error": "#ef4444",
                "success": "#00ff88", "critical": "#ff0000"}.get(self.level, "#94a3b8")

    def format_details(self) -> str:
        if not self.details:
            return ""
        lines = []
        for k, v in self.details.items():
            if isinstance(v, float):
                lines.append(f"  • {k}: {v:.4f}")
            else:
                lines.append(f"  • {k}: {v}")
        return "\n".join(lines)


# ─── Individual Senders ───────────────────────────────────────────────────────

class SlackSender:
    """Slack Incoming Webhook sender."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        payload = {
            "text": f"{alert.emoji} *RL Trader* — {alert.message}",
            "attachments": [
                {
                    "color": alert.color,
                    "fields": [
                        {"title": "Level", "value": alert.level.upper(), "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M UTC"), "short": True},
                    ] + [
                        {"title": k, "value": str(v)[:200], "short": True}
                        for k, v in list(alert.details.items())[:8]
                    ],
                    "footer": "RL Trader",
                    "footer_icon": "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/robot_1f916.png",
                }
            ],
        }

        # Add @channel for critical
        if alert.level == "critical":
            payload["text"] = "<!channel> " + payload["text"]

        try:
            resp = httpx.post(self.webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class EmailSender:
    """SMTP email sender. Works with Gmail, SendGrid, AWS SES."""

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        from_addr: str = "",
        password: str = "",
        to_addrs: List[str] = None,
    ):
        self.host = smtp_host
        self.port = smtp_port
        self.from_addr = from_addr
        self.password = password
        self.to_addrs = to_addrs or []

    def send(self, alert: Alert) -> bool:
        if not self.from_addr or not self.to_addrs:
            return False

        subject = f"[RL Trader {alert.level.upper()}] {alert.message[:60]}"
        body_html = f"""
        <html><body style="font-family: monospace; background: #0d1117; color: #e2e8f0; padding: 20px;">
        <h2 style="color: {alert.color}">{alert.emoji} {alert.message}</h2>
        <p><b>Level:</b> {alert.level.upper()}</p>
        <p><b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><b>Source:</b> {alert.source}</p>
        {"<h3>Details:</h3><pre>" + json.dumps(alert.details, indent=2, default=str) + "</pre>" if alert.details else ""}
        <hr><small>RL Trader Automated Alerts</small>
        </body></html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.host, self.port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.from_addr, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class WebhookSender:
    """Generic HTTP webhook sender (Discord, custom endpoints, etc.)."""

    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, alert: Alert) -> bool:
        payload = {
            "message": alert.message,
            "level": alert.level,
            "emoji": alert.emoji,
            "timestamp": alert.timestamp.isoformat(),
            "details": alert.details,
            "source": alert.source,
        }
        try:
            resp = httpx.post(self.url, json=payload, headers=self.headers, timeout=10)
            return resp.status_code < 300
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False


# ─── Alert System ─────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Unified alert dispatcher. Sends to all configured channels asynchronously.
    Always logs to console regardless of other channel failures.
    """

    def __init__(self):
        self._senders: List = []
        self._history: List[Alert] = []
        self._min_level_order = ["info", "warning", "error", "success", "critical"]

    def add_slack(self, webhook_url: str) -> "AlertSystem":
        if webhook_url:
            self._senders.append(SlackSender(webhook_url))
            logger.info("Alert channel added: Slack")
        return self

    def add_email(self, from_addr: str, password: str, to_addrs: List[str],
                  smtp_host: str = "smtp.gmail.com") -> "AlertSystem":
        if from_addr and to_addrs:
            self._senders.append(EmailSender(smtp_host, 587, from_addr, password, to_addrs))
            logger.info(f"Alert channel added: Email → {to_addrs}")
        return self

    def add_webhook(self, url: str, headers: Dict = None) -> "AlertSystem":
        if url:
            self._senders.append(WebhookSender(url, headers))
            logger.info("Alert channel added: Webhook")
        return self

    def send(
        self,
        message: str,
        level: str = "info",
        details: Dict[str, Any] = None,
        source: str = "rl_trader",
        min_level: str = "info",   # only send if level >= min_level
    ) -> None:
        """Send alert to all configured channels. Non-blocking."""
        alert = Alert(message=message, level=level,
                      details=details or {}, source=source)
        self._history.append(alert)

        # Console log always
        log_fn = {
            "info": logger.info, "warning": logger.warning,
            "error": logger.error, "success": logger.info, "critical": logger.critical,
        }.get(level, logger.info)
        log_fn(f"{alert.emoji} ALERT [{level.upper()}]: {message}")
        if details:
            logger.debug(f"  Details: {details}")

        # Skip if below minimum level
        level_order = self._min_level_order
        if level_order.index(level) < level_order.index(min_level):
            return

        # Send to all channels in background thread
        if self._senders:
            thread = threading.Thread(
                target=self._dispatch_async,
                args=(alert,),
                daemon=True,
            )
            thread.start()

    def _dispatch_async(self, alert: Alert):
        for sender in self._senders:
            try:
                success = sender.send(alert)
                if not success:
                    logger.debug(f"Sender {type(sender).__name__} returned failure")
            except Exception as e:
                logger.debug(f"Sender {type(sender).__name__} error: {e}")

    def get_history(self, level: str = None, last_n: int = 50) -> List[Dict]:
        """Return recent alert history."""
        history = self._history
        if level:
            history = [a for a in history if a.level == level]
        return [
            {
                "timestamp": a.timestamp.isoformat(),
                "level": a.level,
                "message": a.message,
                "details": a.details,
            }
            for a in history[-last_n:]
        ]

    @classmethod
    def from_env(cls) -> "AlertSystem":
        """Build AlertSystem from environment variables."""
        system = cls()

        slack_url = os.getenv("SLACK_WEBHOOK_URL", "")
        if slack_url:
            system.add_slack(slack_url)

        email_to = os.getenv("ALERT_EMAIL_TO", "")
        email_from = os.getenv("ALERT_EMAIL_FROM", "")
        email_pass = os.getenv("ALERT_EMAIL_PASS", "")
        if email_to and email_from:
            system.add_email(
                from_addr=email_from,
                password=email_pass,
                to_addrs=[e.strip() for e in email_to.split(",")],
            )

        webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
        if webhook_url:
            system.add_webhook(webhook_url)

        n = len(system._senders)
        logger.info(f"AlertSystem initialized: {n} channel(s) + console")
        return system

    @classmethod
    def console_only(cls) -> "AlertSystem":
        """Minimal AlertSystem for testing — console log only."""
        return cls()
