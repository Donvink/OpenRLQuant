"""
monitoring/metrics_exporter.py
────────────────────────────────
Prometheus metrics exporter for real-time monitoring.

Metrics exposed at /metrics (scraped by Prometheus, visualized in Grafana):
  rl_trader_portfolio_value         — total portfolio value ($)
  rl_trader_total_return            — cumulative return (ratio)
  rl_trader_daily_return            — today's return (ratio)
  rl_trader_max_drawdown            — current max drawdown (negative ratio)
  rl_trader_sharpe_ratio            — rolling 63-day Sharpe
  rl_trader_position_weight{sym}    — weight per position
  rl_trader_n_trades_today          — trade count
  rl_trader_is_halted               — 0/1 halt flag
  rl_trader_drift_detected          — 0/1 drift flag

Run alongside API server:
  python monitoring/metrics_exporter.py --port 9090

Grafana setup:
  1. Add Prometheus datasource: http://prometheus:9090
  2. Import dashboard JSON from monitoring/grafana_dashboard.json
"""

import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MetricsExporter:
    """
    Simple Prometheus text-format metrics exporter.
    No external dependencies (pure Python stdlib).
    """

    def __init__(self):
        self._metrics: Dict[str, float] = {}
        self._labels: Dict[str, Dict[str, float]] = {}  # metric -> {label -> value}
        self._lock = threading.Lock()

    def set(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a metric value, optionally with labels."""
        with self._lock:
            if labels:
                key = name + "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
                self._labels.setdefault(name, {})[key] = value
            else:
                self._metrics[name] = value

    def update_from_state(self, state, drift_detected: bool = False, sharpe: float = 0.0):
        """Bulk update from engine state."""
        self.set("rl_trader_portfolio_value", state.portfolio_value)
        self.set("rl_trader_total_return", state.total_return)
        self.set("rl_trader_daily_return", state.daily_return)
        self.set("rl_trader_max_drawdown", state.max_drawdown)
        self.set("rl_trader_sharpe_ratio", sharpe)
        self.set("rl_trader_n_trades_today", float(state.n_trades_today))
        self.set("rl_trader_is_halted", 1.0 if state.is_halted else 0.0)
        self.set("rl_trader_drift_detected", 1.0 if drift_detected else 0.0)
        self.set("rl_trader_cycle_count", float(state.cycle_count))

        for sym, weight in state.weights.items():
            self.set("rl_trader_position_weight", weight, {"symbol": sym})

    def render(self) -> str:
        """Render metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for name, value in self._metrics.items():
                lines.append(f"# HELP {name} RL Trader metric")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            for name, label_dict in self._labels.items():
                lines.append(f"# HELP {name} RL Trader metric with labels")
                lines.append(f"# TYPE {name} gauge")
                for key, value in label_dict.items():
                    lines.append(f"{key} {value}")

        return "\n".join(lines) + "\n"

    def start_server(self, port: int = 9090):
        """Start HTTP server exposing /metrics endpoint."""
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    body = exporter.render().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass  # suppress access logs

        server = HTTPServer(("0.0.0.0", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Prometheus metrics server started on :{port}/metrics")
        return server


# ─── Grafana Dashboard JSON ────────────────────────────────────────────────────

GRAFANA_DASHBOARD = {
    "title": "RL Trader Live Dashboard",
    "uid": "rl-trader-v1",
    "schemaVersion": 38,
    "refresh": "10s",
    "time": {"from": "now-6h", "to": "now"},
    "panels": [
        {
            "id": 1, "type": "stat", "title": "Portfolio Value",
            "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
            "targets": [{"expr": "rl_trader_portfolio_value", "legendFormat": "Value"}],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD", "color": {"mode": "thresholds"},
                    "thresholds": {"steps": [{"color": "red", "value": 0}, {"color": "green", "value": 100000}]},
                }
            },
        },
        {
            "id": 2, "type": "stat", "title": "Total Return",
            "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
            "targets": [{"expr": "rl_trader_total_return * 100", "legendFormat": "Return %"}],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {"steps": [{"color": "red", "value": -10}, {"color": "yellow", "value": 0}, {"color": "green", "value": 10}]},
                }
            },
        },
        {
            "id": 3, "type": "stat", "title": "Sharpe Ratio",
            "gridPos": {"h": 4, "w": 4, "x": 8, "y": 0},
            "targets": [{"expr": "rl_trader_sharpe_ratio", "legendFormat": "Sharpe"}],
            "fieldConfig": {
                "defaults": {
                    "thresholds": {"steps": [{"color": "red", "value": 0}, {"color": "yellow", "value": 0.5}, {"color": "green", "value": 1.0}]},
                }
            },
        },
        {
            "id": 4, "type": "stat", "title": "Max Drawdown",
            "gridPos": {"h": 4, "w": 4, "x": 12, "y": 0},
            "targets": [{"expr": "rl_trader_max_drawdown * 100", "legendFormat": "Drawdown %"}],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {"steps": [{"color": "green", "value": -5}, {"color": "yellow", "value": -10}, {"color": "red", "value": -15}]},
                }
            },
        },
        {
            "id": 5, "type": "stat", "title": "Trading Halted",
            "gridPos": {"h": 4, "w": 4, "x": 16, "y": 0},
            "targets": [{"expr": "rl_trader_is_halted", "legendFormat": "Halted"}],
            "fieldConfig": {
                "defaults": {
                    "mappings": [{"type": "value", "options": {"0": {"text": "RUNNING", "color": "green"}, "1": {"text": "HALTED", "color": "red"}}}],
                }
            },
        },
        {
            "id": 6, "type": "timeseries", "title": "Portfolio Value Over Time",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
            "targets": [{"expr": "rl_trader_portfolio_value", "legendFormat": "Portfolio Value"}],
            "fieldConfig": {"defaults": {"unit": "currencyUSD", "color": {"fixedColor": "#00d4ff", "mode": "fixed"}}},
        },
        {
            "id": 7, "type": "timeseries", "title": "Returns (Daily)",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
            "targets": [
                {"expr": "rl_trader_daily_return * 100", "legendFormat": "Daily Return %"},
                {"expr": "rl_trader_total_return * 100", "legendFormat": "Total Return %"},
            ],
            "fieldConfig": {"defaults": {"unit": "percent"}},
        },
        {
            "id": 8, "type": "bargauge", "title": "Position Weights",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
            "targets": [{"expr": "rl_trader_position_weight * 100", "legendFormat": "{{symbol}}"}],
            "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 15}},
            "options": {"orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"]}},
        },
        {
            "id": 9, "type": "timeseries", "title": "Drawdown",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
            "targets": [{"expr": "rl_trader_max_drawdown * 100", "legendFormat": "Drawdown %"}],
            "fieldConfig": {
                "defaults": {"unit": "percent", "color": {"fixedColor": "#ef4444", "mode": "fixed"}},
            },
        },
    ],
    "templating": {"list": []},
    "annotations": {
        "list": [
            {
                "name": "Drift Events",
                "datasource": "prometheus",
                "expr": "changes(rl_trader_drift_detected[1m]) > 0",
                "enable": True,
                "color": "orange",
                "titleFormat": "Drift Detected",
            },
            {
                "name": "Halt Events",
                "datasource": "prometheus",
                "expr": "rl_trader_is_halted == 1",
                "enable": True,
                "color": "red",
                "titleFormat": "Trading Halted",
            },
        ]
    },
}


def save_grafana_dashboard(path: str = "monitoring/grafana_dashboard.json"):
    """Save the Grafana dashboard JSON to disk for import."""
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(GRAFANA_DASHBOARD, f, indent=2)
    logger.info(f"Grafana dashboard saved to {path}")
