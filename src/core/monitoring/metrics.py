"""
Prometheus-based metrics collection for monitoring.
"""

import time
from functools import wraps
from typing import Callable, Optional, Dict, Any
from collections import defaultdict

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


class MetricsCollector:
    """Centralized metrics collection using Prometheus."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.

        Args:
            registry: Optional Prometheus registry
        """
        self.registry = registry or CollectorRegistry()

        # System metrics
        self.system_cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry,
        )
        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry,
        )
        self.system_disk_usage = Gauge(
            "system_disk_usage_percent",
            "System disk usage percentage",
            registry=self.registry,
        )

        # Trading metrics
        self.trades_total = Counter(
            "trades_total",
            "Total number of trades executed",
            ["symbol", "side", "status"],
            registry=self.registry,
        )
        self.trade_volume = Histogram(
            "trade_volume_usd",
            "Trade volume in USD",
            ["symbol"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
            registry=self.registry,
        )
        self.trade_latency = Histogram(
            "trade_execution_latency_seconds",
            "Trade execution latency",
            ["order_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        # Portfolio metrics
        self.portfolio_value = Gauge(
            "portfolio_value_usd",
            "Total portfolio value in USD",
            registry=self.registry,
        )
        self.portfolio_cash = Gauge(
            "portfolio_cash_usd",
            "Portfolio cash balance in USD",
            registry=self.registry,
        )
        self.portfolio_positions = Gauge(
            "portfolio_positions_count",
            "Number of open positions",
            registry=self.registry,
        )
        self.portfolio_pnl = Gauge(
            "portfolio_pnl_usd",
            "Portfolio profit and loss in USD",
            ["timeframe"],
            registry=self.registry,
        )

        # Position metrics
        self.position_size = Gauge(
            "position_size_usd",
            "Position size in USD",
            ["symbol"],
            registry=self.registry,
        )
        self.position_pnl = Gauge(
            "position_pnl_usd",
            "Position P&L in USD",
            ["symbol"],
            registry=self.registry,
        )

        # Risk metrics
        self.portfolio_var = Gauge(
            "portfolio_var",
            "Portfolio Value at Risk",
            ["confidence"],
            registry=self.registry,
        )
        self.portfolio_volatility = Gauge(
            "portfolio_volatility",
            "Portfolio volatility (annualized)",
            registry=self.registry,
        )
        self.portfolio_beta = Gauge(
            "portfolio_beta",
            "Portfolio beta",
            registry=self.registry,
        )
        self.portfolio_sharpe = Gauge(
            "portfolio_sharpe_ratio",
            "Portfolio Sharpe ratio",
            registry=self.registry,
        )
        self.max_drawdown = Gauge(
            "portfolio_max_drawdown",
            "Maximum drawdown",
            registry=self.registry,
        )

        # Model metrics
        self.model_prediction_latency = Histogram(
            "model_prediction_latency_seconds",
            "Model prediction latency",
            ["model_type"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry,
        )
        self.model_accuracy = Gauge(
            "model_accuracy",
            "Model prediction accuracy",
            ["model_name"],
            registry=self.registry,
        )
        self.model_loss = Gauge(
            "model_loss",
            "Model training loss",
            ["model_name"],
            registry=self.registry,
        )
        self.model_updates = Counter(
            "model_updates_total",
            "Total model updates",
            ["model_name"],
            registry=self.registry,
        )

        # Data metrics
        self.data_points_processed = Counter(
            "data_points_processed_total",
            "Total data points processed",
            ["source"],
            registry=self.registry,
        )
        self.data_latency = Histogram(
            "data_fetch_latency_seconds",
            "Data fetch latency",
            ["source"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )
        self.data_errors = Counter(
            "data_errors_total",
            "Total data errors",
            ["source", "error_type"],
            registry=self.registry,
        )

        # API metrics
        self.api_requests = Counter(
            "api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )
        self.api_latency = Histogram(
            "api_latency_seconds",
            "API request latency",
            ["endpoint", "method"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "errors_total",
            "Total errors",
            ["component", "error_type"],
            registry=self.registry,
        )

        # Agent metrics
        self.agent_actions = Counter(
            "agent_actions_total",
            "Total agent actions",
            ["agent_type", "action"],
            registry=self.registry,
        )
        self.agent_rewards = Summary(
            "agent_rewards",
            "Agent rewards distribution",
            ["agent_type"],
            registry=self.registry,
        )

    def record_trade(
        self,
        symbol: str,
        side: str,
        volume: float,
        status: str,
        latency: float,
        order_type: str = "market",
    ) -> None:
        """Record trade metrics."""
        self.trades_total.labels(symbol=symbol, side=side, status=status).inc()
        self.trade_volume.labels(symbol=symbol).observe(volume)
        self.trade_latency.labels(order_type=order_type).observe(latency)

    def update_portfolio(
        self,
        value: float,
        cash: float,
        positions_count: int,
        daily_pnl: float,
        total_pnl: float,
    ) -> None:
        """Update portfolio metrics."""
        self.portfolio_value.set(value)
        self.portfolio_cash.set(cash)
        self.portfolio_positions.set(positions_count)
        self.portfolio_pnl.labels(timeframe="daily").set(daily_pnl)
        self.portfolio_pnl.labels(timeframe="total").set(total_pnl)

    def update_risk_metrics(
        self,
        var_95: float,
        volatility: float,
        beta: float,
        sharpe: float,
        drawdown: float,
    ) -> None:
        """Update risk metrics."""
        self.portfolio_var.labels(confidence="0.95").set(var_95)
        self.portfolio_volatility.set(volatility)
        self.portfolio_beta.set(beta)
        self.portfolio_sharpe.set(sharpe)
        self.max_drawdown.set(drawdown)

    def record_model_prediction(
        self,
        model_type: str,
        latency: float,
    ) -> None:
        """Record model prediction metrics."""
        self.model_prediction_latency.labels(model_type=model_type).observe(latency)

    def update_model_metrics(
        self,
        model_name: str,
        accuracy: float,
        loss: float,
    ) -> None:
        """Update model training metrics."""
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
        self.model_loss.labels(model_name=model_name).set(loss)
        self.model_updates.labels(model_name=model_name).inc()

    def record_data_processing(
        self,
        source: str,
        latency: float,
        points_count: int = 1,
    ) -> None:
        """Record data processing metrics."""
        self.data_points_processed.labels(source=source).inc(points_count)
        self.data_latency.labels(source=source).observe(latency)

    def record_error(
        self,
        component: str,
        error_type: str,
    ) -> None:
        """Record error."""
        self.errors_total.labels(component=component, error_type=error_type).inc()

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)

    def measure_time(self, metric: Histogram, **labels):
        """
        Context manager to measure execution time.

        Args:
            metric: Histogram metric to record to
            **labels: Metric labels
        """

        class Timer:
            def __init__(self, histogram, label_dict):
                self.histogram = histogram
                self.labels = label_dict
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start
                self.histogram.labels(**self.labels).observe(duration)

        return Timer(metric, labels)


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# Decorators for automatic metrics collection
def track_execution_time(metric_name: str = "function_execution_time"):
    """
    Decorator to track function execution time.

    Args:
        metric_name: Name of the metric
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                # Custom histogram or use existing ones
                if hasattr(metrics, metric_name):
                    getattr(metrics, metric_name).observe(duration)

        return wrapper

    return decorator


def track_errors(component: str):
    """
    Decorator to track function errors.

    Args:
        component: Component name
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                metrics.record_error(component, type(e).__name__)
                raise

        return wrapper

    return decorator
