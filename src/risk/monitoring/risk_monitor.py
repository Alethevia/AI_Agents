"""
Real-time risk monitoring module.

This module provides:
- Continuous risk calculation
- Limit breach detection
- Alert generation
- Auto-reduction on breach
- Risk dashboard data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable

import pandas as pd
from pydantic import BaseModel, Field

from ...config.risk_config import RiskConfig
from ...core.types.portfolio import Portfolio
from ..models.var_models import VaRCalculator, VaRResult
from ..models.risk_metrics import RiskMetricsCalculator, RiskMetrics
from ..limits.position_limits import PositionLimitManager, PositionLimitCheck
from ..limits.risk_limits import RiskLimitManager, RiskLimitCheck, LimitViolation


logger = logging.getLogger(__name__)


class RiskAlert(BaseModel):
    """Risk alert notification."""

    alert_id: str = Field(..., description="Unique alert ID")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    violation: Optional[LimitViolation] = Field(None, description="Associated violation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskSnapshot(BaseModel):
    """Point-in-time risk snapshot."""

    # Portfolio info
    portfolio_id: str = Field(..., description="Portfolio ID")
    portfolio_value: float = Field(..., description="Portfolio value")

    # Risk metrics
    metrics: RiskMetrics = Field(..., description="Risk metrics")

    # VaR
    var_results: Dict[str, VaRResult] = Field(
        default_factory=dict,
        description="VaR results by method"
    )

    # Limit checks
    position_limit_check: PositionLimitCheck = Field(
        ..., description="Position limit check results"
    )
    risk_limit_check: RiskLimitCheck = Field(
        ..., description="Risk limit check results"
    )

    # Status
    circuit_breaker_active: bool = Field(
        default=False,
        description="Circuit breaker status"
    )
    trading_allowed: bool = Field(
        default=True,
        description="Whether trading is allowed"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AlertManager:
    """Manage risk alerts."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: Dict[str, RiskAlert] = {}
        self.alert_handlers: List[Callable] = []

    def add_alert_handler(self, handler: Callable) -> None:
        """
        Add alert handler callback.

        Args:
            handler: Async function to handle alerts
        """
        self.alert_handlers.append(handler)

    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        violation: Optional[LimitViolation] = None
    ) -> RiskAlert:
        """
        Create and dispatch a risk alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            violation: Associated violation

        Returns:
            Created RiskAlert
        """
        # Generate unique ID
        alert_id = f"{alert_type}_{datetime.utcnow().timestamp()}"

        # Create alert
        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            violation=violation
        )

        # Store alert
        self.alerts[alert_id] = alert

        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(f"Risk Alert [{severity}]: {message}")

        return alert

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            Whether alert was found and acknowledged
        """
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            return True
        return False

    def get_active_alerts(self, severity: Optional[str] = None) -> List[RiskAlert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            severity: Filter by severity (optional)

        Returns:
            List of active alerts
        """
        alerts = [
            alert for alert in self.alerts.values()
            if not alert.acknowledged
        ]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def clear_old_alerts(self, hours: int = 24) -> None:
        """
        Clear old acknowledged alerts.

        Args:
            hours: Clear alerts older than this many hours
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        to_remove = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.acknowledged and alert.timestamp < cutoff
        ]

        for alert_id in to_remove:
            del self.alerts[alert_id]


class RiskMonitor:
    """Real-time risk monitoring system."""

    def __init__(
        self,
        config: RiskConfig,
        portfolio: Portfolio
    ):
        """
        Initialize risk monitor.

        Args:
            config: Risk configuration
            portfolio: Portfolio to monitor
        """
        self.config = config
        self.portfolio = portfolio

        # Risk calculators
        self.var_calculator = VaRCalculator(config.var_confidence_level)
        self.metrics_calculator = RiskMetricsCalculator(
            config.risk_free_rate
        )

        # Limit managers
        self.position_limits = PositionLimitManager(config)
        self.risk_limits = RiskLimitManager(config)

        # Alert manager
        self.alert_manager = AlertManager()

        # Historical data
        self.returns_history: pd.Series = pd.Series(dtype=float)
        self.price_history: Dict[str, pd.Series] = {}

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_snapshot: Optional[RiskSnapshot] = None

    def update_returns(self, returns: pd.Series) -> None:
        """
        Update returns history.

        Args:
            returns: Returns series to add
        """
        self.returns_history = pd.concat([self.returns_history, returns])

        # Keep only recent history based on lookback period
        lookback = self.config.var_lookback_days + 50  # Extra buffer
        if len(self.returns_history) > lookback:
            self.returns_history = self.returns_history.tail(lookback)

    def update_prices(self, symbol: str, prices: pd.Series) -> None:
        """
        Update price history for a symbol.

        Args:
            symbol: Symbol to update
            prices: Price series
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = prices
        else:
            self.price_history[symbol] = pd.concat([
                self.price_history[symbol],
                prices
            ])

    async def calculate_risk_snapshot(self) -> RiskSnapshot:
        """
        Calculate current risk snapshot.

        Returns:
            RiskSnapshot with current risk state
        """
        portfolio_value = float(self.portfolio.total_value())

        # Calculate risk metrics
        metrics = await self.metrics_calculator.calculate_all_metrics(
            self.returns_history
        )

        # Calculate VaR using all methods
        var_results = await self.var_calculator.calculate_all_methods(
            self.returns_history,
            portfolio_value
        )

        # Check position limits
        position_check = await self.position_limits.check_all_limits(
            self.portfolio
        )

        # Check risk limits
        risk_check = await self.risk_limits.check_all_limits(
            self.portfolio,
            self.returns_history
        )

        # Determine if trading is allowed
        circuit_breaker_active = self.risk_limits.circuit_breaker.is_triggered()
        trading_allowed = (
            not circuit_breaker_active and
            not risk_check.has_critical_violations() and
            not position_check.has_critical_violations()
        )

        return RiskSnapshot(
            portfolio_id=self.portfolio.portfolio_id,
            portfolio_value=portfolio_value,
            metrics=metrics,
            var_results=var_results,
            position_limit_check=position_check,
            risk_limit_check=risk_check,
            circuit_breaker_active=circuit_breaker_active,
            trading_allowed=trading_allowed
        )

    async def process_violations(
        self,
        snapshot: RiskSnapshot
    ) -> None:
        """
        Process limit violations and generate alerts.

        Args:
            snapshot: Risk snapshot with violations
        """
        # Process position limit violations
        for violation in snapshot.position_limit_check.violations:
            await self.alert_manager.create_alert(
                alert_type=violation.limit_type,
                severity=violation.severity,
                message=violation.message,
                violation=violation
            )

            # Auto-reduce if configured
            if self.config.auto_reduce_on_breach and violation.severity == "critical":
                await self._auto_reduce_positions(violation)

        # Process risk limit violations
        for violation in snapshot.risk_limit_check.violations:
            await self.alert_manager.create_alert(
                alert_type=violation.limit_type,
                severity=violation.severity,
                message=violation.message,
                violation=violation
            )

            # Auto-reduce if configured
            if self.config.auto_reduce_on_breach and violation.severity == "critical":
                await self._auto_reduce_positions(violation)

        # Process warnings
        for warning in snapshot.position_limit_check.warnings:
            await self.alert_manager.create_alert(
                alert_type=warning.limit_type,
                severity="warning",
                message=warning.message,
                violation=warning
            )

        for warning in snapshot.risk_limit_check.warnings:
            await self.alert_manager.create_alert(
                alert_type=warning.limit_type,
                severity="warning",
                message=warning.message,
                violation=warning
            )

    async def _auto_reduce_positions(
        self,
        violation: LimitViolation
    ) -> None:
        """
        Automatically reduce positions on critical violation.

        Args:
            violation: Limit violation that triggered reduction
        """
        logger.warning(f"Auto-reducing positions due to: {violation.message}")

        # Reduce all positions by configured percentage
        reduction_pct = self.config.risk_reduction_pct

        for symbol, position in self.portfolio.positions.items():
            if position.is_flat():
                continue

            # Calculate reduced quantity
            current_qty = float(position.quantity)
            reduced_qty = current_qty * (1 - reduction_pct)

            logger.info(
                f"Auto-reducing {symbol}: {current_qty} -> {reduced_qty} "
                f"({reduction_pct:.0%} reduction)"
            )

            # Note: Actual position reduction would be done through execution system
            # This is just logging the intended action

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Risk monitoring started")

        while self.monitoring_active:
            try:
                # Calculate risk snapshot
                snapshot = await self.calculate_risk_snapshot()
                self.last_snapshot = snapshot

                # Process any violations
                await self.process_violations(snapshot)

                # Record P&L for loss tracking
                total_pnl = float(self.portfolio.total_pnl())
                self.risk_limits.record_pnl(total_pnl)

                # Log status
                logger.info(
                    f"Risk Monitor: Portfolio Value: ${snapshot.portfolio_value:.2f}, "
                    f"Sharpe: {snapshot.metrics.sharpe_ratio:.2f}, "
                    f"Max DD: {snapshot.metrics.max_drawdown:.2%}, "
                    f"Trading Allowed: {snapshot.trading_allowed}"
                )

                # Clean up old alerts
                self.alert_manager.clear_old_alerts()

                # Wait for next monitoring interval
                await asyncio.sleep(self.config.risk_monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Brief pause before retry

        logger.info("Risk monitoring stopped")

    async def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self.monitoring_active:
            logger.warning("Risk monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    def get_dashboard_data(self) -> Dict:
        """
        Get risk dashboard data.

        Returns:
            Dictionary with dashboard data
        """
        if not self.last_snapshot:
            return {
                'status': 'no_data',
                'message': 'No risk snapshot available'
            }

        snapshot = self.last_snapshot

        return {
            'status': 'ok' if snapshot.trading_allowed else 'restricted',
            'portfolio_value': snapshot.portfolio_value,
            'circuit_breaker_active': snapshot.circuit_breaker_active,
            'trading_allowed': snapshot.trading_allowed,
            'metrics': {
                'volatility': snapshot.metrics.volatility,
                'sharpe_ratio': snapshot.metrics.sharpe_ratio,
                'sortino_ratio': snapshot.metrics.sortino_ratio,
                'max_drawdown': snapshot.metrics.max_drawdown,
                'current_drawdown': snapshot.metrics.current_drawdown,
                'ulcer_index': snapshot.metrics.ulcer_index
            },
            'var': {
                method: {
                    'var': result.var,
                    'cvar': result.cvar,
                    'var_percentage': result.var_percentage
                }
                for method, result in snapshot.var_results.items()
            },
            'violations': {
                'position_limit_violations': len(snapshot.position_limit_check.violations),
                'risk_limit_violations': len(snapshot.risk_limit_check.violations),
                'warnings': (
                    len(snapshot.position_limit_check.warnings) +
                    len(snapshot.risk_limit_check.warnings)
                )
            },
            'active_alerts': {
                'critical': len(self.alert_manager.get_active_alerts('critical')),
                'warning': len(self.alert_manager.get_active_alerts('warning'))
            },
            'timestamp': snapshot.timestamp.isoformat()
        }

    def get_active_alerts(self) -> List[RiskAlert]:
        """Get active risk alerts."""
        return self.alert_manager.get_active_alerts()

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a risk alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            Whether alert was acknowledged
        """
        return await self.alert_manager.acknowledge_alert(alert_id)

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual override)."""
        self.risk_limits.reset_circuit_breaker()
        logger.warning("Circuit breaker manually reset")

    def add_alert_handler(self, handler: Callable) -> None:
        """
        Add custom alert handler.

        Args:
            handler: Async function to handle alerts
        """
        self.alert_manager.add_alert_handler(handler)

    def set_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """Set sector mapping for exposure limits."""
        self.position_limits.set_sector_mapping(mapping)

    def set_industry_mapping(self, mapping: Dict[str, str]) -> None:
        """Set industry mapping for exposure limits."""
        self.position_limits.set_industry_mapping(mapping)

    def update_correlation_matrix(self, returns: pd.DataFrame) -> None:
        """Update correlation matrix for correlation limits."""
        self.position_limits.update_correlation_matrix(returns)
