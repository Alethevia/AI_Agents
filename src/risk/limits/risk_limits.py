"""
Risk limit enforcement module.

This module enforces risk limits including:
- VaR limits
- Volatility limits
- Drawdown limits
- Loss limits (daily, weekly, monthly)
- Circuit breakers
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from ...config.risk_config import RiskConfig
from ...core.types.portfolio import Portfolio
from ..models.var_models import VaRCalculator, VaRResult
from ..models.risk_metrics import RiskMetricsCalculator


class LimitViolation(BaseModel):
    """Represents a risk limit violation."""

    limit_type: str = Field(..., description="Type of limit violated")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Limit value")
    severity: str = Field(..., description="Violation severity")
    message: str = Field(..., description="Violation message")
    action_required: str = Field(..., description="Required action")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskLimitCheck(BaseModel):
    """Result of risk limit check."""

    passed: bool = Field(..., description="Whether all checks passed")
    violations: List[LimitViolation] = Field(
        default_factory=list,
        description="List of violations"
    )
    warnings: List[LimitViolation] = Field(
        default_factory=list,
        description="List of warnings"
    )
    circuit_breaker_triggered: bool = Field(
        default=False,
        description="Whether circuit breaker was triggered"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations."""
        return any(v.severity == "critical" for v in self.violations)


class VaRLimitEnforcer:
    """Enforce VaR limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize VaR limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.var_calculator = VaRCalculator(config.var_confidence_level)

    async def check_portfolio_var(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> Optional[LimitViolation]:
        """
        Check portfolio VaR against limit.

        Args:
            returns: Portfolio returns series
            portfolio_value: Current portfolio value

        Returns:
            LimitViolation if limit exceeded
        """
        # Calculate VaR using configured method
        var_result = await self.var_calculator.calculate(
            returns,
            portfolio_value,
            method=self.config.var_method
        )

        # Check against limit
        var_limit = self.config.max_portfolio_var * portfolio_value

        if var_result.var > var_limit:
            return LimitViolation(
                limit_type="portfolio_var",
                current_value=var_result.var,
                limit_value=var_limit,
                severity="critical",
                message=f"Portfolio VaR ${var_result.var:.2f} "
                        f"({var_result.var_percentage:.2%}) exceeds limit "
                        f"${var_limit:.2f} ({self.config.max_portfolio_var:.2%})",
                action_required="Reduce positions to lower VaR"
            )

        return None

    async def check_portfolio_cvar(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> Optional[LimitViolation]:
        """
        Check portfolio CVaR against limit.

        Args:
            returns: Portfolio returns series
            portfolio_value: Current portfolio value

        Returns:
            LimitViolation if limit exceeded
        """
        # Calculate CVaR
        var_result = await self.var_calculator.calculate(
            returns,
            portfolio_value,
            method=self.config.var_method
        )

        if var_result.cvar is None:
            return None

        # Check against limit
        cvar_limit = self.config.max_portfolio_cvar * portfolio_value

        if var_result.cvar > cvar_limit:
            return LimitViolation(
                limit_type="portfolio_cvar",
                current_value=var_result.cvar,
                limit_value=cvar_limit,
                severity="critical",
                message=f"Portfolio CVaR ${var_result.cvar:.2f} "
                        f"({var_result.cvar_percentage:.2%}) exceeds limit "
                        f"${cvar_limit:.2f} ({self.config.max_portfolio_cvar:.2%})",
                action_required="Reduce positions to lower CVaR"
            )

        return None


class VolatilityLimitEnforcer:
    """Enforce volatility limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize volatility limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.metrics_calculator = RiskMetricsCalculator(
            config.risk_free_rate
        )

    async def check_portfolio_volatility(
        self,
        returns: pd.Series
    ) -> Optional[LimitViolation]:
        """
        Check portfolio volatility against limit.

        Args:
            returns: Portfolio returns series

        Returns:
            LimitViolation if limit exceeded
        """
        # Calculate volatility
        vol = await self.metrics_calculator.vol_calculator.historical_volatility(
            returns
        )

        # Check against limit
        if vol > self.config.max_portfolio_volatility:
            return LimitViolation(
                limit_type="portfolio_volatility",
                current_value=vol,
                limit_value=self.config.max_portfolio_volatility,
                severity="warning",
                message=f"Portfolio volatility {vol:.2%} exceeds limit "
                        f"{self.config.max_portfolio_volatility:.2%}",
                action_required="Reduce position sizes or hedge volatility"
            )

        return None


class DrawdownLimitEnforcer:
    """Enforce drawdown limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize drawdown limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.metrics_calculator = RiskMetricsCalculator(
            config.risk_free_rate
        )

    async def check_max_drawdown(
        self,
        returns: pd.Series
    ) -> Optional[LimitViolation]:
        """
        Check maximum drawdown against limit.

        Args:
            returns: Portfolio returns series

        Returns:
            LimitViolation if limit exceeded
        """
        # Calculate max drawdown
        max_dd = await self.metrics_calculator.drawdown.max_drawdown(returns)

        # Check against limit
        if abs(max_dd) > self.config.max_drawdown:
            return LimitViolation(
                limit_type="max_drawdown",
                current_value=abs(max_dd),
                limit_value=self.config.max_drawdown,
                severity="critical",
                message=f"Maximum drawdown {abs(max_dd):.2%} exceeds limit "
                        f"{self.config.max_drawdown:.2%}",
                action_required="Close losing positions and pause trading"
            )

        return None

    async def check_current_drawdown(
        self,
        returns: pd.Series
    ) -> Optional[LimitViolation]:
        """
        Check current drawdown against limit.

        Args:
            returns: Portfolio returns series

        Returns:
            LimitViolation if approaching limit
        """
        # Calculate current drawdown
        current_dd = await self.metrics_calculator.drawdown.current_drawdown(returns)

        # Check if approaching limit (80% of max)
        warning_threshold = self.config.max_drawdown * 0.8

        if abs(current_dd) > warning_threshold:
            return LimitViolation(
                limit_type="current_drawdown",
                current_value=abs(current_dd),
                limit_value=warning_threshold,
                severity="warning",
                message=f"Current drawdown {abs(current_dd):.2%} approaching limit "
                        f"(threshold: {warning_threshold:.2%}, limit: {self.config.max_drawdown:.2%})",
                action_required="Monitor closely and prepare to reduce positions"
            )

        return None


class LossLimitEnforcer:
    """Enforce loss limits (daily, weekly, monthly)."""

    def __init__(self, config: RiskConfig):
        """
        Initialize loss limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.loss_history: Dict[str, List[Tuple[datetime, float]]] = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }

    def record_pnl(self, pnl: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record P&L for loss tracking.

        Args:
            pnl: P&L value
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.loss_history['daily'].append((timestamp, pnl))
        self.loss_history['weekly'].append((timestamp, pnl))
        self.loss_history['monthly'].append((timestamp, pnl))

    def _clean_old_records(self) -> None:
        """Remove old records beyond tracking periods."""
        now = datetime.utcnow()

        # Clean daily (keep 1 day)
        cutoff_daily = now - timedelta(days=1)
        self.loss_history['daily'] = [
            (ts, pnl) for ts, pnl in self.loss_history['daily']
            if ts > cutoff_daily
        ]

        # Clean weekly (keep 7 days)
        cutoff_weekly = now - timedelta(days=7)
        self.loss_history['weekly'] = [
            (ts, pnl) for ts, pnl in self.loss_history['weekly']
            if ts > cutoff_weekly
        ]

        # Clean monthly (keep 30 days)
        cutoff_monthly = now - timedelta(days=30)
        self.loss_history['monthly'] = [
            (ts, pnl) for ts, pnl in self.loss_history['monthly']
            if ts > cutoff_monthly
        ]

    async def check_daily_loss(
        self,
        portfolio_value: float
    ) -> Optional[LimitViolation]:
        """
        Check daily loss limit.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            LimitViolation if limit exceeded
        """
        self._clean_old_records()

        # Calculate daily P&L
        daily_pnl = sum(pnl for _, pnl in self.loss_history['daily'])
        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0

        # Check if loss (negative P&L)
        if daily_pnl < 0 and daily_loss_pct > self.config.daily_loss_limit:
            return LimitViolation(
                limit_type="daily_loss",
                current_value=daily_loss_pct,
                limit_value=self.config.daily_loss_limit,
                severity="critical",
                message=f"Daily loss {daily_loss_pct:.2%} exceeds limit "
                        f"{self.config.daily_loss_limit:.2%}",
                action_required="CIRCUIT BREAKER: Stop all trading immediately"
            )

        return None

    async def check_weekly_loss(
        self,
        portfolio_value: float
    ) -> Optional[LimitViolation]:
        """
        Check weekly loss limit.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            LimitViolation if limit exceeded
        """
        self._clean_old_records()

        # Calculate weekly P&L
        weekly_pnl = sum(pnl for _, pnl in self.loss_history['weekly'])
        weekly_loss_pct = abs(weekly_pnl) / portfolio_value if portfolio_value > 0 else 0

        # Check if loss
        if weekly_pnl < 0 and weekly_loss_pct > self.config.weekly_loss_limit:
            return LimitViolation(
                limit_type="weekly_loss",
                current_value=weekly_loss_pct,
                limit_value=self.config.weekly_loss_limit,
                severity="critical",
                message=f"Weekly loss {weekly_loss_pct:.2%} exceeds limit "
                        f"{self.config.weekly_loss_limit:.2%}",
                action_required="Reduce positions and review strategy"
            )

        return None

    async def check_monthly_loss(
        self,
        portfolio_value: float
    ) -> Optional[LimitViolation]:
        """
        Check monthly loss limit.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            LimitViolation if limit exceeded
        """
        self._clean_old_records()

        # Calculate monthly P&L
        monthly_pnl = sum(pnl for _, pnl in self.loss_history['monthly'])
        monthly_loss_pct = abs(monthly_pnl) / portfolio_value if portfolio_value > 0 else 0

        # Check if loss
        if monthly_pnl < 0 and monthly_loss_pct > self.config.monthly_loss_limit:
            return LimitViolation(
                limit_type="monthly_loss",
                current_value=monthly_loss_pct,
                limit_value=self.config.monthly_loss_limit,
                severity="critical",
                message=f"Monthly loss {monthly_loss_pct:.2%} exceeds limit "
                        f"{self.config.monthly_loss_limit:.2%}",
                action_required="Stop trading and conduct comprehensive review"
            )

        return None


class CircuitBreaker:
    """Circuit breaker logic for risk management."""

    def __init__(self, config: RiskConfig):
        """
        Initialize circuit breaker.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None

    def trigger(self, reason: str) -> None:
        """
        Trigger the circuit breaker.

        Args:
            reason: Reason for triggering
        """
        if not self.triggered:
            self.triggered = True
            self.trigger_time = datetime.utcnow()
            self.trigger_reason = reason

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = None

    def is_triggered(self) -> bool:
        """Check if circuit breaker is triggered."""
        return self.triggered

    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            'triggered': self.triggered,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'trigger_reason': self.trigger_reason
        }


class RiskLimitManager:
    """Unified risk limit manager."""

    def __init__(self, config: RiskConfig):
        """
        Initialize risk limit manager.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.var_enforcer = VaRLimitEnforcer(config)
        self.vol_enforcer = VolatilityLimitEnforcer(config)
        self.drawdown_enforcer = DrawdownLimitEnforcer(config)
        self.loss_enforcer = LossLimitEnforcer(config)
        self.circuit_breaker = CircuitBreaker(config)

    async def check_all_limits(
        self,
        portfolio: Portfolio,
        returns: pd.Series
    ) -> RiskLimitCheck:
        """
        Check all risk limits.

        Args:
            portfolio: Portfolio to check
            returns: Portfolio returns series

        Returns:
            RiskLimitCheck with all violations
        """
        portfolio_value = float(portfolio.total_value())

        # Run all checks concurrently
        (
            var_violation,
            cvar_violation,
            vol_violation,
            max_dd_violation,
            current_dd_violation,
            daily_loss_violation,
            weekly_loss_violation,
            monthly_loss_violation
        ) = await asyncio.gather(
            self.var_enforcer.check_portfolio_var(returns, portfolio_value),
            self.var_enforcer.check_portfolio_cvar(returns, portfolio_value),
            self.vol_enforcer.check_portfolio_volatility(returns),
            self.drawdown_enforcer.check_max_drawdown(returns),
            self.drawdown_enforcer.check_current_drawdown(returns),
            self.loss_enforcer.check_daily_loss(portfolio_value),
            self.loss_enforcer.check_weekly_loss(portfolio_value),
            self.loss_enforcer.check_monthly_loss(portfolio_value)
        )

        # Collect violations
        violations = []
        warnings = []

        # Add VaR violations
        if var_violation:
            violations.append(var_violation)
        if cvar_violation:
            violations.append(cvar_violation)

        # Add volatility violations (warnings)
        if vol_violation:
            warnings.append(vol_violation)

        # Add drawdown violations
        if max_dd_violation:
            violations.append(max_dd_violation)
        if current_dd_violation:
            warnings.append(current_dd_violation)

        # Add loss violations (critical - trigger circuit breaker)
        circuit_breaker_triggered = False
        if daily_loss_violation:
            violations.append(daily_loss_violation)
            circuit_breaker_triggered = True
            self.circuit_breaker.trigger(daily_loss_violation.message)

        if weekly_loss_violation:
            violations.append(weekly_loss_violation)
            if not circuit_breaker_triggered:
                circuit_breaker_triggered = True
                self.circuit_breaker.trigger(weekly_loss_violation.message)

        if monthly_loss_violation:
            violations.append(monthly_loss_violation)
            if not circuit_breaker_triggered:
                circuit_breaker_triggered = True
                self.circuit_breaker.trigger(monthly_loss_violation.message)

        # Determine if passed
        passed = len(violations) == 0

        return RiskLimitCheck(
            passed=passed,
            violations=violations,
            warnings=warnings,
            circuit_breaker_triggered=circuit_breaker_triggered
        )

    def record_pnl(self, pnl: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record P&L for loss tracking.

        Args:
            pnl: P&L value
            timestamp: Timestamp (defaults to now)
        """
        self.loss_enforcer.record_pnl(pnl, timestamp)

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker."""
        self.circuit_breaker.reset()

    def get_circuit_breaker_status(self) -> Dict:
        """Get circuit breaker status."""
        return self.circuit_breaker.get_status()
