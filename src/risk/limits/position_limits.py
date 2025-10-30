"""
Position limit enforcement module.

This module enforces position limits including:
- Position size limits (per symbol, total)
- Sector/industry exposure limits
- Correlation limits
- Leverage limits
- Concentration limits
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ...config.risk_config import RiskConfig
from ...core.types.portfolio import Portfolio
from ...core.types.positions import Position


class LimitViolation(BaseModel):
    """Represents a limit violation."""

    limit_type: str = Field(..., description="Type of limit violated")
    symbol: Optional[str] = Field(None, description="Symbol (if applicable)")
    current_value: float = Field(..., description="Current value")
    limit_value: float = Field(..., description="Limit value")
    severity: str = Field(..., description="Violation severity (warning, critical)")
    message: str = Field(..., description="Violation message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PositionLimitCheck(BaseModel):
    """Result of position limit check."""

    passed: bool = Field(..., description="Whether all checks passed")
    violations: List[LimitViolation] = Field(
        default_factory=list,
        description="List of violations"
    )
    warnings: List[LimitViolation] = Field(
        default_factory=list,
        description="List of warnings"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations."""
        return any(v.severity == "critical" for v in self.violations)

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class PositionSizeEnforcer:
    """Enforce position size limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize position size enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config

    async def check_position_size(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float
    ) -> Optional[LimitViolation]:
        """
        Check if position size is within limits.

        Args:
            symbol: Symbol to check
            position_value: Position market value
            portfolio_value: Total portfolio value

        Returns:
            LimitViolation if limit exceeded, None otherwise
        """
        if portfolio_value <= 0:
            return None

        # Calculate position size as percentage
        position_pct = abs(position_value) / portfolio_value

        # Check against limit
        if position_pct > self.config.max_position_size_pct:
            return LimitViolation(
                limit_type="position_size",
                symbol=symbol,
                current_value=position_pct,
                limit_value=self.config.max_position_size_pct,
                severity="critical",
                message=f"Position size {position_pct:.2%} exceeds limit {self.config.max_position_size_pct:.2%}"
            )

        return None

    async def check_concentration(
        self,
        portfolio: Portfolio,
        top_n: int = 5
    ) -> Optional[LimitViolation]:
        """
        Check concentration in top N positions.

        Args:
            portfolio: Portfolio to check
            top_n: Number of top positions to check

        Returns:
            LimitViolation if concentration limit exceeded
        """
        portfolio_value = float(portfolio.total_value())

        if portfolio_value <= 0:
            return None

        # Get position values
        position_values = [
            abs(float(pos.market_value()))
            for pos in portfolio.positions.values()
            if not pos.is_flat()
        ]

        if len(position_values) < top_n:
            return None

        # Sort and get top N
        top_values = sorted(position_values, reverse=True)[:top_n]
        top_concentration = sum(top_values) / portfolio_value

        # Check against limit
        if top_concentration > self.config.max_position_concentration:
            return LimitViolation(
                limit_type="concentration",
                symbol=None,
                current_value=top_concentration,
                limit_value=self.config.max_position_concentration,
                severity="warning",
                message=f"Top {top_n} positions concentration {top_concentration:.2%} "
                        f"exceeds limit {self.config.max_position_concentration:.2%}"
            )

        return None

    async def check_all_position_sizes(
        self,
        portfolio: Portfolio
    ) -> List[LimitViolation]:
        """
        Check all position sizes in portfolio.

        Args:
            portfolio: Portfolio to check

        Returns:
            List of violations
        """
        violations = []
        portfolio_value = float(portfolio.total_value())

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = abs(float(position.market_value()))
            violation = await self.check_position_size(
                symbol, position_value, portfolio_value
            )

            if violation:
                violations.append(violation)

        return violations


class ExposureLimitEnforcer:
    """Enforce sector and industry exposure limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize exposure limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.sector_map: Dict[str, str] = {}
        self.industry_map: Dict[str, str] = {}

    def set_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set symbol to sector mapping.

        Args:
            mapping: Dictionary mapping symbol to sector
        """
        self.sector_map = mapping

    def set_industry_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set symbol to industry mapping.

        Args:
            mapping: Dictionary mapping symbol to industry
        """
        self.industry_map = mapping

    async def check_sector_exposure(
        self,
        portfolio: Portfolio
    ) -> List[LimitViolation]:
        """
        Check sector exposure limits.

        Args:
            portfolio: Portfolio to check

        Returns:
            List of violations
        """
        violations = []
        portfolio_value = float(portfolio.total_value())

        if portfolio_value <= 0:
            return violations

        # Calculate sector exposures
        sector_exposures: Dict[str, float] = {}

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            sector = self.sector_map.get(symbol, "Unknown")
            position_value = abs(float(position.market_value()))

            sector_exposures[sector] = sector_exposures.get(sector, 0) + position_value

        # Check each sector against limit
        for sector, exposure in sector_exposures.items():
            exposure_pct = exposure / portfolio_value

            if exposure_pct > self.config.max_sector_exposure:
                violations.append(LimitViolation(
                    limit_type="sector_exposure",
                    symbol=sector,
                    current_value=exposure_pct,
                    limit_value=self.config.max_sector_exposure,
                    severity="critical",
                    message=f"Sector {sector} exposure {exposure_pct:.2%} "
                            f"exceeds limit {self.config.max_sector_exposure:.2%}"
                ))

        return violations

    async def check_industry_exposure(
        self,
        portfolio: Portfolio
    ) -> List[LimitViolation]:
        """
        Check industry exposure limits.

        Args:
            portfolio: Portfolio to check

        Returns:
            List of violations
        """
        violations = []
        portfolio_value = float(portfolio.total_value())

        if portfolio_value <= 0:
            return violations

        # Calculate industry exposures
        industry_exposures: Dict[str, float] = {}

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            industry = self.industry_map.get(symbol, "Unknown")
            position_value = abs(float(position.market_value()))

            industry_exposures[industry] = industry_exposures.get(industry, 0) + position_value

        # Check each industry against limit
        for industry, exposure in industry_exposures.items():
            exposure_pct = exposure / portfolio_value

            if exposure_pct > self.config.max_industry_exposure:
                violations.append(LimitViolation(
                    limit_type="industry_exposure",
                    symbol=industry,
                    current_value=exposure_pct,
                    limit_value=self.config.max_industry_exposure,
                    severity="warning",
                    message=f"Industry {industry} exposure {exposure_pct:.2%} "
                            f"exceeds limit {self.config.max_industry_exposure:.2%}"
                ))

        return violations


class CorrelationLimitEnforcer:
    """Enforce correlation limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize correlation limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def update_correlation_matrix(
        self,
        returns: pd.DataFrame
    ) -> None:
        """
        Update correlation matrix from returns.

        Args:
            returns: DataFrame of returns with symbols as columns
        """
        self.correlation_matrix = returns.corr()

    async def check_pairwise_correlation(
        self,
        portfolio: Portfolio
    ) -> List[LimitViolation]:
        """
        Check pairwise correlation limits.

        Args:
            portfolio: Portfolio to check

        Returns:
            List of violations
        """
        violations = []

        if self.correlation_matrix is None:
            return violations

        # Get active positions
        active_symbols = [
            symbol for symbol, pos in portfolio.positions.items()
            if not pos.is_flat()
        ]

        # Check all pairs
        for i, symbol1 in enumerate(active_symbols):
            for symbol2 in active_symbols[i+1:]:
                if symbol1 in self.correlation_matrix.index and \
                   symbol2 in self.correlation_matrix.columns:

                    corr = self.correlation_matrix.loc[symbol1, symbol2]

                    if abs(corr) > self.config.max_position_correlation:
                        violations.append(LimitViolation(
                            limit_type="position_correlation",
                            symbol=f"{symbol1}-{symbol2}",
                            current_value=abs(corr),
                            limit_value=self.config.max_position_correlation,
                            severity="warning",
                            message=f"Correlation between {symbol1} and {symbol2} "
                                    f"({corr:.2f}) exceeds limit "
                                    f"{self.config.max_position_correlation:.2f}"
                        ))

        return violations

    async def check_portfolio_correlation(
        self,
        portfolio: Portfolio
    ) -> Optional[LimitViolation]:
        """
        Check average portfolio correlation.

        Args:
            portfolio: Portfolio to check

        Returns:
            LimitViolation if limit exceeded
        """
        if self.correlation_matrix is None:
            return None

        # Get active positions
        active_symbols = [
            symbol for symbol, pos in portfolio.positions.items()
            if not pos.is_flat() and symbol in self.correlation_matrix.index
        ]

        if len(active_symbols) < 2:
            return None

        # Calculate average correlation
        correlations = []
        for i, symbol1 in enumerate(active_symbols):
            for symbol2 in active_symbols[i+1:]:
                corr = self.correlation_matrix.loc[symbol1, symbol2]
                correlations.append(abs(corr))

        avg_correlation = np.mean(correlations)

        if avg_correlation > self.config.max_portfolio_correlation:
            return LimitViolation(
                limit_type="portfolio_correlation",
                symbol=None,
                current_value=avg_correlation,
                limit_value=self.config.max_portfolio_correlation,
                severity="warning",
                message=f"Average portfolio correlation {avg_correlation:.2f} "
                        f"exceeds limit {self.config.max_portfolio_correlation:.2f}"
            )

        return None


class LeverageLimitEnforcer:
    """Enforce leverage limits."""

    def __init__(self, config: RiskConfig):
        """
        Initialize leverage limit enforcer.

        Args:
            config: Risk configuration
        """
        self.config = config

    async def check_gross_leverage(
        self,
        portfolio: Portfolio
    ) -> Optional[LimitViolation]:
        """
        Check gross leverage limit.

        Args:
            portfolio: Portfolio to check

        Returns:
            LimitViolation if limit exceeded
        """
        equity = float(portfolio.equity())

        if equity <= 0:
            return LimitViolation(
                limit_type="gross_leverage",
                symbol=None,
                current_value=0,
                limit_value=self.config.max_gross_leverage,
                severity="critical",
                message="Portfolio equity is zero or negative"
            )

        # Calculate gross leverage
        gross_exposure = float(portfolio.gross_exposure())
        gross_leverage = gross_exposure / equity

        if gross_leverage > self.config.max_gross_leverage:
            return LimitViolation(
                limit_type="gross_leverage",
                symbol=None,
                current_value=gross_leverage,
                limit_value=self.config.max_gross_leverage,
                severity="critical",
                message=f"Gross leverage {gross_leverage:.2f}x exceeds limit "
                        f"{self.config.max_gross_leverage:.2f}x"
            )

        return None

    async def check_net_leverage(
        self,
        portfolio: Portfolio
    ) -> Optional[LimitViolation]:
        """
        Check net leverage limit.

        Args:
            portfolio: Portfolio to check

        Returns:
            LimitViolation if limit exceeded
        """
        equity = float(portfolio.equity())

        if equity <= 0:
            return None

        # Calculate net leverage
        net_exposure = abs(float(portfolio.net_exposure()))
        net_leverage = net_exposure / equity

        if net_leverage > self.config.max_net_leverage:
            return LimitViolation(
                limit_type="net_leverage",
                symbol=None,
                current_value=net_leverage,
                limit_value=self.config.max_net_leverage,
                severity="warning",
                message=f"Net leverage {net_leverage:.2f}x exceeds limit "
                        f"{self.config.max_net_leverage:.2f}x"
            )

        return None


class PositionLimitManager:
    """Unified position limit manager."""

    def __init__(self, config: RiskConfig):
        """
        Initialize position limit manager.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.position_size = PositionSizeEnforcer(config)
        self.exposure = ExposureLimitEnforcer(config)
        self.correlation = CorrelationLimitEnforcer(config)
        self.leverage = LeverageLimitEnforcer(config)

    async def check_all_limits(
        self,
        portfolio: Portfolio
    ) -> PositionLimitCheck:
        """
        Check all position limits.

        Args:
            portfolio: Portfolio to check

        Returns:
            PositionLimitCheck with all violations
        """
        # Run all checks concurrently
        (
            size_violations,
            concentration_violation,
            sector_violations,
            industry_violations,
            pairwise_corr_violations,
            portfolio_corr_violation,
            gross_leverage_violation,
            net_leverage_violation
        ) = await asyncio.gather(
            self.position_size.check_all_position_sizes(portfolio),
            self.position_size.check_concentration(portfolio),
            self.exposure.check_sector_exposure(portfolio),
            self.exposure.check_industry_exposure(portfolio),
            self.correlation.check_pairwise_correlation(portfolio),
            self.correlation.check_portfolio_correlation(portfolio),
            self.leverage.check_gross_leverage(portfolio),
            self.leverage.check_net_leverage(portfolio)
        )

        # Collect all violations
        violations = []
        warnings = []

        # Add size violations
        violations.extend(size_violations)

        # Add concentration violation
        if concentration_violation:
            warnings.append(concentration_violation)

        # Add sector violations
        violations.extend(sector_violations)

        # Add industry violations (warnings)
        warnings.extend(industry_violations)

        # Add correlation violations (warnings)
        warnings.extend(pairwise_corr_violations)
        if portfolio_corr_violation:
            warnings.append(portfolio_corr_violation)

        # Add leverage violations
        if gross_leverage_violation:
            violations.append(gross_leverage_violation)
        if net_leverage_violation:
            warnings.append(net_leverage_violation)

        # Determine if passed
        passed = len(violations) == 0

        return PositionLimitCheck(
            passed=passed,
            violations=violations,
            warnings=warnings
        )

    async def can_open_position(
        self,
        portfolio: Portfolio,
        symbol: str,
        size: float
    ) -> Tuple[bool, List[LimitViolation]]:
        """
        Check if a new position can be opened without violating limits.

        Args:
            portfolio: Current portfolio
            symbol: Symbol to open
            size: Position size to open

        Returns:
            Tuple of (can_open, violations)
        """
        # Create hypothetical position
        portfolio_value = float(portfolio.total_value())

        # Check position size
        violations = []

        size_violation = await self.position_size.check_position_size(
            symbol, abs(size), portfolio_value
        )

        if size_violation:
            violations.append(size_violation)

        # Check if new position would cause leverage issues
        # This requires simulating the new portfolio state

        can_open = len(violations) == 0

        return can_open, violations

    def set_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """Set sector mapping."""
        self.exposure.set_sector_mapping(mapping)

    def set_industry_mapping(self, mapping: Dict[str, str]) -> None:
        """Set industry mapping."""
        self.exposure.set_industry_mapping(mapping)

    def update_correlation_matrix(self, returns: pd.DataFrame) -> None:
        """Update correlation matrix."""
        self.correlation.update_correlation_matrix(returns)
