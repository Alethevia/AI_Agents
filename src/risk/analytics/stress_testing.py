"""
Stress testing and scenario analysis module.

This module provides:
- Historical scenario analysis
- Hypothetical scenarios
- Factor shock analysis
- Extreme value scenarios
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from ...config.risk_config import RiskConfig
from ...core.types.portfolio import Portfolio


class StressTestResult(BaseModel):
    """Result of a stress test scenario."""

    scenario_name: str = Field(..., description="Scenario name")
    scenario_type: str = Field(..., description="Scenario type")

    # Portfolio impact
    initial_value: float = Field(..., description="Initial portfolio value")
    stressed_value: float = Field(..., description="Stressed portfolio value")
    loss_amount: float = Field(..., description="Loss amount")
    loss_percentage: float = Field(..., description="Loss percentage")

    # Position impacts
    position_impacts: Dict[str, float] = Field(
        default_factory=dict,
        description="Impact by position"
    )

    # Metrics
    max_drawdown: float = Field(..., description="Maximum drawdown in scenario")
    recovery_time: Optional[int] = Field(None, description="Recovery time in days")

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StressScenario(BaseModel):
    """Definition of a stress test scenario."""

    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    scenario_type: str = Field(..., description="Type of scenario")

    # Scenario parameters
    shocks: Dict[str, float] = Field(
        default_factory=dict,
        description="Shocks to apply (symbol -> percentage change)"
    )
    market_shock: Optional[float] = Field(
        None,
        description="General market shock percentage"
    )
    volatility_multiplier: float = Field(
        default=1.0,
        description="Volatility multiplier"
    )
    correlation_adjustment: float = Field(
        default=0.0,
        description="Correlation adjustment"
    )


class HistoricalScenarioAnalyzer:
    """Analyze historical crisis scenarios."""

    def __init__(self):
        """Initialize historical scenario analyzer."""
        # Define historical crisis periods
        self.scenarios = {
            "black_monday_1987": {
                "name": "Black Monday 1987",
                "start_date": "1987-10-19",
                "end_date": "1987-10-19",
                "market_shock": -0.22,  # -22% single day
            },
            "dot_com_crash_2000": {
                "name": "Dot-com Crash 2000-2002",
                "start_date": "2000-03-10",
                "end_date": "2002-10-09",
                "market_shock": -0.49,  # -49% total
            },
            "financial_crisis_2008": {
                "name": "Financial Crisis 2008",
                "start_date": "2007-10-09",
                "end_date": "2009-03-09",
                "market_shock": -0.57,  # -57% total
            },
            "flash_crash_2010": {
                "name": "Flash Crash 2010",
                "start_date": "2010-05-06",
                "end_date": "2010-05-06",
                "market_shock": -0.09,  # -9% intraday
            },
            "covid_crash_2020": {
                "name": "COVID-19 Crash 2020",
                "start_date": "2020-02-19",
                "end_date": "2020-03-23",
                "market_shock": -0.34,  # -34% total
            },
            "crypto_winter_2022": {
                "name": "Crypto Winter 2022",
                "start_date": "2021-11-10",
                "end_date": "2022-11-21",
                "market_shock": -0.77,  # -77% for crypto
            }
        }

    async def run_historical_scenario(
        self,
        portfolio: Portfolio,
        scenario_name: str,
        historical_returns: Optional[pd.DataFrame] = None
    ) -> StressTestResult:
        """
        Run historical scenario analysis.

        Args:
            portfolio: Portfolio to test
            scenario_name: Name of historical scenario
            historical_returns: Historical returns data (optional)

        Returns:
            StressTestResult
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        initial_value = float(portfolio.total_value())

        # Calculate stressed portfolio value
        if historical_returns is not None:
            # Use actual historical returns if available
            stressed_value = await self._apply_historical_returns(
                portfolio, historical_returns, scenario
            )
        else:
            # Use scenario shock
            market_shock = scenario["market_shock"]
            stressed_value = initial_value * (1 + market_shock)

        # Calculate position impacts
        position_impacts = {}
        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            # Assume positions move with market (simplified)
            shocked_value = position_value * (1 + scenario["market_shock"])
            impact = shocked_value - position_value
            position_impacts[symbol] = impact

        # Calculate metrics
        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name=scenario["name"],
            scenario_type="historical",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=min(0, loss_percentage)
        )

    async def _apply_historical_returns(
        self,
        portfolio: Portfolio,
        returns: pd.DataFrame,
        scenario: Dict
    ) -> float:
        """Apply historical returns to portfolio."""
        # This would require actual historical return data
        # Simplified implementation
        initial_value = float(portfolio.total_value())
        market_shock = scenario["market_shock"]
        return initial_value * (1 + market_shock)


class HypotheticalScenarioAnalyzer:
    """Analyze hypothetical scenarios."""

    async def run_market_crash(
        self,
        portfolio: Portfolio,
        crash_percentage: float = -0.20
    ) -> StressTestResult:
        """
        Simulate market crash scenario.

        Args:
            portfolio: Portfolio to test
            crash_percentage: Market crash percentage

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        # Apply crash to all positions
        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            shocked_value = position_value * (1 + crash_percentage)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        # Add cash
        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name=f"Market Crash {crash_percentage:.0%}",
            scenario_type="hypothetical",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=loss_percentage
        )

    async def run_volatility_spike(
        self,
        portfolio: Portfolio,
        volatility_multiplier: float = 3.0,
        current_volatility: float = 0.20
    ) -> StressTestResult:
        """
        Simulate volatility spike scenario.

        Args:
            portfolio: Portfolio to test
            volatility_multiplier: Volatility increase multiplier
            current_volatility: Current volatility level

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        # Increased volatility leads to potential larger moves
        # Simulate worst-case move based on increased volatility
        stressed_vol = current_volatility * volatility_multiplier

        # Use 3-sigma move as stress
        worst_case_move = -3 * stressed_vol

        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            shocked_value = position_value * (1 + worst_case_move)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name=f"Volatility Spike {volatility_multiplier}x",
            scenario_type="hypothetical",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=loss_percentage
        )

    async def run_sector_rotation(
        self,
        portfolio: Portfolio,
        sector_shocks: Dict[str, float],
        sector_map: Dict[str, str]
    ) -> StressTestResult:
        """
        Simulate sector rotation scenario.

        Args:
            portfolio: Portfolio to test
            sector_shocks: Shocks by sector
            sector_map: Symbol to sector mapping

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            sector = sector_map.get(symbol, "Unknown")
            shock = sector_shocks.get(sector, 0.0)

            shocked_value = position_value * (1 + shock)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name="Sector Rotation",
            scenario_type="hypothetical",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=min(0, loss_percentage)
        )

    async def run_liquidity_crisis(
        self,
        portfolio: Portfolio,
        liquidity_impact: float = -0.15
    ) -> StressTestResult:
        """
        Simulate liquidity crisis scenario.

        Args:
            portfolio: Portfolio to test
            liquidity_impact: Impact from reduced liquidity

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        # Liquidity crisis affects positions differently
        # Larger positions face greater impact
        total_position_value = float(portfolio.total_position_value())

        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())

            # Scale impact by position size
            position_pct = position_value / total_position_value if total_position_value > 0 else 0
            scaled_impact = liquidity_impact * (1 + position_pct)

            shocked_value = position_value * (1 + scaled_impact)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name=f"Liquidity Crisis {liquidity_impact:.0%}",
            scenario_type="hypothetical",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=loss_percentage
        )


class FactorShockAnalyzer:
    """Analyze factor shocks."""

    async def interest_rate_shock(
        self,
        portfolio: Portfolio,
        rate_change: float,
        duration_map: Dict[str, float]
    ) -> StressTestResult:
        """
        Simulate interest rate shock.

        Args:
            portfolio: Portfolio to test
            rate_change: Interest rate change in percentage points
            duration_map: Symbol to duration mapping

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            duration = duration_map.get(symbol, 0.0)

            # Price change ≈ -Duration × Rate Change
            price_change = -duration * rate_change

            shocked_value = position_value * (1 + price_change)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name=f"Interest Rate Shock {rate_change:+.2%}",
            scenario_type="factor_shock",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=min(0, loss_percentage)
        )

    async def currency_shock(
        self,
        portfolio: Portfolio,
        currency_shocks: Dict[str, float],
        currency_map: Dict[str, str]
    ) -> StressTestResult:
        """
        Simulate currency shock.

        Args:
            portfolio: Portfolio to test
            currency_shocks: Shocks by currency
            currency_map: Symbol to currency mapping

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            currency = currency_map.get(symbol, "USD")
            shock = currency_shocks.get(currency, 0.0)

            shocked_value = position_value * (1 + shock)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name="Currency Shock",
            scenario_type="factor_shock",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=min(0, loss_percentage)
        )


class ExtremeValueAnalyzer:
    """Analyze extreme value scenarios."""

    async def tail_risk_scenario(
        self,
        portfolio: Portfolio,
        returns: pd.Series,
        confidence_level: float = 0.99
    ) -> StressTestResult:
        """
        Simulate extreme tail risk scenario.

        Args:
            portfolio: Portfolio to test
            returns: Historical returns
            confidence_level: Confidence level for tail event

        Returns:
            StressTestResult
        """
        initial_value = float(portfolio.total_value())

        # Find extreme tail return
        alpha = 1 - confidence_level
        tail_return = np.percentile(returns, alpha * 100)

        # Apply to portfolio
        stressed_value = initial_value * (1 + tail_return)

        position_impacts = {}
        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            shocked_value = position_value * (1 + tail_return)
            impact = shocked_value - position_value
            position_impacts[symbol] = impact

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name=f"Tail Risk {confidence_level:.0%}",
            scenario_type="extreme_value",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=loss_percentage
        )

    async def correlation_breakdown(
        self,
        portfolio: Portfolio,
        normal_correlation: float = 0.5,
        stressed_correlation: float = 0.9
    ) -> StressTestResult:
        """
        Simulate correlation breakdown scenario.

        Args:
            portfolio: Portfolio to test
            normal_correlation: Normal correlation
            stressed_correlation: Stressed correlation

        Returns:
            StressTestResult
        """
        # In a correlation breakdown, diversification benefits disappear
        # All positions move together

        initial_value = float(portfolio.total_value())

        # Estimate diversification benefit
        n_positions = portfolio.position_count()
        if n_positions <= 1:
            diversification_factor = 1.0
        else:
            # Normal diversification effect
            normal_factor = np.sqrt(
                1 + (n_positions - 1) * normal_correlation
            ) / np.sqrt(n_positions)

            # Stressed (no diversification)
            stressed_factor = np.sqrt(
                1 + (n_positions - 1) * stressed_correlation
            ) / np.sqrt(n_positions)

            diversification_factor = stressed_factor / normal_factor

        # Assume a -2 sigma move with reduced diversification
        market_move = -0.04  # -4% move
        amplified_move = market_move * diversification_factor

        position_impacts = {}
        stressed_value = 0

        for symbol, position in portfolio.positions.items():
            if position.is_flat():
                continue

            position_value = float(position.market_value())
            shocked_value = position_value * (1 + amplified_move)
            impact = shocked_value - position_value

            position_impacts[symbol] = impact
            stressed_value += shocked_value

        stressed_value += float(portfolio.cash)

        loss_amount = stressed_value - initial_value
        loss_percentage = loss_amount / initial_value if initial_value > 0 else 0

        return StressTestResult(
            scenario_name="Correlation Breakdown",
            scenario_type="extreme_value",
            initial_value=initial_value,
            stressed_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            position_impacts=position_impacts,
            max_drawdown=loss_percentage
        )


class StressTestManager:
    """Unified stress testing manager."""

    def __init__(self, config: RiskConfig):
        """
        Initialize stress test manager.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.historical = HistoricalScenarioAnalyzer()
        self.hypothetical = HypotheticalScenarioAnalyzer()
        self.factor_shock = FactorShockAnalyzer()
        self.extreme_value = ExtremeValueAnalyzer()

    async def run_all_scenarios(
        self,
        portfolio: Portfolio,
        returns: Optional[pd.Series] = None
    ) -> Dict[str, StressTestResult]:
        """
        Run all configured stress test scenarios.

        Args:
            portfolio: Portfolio to test
            returns: Historical returns (optional)

        Returns:
            Dictionary of results by scenario name
        """
        results = {}

        # Run scenarios based on configuration
        if self.config.enable_stress_tests:
            tasks = []

            # Historical scenarios
            if "market_crash" in self.config.stress_test_scenarios:
                tasks.append(("Market Crash -20%", self.hypothetical.run_market_crash(portfolio, -0.20)))

            if "flash_crash" in self.config.stress_test_scenarios:
                tasks.append(("Flash Crash -10%", self.hypothetical.run_market_crash(portfolio, -0.10)))

            if "volatility_spike" in self.config.stress_test_scenarios:
                tasks.append(("Volatility Spike 3x", self.hypothetical.run_volatility_spike(portfolio, 3.0)))

            # Run tail risk if returns provided
            if returns is not None:
                tasks.append(("Tail Risk 99%", self.extreme_value.tail_risk_scenario(portfolio, returns, 0.99)))

            # Execute all scenarios concurrently
            if tasks:
                scenario_results = await asyncio.gather(*[task for _, task in tasks])
                results = {name: result for (name, _), result in zip(tasks, scenario_results)}

        return results

    async def check_stress_limits(
        self,
        results: Dict[str, StressTestResult]
    ) -> List[str]:
        """
        Check if stress test results violate limits.

        Args:
            results: Stress test results

        Returns:
            List of violations
        """
        violations = []

        max_stress_loss = self.config.max_stress_loss

        for scenario_name, result in results.items():
            if abs(result.loss_percentage) > max_stress_loss:
                violations.append(
                    f"Scenario '{scenario_name}' loss {abs(result.loss_percentage):.2%} "
                    f"exceeds limit {max_stress_loss:.2%}"
                )

        return violations
