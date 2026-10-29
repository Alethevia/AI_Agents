"""
Example usage of the risk management system.

This file demonstrates how to use the various components of the risk management system.
"""

import asyncio
from decimal import Decimal
from datetime import datetime

import numpy as np
import pandas as pd

from ..config.risk_config import RiskConfig
from ..core.types.portfolio import Portfolio
from ..core.types.positions import Position, PositionSide

from .models import VaRCalculator, RiskMetricsCalculator
from .limits import PositionLimitManager, RiskLimitManager
from .monitoring import RiskMonitor
from .analytics import StressTestManager


async def example_var_calculation():
    """Example: Calculate VaR using different methods."""
    print("=== VaR Calculation Example ===\n")

    # Generate sample returns data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
    portfolio_value = 1000000.0  # $1M portfolio

    # Initialize VaR calculator
    var_calculator = VaRCalculator(confidence_level=0.95)

    # Calculate VaR using all methods
    results = await var_calculator.calculate_all_methods(returns, portfolio_value)

    print(f"Portfolio Value: ${portfolio_value:,.2f}\n")

    for method, result in results.items():
        print(f"{method.upper()} VaR:")
        print(f"  VaR (95%): ${result.var:,.2f} ({result.var_percentage:.2%})")
        if result.cvar:
            print(f"  CVaR (95%): ${result.cvar:,.2f} ({result.cvar_percentage:.2%})")
        print()


async def example_risk_metrics():
    """Example: Calculate risk metrics."""
    print("=== Risk Metrics Example ===\n")

    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    # Initialize metrics calculator
    metrics_calc = RiskMetricsCalculator(risk_free_rate=0.045)

    # Calculate all metrics
    metrics = await metrics_calc.calculate_all_metrics(returns)

    print(f"Volatility: {metrics.volatility:.2%}")
    print(f"Downside Deviation: {metrics.downside_deviation:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Current Drawdown: {metrics.current_drawdown:.2%}")
    print(f"Ulcer Index: {metrics.ulcer_index:.4f}")
    print()


async def example_position_limits():
    """Example: Check position limits."""
    print("=== Position Limits Example ===\n")

    # Create a sample portfolio
    portfolio = Portfolio(
        portfolio_id="test_portfolio",
        name="Test Portfolio",
        cash=Decimal('500000'),
        initial_capital=Decimal('1000000')
    )

    # Add some positions
    positions = [
        Position(
            symbol="AAPL",
            quantity=Decimal('1000'),
            entry_price=Decimal('150.0'),
            current_price=Decimal('155.0'),
            side=PositionSide.LONG
        ),
        Position(
            symbol="GOOGL",
            quantity=Decimal('500'),
            entry_price=Decimal('120.0'),
            current_price=Decimal('125.0'),
            side=PositionSide.LONG
        ),
        Position(
            symbol="MSFT",
            quantity=Decimal('800'),
            entry_price=Decimal('300.0'),
            current_price=Decimal('310.0'),
            side=PositionSide.LONG
        )
    ]

    for position in positions:
        portfolio.add_position(position)

    # Initialize limit manager
    config = RiskConfig()
    limit_manager = PositionLimitManager(config)

    # Check all limits
    check_result = await limit_manager.check_all_limits(portfolio)

    print(f"Limit Check Passed: {check_result.passed}")
    print(f"Violations: {len(check_result.violations)}")
    print(f"Warnings: {len(check_result.warnings)}")
    print()

    if check_result.violations:
        print("Violations:")
        for violation in check_result.violations:
            print(f"  - {violation.message}")
        print()

    if check_result.warnings:
        print("Warnings:")
        for warning in check_result.warnings:
            print(f"  - {warning.message}")
        print()


async def example_risk_limits():
    """Example: Check risk limits."""
    print("=== Risk Limits Example ===\n")

    # Create a sample portfolio
    portfolio = Portfolio(
        portfolio_id="test_portfolio",
        name="Test Portfolio",
        cash=Decimal('500000'),
        initial_capital=Decimal('1000000')
    )

    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    # Initialize risk limit manager
    config = RiskConfig()
    risk_manager = RiskLimitManager(config)

    # Record some P&L
    risk_manager.record_pnl(5000.0)  # Profit
    risk_manager.record_pnl(-2000.0)  # Loss

    # Check all limits
    check_result = await risk_manager.check_all_limits(portfolio, returns)

    print(f"Risk Check Passed: {check_result.passed}")
    print(f"Circuit Breaker: {check_result.circuit_breaker_triggered}")
    print(f"Violations: {len(check_result.violations)}")
    print()

    if check_result.violations:
        print("Risk Violations:")
        for violation in check_result.violations:
            print(f"  - {violation.message}")
            print(f"    Action Required: {violation.action_required}")
        print()


async def example_risk_monitoring():
    """Example: Real-time risk monitoring."""
    print("=== Risk Monitoring Example ===\n")

    # Create a sample portfolio
    portfolio = Portfolio(
        portfolio_id="test_portfolio",
        name="Test Portfolio",
        cash=Decimal('500000'),
        initial_capital=Decimal('1000000')
    )

    # Add a position
    position = Position(
        symbol="AAPL",
        quantity=Decimal('1000'),
        entry_price=Decimal('150.0'),
        current_price=Decimal('155.0'),
        side=PositionSide.LONG
    )
    portfolio.add_position(position)

    # Initialize risk monitor
    config = RiskConfig()
    config.risk_monitoring_interval_seconds = 5  # Check every 5 seconds
    monitor = RiskMonitor(config, portfolio)

    # Generate and update returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    monitor.update_returns(returns)

    # Calculate a snapshot (without starting continuous monitoring)
    snapshot = await monitor.calculate_risk_snapshot()

    print(f"Portfolio Value: ${snapshot.portfolio_value:,.2f}")
    print(f"Trading Allowed: {snapshot.trading_allowed}")
    print(f"Circuit Breaker Active: {snapshot.circuit_breaker_active}")
    print()

    print("Risk Metrics:")
    print(f"  Sharpe Ratio: {snapshot.metrics.sharpe_ratio:.2f}")
    print(f"  Volatility: {snapshot.metrics.volatility:.2%}")
    print(f"  Max Drawdown: {snapshot.metrics.max_drawdown:.2%}")
    print()

    print("VaR Results:")
    for method, result in snapshot.var_results.items():
        print(f"  {method}: ${result.var:,.2f} ({result.var_percentage:.2%})")
    print()

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print("Dashboard Status:", dashboard['status'])
    print()


async def example_stress_testing():
    """Example: Stress testing."""
    print("=== Stress Testing Example ===\n")

    # Create a sample portfolio
    portfolio = Portfolio(
        portfolio_id="test_portfolio",
        name="Test Portfolio",
        cash=Decimal('500000'),
        initial_capital=Decimal('1000000')
    )

    # Add positions
    positions = [
        Position(
            symbol="AAPL",
            quantity=Decimal('1000'),
            entry_price=Decimal('150.0'),
            current_price=Decimal('155.0'),
            side=PositionSide.LONG
        ),
        Position(
            symbol="GOOGL",
            quantity=Decimal('500'),
            entry_price=Decimal('120.0'),
            current_price=Decimal('125.0'),
            side=PositionSide.LONG
        )
    ]

    for position in positions:
        portfolio.add_position(position)

    # Initialize stress test manager
    config = RiskConfig()
    stress_manager = StressTestManager(config)

    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    # Run market crash scenario
    crash_result = await stress_manager.hypothetical.run_market_crash(
        portfolio, crash_percentage=-0.20
    )

    print("Market Crash Scenario (-20%):")
    print(f"  Initial Value: ${crash_result.initial_value:,.2f}")
    print(f"  Stressed Value: ${crash_result.stressed_value:,.2f}")
    print(f"  Loss: ${crash_result.loss_amount:,.2f} ({crash_result.loss_percentage:.2%})")
    print()

    # Run volatility spike scenario
    vol_result = await stress_manager.hypothetical.run_volatility_spike(
        portfolio, volatility_multiplier=3.0
    )

    print("Volatility Spike Scenario (3x):")
    print(f"  Initial Value: ${vol_result.initial_value:,.2f}")
    print(f"  Stressed Value: ${vol_result.stressed_value:,.2f}")
    print(f"  Loss: ${vol_result.loss_amount:,.2f} ({vol_result.loss_percentage:.2%})")
    print()

    # Run all scenarios
    all_results = await stress_manager.run_all_scenarios(portfolio, returns)

    print(f"Total Scenarios Run: {len(all_results)}")
    for name, result in all_results.items():
        print(f"  {name}: {result.loss_percentage:.2%} loss")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Risk Management System - Example Usage")
    print("=" * 60)
    print()

    await example_var_calculation()
    await example_risk_metrics()
    await example_position_limits()
    await example_risk_limits()
    await example_risk_monitoring()
    await example_stress_testing()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
