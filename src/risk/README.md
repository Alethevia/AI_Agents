# Risk Management System

A comprehensive, production-ready risk management system for algorithmic trading with real-time monitoring, multi-method VaR calculations, limit enforcement, and stress testing.

## Features

### 1. Value at Risk (VaR) Models (`models/var_models.py`)

Multiple VaR calculation methods for comprehensive risk assessment:

- **Historical VaR**: Non-parametric VaR using empirical distribution
- **Parametric VaR**: Variance-covariance method assuming normal distribution
- **Monte Carlo VaR**: Simulation-based VaR with configurable scenarios
- **CVaR (Conditional VaR)**: Expected Shortfall for tail risk
- **Cornish-Fisher VaR**: Accounts for skewness and kurtosis

**Key Features:**
- Async operation for all calculations
- Multiple time horizons support
- Portfolio-level and position-level VaR
- Confidence level configuration
- Aggregated VaR across methods

### 2. Risk Metrics (`models/risk_metrics.py`)

Comprehensive risk metrics calculation:

**Volatility Metrics:**
- Historical volatility
- EWMA (Exponentially Weighted Moving Average)
- GARCH(1,1) volatility forecasting
- Realized volatility from high-frequency data
- Downside deviation

**Performance Metrics:**
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Information ratio

**Drawdown Metrics:**
- Maximum drawdown
- Current drawdown
- Drawdown duration
- Ulcer index

**Correlation Metrics:**
- Portfolio beta
- Correlation with benchmark
- Rolling beta

### 3. Position Limits (`limits/position_limits.py`)

Enforce position size and exposure limits:

- **Position Size Limits**: Per-symbol and total portfolio
- **Concentration Limits**: Top N positions concentration
- **Sector/Industry Exposure**: Configurable sector and industry limits
- **Correlation Limits**: Pairwise and portfolio-wide correlation
- **Leverage Limits**: Gross and net leverage constraints

**Features:**
- Real-time limit checking
- Violation severity levels (warning/critical)
- Pre-trade limit checks
- Automatic sector/industry classification

### 4. Risk Limits (`limits/risk_limits.py`)

Enforce risk-based limits:

- **VaR Limits**: Portfolio VaR and CVaR thresholds
- **Volatility Limits**: Portfolio volatility constraints
- **Drawdown Limits**: Maximum and current drawdown limits
- **Loss Limits**: Daily, weekly, and monthly loss limits
- **Circuit Breakers**: Automatic trading halt on critical violations

**Features:**
- Time-based loss tracking
- Automatic circuit breaker triggering
- Configurable severity levels
- Action recommendations

### 5. Real-time Risk Monitoring (`monitoring/risk_monitor.py`)

Continuous risk monitoring and alerting:

- **Continuous Calculation**: Real-time risk metrics updates
- **Limit Breach Detection**: Automatic violation detection
- **Alert Generation**: Multi-level alert system
- **Auto-reduction**: Automatic position reduction on breach
- **Risk Dashboard**: Real-time risk visualization data

**Features:**
- Async monitoring loop
- Configurable monitoring intervals
- Alert handlers and callbacks
- Historical alert tracking
- Circuit breaker management

### 6. Stress Testing (`analytics/stress_testing.py`)

Comprehensive stress testing and scenario analysis:

**Historical Scenarios:**
- Black Monday 1987
- Dot-com Crash 2000-2002
- Financial Crisis 2008
- Flash Crash 2010
- COVID-19 Crash 2020
- Crypto Winter 2022

**Hypothetical Scenarios:**
- Market crash (configurable magnitude)
- Volatility spike
- Sector rotation
- Liquidity crisis

**Factor Shocks:**
- Interest rate shocks
- Currency shocks
- Commodity price shocks

**Extreme Value Analysis:**
- Tail risk scenarios
- Correlation breakdown
- Black swan events

## Installation

The risk management system requires the following dependencies:

```bash
pip install numpy pandas scipy arch pydantic
```

## Quick Start

### Basic VaR Calculation

```python
import asyncio
import pandas as pd
from src.risk import VaRCalculator

async def calculate_var():
    # Historical returns
    returns = pd.Series([...])  # Your returns data
    portfolio_value = 1000000.0

    # Initialize calculator
    var_calc = VaRCalculator(confidence_level=0.95)

    # Calculate VaR
    result = await var_calc.calculate(
        returns,
        portfolio_value,
        method="historical"
    )

    print(f"VaR (95%): ${result.var:,.2f}")
    print(f"CVaR (95%): ${result.cvar:,.2f}")

asyncio.run(calculate_var())
```

### Risk Metrics Calculation

```python
from src.risk import RiskMetricsCalculator

async def calculate_metrics():
    metrics_calc = RiskMetricsCalculator(risk_free_rate=0.045)

    metrics = await metrics_calc.calculate_all_metrics(returns)

    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Volatility: {metrics.volatility:.2%}")
```

### Position Limit Checking

```python
from src.risk import PositionLimitManager
from src.config.risk_config import RiskConfig

async def check_limits():
    config = RiskConfig()
    limit_manager = PositionLimitManager(config)

    # Check all limits
    result = await limit_manager.check_all_limits(portfolio)

    if not result.passed:
        for violation in result.violations:
            print(f"Violation: {violation.message}")
```

### Real-time Risk Monitoring

```python
from src.risk import RiskMonitor

async def start_monitoring():
    config = RiskConfig()
    monitor = RiskMonitor(config, portfolio)

    # Update returns history
    monitor.update_returns(returns)

    # Add alert handler
    async def alert_handler(alert):
        print(f"ALERT: {alert.message}")

    monitor.add_alert_handler(alert_handler)

    # Start monitoring
    await monitor.start_monitoring()

    # Let it run...
    await asyncio.sleep(3600)

    # Stop monitoring
    await monitor.stop_monitoring()
```

### Stress Testing

```python
from src.risk import StressTestManager

async def run_stress_tests():
    config = RiskConfig()
    stress_manager = StressTestManager(config)

    # Run market crash scenario
    result = await stress_manager.hypothetical.run_market_crash(
        portfolio,
        crash_percentage=-0.20
    )

    print(f"Loss in crash: {result.loss_percentage:.2%}")

    # Run all scenarios
    all_results = await stress_manager.run_all_scenarios(portfolio, returns)

    # Check against limits
    violations = await stress_manager.check_stress_limits(all_results)
```

## Configuration

The system uses `RiskConfig` from `src/config/risk_config.py`. Key parameters:

```python
config = RiskConfig(
    # VaR settings
    var_confidence_level=0.95,
    max_portfolio_var=0.02,
    max_portfolio_cvar=0.03,

    # Position limits
    max_position_size_pct=0.1,
    max_sector_exposure=0.3,
    max_gross_leverage=2.0,

    # Risk limits
    max_drawdown=0.15,
    daily_loss_limit=0.03,

    # Monitoring
    risk_monitoring_interval_seconds=60,
    alert_on_limit_breach=True,
    auto_reduce_on_breach=True,
    risk_reduction_pct=0.5,

    # Circuit breakers
    enable_circuit_breakers=True
)
```

## Architecture

```
src/risk/
├── __init__.py                      # Main exports
├── models/
│   ├── __init__.py
│   ├── var_models.py               # VaR calculations
│   └── risk_metrics.py             # Risk metrics
├── limits/
│   ├── __init__.py
│   ├── position_limits.py          # Position limit enforcement
│   └── risk_limits.py              # Risk limit enforcement
├── monitoring/
│   ├── __init__.py
│   └── risk_monitor.py             # Real-time monitoring
├── analytics/
│   ├── __init__.py
│   └── stress_testing.py           # Stress testing
├── example_usage.py                 # Usage examples
└── README.md                        # This file
```

## Production Considerations

### Performance
- All calculations are async-first
- Concurrent execution of independent checks
- Efficient numpy/pandas operations
- Configurable monitoring intervals

### Reliability
- Comprehensive error handling
- Circuit breaker for critical violations
- Automatic position reduction
- Alert acknowledgment system

### Observability
- Detailed logging throughout
- Real-time dashboard data
- Historical alert tracking
- Violation audit trail

### Scalability
- Modular design
- Independent limit checkers
- Pluggable alert handlers
- Extensible scenario framework

## Alert System

The monitoring system includes a comprehensive alerting framework:

```python
# Add custom alert handler
async def send_email_alert(alert):
    if alert.severity == "critical":
        # Send email notification
        await send_email(alert.message)

monitor.add_alert_handler(send_email_alert)
```

Alert types:
- Position limit violations
- Risk limit violations
- Circuit breaker triggers
- Drawdown warnings
- Volatility spikes

## Circuit Breakers

Circuit breakers automatically halt trading on critical violations:

- Daily loss limit exceeded
- Weekly loss limit exceeded
- Monthly loss limit exceeded
- Maximum drawdown reached

Manual reset required:
```python
monitor.reset_circuit_breaker()
```

## Best Practices

1. **Regular Calibration**: Update VaR models with recent data
2. **Multiple Methods**: Use multiple VaR methods for robust estimates
3. **Conservative Limits**: Set limits with adequate safety margins
4. **Stress Testing**: Run stress tests regularly
5. **Alert Response**: Document and follow alert response procedures
6. **Monitoring**: Monitor the monitoring system itself
7. **Audit Trail**: Maintain logs of all violations and actions
8. **Backtesting**: Backtest risk models on historical data

## Testing

Run the example usage file to test all components:

```bash
python -m src.risk.example_usage
```

## Integration with Trading System

The risk management system integrates with:

- Portfolio management (`src/portfolio`)
- Execution system (`src/execution`)
- Strategy modules (`src/strategy`)
- Configuration (`src/config`)

Example integration:

```python
from src.risk import RiskMonitor
from src.portfolio import PortfolioManager

class TradingSystem:
    def __init__(self):
        self.portfolio = PortfolioManager()
        self.risk_monitor = RiskMonitor(config, self.portfolio.portfolio)

    async def execute_trade(self, signal):
        # Check if trading is allowed
        if not self.risk_monitor.last_snapshot.trading_allowed:
            logger.warning("Trading not allowed due to risk limits")
            return

        # Check position limits before trade
        can_trade, violations = await self.risk_monitor.position_limits.can_open_position(
            self.portfolio.portfolio,
            signal.symbol,
            signal.size
        )

        if not can_trade:
            logger.warning(f"Cannot execute trade: {violations}")
            return

        # Execute trade
        await self.execute(signal)
```

## License

This risk management system is part of the AI_Agents trading framework.

## Support

For issues or questions, refer to the main project documentation or create an issue in the repository.
