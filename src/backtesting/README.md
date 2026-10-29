# Comprehensive Backtesting Framework

A production-ready, event-driven backtesting framework for quantitative trading strategies.

## Features

### Core Engine
- **Event-Driven Architecture**: Realistic simulation with proper event ordering
- **Historical Data Replay**: Time-series accurate backtesting
- **Realistic Fill Simulation**: Multiple execution models (market, close, realistic)
- **Commission & Slippage Models**: Fixed, percentage, volume-based, and spread-based
- **Multi-Strategy Support**: Test multiple strategies simultaneously
- **Walk-Forward Analysis**: Out-of-sample testing framework
- **Concurrent Backtesting**: Parallel execution for parameter optimization

### Analytics
- **Comprehensive Performance Metrics**:
  - Returns: Total, CAGR, daily, monthly, yearly
  - Risk: Volatility, VaR, CVaR, max drawdown
  - Risk-Adjusted: Sharpe, Sortino, Calmar, Omega ratios
  - Trade Statistics: Win rate, profit factor, expectancy

- **Trade-Level Analysis**:
  - Individual trade statistics
  - Duration analysis
  - Entry/exit analysis
  - Slippage and commission impact
  - Hourly/daily patterns

- **Advanced Analytics**:
  - Drawdown periods analysis
  - Rolling metrics
  - Monte Carlo simulations
  - Benchmark comparisons

### Visualization
- Equity curves with benchmark comparison
- Drawdown charts
- Returns distributions
- Monthly returns heatmaps
- Trade analysis plots
- Comprehensive metrics dashboards
- Rolling performance metrics

## Installation

The backtesting framework is part of the main project. No additional installation required.

```python
from src.backtesting import BacktestEngine, BacktestConfig
```

## Quick Start

### 1. Basic Backtest

```python
import pandas as pd
from src.backtesting import BacktestEngine, BacktestConfig

# Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_bps=5.0
)

# Create engine
engine = BacktestEngine(config)

# Load data
data = pd.read_csv('AAPL.csv', index_col='date', parse_dates=True)
engine.load_data('AAPL', data)

# Define strategy
def my_strategy(engine, symbol, data):
    # Your strategy logic
    if should_buy:
        return {'type': 'BUY', 'strength': 1.0}
    elif should_sell:
        return {'type': 'SELL', 'strength': 1.0}
    return None

# Run backtest
engine.set_strategy(my_strategy)
results = engine.run()

# View results
print(engine.analyzer.get_summary_report(results))
```

### 2. Advanced Configuration

```python
from src.backtesting import BacktestConfig, ExecutionModel

config = BacktestConfig(
    # Capital
    initial_capital=100000.0,

    # Execution
    execution_model=ExecutionModel.REALISTIC,
    fill_delay_bars=1,

    # Costs
    commission_model="percentage",  # or "fixed", "tiered"
    commission_rate=0.001,          # 0.1%
    slippage_model="volume",        # or "fixed", "spread"
    slippage_bps=5.0,

    # Risk Management
    max_position_size=0.2,          # Max 20% per position
    max_leverage=1.0,               # No leverage
    stop_loss_pct=0.05,            # 5% stop loss
    take_profit_pct=0.10,          # 10% take profit

    # Analysis
    risk_free_rate=0.02,           # 2% annual
)
```

### 3. Comprehensive Analysis

```python
from src.backtesting import BacktestAnalyzer

# Run backtest
results = engine.run()

# Get detailed metrics
perf = results.performance_metrics
print(f"Total Return: {perf.total_return:.2%}")
print(f"Sharpe Ratio: {perf.sharpe_ratio:.2f}")
print(f"Max Drawdown: {perf.max_drawdown:.2%}")

# Analyze trades
trades = results.trade_analytics
print(f"Total Trades: {trades.total_trades}")
print(f"Win Rate: {trades.win_rate:.2%}")
print(f"Profit Factor: {trades.profit_factor:.2f}")

# Analyze drawdowns
analyzer = BacktestAnalyzer(initial_capital=100000)
# ... add equity points and trades ...
results = analyzer.analyze()
drawdowns = analyzer.analyze_drawdowns(min_duration_days=5)

for dd in drawdowns[:5]:
    print(f"Drawdown: {dd.drawdown_pct:.2%}, Duration: {dd.duration_days} days")

# Monte Carlo simulation
mc_results = analyzer.monte_carlo_simulation(n_simulations=1000)
print(f"Expected Return (95% CI): {mc_results['percentile_5']:.2%} to {mc_results['percentile_95']:.2%}")
```

### 4. Visualization

```python
from src.backtesting import BacktestPlotter

# Create plotter
plotter = BacktestPlotter(results)

# Generate all plots
plotter.plot_all(output_dir='./output', prefix='my_strategy')

# Or generate individual plots
plotter.plot_equity_curve(save_path='equity.png')
plotter.plot_drawdown(save_path='drawdown.png')
plotter.plot_returns_distribution(save_path='returns.png')
plotter.plot_monthly_returns_heatmap(save_path='monthly.png')
plotter.plot_trade_analysis(save_path='trades.png')
plotter.plot_metrics_dashboard(save_path='dashboard.png')
```

### 5. Strategy Comparison

```python
from src.backtesting import compare_strategies_plot

# Run multiple strategies
results_dict = {
    'MA Crossover': run_backtest(ma_strategy),
    'RSI Strategy': run_backtest(rsi_strategy),
    'Mean Reversion': run_backtest(mean_reversion_strategy),
}

# Compare
compare_strategies_plot(
    results_dict,
    save_path='strategy_comparison.png'
)
```

### 6. Parallel Backtesting

```python
from src.backtesting import run_backtest_parallel

# Define parameter combinations
configs = []
for fast_window in [10, 20, 30]:
    for slow_window in [50, 100, 200]:
        config = BacktestConfig(initial_capital=100000)
        strategy = lambda e, s, d: ma_strategy(e, s, d, fast_window, slow_window)
        configs.append((config, {'AAPL': data}, strategy))

# Run in parallel
results_list = run_backtest_parallel(configs, max_workers=4)

# Find best parameters
best_result = max(results_list, key=lambda r: r.performance_metrics.sharpe_ratio)
```

## Strategy Development

### Strategy Function Signature

```python
def my_strategy(engine, symbol, data):
    """
    Strategy function called on each new market data event

    Args:
        engine: BacktestEngine instance (access portfolio state)
        symbol: Trading symbol
        data: Historical DataFrame up to current point

    Returns:
        Signal dictionary or None:
        {
            'type': 'BUY' or 'SELL',
            'strength': float (0-1),  # Optional: position sizing
            'target_price': Decimal,   # Optional
            'stop_loss': Decimal,      # Optional
            'take_profit': Decimal,    # Optional
            'rationale': str,          # Optional: why this signal
            'strategy_id': str         # Optional: strategy identifier
        }
    """
    # Your logic here
    return signal_dict or None
```

### Example Strategies

#### Moving Average Crossover

```python
def ma_crossover(engine, symbol, data, fast=20, slow=50):
    if len(data) < slow:
        return None

    fast_ma = data['close'].rolling(fast).mean()
    slow_ma = data['close'].rolling(slow).mean()

    current_fast = fast_ma.iloc[-1]
    current_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2]
    prev_slow = slow_ma.iloc[-2]

    # Buy on golden cross
    if prev_fast <= prev_slow and current_fast > current_slow:
        return {'type': 'BUY', 'strength': 1.0}

    # Sell on death cross
    elif prev_fast >= prev_slow and current_fast < current_slow:
        return {'type': 'SELL', 'strength': 1.0}

    return None
```

#### RSI Strategy

```python
def rsi_strategy(engine, symbol, data, period=14, oversold=30, overbought=70):
    if len(data) < period + 1:
        return None

    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]

    # Buy when oversold
    if current_rsi < oversold:
        return {'type': 'BUY', 'strength': 1.0}

    # Sell when overbought
    elif current_rsi > overbought:
        return {'type': 'SELL', 'strength': 1.0}

    return None
```

#### Mean Reversion

```python
def mean_reversion(engine, symbol, data, window=20, std_dev=2):
    if len(data) < window:
        return None

    # Bollinger Bands
    sma = data['close'].rolling(window).mean()
    std = data['close'].rolling(window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    current_price = data['close'].iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]

    # Buy when price touches lower band
    if current_price <= current_lower:
        return {'type': 'BUY', 'strength': 1.0, 'target_price': sma.iloc[-1]}

    # Sell when price touches upper band
    elif current_price >= current_upper:
        return {'type': 'SELL', 'strength': 1.0, 'target_price': sma.iloc[-1]}

    return None
```

## Performance Metrics Reference

### Returns Metrics
- **Total Return**: Overall return from start to end
- **CAGR**: Compound Annual Growth Rate
- **Daily/Monthly Returns**: Average and standard deviation

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Downside Volatility**: Standard deviation of negative returns only
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional Value at Risk (expected shortfall)
- **Max Drawdown**: Largest peak-to-trough decline
- **Max DD Duration**: Longest drawdown period in days

### Risk-Adjusted Returns
- **Sharpe Ratio**: Excess return per unit of total risk
- **Sortino Ratio**: Excess return per unit of downside risk
- **Calmar Ratio**: CAGR / Max Drawdown
- **Omega Ratio**: Probability-weighted gains vs losses

### Trade Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Expectancy**: Average expected profit per trade
- **Average Win/Loss**: Mean profit/loss per winning/losing trade
- **Win/Loss Ratio**: Average win / Average loss

## Architecture

```
src/backtesting/
├── __init__.py                 # Main exports
│
├── engine/
│   ├── __init__.py
│   ├── event_queue.py          # Event system (queue, dispatcher, event types)
│   └── backtest_engine.py      # Core backtesting engine
│
├── analytics/
│   ├── __init__.py
│   ├── trade_analyzer.py       # Trade-level analysis
│   └── backtest_analyzer.py    # Performance analysis
│
├── metrics/
│   ├── __init__.py
│   └── performance_metrics.py  # Metrics calculations
│
└── visualization/
    ├── __init__.py
    └── plotter.py              # Plotting and visualization
```

## Event Flow

```
1. Market Data Event
   ↓
2. Update Positions & Portfolio
   ↓
3. Call Strategy Function
   ↓
4. Generate Signal Event (if applicable)
   ↓
5. Create Order Event
   ↓
6. Add to Pending Orders
   ↓
7. Check Fill Conditions
   ↓
8. Generate Fill Event
   ↓
9. Update Portfolio & Record Trade
   ↓
10. Update Analytics
```

## Best Practices

### 1. Data Preparation
- Ensure data is properly indexed with timestamps
- Remove or handle missing values
- Verify data quality (no look-ahead bias)
- Include all necessary columns (OHLCV minimum)

### 2. Strategy Development
- Start simple, iterate to complexity
- Avoid look-ahead bias (only use data up to current bar)
- Test on out-of-sample data
- Consider transaction costs from the start

### 3. Backtesting
- Use realistic execution models
- Include commission and slippage
- Test across different market conditions
- Verify results manually for small samples

### 4. Analysis
- Focus on risk-adjusted returns, not just returns
- Analyze drawdown periods in detail
- Check trade distribution and patterns
- Compare against relevant benchmarks

### 5. Validation
- Walk-forward analysis for robustness
- Monte Carlo simulation for confidence intervals
- Test multiple parameter sets
- Check for overfitting

## Common Pitfalls

1. **Look-Ahead Bias**: Using future information in decisions
2. **Survivorship Bias**: Testing only on currently active securities
3. **Overfitting**: Optimizing too heavily on historical data
4. **Ignoring Costs**: Not accounting for realistic transaction costs
5. **Data Quality**: Poor or inconsistent data
6. **Unrealistic Execution**: Assuming perfect fills at desired prices

## Example Output

See `/examples/backtest_example.py` for a complete working example.

## Performance

The framework is optimized for:
- **Memory Efficiency**: Streaming event processing
- **Speed**: Vectorized calculations where possible
- **Scalability**: Parallel backtesting support
- **Production-Ready**: Comprehensive error handling and logging

## Contributing

When adding new features:
1. Follow the existing event-driven architecture
2. Add comprehensive docstrings
3. Include type hints
4. Add tests for new functionality
5. Update this README

## License

Part of the AI_Agents trading system project.

## Support

For issues or questions, refer to the main project documentation.
