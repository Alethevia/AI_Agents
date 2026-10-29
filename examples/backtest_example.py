"""
Example: Using the Backtesting Framework

This example demonstrates how to use the comprehensive backtesting framework
to test a simple moving average crossover strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import backtesting components
from src.backtesting import (
    BacktestEngine,
    BacktestConfig,
    ExecutionModel,
    BacktestPlotter,
)


def generate_sample_data(symbol: str, days: int = 252 * 3) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing

    Args:
        symbol: Trading symbol
        days: Number of days of data

    Returns:
        DataFrame with OHLCV data
    """
    # Generate dates
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(end=end_date, periods=days, freq='D')

    # Generate synthetic price data (random walk with trend)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'close': prices,
        'volume': np.random.uniform(1e6, 5e6, days)
    }, index=dates)

    return data


def simple_ma_crossover_strategy(engine, symbol, data):
    """
    Simple moving average crossover strategy

    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA

    Args:
        engine: BacktestEngine instance
        symbol: Trading symbol
        data: Historical price data

    Returns:
        Signal dictionary or None
    """
    # Need at least 50 bars for calculation
    if len(data) < 50:
        return None

    # Calculate moving averages
    fast_window = 20
    slow_window = 50

    fast_ma = data['close'].rolling(window=fast_window).mean()
    slow_ma = data['close'].rolling(window=slow_window).mean()

    # Get current and previous values
    current_fast = fast_ma.iloc[-1]
    current_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2]
    prev_slow = slow_ma.iloc[-2]

    # Get current position
    portfolio_state = engine.get_portfolio_state()
    current_position = portfolio_state['positions'].get(symbol, {})
    has_position = current_position.get('quantity', 0) > 0

    # Buy signal: fast crosses above slow
    if prev_fast <= prev_slow and current_fast > current_slow:
        if not has_position:
            return {
                'type': 'BUY',
                'strength': 1.0,
                'rationale': f'Fast MA ({current_fast:.2f}) crossed above Slow MA ({current_slow:.2f})'
            }

    # Sell signal: fast crosses below slow
    elif prev_fast >= prev_slow and current_fast < current_slow:
        if has_position:
            return {
                'type': 'SELL',
                'strength': 1.0,
                'rationale': f'Fast MA ({current_fast:.2f}) crossed below Slow MA ({current_slow:.2f})'
            }

    return None


def run_example_backtest():
    """Run an example backtest"""

    print("=" * 80)
    print("BACKTESTING FRAMEWORK EXAMPLE")
    print("=" * 80)
    print()

    # 1. Generate sample data
    print("1. Generating sample market data...")
    symbol = "AAPL"
    data = generate_sample_data(symbol, days=252 * 3)  # 3 years
    print(f"   Generated {len(data)} days of data for {symbol}")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print()

    # 2. Configure backtest
    print("2. Configuring backtest...")
    config = BacktestConfig(
        initial_capital=100000.0,
        execution_model=ExecutionModel.REALISTIC,
        commission_model="percentage",
        commission_rate=0.001,  # 0.1% per trade
        slippage_model="fixed",
        slippage_bps=5.0,  # 5 basis points
        max_position_size=0.2,  # Max 20% per position
        risk_free_rate=0.02
    )
    print(f"   Initial Capital: ${config.initial_capital:,.0f}")
    print(f"   Commission: {config.commission_rate:.2%}")
    print(f"   Slippage: {config.slippage_bps} bps")
    print()

    # 3. Create engine and load data
    print("3. Initializing backtest engine...")
    engine = BacktestEngine(config)
    engine.load_data(symbol, data)
    print(f"   Loaded data for {symbol}")
    print()

    # 4. Set strategy
    print("4. Setting trading strategy...")
    engine.set_strategy(simple_ma_crossover_strategy)
    print("   Strategy: Simple Moving Average Crossover (20/50)")
    print()

    # 5. Run backtest
    print("5. Running backtest...")
    results = engine.run()
    print("   Backtest completed!")
    print()

    # 6. Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Summary report
    print(engine.analyzer.get_summary_report(results))

    # Key metrics
    perf = results.performance_metrics
    trades = results.trade_analytics

    print()
    print("=" * 80)
    print("KEY HIGHLIGHTS")
    print("=" * 80)
    print()
    print(f"Total Return:        {perf.total_return:>12.2%}")
    print(f"CAGR:                {perf.cagr:>12.2%}")
    print(f"Sharpe Ratio:        {perf.sharpe_ratio:>12.2f}")
    print(f"Max Drawdown:        {perf.max_drawdown:>12.2%}")
    print()
    print(f"Total Trades:        {trades.total_trades:>12}")
    print(f"Win Rate:            {trades.win_rate:>12.2%}")
    print(f"Profit Factor:       {trades.profit_factor:>12.2f}")
    print()

    # 7. Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    output_dir = Path("./backtest_output")
    output_dir.mkdir(exist_ok=True)

    plotter = BacktestPlotter(results)

    print("Generating plots...")
    plots = plotter.plot_all(output_dir=str(output_dir), prefix="ma_crossover")

    print(f"   Created {len(plots)} visualizations:")
    for plot_name in plots.keys():
        print(f"   - {plot_name}")

    print()
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print()

    # 8. Export results
    print("=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    print()

    # Export trades
    if not results.trades.empty:
        trades_file = output_dir / "trades.csv"
        results.trades.to_csv(trades_file, index=False)
        print(f"Trades exported to: {trades_file}")

    # Export equity curve
    equity_file = output_dir / "equity_curve.csv"
    results.equity_curve.to_csv(equity_file)
    print(f"Equity curve exported to: {equity_file}")

    # Export results summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(engine.analyzer.get_detailed_report(results))
    print(f"Summary report exported to: {summary_file}")

    print()
    print("=" * 80)
    print("BACKTEST COMPLETE!")
    print("=" * 80)
    print()

    return results


if __name__ == "__main__":
    # Run the example
    results = run_example_backtest()

    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("1. Check the './backtest_output' directory for visualizations and reports")
    print("2. Modify the strategy function to test your own ideas")
    print("3. Adjust BacktestConfig parameters to test different scenarios")
    print("4. Load real market data instead of synthetic data")
