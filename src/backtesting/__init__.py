"""
Comprehensive Backtesting Framework

This package provides a complete event-driven backtesting framework with:
- Event-driven simulation for realistic backtesting
- Historical data replay with proper ordering
- Realistic order fill simulation
- Commission and slippage modeling
- Multi-strategy support
- Walk-forward analysis
- Comprehensive performance analytics
- Professional visualizations

Main Components:
    Engine: Core backtesting engine and event system
    Analytics: Performance and trade analysis
    Metrics: Performance metric calculations
    Visualization: Plotting and visualization tools

Example Usage:
    ```python
    from src.backtesting import BacktestEngine, BacktestConfig
    import pandas as pd

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_bps=5.0
    )

    # Create engine
    engine = BacktestEngine(config)

    # Load data
    data = pd.read_csv('market_data.csv', index_col='date', parse_dates=True)
    engine.load_data('AAPL', data)

    # Define strategy
    def my_strategy(engine, symbol, data):
        # Strategy logic here
        # Return signal dict or None
        if should_buy:
            return {'type': 'BUY', 'strength': 1.0}
        return None

    # Set strategy and run
    engine.set_strategy(my_strategy)
    results = engine.run()

    # Analyze results
    print(results.get_summary_report())

    # Visualize
    from src.backtesting import BacktestPlotter
    plotter = BacktestPlotter(results)
    plotter.plot_all(output_dir='./output')
    ```
"""

# Engine components
from .engine import (
    BacktestEngine,
    BacktestConfig,
    ExecutionModel,
    Position,
    Portfolio,
    EventQueue,
    EventDispatcher,
    Event,
    EventType,
    EventPriority,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PositionUpdateEvent,
    RiskCheckEvent,
    PortfolioUpdateEvent,
    run_backtest_parallel,
)

# Analytics components
from .analytics import (
    BacktestAnalyzer,
    BacktestResults,
    DrawdownPeriod,
    TradeAnalyzer,
    Trade,
    TradeAnalytics,
)

# Metrics components
from .metrics import (
    MetricsCalculator,
    PerformanceMetrics,
    compare_strategies,
)

# Visualization components
from .visualization import (
    BacktestPlotter,
    compare_strategies_plot,
)

__version__ = "1.0.0"

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "ExecutionModel",
    "Position",
    "Portfolio",
    "EventQueue",
    "EventDispatcher",
    "Event",
    "EventType",
    "EventPriority",
    "MarketDataEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "PositionUpdateEvent",
    "RiskCheckEvent",
    "PortfolioUpdateEvent",
    "run_backtest_parallel",
    # Analytics
    "BacktestAnalyzer",
    "BacktestResults",
    "DrawdownPeriod",
    "TradeAnalyzer",
    "Trade",
    "TradeAnalytics",
    # Metrics
    "MetricsCalculator",
    "PerformanceMetrics",
    "compare_strategies",
    # Visualization
    "BacktestPlotter",
    "compare_strategies_plot",
]
