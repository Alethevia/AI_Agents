"""Backtesting engine components"""

from .event_queue import (
    Event,
    EventType,
    EventPriority,
    EventQueue,
    EventDispatcher,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PositionUpdateEvent,
    RiskCheckEvent,
    PortfolioUpdateEvent,
)

from .backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    ExecutionModel,
    Position,
    Portfolio,
    run_backtest_parallel,
)

__all__ = [
    # Event system
    "Event",
    "EventType",
    "EventPriority",
    "EventQueue",
    "EventDispatcher",
    "MarketDataEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "PositionUpdateEvent",
    "RiskCheckEvent",
    "PortfolioUpdateEvent",
    # Backtest engine
    "BacktestEngine",
    "BacktestConfig",
    "ExecutionModel",
    "Position",
    "Portfolio",
    "run_backtest_parallel",
]
