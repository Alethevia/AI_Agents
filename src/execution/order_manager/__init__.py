"""
Order Manager Module

Provides order management and splitting functionality.
"""

from .order_manager import (
    OrderManager,
    Order,
    Fill,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce
)

from .order_splitter import (
    OrderSplitter,
    TWAPSplitter,
    VWAPSplitter,
    POVSplitter,
    ImplementationShortfallSplitter,
    AdaptiveSplitter,
    OrderSplitterManager,
    SplitStrategy,
    ChildOrder,
    MarketState
)


__all__ = [
    # Order Manager
    "OrderManager",
    "Order",
    "Fill",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",

    # Order Splitter
    "OrderSplitter",
    "TWAPSplitter",
    "VWAPSplitter",
    "POVSplitter",
    "ImplementationShortfallSplitter",
    "AdaptiveSplitter",
    "OrderSplitterManager",
    "SplitStrategy",
    "ChildOrder",
    "MarketState"
]
