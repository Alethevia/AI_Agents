"""
Execution Engine

A sophisticated order execution system providing:
- Order management and lifecycle tracking
- Smart order splitting (TWAP, VWAP, POV, etc.)
- Intelligent order routing across multiple brokers
- Slippage estimation and analysis
- Realistic fill simulation for paper trading
- Broker adapters (Alpaca, Interactive Brokers)

Example Usage:
    >>> from execution import OrderManager, SmartRouter, FillSimulator
    >>> from execution.order_manager import OrderType, OrderSide, TimeInForce
    >>> from execution.routing import AlpacaBrokerAdapter, BrokerConfig
    >>>
    >>> # Setup broker
    >>> config = BrokerConfig(
    ...     broker_id="alpaca",
    ...     api_key="your_key",
    ...     api_secret="your_secret",
    ...     base_url="https://paper-api.alpaca.markets",
    ...     paper_trading=True
    ... )
    >>> broker = AlpacaBrokerAdapter(config)
    >>> await broker.connect()
    >>>
    >>> # Create order manager
    >>> order_manager = OrderManager(broker_adapter=broker)
    >>>
    >>> # Create and submit order
    >>> order = await order_manager.create_order(
    ...     symbol="AAPL",
    ...     side=OrderSide.BUY,
    ...     order_type=OrderType.MARKET,
    ...     quantity=100
    ... )
    >>> await order_manager.submit_order(order.order_id)
    >>>
    >>> # Smart order splitting
    >>> from execution.order_manager import OrderSplitterManager, SplitStrategy
    >>> splitter = OrderSplitterManager()
    >>> child_orders = await splitter.split_order(
    ...     strategy=SplitStrategy.VWAP,
    ...     order_id=order.order_id,
    ...     symbol="AAPL",
    ...     side="buy",
    ...     total_quantity=1000,
    ...     start_time=datetime.now(),
    ...     end_time=datetime.now() + timedelta(hours=1)
    ... )
"""

# Order Management
from .order_manager import (
    OrderManager,
    Order,
    Fill,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    OrderSplitter,
    TWAPSplitter,
    VWAPSplitter,
    POVSplitter,
    ImplementationShortfallSplitter,
    AdaptiveSplitter,
    OrderSplitterManager,
    SplitStrategy,
    ChildOrder
)

# Routing
from .routing import (
    SmartRouter,
    RoutingStrategy,
    RoutingDecision,
    BrokerQuote,
    BrokerPerformance,
    BrokerAdapter,
    AlpacaBrokerAdapter,
    InteractiveBrokerAdapter,
    BrokerAdapterManager,
    BrokerConfig,
    BrokerOrder,
    BrokerStatus,
    BrokerError,
    BrokerConnectionError,
    BrokerAuthenticationError,
    BrokerOrderError,
    BrokerRateLimitError
)

# Slippage
from .slippage import (
    SlippageModel,
    LinearImpactModel,
    SquareRootImpactModel,
    VolumeBasedSlippageModel,
    SpreadBasedSlippageModel,
    HistoricalSlippageModel,
    EnsembleSlippageModel,
    SlippageAnalyzer,
    SlippageEstimate,
    HistoricalSlippage
)

# Fill Simulation
from .fill_simulator import (
    FillSimulator,
    RealisticFillSimulator,
    FillMode,
    SimulatedOrder,
    SimulatedFill,
    MarketSnapshot
)


__version__ = "1.0.0"


__all__ = [
    # Order Management
    "OrderManager",
    "Order",
    "Fill",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",

    # Order Splitting
    "OrderSplitter",
    "TWAPSplitter",
    "VWAPSplitter",
    "POVSplitter",
    "ImplementationShortfallSplitter",
    "AdaptiveSplitter",
    "OrderSplitterManager",
    "SplitStrategy",
    "ChildOrder",

    # Smart Routing
    "SmartRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "BrokerQuote",
    "BrokerPerformance",

    # Broker Adapters
    "BrokerAdapter",
    "AlpacaBrokerAdapter",
    "InteractiveBrokerAdapter",
    "BrokerAdapterManager",
    "BrokerConfig",
    "BrokerOrder",
    "BrokerStatus",
    "BrokerError",
    "BrokerConnectionError",
    "BrokerAuthenticationError",
    "BrokerOrderError",
    "BrokerRateLimitError",

    # Slippage Models
    "SlippageModel",
    "LinearImpactModel",
    "SquareRootImpactModel",
    "VolumeBasedSlippageModel",
    "SpreadBasedSlippageModel",
    "HistoricalSlippageModel",
    "EnsembleSlippageModel",
    "SlippageAnalyzer",
    "SlippageEstimate",
    "HistoricalSlippage",

    # Fill Simulation
    "FillSimulator",
    "RealisticFillSimulator",
    "FillMode",
    "SimulatedOrder",
    "SimulatedFill",
    "MarketSnapshot"
]
