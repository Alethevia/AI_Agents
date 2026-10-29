"""
Routing Module

Provides smart order routing and broker adapter functionality.
"""

from .smart_router import (
    SmartRouter,
    RoutingStrategy,
    RoutingDecision,
    BrokerQuote,
    BrokerPerformance
)

from .broker_adapters import (
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


__all__ = [
    # Smart Router
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
    "BrokerRateLimitError"
]
