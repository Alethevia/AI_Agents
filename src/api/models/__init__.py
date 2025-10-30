"""
API models package.

This package contains all Pydantic models for request validation and response serialization.
"""

from .api_models import (
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    StrategyStatus,
    TimeInterval,
    WebSocketMessageType,

    # Auth models
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    APIKeyRequest,
    APIKeyResponse,

    # Order models
    CreateOrderRequest,
    OrderResponse,
    CancelOrderRequest,
    OrderStatusResponse,

    # Portfolio models
    PortfolioResponse,
    PositionResponse,
    PortfolioPerformanceResponse,
    PortfolioHistoryResponse,
    PerformanceMetrics,

    # Strategy models
    StrategyConfig,
    StrategyResponse,
    StrategyPerformanceResponse,
    StartStrategyRequest,
    StopStrategyRequest,

    # Market data models
    MarketDataRequest,
    MarketDataResponse,
    Candlestick,
    TickerResponse,
    OrderBookResponse,
    OrderBookLevel,

    # Risk models
    RiskMetrics,
    RiskAlert,
    RiskLimits,
    UpdateRiskLimitsRequest,

    # WebSocket models
    WebSocketSubscribeRequest,
    WebSocketUnsubscribeRequest,
    WebSocketMessage,

    # Error models
    ErrorDetail,
    ErrorResponse,

    # Generic models
    HealthCheckResponse,
    MessageResponse,
    PaginatedResponse,
    BulkOperationResponse,
)

__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "StrategyStatus",
    "TimeInterval",
    "WebSocketMessageType",

    # Auth models
    "LoginRequest",
    "TokenResponse",
    "RefreshTokenRequest",
    "APIKeyRequest",
    "APIKeyResponse",

    # Order models
    "CreateOrderRequest",
    "OrderResponse",
    "CancelOrderRequest",
    "OrderStatusResponse",

    # Portfolio models
    "PortfolioResponse",
    "PositionResponse",
    "PortfolioPerformanceResponse",
    "PortfolioHistoryResponse",
    "PerformanceMetrics",

    # Strategy models
    "StrategyConfig",
    "StrategyResponse",
    "StrategyPerformanceResponse",
    "StartStrategyRequest",
    "StopStrategyRequest",

    # Market data models
    "MarketDataRequest",
    "MarketDataResponse",
    "Candlestick",
    "TickerResponse",
    "OrderBookResponse",
    "OrderBookLevel",

    # Risk models
    "RiskMetrics",
    "RiskAlert",
    "RiskLimits",
    "UpdateRiskLimitsRequest",

    # WebSocket models
    "WebSocketSubscribeRequest",
    "WebSocketUnsubscribeRequest",
    "WebSocketMessage",

    # Error models
    "ErrorDetail",
    "ErrorResponse",

    # Generic models
    "HealthCheckResponse",
    "MessageResponse",
    "PaginatedResponse",
    "BulkOperationResponse",
]
