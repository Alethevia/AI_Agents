"""
Pydantic models for API request/response validation.

This module defines all API models for request validation, response serialization,
and error handling in the trading robot API.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, ConfigDict


# ============================================================================
# Enums
# ============================================================================

class OrderSide(str, Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enum."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order status enum."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class StrategyStatus(str, Enum):
    """Strategy status enum."""
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class TimeInterval(str, Enum):
    """Time interval enum for market data."""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


# ============================================================================
# Base Models
# ============================================================================

class BaseAPIModel(BaseModel):
    """Base model with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }
    )


# ============================================================================
# Authentication Models
# ============================================================================

class LoginRequest(BaseAPIModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class TokenResponse(BaseAPIModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class RefreshTokenRequest(BaseAPIModel):
    """Refresh token request model."""
    refresh_token: str


class APIKeyRequest(BaseAPIModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    permissions: List[str] = Field(default_factory=list)


class APIKeyResponse(BaseAPIModel):
    """API key response model."""
    key_id: str
    api_key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    permissions: List[str]


# ============================================================================
# Order Models
# ============================================================================

class CreateOrderRequest(BaseAPIModel):
    """Create order request model."""
    symbol: str = Field(..., min_length=1)
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = Field(None, gt=0)
    stop_price: Optional[Decimal] = Field(None, gt=0)
    time_in_force: str = Field(default="GTC")
    client_order_id: Optional[str] = None

    @validator('price')
    def validate_limit_price(cls, v, values):
        """Validate that limit orders have a price."""
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Price is required for LIMIT and STOP_LIMIT orders")
        return v

    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """Validate that stop orders have a stop price."""
        if values.get('order_type') in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Stop price is required for STOP and STOP_LIMIT orders")
        return v


class OrderResponse(BaseAPIModel):
    """Order response model."""
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    filled_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    status: OrderStatus
    time_in_force: str
    created_at: datetime
    updated_at: datetime
    filled_at: Optional[datetime] = None


class CancelOrderRequest(BaseAPIModel):
    """Cancel order request model."""
    order_id: str


class OrderStatusResponse(BaseAPIModel):
    """Order status response model."""
    order_id: str
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_price: Optional[Decimal] = None
    updated_at: datetime


# ============================================================================
# Portfolio Models
# ============================================================================

class PositionResponse(BaseAPIModel):
    """Position response model."""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: Decimal
    cost_basis: Decimal
    updated_at: datetime


class PortfolioResponse(BaseAPIModel):
    """Portfolio response model."""
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    positions: List[PositionResponse]
    updated_at: datetime


class PerformanceMetrics(BaseAPIModel):
    """Performance metrics model."""
    total_return: Decimal
    total_return_percent: Decimal
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: Optional[Decimal] = None
    average_loss: Optional[Decimal] = None
    largest_win: Optional[Decimal] = None
    largest_loss: Optional[Decimal] = None


class PortfolioPerformanceResponse(BaseAPIModel):
    """Portfolio performance response model."""
    start_date: datetime
    end_date: datetime
    initial_value: Decimal
    final_value: Decimal
    metrics: PerformanceMetrics


class PortfolioHistoryResponse(BaseAPIModel):
    """Portfolio history response model."""
    timestamp: datetime
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


# ============================================================================
# Strategy Models
# ============================================================================

class StrategyConfig(BaseAPIModel):
    """Strategy configuration model."""
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    symbols: List[str] = Field(default_factory=list)
    enabled: bool = True


class StrategyResponse(BaseAPIModel):
    """Strategy response model."""
    strategy_id: str
    name: str
    status: StrategyStatus
    symbols: List[str]
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None


class StrategyPerformanceResponse(BaseAPIModel):
    """Strategy performance response model."""
    strategy_id: str
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: Decimal
    win_rate: Decimal
    average_pnl_per_trade: Decimal
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None


class StartStrategyRequest(BaseAPIModel):
    """Start strategy request model."""
    strategy_id: str
    config: Optional[Dict[str, Any]] = None


class StopStrategyRequest(BaseAPIModel):
    """Stop strategy request model."""
    strategy_id: str
    reason: Optional[str] = None


# ============================================================================
# Market Data Models
# ============================================================================

class MarketDataRequest(BaseAPIModel):
    """Market data request model."""
    symbol: str
    interval: TimeInterval
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)


class Candlestick(BaseAPIModel):
    """Candlestick data model."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class MarketDataResponse(BaseAPIModel):
    """Market data response model."""
    symbol: str
    interval: TimeInterval
    data: List[Candlestick]


class TickerResponse(BaseAPIModel):
    """Ticker response model."""
    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    volume_24h: Decimal
    price_change_24h: Decimal
    price_change_percent_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime


class OrderBookLevel(BaseAPIModel):
    """Order book level model."""
    price: Decimal
    quantity: Decimal


class OrderBookResponse(BaseAPIModel):
    """Order book response model."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime


# ============================================================================
# Risk Models
# ============================================================================

class RiskMetrics(BaseAPIModel):
    """Risk metrics model."""
    portfolio_value: Decimal
    max_position_size: Decimal
    current_exposure: Decimal
    available_buying_power: Decimal
    margin_used: Decimal
    margin_available: Decimal
    risk_score: Decimal
    var_95: Optional[Decimal] = None  # Value at Risk 95%
    var_99: Optional[Decimal] = None  # Value at Risk 99%


class RiskAlert(BaseAPIModel):
    """Risk alert model."""
    alert_id: str
    severity: str  # INFO, WARNING, CRITICAL
    category: str  # POSITION_SIZE, DRAWDOWN, EXPOSURE, etc.
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    acknowledged: bool = False


class RiskLimits(BaseAPIModel):
    """Risk limits configuration model."""
    max_position_size: Decimal
    max_portfolio_allocation: Decimal
    max_daily_loss: Decimal
    max_drawdown: Decimal
    max_leverage: Decimal


class UpdateRiskLimitsRequest(BaseAPIModel):
    """Update risk limits request model."""
    limits: RiskLimits


# ============================================================================
# WebSocket Models
# ============================================================================

class WebSocketSubscribeRequest(BaseAPIModel):
    """WebSocket subscribe request model."""
    topics: List[str]


class WebSocketUnsubscribeRequest(BaseAPIModel):
    """WebSocket unsubscribe request model."""
    topics: List[str]


class WebSocketMessage(BaseAPIModel):
    """WebSocket message model."""
    type: WebSocketMessageType
    topic: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseAPIModel):
    """Error detail model."""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseAPIModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# ============================================================================
# Generic Response Models
# ============================================================================

class HealthCheckResponse(BaseAPIModel):
    """Health check response model."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime: float
    services: Dict[str, str] = Field(default_factory=dict)


class MessageResponse(BaseAPIModel):
    """Generic message response model."""
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseAPIModel):
    """Paginated response model."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkOperationResponse(BaseAPIModel):
    """Bulk operation response model."""
    success_count: int
    failure_count: int
    results: List[Dict[str, Any]]
    errors: List[ErrorDetail] = Field(default_factory=list)
