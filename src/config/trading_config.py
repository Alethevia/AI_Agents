"""
Trading-specific configuration.
"""

from typing import Dict, List, Optional
from datetime import time
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class MarketType(str, Enum):
    """Market types."""
    STOCKS = "stocks"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class TradingMode(str, Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class TradingConfig(BaseModel):
    """Trading configuration."""

    # Trading mode
    mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode"
    )

    # Market configuration
    markets: List[MarketType] = Field(
        default=[MarketType.STOCKS],
        description="Markets to trade"
    )

    # Trading hours
    market_open: time = Field(
        default=time(9, 30),
        description="Market open time"
    )
    market_close: time = Field(
        default=time(16, 0),
        description="Market close time"
    )
    pre_market_start: Optional[time] = Field(
        default=time(4, 0),
        description="Pre-market trading start"
    )
    after_market_end: Optional[time] = Field(
        default=time(20, 0),
        description="After-hours trading end"
    )

    # Position limits
    max_positions: int = Field(
        default=50,
        description="Maximum number of concurrent positions",
        ge=1
    )
    max_position_size_usd: float = Field(
        default=100000.0,
        description="Maximum position size in USD",
        gt=0
    )
    max_position_pct: float = Field(
        default=0.1,
        description="Maximum position size as % of portfolio",
        gt=0,
        le=1.0
    )

    # Risk limits
    max_portfolio_risk: float = Field(
        default=0.02,
        description="Maximum portfolio risk per trade",
        gt=0,
        le=1.0
    )
    max_sector_exposure: float = Field(
        default=0.3,
        description="Maximum exposure to single sector",
        gt=0,
        le=1.0
    )
    max_correlation: float = Field(
        default=0.7,
        description="Maximum correlation between positions",
        ge=-1.0,
        le=1.0
    )

    # Stop loss / Take profit
    default_stop_loss_pct: float = Field(
        default=0.02,
        description="Default stop loss percentage",
        gt=0,
        le=1.0
    )
    default_take_profit_pct: float = Field(
        default=0.05,
        description="Default take profit percentage",
        gt=0
    )
    trailing_stop_activation: float = Field(
        default=0.03,
        description="Trailing stop activation threshold",
        gt=0
    )
    trailing_stop_distance: float = Field(
        default=0.015,
        description="Trailing stop distance",
        gt=0
    )

    # Order execution
    default_order_type: OrderType = Field(
        default=OrderType.LIMIT,
        description="Default order type"
    )
    default_time_in_force: TimeInForce = Field(
        default=TimeInForce.DAY,
        description="Default time in force"
    )
    limit_order_offset_pct: float = Field(
        default=0.001,
        description="Limit order price offset percentage",
        ge=0
    )
    max_slippage_pct: float = Field(
        default=0.005,
        description="Maximum allowed slippage",
        ge=0
    )

    # Order sizing
    min_order_size_usd: float = Field(
        default=100.0,
        description="Minimum order size in USD",
        gt=0
    )
    max_order_size_usd: float = Field(
        default=50000.0,
        description="Maximum order size in USD",
        gt=0
    )
    order_chunk_size_usd: float = Field(
        default=10000.0,
        description="Order chunk size for splitting large orders",
        gt=0
    )

    # Leverage
    max_leverage: float = Field(
        default=1.0,
        description="Maximum leverage",
        ge=1.0,
        le=10.0
    )
    margin_requirement: float = Field(
        default=0.25,
        description="Margin requirement",
        gt=0,
        le=1.0
    )

    # Portfolio constraints
    min_cash_balance_pct: float = Field(
        default=0.05,
        description="Minimum cash balance as % of portfolio",
        ge=0,
        le=1.0
    )
    max_turnover_daily: float = Field(
        default=0.5,
        description="Maximum daily portfolio turnover",
        ge=0
    )

    # Rebalancing
    rebalance_frequency_hours: int = Field(
        default=24,
        description="Rebalancing frequency in hours",
        ge=1
    )
    rebalance_threshold_pct: float = Field(
        default=0.05,
        description="Rebalancing threshold",
        gt=0
    )

    # Filters
    min_price: float = Field(
        default=1.0,
        description="Minimum stock price",
        gt=0
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum stock price"
    )
    min_volume: int = Field(
        default=100000,
        description="Minimum daily volume",
        ge=0
    )
    min_market_cap: float = Field(
        default=100_000_000,
        description="Minimum market cap",
        ge=0
    )

    # Commission and fees
    commission_per_share: float = Field(
        default=0.005,
        description="Commission per share",
        ge=0
    )
    commission_min: float = Field(
        default=1.0,
        description="Minimum commission per trade",
        ge=0
    )
    commission_pct: float = Field(
        default=0.0,
        description="Commission as percentage of trade value",
        ge=0
    )

    # Watchlist
    max_watchlist_size: int = Field(
        default=500,
        description="Maximum watchlist size",
        ge=1
    )
    watchlist_update_interval_minutes: int = Field(
        default=5,
        description="Watchlist update interval in minutes",
        ge=1
    )

    # Data requirements
    min_data_points: int = Field(
        default=100,
        description="Minimum data points for trading decision",
        ge=10
    )
    data_lookback_days: int = Field(
        default=252,
        description="Data lookback period in days",
        ge=1
    )

    @field_validator("max_order_size_usd")
    @classmethod
    def validate_max_order_size(cls, v: float, info) -> float:
        """Validate max order size is greater than min."""
        if "min_order_size_usd" in info.data and v < info.data["min_order_size_usd"]:
            raise ValueError("max_order_size_usd must be >= min_order_size_usd")
        return v

    class Config:
        use_enum_values = True
