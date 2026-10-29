"""Order type definitions for trading system.

This module defines all order-related types including order sides, types,
statuses, and execution details.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"  # Created but not yet submitted
    SUBMITTED = "SUBMITTED"  # Submitted to broker/exchange
    PARTIAL = "PARTIAL"  # Partially filled
    FILLED = "FILLED"  # Completely filled
    CANCELLED = "CANCELLED"  # Cancelled by user or system
    REJECTED = "REJECTED"  # Rejected by broker/exchange


class TimeInForce(str, Enum):
    """Time in force enumeration."""

    DAY = "DAY"  # Valid for the day
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill


class Order(BaseModel):
    """Order class representing a trading order.

    Complete order information including identification, execution details,
    and lifecycle tracking.
    """

    # Identification
    order_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique order ID")
    client_order_id: Optional[str] = Field(None, description="Client-assigned order ID")
    broker_order_id: Optional[str] = Field(None, description="Broker/exchange order ID")

    # Instrument
    symbol: str = Field(..., description="Trading symbol/ticker")

    # Order parameters
    side: OrderSide = Field(..., description="Order side (BUY/SELL)")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")

    # Price parameters (optional depending on order type)
    limit_price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price")
    trailing_amount: Optional[Decimal] = Field(None, gt=0, description="Trailing stop amount")
    trailing_percent: Optional[Decimal] = Field(
        None, gt=0, le=100, description="Trailing stop percent"
    )

    # Time in force
    time_in_force: TimeInForce = Field(default=TimeInForce.DAY, description="Time in force")

    # Execution details
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: Decimal = Field(default=Decimal('0'), ge=0, description="Filled quantity")
    remaining_quantity: Optional[Decimal] = Field(None, ge=0, description="Remaining quantity")
    average_fill_price: Optional[Decimal] = Field(None, gt=0, description="Average fill price")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    submitted_at: Optional[datetime] = Field(None, description="Submission time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    filled_at: Optional[datetime] = Field(None, description="Fill completion time")
    cancelled_at: Optional[datetime] = Field(None, description="Cancellation time")

    # Additional info
    commission: Decimal = Field(default=Decimal('0'), ge=0, description="Commission paid")
    notes: Optional[str] = Field(None, description="Additional notes")
    error_message: Optional[str] = Field(None, description="Error message if rejected")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_order(self) -> 'Order':
        """Validate order constraints."""
        # Validate price parameters based on order type
        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("LIMIT orders require limit_price")
        elif self.order_type == OrderType.STOP:
            if self.stop_price is None:
                raise ValueError("STOP orders require stop_price")
        elif self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("STOP_LIMIT orders require both limit_price and stop_price")
        elif self.order_type == OrderType.TRAILING_STOP:
            if self.trailing_amount is None and self.trailing_percent is None:
                raise ValueError(
                    "TRAILING_STOP orders require either trailing_amount or trailing_percent"
                )

        # Validate filled quantity
        if self.filled_quantity > self.quantity:
            raise ValueError(
                f"Filled quantity ({self.filled_quantity}) cannot exceed "
                f"order quantity ({self.quantity})"
            )

        # Validate remaining quantity
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity - self.filled_quantity
        elif self.remaining_quantity != self.quantity - self.filled_quantity:
            raise ValueError(
                f"Remaining quantity ({self.remaining_quantity}) must equal "
                f"quantity ({self.quantity}) - filled_quantity ({self.filled_quantity})"
            )

        # Validate status transitions
        if self.status == OrderStatus.FILLED and self.filled_quantity != self.quantity:
            raise ValueError("FILLED orders must have filled_quantity equal to quantity")

        if self.status == OrderStatus.PARTIAL:
            if self.filled_quantity <= 0:
                raise ValueError("PARTIAL orders must have positive filled_quantity")
            if self.filled_quantity >= self.quantity:
                raise ValueError("PARTIAL orders must have filled_quantity < quantity")

        return self

    def is_active(self) -> bool:
        """Check if order is in an active state."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]

    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY

    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL

    def fill_rate(self) -> Decimal:
        """Calculate fill rate as percentage."""
        if self.quantity > 0:
            return (self.filled_quantity / self.quantity) * Decimal('100')
        return Decimal('0')

    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value if price is available."""
        price = self.limit_price or self.average_fill_price
        if price is not None:
            return self.quantity * price
        return None

    def filled_notional_value(self) -> Optional[Decimal]:
        """Calculate filled notional value."""
        if self.average_fill_price is not None:
            return self.filled_quantity * self.average_fill_price
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'broker_order_id': self.broker_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'limit_price': str(self.limit_price) if self.limit_price else None,
            'stop_price': str(self.stop_price) if self.stop_price else None,
            'trailing_amount': str(self.trailing_amount) if self.trailing_amount else None,
            'trailing_percent': str(self.trailing_percent) if self.trailing_percent else None,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity),
            'remaining_quantity': str(self.remaining_quantity) if self.remaining_quantity else None,
            'average_fill_price': str(self.average_fill_price) if self.average_fill_price else None,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'commission': str(self.commission),
            'notes': self.notes,
            'error_message': self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create Order from dictionary."""
        data_copy = data.copy()

        # Convert datetime fields
        for field in ['created_at', 'submitted_at', 'updated_at', 'filled_at', 'cancelled_at']:
            if field in data_copy and data_copy[field] is not None:
                if isinstance(data_copy[field], str):
                    data_copy[field] = datetime.fromisoformat(data_copy[field])

        # Convert decimal fields
        decimal_fields = [
            'quantity', 'limit_price', 'stop_price', 'trailing_amount',
            'trailing_percent', 'filled_quantity', 'remaining_quantity',
            'average_fill_price', 'commission'
        ]
        for field in decimal_fields:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        # Convert enum fields
        if 'side' in data_copy and isinstance(data_copy['side'], str):
            data_copy['side'] = OrderSide(data_copy['side'])
        if 'order_type' in data_copy and isinstance(data_copy['order_type'], str):
            data_copy['order_type'] = OrderType(data_copy['order_type'])
        if 'status' in data_copy and isinstance(data_copy['status'], str):
            data_copy['status'] = OrderStatus(data_copy['status'])
        if 'time_in_force' in data_copy and isinstance(data_copy['time_in_force'], str):
            data_copy['time_in_force'] = TimeInForce(data_copy['time_in_force'])

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class Fill(BaseModel):
    """Fill class representing a partial or complete order execution.

    Details about a single execution/fill of an order.
    """

    # Identification
    fill_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique fill ID")
    order_id: str = Field(..., description="Associated order ID")
    execution_id: Optional[str] = Field(None, description="Exchange execution ID")

    # Instrument
    symbol: str = Field(..., description="Trading symbol/ticker")

    # Execution details
    side: OrderSide = Field(..., description="Fill side (BUY/SELL)")
    quantity: Decimal = Field(..., gt=0, description="Fill quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Fill timestamp")

    # Costs
    commission: Decimal = Field(default=Decimal('0'), ge=0, description="Commission for this fill")
    fee: Decimal = Field(default=Decimal('0'), ge=0, description="Additional fees")

    # Additional info
    exchange: Optional[str] = Field(None, description="Exchange identifier")
    liquidity: Optional[str] = Field(None, description="Liquidity flag (maker/taker)")
    notes: Optional[str] = Field(None, description="Additional notes")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator('order_id')
    @classmethod
    def validate_order_id(cls, v: str) -> str:
        """Validate order ID."""
        if not v or not v.strip():
            raise ValueError("Order ID cannot be empty")
        return v.strip()

    def notional_value(self) -> Decimal:
        """Calculate notional value of the fill."""
        return self.quantity * self.price

    def total_cost(self) -> Decimal:
        """Calculate total cost including fees."""
        return self.commission + self.fee

    def net_proceeds(self) -> Decimal:
        """Calculate net proceeds (for sells) or cost (for buys)."""
        notional = self.notional_value()
        total_fees = self.total_cost()

        if self.side == OrderSide.SELL:
            return notional - total_fees
        else:  # BUY
            return notional + total_fees

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'execution_id': self.execution_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price),
            'timestamp': self.timestamp.isoformat(),
            'commission': str(self.commission),
            'fee': str(self.fee),
            'exchange': self.exchange,
            'liquidity': self.liquidity,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fill':
        """Create Fill from dictionary."""
        data_copy = data.copy()

        # Convert datetime
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert decimal fields
        for field in ['quantity', 'price', 'commission', 'fee']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        # Convert enum
        if 'side' in data_copy and isinstance(data_copy['side'], str):
            data_copy['side'] = OrderSide(data_copy['side'])

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }
