"""Position type definitions for trading system.

This module defines position-related types including position sides and
position tracking with P&L calculations.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PositionSide(str, Enum):
    """Position side enumeration."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Position(BaseModel):
    """Position class representing a trading position.

    Tracks position size, entry details, current market value, and P&L.
    """

    # Identification
    symbol: str = Field(..., description="Trading symbol/ticker")

    # Position details
    side: PositionSide = Field(..., description="Position side (LONG/SHORT/FLAT)")
    quantity: Decimal = Field(default=Decimal('0'), ge=0, description="Position quantity")

    # Entry details
    entry_price: Optional[Decimal] = Field(None, gt=0, description="Average entry price")
    entry_value: Optional[Decimal] = Field(None, description="Total entry value")
    entry_time: Optional[datetime] = Field(None, description="Initial entry timestamp")

    # Current market data
    current_price: Optional[Decimal] = Field(None, gt=0, description="Current market price")
    last_updated: Optional[datetime] = Field(None, description="Last price update timestamp")

    # P&L tracking
    unrealized_pnl: Decimal = Field(default=Decimal('0'), description="Unrealized profit/loss")
    realized_pnl: Decimal = Field(default=Decimal('0'), description="Realized profit/loss")

    # Cost tracking
    total_commission: Decimal = Field(default=Decimal('0'), ge=0, description="Total commissions paid")
    total_fees: Decimal = Field(default=Decimal('0'), ge=0, description="Total fees paid")

    # Additional info
    notes: Optional[str] = Field(None, description="Additional notes")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_position(self) -> 'Position':
        """Validate position constraints."""
        # Validate FLAT positions have zero quantity
        if self.side == PositionSide.FLAT and self.quantity != 0:
            raise ValueError("FLAT positions must have zero quantity")

        # Validate non-FLAT positions have positive quantity
        if self.side != PositionSide.FLAT and self.quantity <= 0:
            raise ValueError(f"{self.side} positions must have positive quantity")

        # If we have a quantity, we should have entry price
        if self.quantity > 0 and self.entry_price is None:
            raise ValueError("Positions with quantity must have entry_price")

        return self

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT

    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.side == PositionSide.FLAT or self.quantity == 0

    def market_value(self) -> Decimal:
        """Calculate current market value of the position."""
        if self.current_price is None or self.quantity == 0:
            return Decimal('0')
        return self.quantity * self.current_price

    def cost_basis(self) -> Decimal:
        """Calculate cost basis of the position."""
        if self.entry_value is not None:
            return self.entry_value
        if self.entry_price is not None:
            return self.quantity * self.entry_price
        return Decimal('0')

    def calculate_unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L based on current price.

        For LONG: (current_price - entry_price) * quantity
        For SHORT: (entry_price - current_price) * quantity
        """
        if self.current_price is None or self.entry_price is None or self.quantity == 0:
            return Decimal('0')

        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - self.current_price) * self.quantity
        else:
            return Decimal('0')

    def update_market_price(self, price: Decimal) -> None:
        """Update current market price and recalculate unrealized P&L."""
        self.current_price = price
        self.last_updated = datetime.utcnow()
        self.unrealized_pnl = self.calculate_unrealized_pnl()

    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def total_costs(self) -> Decimal:
        """Calculate total costs (commissions + fees)."""
        return self.total_commission + self.total_fees

    def net_pnl(self) -> Decimal:
        """Calculate net P&L after costs."""
        return self.total_pnl() - self.total_costs()

    def pnl_percent(self) -> Decimal:
        """Calculate P&L as percentage of cost basis."""
        cost = self.cost_basis()
        if cost > 0:
            return (self.unrealized_pnl / cost) * Decimal('100')
        return Decimal('0')

    def return_on_investment(self) -> Decimal:
        """Calculate ROI including realized P&L."""
        cost = self.cost_basis()
        if cost > 0:
            return (self.total_pnl() / cost) * Decimal('100')
        return Decimal('0')

    def duration_seconds(self) -> Optional[int]:
        """Calculate position duration in seconds."""
        if self.entry_time and self.last_updated:
            return int((self.last_updated - self.entry_time).total_seconds())
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'entry_price': str(self.entry_price) if self.entry_price else None,
            'entry_value': str(self.entry_value) if self.entry_value else None,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'current_price': str(self.current_price) if self.current_price else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'unrealized_pnl': str(self.unrealized_pnl),
            'realized_pnl': str(self.realized_pnl),
            'total_commission': str(self.total_commission),
            'total_fees': str(self.total_fees),
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from dictionary."""
        data_copy = data.copy()

        # Convert datetime fields
        for field in ['entry_time', 'last_updated']:
            if field in data_copy and data_copy[field] is not None:
                if isinstance(data_copy[field], str):
                    data_copy[field] = datetime.fromisoformat(data_copy[field])

        # Convert decimal fields
        decimal_fields = [
            'quantity', 'entry_price', 'entry_value', 'current_price',
            'unrealized_pnl', 'realized_pnl', 'total_commission', 'total_fees'
        ]
        for field in decimal_fields:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        # Convert enum
        if 'side' in data_copy and isinstance(data_copy['side'], str):
            data_copy['side'] = PositionSide(data_copy['side'])

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class PositionUpdate(BaseModel):
    """Position update event.

    Represents a change to a position from an order fill.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    side: PositionSide = Field(..., description="New position side")
    quantity_delta: Decimal = Field(..., description="Change in quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    commission: Decimal = Field(default=Decimal('0'), ge=0, description="Commission paid")
    fee: Decimal = Field(default=Decimal('0'), ge=0, description="Fees paid")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    def is_opening(self) -> bool:
        """Check if this update is opening a position."""
        return self.quantity_delta > 0

    def is_closing(self) -> bool:
        """Check if this update is closing a position."""
        return self.quantity_delta < 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity_delta': str(self.quantity_delta),
            'price': str(self.price),
            'timestamp': self.timestamp.isoformat(),
            'commission': str(self.commission),
            'fee': str(self.fee),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionUpdate':
        """Create PositionUpdate from dictionary."""
        data_copy = data.copy()

        # Convert datetime
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert decimal fields
        for field in ['quantity_delta', 'price', 'commission', 'fee']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        # Convert enum
        if 'side' in data_copy and isinstance(data_copy['side'], str):
            data_copy['side'] = PositionSide(data_copy['side'])

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }
