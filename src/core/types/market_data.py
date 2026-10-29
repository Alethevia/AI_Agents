"""Market data type definitions for trading system.

This module defines all market data types including bars, ticks, quotes,
trades, and order book data structures.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Bar(BaseModel):
    """OHLCV bar data with timestamp and symbol.

    Represents aggregated price data over a time period with open, high,
    low, close prices and volume information.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    timestamp: datetime = Field(..., description="Bar timestamp")
    open: Decimal = Field(..., gt=0, description="Opening price")
    high: Decimal = Field(..., gt=0, description="Highest price")
    low: Decimal = Field(..., gt=0, description="Lowest price")
    close: Decimal = Field(..., gt=0, description="Closing price")
    volume: Decimal = Field(..., ge=0, description="Trading volume")
    vwap: Optional[Decimal] = Field(None, gt=0, description="Volume-weighted average price")
    trade_count: Optional[int] = Field(None, ge=0, description="Number of trades")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_ohlc(self) -> 'Bar':
        """Validate OHLC relationships."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) cannot be less than open ({self.open})")
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) cannot be less than close ({self.close})")
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) cannot be greater than open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) cannot be greater than close ({self.close})")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': str(self.open),
            'high': str(self.high),
            'low': str(self.low),
            'close': str(self.close),
            'volume': str(self.volume),
            'vwap': str(self.vwap) if self.vwap else None,
            'trade_count': self.trade_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bar':
        """Create Bar from dictionary."""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert string decimals
        for field in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class Tick(BaseModel):
    """Real-time price tick data.

    Represents a single price update with bid/ask information.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    timestamp: datetime = Field(..., description="Tick timestamp")
    price: Decimal = Field(..., gt=0, description="Last traded price")
    size: Decimal = Field(..., ge=0, description="Size of last trade")
    bid: Optional[Decimal] = Field(None, gt=0, description="Current bid price")
    ask: Optional[Decimal] = Field(None, gt=0, description="Current ask price")
    bid_size: Optional[Decimal] = Field(None, ge=0, description="Bid size")
    ask_size: Optional[Decimal] = Field(None, ge=0, description="Ask size")
    exchange: Optional[str] = Field(None, description="Exchange identifier")
    conditions: Optional[List[str]] = Field(default_factory=list, description="Trade conditions")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_bid_ask(self) -> 'Tick':
        """Validate bid/ask relationship."""
        if self.bid is not None and self.ask is not None:
            if self.bid > self.ask:
                raise ValueError(f"Bid ({self.bid}) cannot be greater than ask ({self.ask})")
        return self

    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price between bid and ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / Decimal('2')
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': str(self.price),
            'size': str(self.size),
            'bid': str(self.bid) if self.bid else None,
            'ask': str(self.ask) if self.ask else None,
            'bid_size': str(self.bid_size) if self.bid_size else None,
            'ask_size': str(self.ask_size) if self.ask_size else None,
            'exchange': self.exchange,
            'conditions': self.conditions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tick':
        """Create Tick from dictionary."""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert string decimals
        for field in ['price', 'size', 'bid', 'ask', 'bid_size', 'ask_size']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class Quote(BaseModel):
    """Bid/Ask quote data.

    Represents the best bid and ask prices with sizes.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    timestamp: datetime = Field(..., description="Quote timestamp")
    bid_price: Decimal = Field(..., gt=0, description="Bid price")
    bid_size: Decimal = Field(..., ge=0, description="Bid size")
    ask_price: Decimal = Field(..., gt=0, description="Ask price")
    ask_size: Decimal = Field(..., ge=0, description="Ask size")
    exchange: Optional[str] = Field(None, description="Exchange identifier")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_bid_ask(self) -> 'Quote':
        """Validate bid/ask relationship."""
        if self.bid_price > self.ask_price:
            raise ValueError(
                f"Bid price ({self.bid_price}) cannot be greater than ask price ({self.ask_price})"
            )
        return self

    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    def mid_price(self) -> Decimal:
        """Calculate mid price between bid and ask."""
        return (self.bid_price + self.ask_price) / Decimal('2')

    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        mid = self.mid_price()
        if mid > 0:
            return (self.spread() / mid) * Decimal('10000')
        return Decimal('0')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid_price': str(self.bid_price),
            'bid_size': str(self.bid_size),
            'ask_price': str(self.ask_price),
            'ask_size': str(self.ask_size),
            'exchange': self.exchange,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Quote':
        """Create Quote from dictionary."""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert string decimals
        for field in ['bid_price', 'bid_size', 'ask_price', 'ask_size']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class TradeSide(str, Enum):
    """Trade side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class Trade(BaseModel):
    """Executed trade data.

    Represents a completed trade with execution details.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    timestamp: datetime = Field(..., description="Trade timestamp")
    trade_id: str = Field(..., description="Unique trade identifier")
    price: Decimal = Field(..., gt=0, description="Execution price")
    size: Decimal = Field(..., gt=0, description="Trade size/quantity")
    side: TradeSide = Field(..., description="Trade side (BUY/SELL)")
    exchange: Optional[str] = Field(None, description="Exchange identifier")
    conditions: Optional[List[str]] = Field(default_factory=list, description="Trade conditions")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator('trade_id')
    @classmethod
    def validate_trade_id(cls, v: str) -> str:
        """Validate trade ID."""
        if not v or not v.strip():
            raise ValueError("Trade ID cannot be empty")
        return v.strip()

    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return self.price * self.size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id,
            'price': str(self.price),
            'size': str(self.size),
            'side': self.side.value,
            'exchange': self.exchange,
            'conditions': self.conditions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create Trade from dictionary."""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert string decimals
        for field in ['price', 'size']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        # Convert side enum
        if 'side' in data_copy and isinstance(data_copy['side'], str):
            data_copy['side'] = TradeSide(data_copy['side'])

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class OrderBookLevel(BaseModel):
    """Single level in order book.

    Represents a price level with size and order count.
    """

    price: Decimal = Field(..., gt=0, description="Price level")
    size: Decimal = Field(..., ge=0, description="Total size at this level")
    order_count: Optional[int] = Field(None, ge=0, description="Number of orders")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'price': str(self.price),
            'size': str(self.size),
            'order_count': self.order_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBookLevel':
        """Create OrderBookLevel from dictionary."""
        data_copy = data.copy()

        # Convert string decimals
        for field in ['price', 'size']:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {Decimal: str}


class OrderBook(BaseModel):
    """Full order book with bids and asks.

    Represents the complete order book with multiple price levels.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    timestamp: datetime = Field(..., description="Order book timestamp")
    bids: List[OrderBookLevel] = Field(default_factory=list, description="Bid levels")
    asks: List[OrderBookLevel] = Field(default_factory=list, description="Ask levels")
    exchange: Optional[str] = Field(None, description="Exchange identifier")
    sequence: Optional[int] = Field(None, description="Sequence number for updates")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_order_book(self) -> 'OrderBook':
        """Validate order book structure."""
        # Validate bids are sorted descending
        for i in range(len(self.bids) - 1):
            if self.bids[i].price < self.bids[i + 1].price:
                raise ValueError("Bid levels must be sorted in descending order")

        # Validate asks are sorted ascending
        for i in range(len(self.asks) - 1):
            if self.asks[i].price > self.asks[i + 1].price:
                raise ValueError("Ask levels must be sorted in ascending order")

        # Validate no cross
        if self.bids and self.asks:
            if self.bids[0].price >= self.asks[0].price:
                raise ValueError(
                    f"Order book crossed: best bid ({self.bids[0].price}) >= "
                    f"best ask ({self.asks[0].price})"
                )

        return self

    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid."""
        return self.bids[0] if self.bids else None

    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask."""
        return self.asks[0] if self.asks else None

    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / Decimal('2')
        return None

    def total_bid_volume(self, levels: Optional[int] = None) -> Decimal:
        """Calculate total bid volume."""
        bids_to_sum = self.bids[:levels] if levels else self.bids
        return sum(level.size for level in bids_to_sum)

    def total_ask_volume(self, levels: Optional[int] = None) -> Decimal:
        """Calculate total ask volume."""
        asks_to_sum = self.asks[:levels] if levels else self.asks
        return sum(level.size for level in asks_to_sum)

    def imbalance(self, levels: Optional[int] = None) -> Decimal:
        """Calculate order book imbalance."""
        bid_vol = self.total_bid_volume(levels)
        ask_vol = self.total_ask_volume(levels)
        total = bid_vol + ask_vol
        if total > 0:
            return (bid_vol - ask_vol) / total
        return Decimal('0')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bids': [level.to_dict() for level in self.bids],
            'asks': [level.to_dict() for level in self.asks],
            'exchange': self.exchange,
            'sequence': self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBook':
        """Create OrderBook from dictionary."""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert bid and ask levels
        if 'bids' in data_copy:
            data_copy['bids'] = [
                OrderBookLevel.from_dict(level) if isinstance(level, dict) else level
                for level in data_copy['bids']
            ]

        if 'asks' in data_copy:
            data_copy['asks'] = [
                OrderBookLevel.from_dict(level) if isinstance(level, dict) else level
                for level in data_copy['asks']
            ]

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class MarketDepth(BaseModel):
    """Market depth aggregated by price levels.

    Simplified representation of market depth showing cumulative volumes.
    """

    symbol: str = Field(..., description="Trading symbol/ticker")
    timestamp: datetime = Field(..., description="Market depth timestamp")
    bid_prices: List[Decimal] = Field(default_factory=list, description="Bid price levels")
    bid_sizes: List[Decimal] = Field(default_factory=list, description="Cumulative bid sizes")
    ask_prices: List[Decimal] = Field(default_factory=list, description="Ask price levels")
    ask_sizes: List[Decimal] = Field(default_factory=list, description="Cumulative ask sizes")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_depth(self) -> 'MarketDepth':
        """Validate market depth structure."""
        if len(self.bid_prices) != len(self.bid_sizes):
            raise ValueError("Bid prices and sizes must have same length")
        if len(self.ask_prices) != len(self.ask_sizes):
            raise ValueError("Ask prices and sizes must have same length")

        # Validate bids are sorted descending
        for i in range(len(self.bid_prices) - 1):
            if self.bid_prices[i] < self.bid_prices[i + 1]:
                raise ValueError("Bid prices must be sorted in descending order")

        # Validate asks are sorted ascending
        for i in range(len(self.ask_prices) - 1):
            if self.ask_prices[i] > self.ask_prices[i + 1]:
                raise ValueError("Ask prices must be sorted in ascending order")

        # Validate cumulative sizes are non-decreasing
        for i in range(len(self.bid_sizes) - 1):
            if self.bid_sizes[i] > self.bid_sizes[i + 1]:
                raise ValueError("Bid cumulative sizes must be non-decreasing")

        for i in range(len(self.ask_sizes) - 1):
            if self.ask_sizes[i] > self.ask_sizes[i + 1]:
                raise ValueError("Ask cumulative sizes must be non-decreasing")

        return self

    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        return self.bid_prices[0] if self.bid_prices else None

    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        return self.ask_prices[0] if self.ask_prices else None

    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None

    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / Decimal('2')
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid_prices': [str(p) for p in self.bid_prices],
            'bid_sizes': [str(s) for s in self.bid_sizes],
            'ask_prices': [str(p) for p in self.ask_prices],
            'ask_sizes': [str(s) for s in self.ask_sizes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketDepth':
        """Create MarketDepth from dictionary."""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert string decimals
        for field in ['bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes']:
            if field in data_copy:
                data_copy[field] = [
                    Decimal(str(v)) if not isinstance(v, Decimal) else v
                    for v in data_copy[field]
                ]

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }
