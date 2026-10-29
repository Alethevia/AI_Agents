"""
Event Queue System for Event-Driven Backtesting

This module provides the event queue and event types for the backtesting engine.
Supports different event types with priorities for realistic market simulation.
"""

import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from heapq import heappush, heappop
import uuid

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events in the backtesting system"""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    POSITION_UPDATE = "position_update"
    RISK_CHECK = "risk_check"
    PORTFOLIO_UPDATE = "portfolio_update"
    CUSTOM = "custom"


class EventPriority(int, Enum):
    """Event priorities for processing order"""
    CRITICAL = 0  # Risk checks, position updates
    HIGH = 1      # Market data updates
    NORMAL = 2    # Signals, orders
    LOW = 3       # Portfolio updates, logging


@dataclass
class Event:
    """Base event class for backtesting"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None  # Source component/strategy

    def __lt__(self, other: 'Event') -> bool:
        """Compare events for priority queue"""
        # Time must precede priority, or all future quotes run before today's signal.
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.event_id < other.event_id

    def __repr__(self) -> str:
        return (
            f"Event(id={self.event_id[:8]}, type={self.event_type.value}, "
            f"time={self.timestamp.isoformat()}, priority={self.priority.value})"
        )


@dataclass
class MarketDataEvent(Event):
    """Market data event"""
    symbol: str = ""
    price: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    open: Optional[Decimal] = None
    close: Optional[Decimal] = None

    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA
        self.priority = EventPriority.HIGH

        # Ensure Decimal types
        self.price = Decimal(str(self.price))
        self.volume = Decimal(str(self.volume))
        if self.bid is not None:
            self.bid = Decimal(str(self.bid))
        if self.ask is not None:
            self.ask = Decimal(str(self.ask))
        if self.high is not None:
            self.high = Decimal(str(self.high))
        if self.low is not None:
            self.low = Decimal(str(self.low))
        if self.open is not None:
            self.open = Decimal(str(self.open))
        if self.close is not None:
            self.close = Decimal(str(self.close))


@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    strategy_id: str = ""
    symbol: str = ""
    signal_type: str = ""  # "BUY", "SELL", "HOLD"
    strength: float = 0.0  # 0-1
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    rationale: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        self.priority = EventPriority.NORMAL

        if self.target_price is not None:
            self.target_price = Decimal(str(self.target_price))
        if self.stop_loss is not None:
            self.stop_loss = Decimal(str(self.stop_loss))
        if self.take_profit is not None:
            self.take_profit = Decimal(str(self.take_profit))


@dataclass
class OrderEvent(Event):
    """Order event"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    order_type: str = ""  # "MARKET", "LIMIT", "STOP"
    side: str = ""  # "BUY", "SELL"
    quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None  # For limit orders
    stop_price: Optional[Decimal] = None  # For stop orders
    time_in_force: str = "GTC"  # GTC, DAY, IOC, FOK
    strategy_id: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.ORDER
        self.priority = EventPriority.NORMAL

        self.quantity = Decimal(str(self.quantity))
        if self.price is not None:
            self.price = Decimal(str(self.price))
        if self.stop_price is not None:
            self.stop_price = Decimal(str(self.stop_price))


@dataclass
class FillEvent(Event):
    """Order fill event"""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: str = ""  # "BUY", "SELL"
    quantity: Decimal = Decimal("0")
    fill_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    strategy_id: Optional[str] = None

    def __post_init__(self):
        self.event_type = EventType.FILL
        self.priority = EventPriority.HIGH

        self.quantity = Decimal(str(self.quantity))
        self.fill_price = Decimal(str(self.fill_price))
        self.commission = Decimal(str(self.commission))
        self.slippage = Decimal(str(self.slippage))

    @property
    def total_cost(self) -> Decimal:
        """Total cost including commission and slippage"""
        return (self.fill_price * self.quantity) + self.commission

    @property
    def average_price(self) -> Decimal:
        """Average price per unit including costs"""
        if self.quantity == 0:
            return Decimal("0")
        return self.total_cost / self.quantity


@dataclass
class PositionUpdateEvent(Event):
    """Position update event"""
    symbol: str = ""
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    def __post_init__(self):
        self.event_type = EventType.POSITION_UPDATE
        self.priority = EventPriority.CRITICAL

        self.quantity = Decimal(str(self.quantity))
        self.average_price = Decimal(str(self.average_price))
        self.current_price = Decimal(str(self.current_price))
        self.unrealized_pnl = Decimal(str(self.unrealized_pnl))
        self.realized_pnl = Decimal(str(self.realized_pnl))


@dataclass
class RiskCheckEvent(Event):
    """Risk check event"""
    check_type: str = ""  # "PRE_TRADE", "POST_TRADE", "POSITION_LIMIT"
    passed: bool = False
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.RISK_CHECK
        self.priority = EventPriority.CRITICAL


@dataclass
class PortfolioUpdateEvent(Event):
    """Portfolio update event"""
    total_value: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    positions_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")

    def __post_init__(self):
        self.event_type = EventType.PORTFOLIO_UPDATE
        self.priority = EventPriority.LOW

        self.total_value = Decimal(str(self.total_value))
        self.cash = Decimal(str(self.cash))
        self.positions_value = Decimal(str(self.positions_value))
        self.total_pnl = Decimal(str(self.total_pnl))
        self.daily_pnl = Decimal(str(self.daily_pnl))


class EventQueue:
    """
    Priority-based event queue for backtesting

    Events are processed based on:
    1. Timestamp (chronological order)
    2. Priority within the same timestamp
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize event queue

        Args:
            max_size: Maximum queue size (None for unlimited)
        """
        self._queue: List[tuple] = []  # Min-heap of (priority_tuple, event)
        self._event_count = 0
        self._max_size = max_size
        self._event_history: List[Event] = []
        self._processed_count = 0

    def put(self, event: Event) -> None:
        """
        Add event to queue

        Args:
            event: Event to add

        Raises:
            ValueError: If queue is full
        """
        if self._max_size and len(self._queue) >= self._max_size:
            raise ValueError(f"Queue is full (max_size={self._max_size})")

        # Create priority tuple for sorting
        # (timestamp, priority_value, counter) ensures stable sorting
        priority_tuple = (
            event.timestamp,
            event.priority.value,
            self._event_count
        )

        heappush(self._queue, (priority_tuple, event))
        self._event_count += 1

        logger.debug(
            f"Event added to queue: {event.event_type.value} "
            f"(priority={event.priority.value}, queue_size={len(self._queue)})"
        )

    def get(self) -> Optional[Event]:
        """
        Get next event from queue

        Returns:
            Next event or None if queue is empty
        """
        if not self._queue:
            return None

        _, event = heappop(self._queue)
        self._processed_count += 1

        # Store in history
        self._event_history.append(event)

        logger.debug(
            f"Event retrieved from queue: {event.event_type.value} "
            f"(remaining={len(self._queue)})"
        )

        return event

    def peek(self) -> Optional[Event]:
        """
        Peek at next event without removing it

        Returns:
            Next event or None if queue is empty
        """
        if not self._queue:
            return None

        _, event = self._queue[0]
        return event

    def empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0

    def size(self) -> int:
        """Get current queue size"""
        return len(self._queue)

    def clear(self) -> None:
        """Clear all events from queue"""
        self._queue.clear()
        logger.info("Event queue cleared")

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """
        Get all events of a specific type (without removing them)

        Args:
            event_type: Event type to filter by

        Returns:
            List of matching events
        """
        return [event for _, event in self._queue if event.event_type == event_type]

    def get_events_by_symbol(self, symbol: str) -> List[Event]:
        """
        Get all events for a specific symbol (without removing them)

        Args:
            symbol: Symbol to filter by

        Returns:
            List of matching events
        """
        events = []
        for _, event in self._queue:
            if hasattr(event, 'symbol') and event.symbol == symbol:
                events.append(event)
        return events

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get event history

        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return

        Returns:
            List of historical events
        """
        history = self._event_history

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        if limit:
            history = history[-limit:]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics

        Returns:
            Dictionary with queue statistics
        """
        event_counts = {}
        for _, event in self._queue:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            'current_size': len(self._queue),
            'max_size': self._max_size,
            'total_events_added': self._event_count,
            'total_events_processed': self._processed_count,
            'events_in_history': len(self._event_history),
            'events_by_type': event_counts,
        }

    def __len__(self) -> int:
        """Get queue size"""
        return len(self._queue)

    def __repr__(self) -> str:
        """String representation"""
        return f"EventQueue(size={len(self._queue)}, processed={self._processed_count})"


class EventDispatcher:
    """
    Event dispatcher for routing events to handlers
    """

    def __init__(self):
        """Initialize event dispatcher"""
        self._handlers: Dict[EventType, List[callable]] = {
            event_type: [] for event_type in EventType
        }
        self._global_handlers: List[callable] = []

    def register_handler(
        self,
        event_type: EventType,
        handler: callable
    ) -> None:
        """
        Register an event handler

        Args:
            event_type: Event type to handle
            handler: Handler function (takes Event as argument)
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    def register_global_handler(self, handler: callable) -> None:
        """
        Register a global handler (receives all events)

        Args:
            handler: Handler function (takes Event as argument)
        """
        self._global_handlers.append(handler)
        logger.info("Registered global event handler")

    def unregister_handler(
        self,
        event_type: EventType,
        handler: callable
    ) -> None:
        """
        Unregister an event handler

        Args:
            event_type: Event type
            handler: Handler function to remove
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.info(f"Unregistered handler for {event_type.value}")

    def dispatch(self, event: Event) -> None:
        """
        Dispatch event to registered handlers

        Args:
            event: Event to dispatch
        """
        # Call global handlers first
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in global handler: {e}",
                    exc_info=True
                )

        # Call specific handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(
                        f"Error in {event.event_type.value} handler: {e}",
                        exc_info=True
                    )

    def get_handler_count(self, event_type: Optional[EventType] = None) -> int:
        """
        Get number of registered handlers

        Args:
            event_type: Event type (None for total count)

        Returns:
            Number of handlers
        """
        if event_type is None:
            return sum(len(handlers) for handlers in self._handlers.values()) + len(self._global_handlers)

        return len(self._handlers.get(event_type, []))

    def clear_handlers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear registered handlers

        Args:
            event_type: Event type to clear (None for all)
        """
        if event_type is None:
            self._handlers = {et: [] for et in EventType}
            self._global_handlers.clear()
            logger.info("Cleared all event handlers")
        else:
            self._handlers[event_type].clear()
            logger.info(f"Cleared handlers for {event_type.value}")
