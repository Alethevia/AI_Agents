"""
Centralized Order Management System

This module provides a comprehensive order management system for handling
order lifecycle, state tracking, fills, and order book management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
from decimal import Decimal


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the execution engine"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MOC = "market_on_close"
    LOC = "limit_on_close"


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"  # Created but not submitted
    SUBMITTED = "submitted"  # Submitted to broker
    ACCEPTED = "accepted"  # Accepted by broker
    PARTIAL_FILL = "partial_fill"  # Partially filled
    FILLED = "filled"  # Completely filled
    CANCELLED = "cancelled"  # Cancelled
    REJECTED = "rejected"  # Rejected by broker
    EXPIRED = "expired"  # Expired
    REPLACED = "replaced"  # Replaced by another order


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close


@dataclass
class Fill:
    """Represents a single fill for an order"""
    fill_id: str
    order_id: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    commission: Decimal = Decimal("0")
    exchange: Optional[str] = None
    liquidity_flag: Optional[str] = None  # "added" or "removed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure decimals and datetime are properly typed"""
        self.price = Decimal(str(self.price))
        self.quantity = Decimal(str(self.quantity))
        self.commission = Decimal(str(self.commission))
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    @property
    def notional_value(self) -> Decimal:
        """Calculate the notional value of this fill"""
        return self.price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert fill to dictionary"""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "price": str(self.price),
            "quantity": str(self.quantity),
            "commission": str(self.commission),
            "exchange": self.exchange,
            "liquidity_flag": self.liquidity_flag,
            "metadata": self.metadata
        }


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional fields
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    broker: Optional[str] = None
    broker_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For split orders

    # Timestamps
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Fills
    fills: List[Fill] = field(default_factory=list)

    # Metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None

    # Trading mode
    paper_trading: bool = False

    def __post_init__(self):
        """Ensure proper types"""
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)
        if isinstance(self.time_in_force, str):
            self.time_in_force = TimeInForce(self.time_in_force)

        self.quantity = Decimal(str(self.quantity))
        self.filled_quantity = Decimal(str(self.filled_quantity))

        if self.limit_price is not None:
            self.limit_price = Decimal(str(self.limit_price))
        if self.stop_price is not None:
            self.stop_price = Decimal(str(self.stop_price))

    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to be filled"""
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is active (can still be filled)"""
        return self.status in [
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL_FILL
        ]

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.REPLACED
        ]

    def add_fill(self, fill: Fill) -> None:
        """Add a fill to the order and update status"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity

        # Update average fill price
        total_value = sum(f.price * f.quantity for f in self.fills)
        self.avg_fill_price = total_value / self.filled_quantity

        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.timestamp
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL_FILL

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "limit_price": str(self.limit_price) if self.limit_price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force.value,
            "filled_quantity": str(self.filled_quantity),
            "avg_fill_price": str(self.avg_fill_price) if self.avg_fill_price else None,
            "broker": self.broker,
            "broker_order_id": self.broker_order_id,
            "parent_order_id": self.parent_order_id,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "fills": [f.to_dict() for f in self.fills],
            "tags": self.tags,
            "strategy_id": self.strategy_id,
            "account_id": self.account_id,
            "paper_trading": self.paper_trading
        }


class OrderManager:
    """
    Centralized order management system

    Handles:
    - Order creation and validation
    - Order submission and cancellation
    - Order state tracking
    - Fill management
    - Order book management
    - Order history
    """

    def __init__(
        self,
        broker_adapter=None,
        fill_simulator=None,
        order_validators: Optional[List[Callable]] = None,
        max_history_size: int = 10000
    ):
        """
        Initialize order manager

        Args:
            broker_adapter: Broker adapter for order execution
            fill_simulator: Fill simulator for paper trading
            order_validators: List of validation functions
            max_history_size: Maximum number of orders to keep in history
        """
        self.broker_adapter = broker_adapter
        self.fill_simulator = fill_simulator
        self.order_validators = order_validators or []
        self.max_history_size = max_history_size

        # Order storage
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []

        # Indices for fast lookup
        self._orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self._orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        self._orders_by_strategy: Dict[str, List[str]] = defaultdict(list)

        # Event callbacks
        self._order_callbacks: List[Callable] = []
        self._fill_callbacks: List[Callable] = []

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("OrderManager initialized")

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        strategy_id: Optional[str] = None,
        account_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        paper_trading: bool = False,
        **kwargs
    ) -> Order:
        """
        Create a new order

        Args:
            symbol: Trading symbol
            side: Buy or sell
            order_type: Type of order
            quantity: Quantity to trade
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            strategy_id: Strategy identifier
            account_id: Account identifier
            tags: Additional metadata
            paper_trading: Whether this is a paper trading order
            **kwargs: Additional order parameters

        Returns:
            Created Order object

        Raises:
            ValueError: If order validation fails
        """
        async with self._lock:
            # Generate order ID
            order_id = self._generate_order_id()

            # Create order object
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=Decimal(str(quantity)),
                limit_price=Decimal(str(limit_price)) if limit_price else None,
                stop_price=Decimal(str(stop_price)) if stop_price else None,
                time_in_force=time_in_force,
                strategy_id=strategy_id,
                account_id=account_id,
                tags=tags or {},
                paper_trading=paper_trading
            )

            # Validate order
            await self._validate_order(order)

            # Store order
            self._add_order(order)

            logger.info(f"Created order: {order_id} {side.value} {quantity} {symbol} @ {order_type.value}")

            # Trigger callbacks
            await self._trigger_order_callbacks(order, "created")

            return order

    async def submit_order(self, order_id: str) -> Order:
        """
        Submit an order to the broker

        Args:
            order_id: Order ID to submit

        Returns:
            Updated Order object

        Raises:
            ValueError: If order cannot be submitted
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            if order.status != OrderStatus.PENDING:
                raise ValueError(f"Order {order_id} is not in PENDING state")

            try:
                # Submit to broker or simulator
                if order.paper_trading and self.fill_simulator:
                    # Paper trading - use simulator
                    broker_order_id = await self.fill_simulator.submit_order(order)
                elif self.broker_adapter:
                    # Live trading - use broker
                    broker_order_id = await self.broker_adapter.submit_order(order)
                else:
                    raise ValueError("No broker adapter or fill simulator configured")

                # Update order
                order.broker_order_id = broker_order_id
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now(timezone.utc)

                # Update indices
                self._update_order_index(order, OrderStatus.PENDING, OrderStatus.SUBMITTED)

                logger.info(f"Submitted order: {order_id} -> broker order: {broker_order_id}")

                # Trigger callbacks
                await self._trigger_order_callbacks(order, "submitted")

                return order

            except Exception as e:
                logger.error(f"Failed to submit order {order_id}: {e}")
                order.status = OrderStatus.REJECTED
                self._update_order_index(order, OrderStatus.PENDING, OrderStatus.REJECTED)
                await self._trigger_order_callbacks(order, "rejected")
                raise

    async def cancel_order(self, order_id: str, reason: Optional[str] = None) -> Order:
        """
        Cancel an order

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            Updated Order object

        Raises:
            ValueError: If order cannot be cancelled
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            if not order.is_active:
                raise ValueError(f"Order {order_id} is not active")

            try:
                # Cancel with broker or simulator
                if order.paper_trading and self.fill_simulator:
                    await self.fill_simulator.cancel_order(order.broker_order_id)
                elif self.broker_adapter:
                    await self.broker_adapter.cancel_order(order.broker_order_id)

                # Update order
                old_status = order.status
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now(timezone.utc)
                if reason:
                    order.tags["cancellation_reason"] = reason

                # Update indices
                self._update_order_index(order, old_status, OrderStatus.CANCELLED)

                logger.info(f"Cancelled order: {order_id}" + (f" - {reason}" if reason else ""))

                # Trigger callbacks
                await self._trigger_order_callbacks(order, "cancelled")

                return order

            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                raise

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        broker_order_id: Optional[str] = None
    ) -> Order:
        """
        Update order status (typically called by broker adapter)

        Args:
            order_id: Order ID
            status: New status
            broker_order_id: Broker order ID

        Returns:
            Updated Order object
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            old_status = order.status
            order.status = status

            if broker_order_id:
                order.broker_order_id = broker_order_id

            # Update timestamps
            now = datetime.now(timezone.utc)
            if status == OrderStatus.ACCEPTED:
                order.accepted_at = now
            elif status == OrderStatus.FILLED:
                order.filled_at = now
            elif status == OrderStatus.CANCELLED:
                order.cancelled_at = now

            # Update indices
            self._update_order_index(order, old_status, status)

            logger.debug(f"Updated order {order_id} status: {old_status.value} -> {status.value}")

            # Trigger callbacks
            await self._trigger_order_callbacks(order, "status_updated")

            return order

    async def add_fill(
        self,
        order_id: str,
        fill_id: str,
        price: Decimal,
        quantity: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Decimal = Decimal("0"),
        exchange: Optional[str] = None,
        liquidity_flag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Fill:
        """
        Add a fill to an order

        Args:
            order_id: Order ID
            fill_id: Fill identifier
            price: Fill price
            quantity: Fill quantity
            timestamp: Fill timestamp
            commission: Commission charged
            exchange: Exchange where fill occurred
            liquidity_flag: "added" or "removed"
            metadata: Additional fill metadata

        Returns:
            Created Fill object
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            # Create fill
            fill = Fill(
                fill_id=fill_id,
                order_id=order_id,
                timestamp=timestamp or datetime.now(timezone.utc),
                price=Decimal(str(price)),
                quantity=Decimal(str(quantity)),
                commission=Decimal(str(commission)),
                exchange=exchange,
                liquidity_flag=liquidity_flag,
                metadata=metadata or {}
            )

            # Update order
            old_status = order.status
            order.add_fill(fill)

            # Update indices if status changed
            if old_status != order.status:
                self._update_order_index(order, old_status, order.status)

            logger.info(
                f"Fill added to order {order_id}: {quantity} @ {price} "
                f"({order.filled_quantity}/{order.quantity})"
            )

            # Trigger callbacks
            await self._trigger_fill_callbacks(order, fill)
            await self._trigger_order_callbacks(order, "fill_added")

            return fill

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with a specific status"""
        order_ids = self._orders_by_status.get(status, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        active = []
        for status in [OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIAL_FILL]:
            active.extend(self.get_orders_by_status(status))
        return active

    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a strategy"""
        order_ids = self._orders_by_strategy.get(strategy_id, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_order_history(self, limit: Optional[int] = None) -> List[Order]:
        """Get order history"""
        if limit:
            return self._order_history[-limit:]
        return self._order_history.copy()

    def register_order_callback(self, callback: Callable) -> None:
        """Register a callback for order events"""
        self._order_callbacks.append(callback)

    def register_fill_callback(self, callback: Callable) -> None:
        """Register a callback for fill events"""
        self._fill_callbacks.append(callback)

    def _generate_order_id(self) -> str:
        """Generate a unique order ID"""
        return f"ORD-{uuid.uuid4().hex[:12].upper()}"

    def _add_order(self, order: Order) -> None:
        """Add order to storage and indices"""
        self._orders[order.order_id] = order

        # Update indices
        self._orders_by_symbol[order.symbol].append(order.order_id)
        self._orders_by_status[order.status].append(order.order_id)
        if order.strategy_id:
            self._orders_by_strategy[order.strategy_id].append(order.order_id)

        # Add to history when terminal
        if order.is_terminal:
            self._add_to_history(order)

    def _update_order_index(
        self,
        order: Order,
        old_status: OrderStatus,
        new_status: OrderStatus
    ) -> None:
        """Update order status index"""
        # Remove from old status
        if order.order_id in self._orders_by_status[old_status]:
            self._orders_by_status[old_status].remove(order.order_id)

        # Add to new status
        if order.order_id not in self._orders_by_status[new_status]:
            self._orders_by_status[new_status].append(order.order_id)

        # Add to history if now terminal
        if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._add_to_history(order)

    def _add_to_history(self, order: Order) -> None:
        """Add order to history"""
        if order not in self._order_history:
            self._order_history.append(order)

            # Trim history if needed
            if len(self._order_history) > self.max_history_size:
                self._order_history = self._order_history[-self.max_history_size:]

    async def _validate_order(self, order: Order) -> None:
        """
        Validate order before creation

        Raises:
            ValueError: If validation fails
        """
        # Basic validation
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.LOC]:
            if not order.limit_price or order.limit_price <= 0:
                raise ValueError(f"{order.order_type.value} orders require a positive limit price")

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
            if not order.stop_price or order.stop_price <= 0:
                raise ValueError(f"{order.order_type.value} orders require a positive stop price")

        # Run custom validators
        for validator in self.order_validators:
            await validator(order)

    async def _trigger_order_callbacks(self, order: Order, event_type: str) -> None:
        """Trigger order event callbacks"""
        for callback in self._order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, event_type)
                else:
                    callback(order, event_type)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    async def _trigger_fill_callbacks(self, order: Order, fill: Fill) -> None:
        """Trigger fill event callbacks"""
        for callback in self._fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, fill)
                else:
                    callback(order, fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics"""
        return {
            "total_orders": len(self._orders),
            "active_orders": len(self.get_active_orders()),
            "orders_by_status": {
                status.value: len(self.get_orders_by_status(status))
                for status in OrderStatus
            },
            "symbols_traded": len(self._orders_by_symbol),
            "history_size": len(self._order_history)
        }
