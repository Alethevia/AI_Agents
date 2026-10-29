"""
Order Fill Simulator

This module provides realistic order fill simulation for paper trading:
- Market order fills
- Limit order fills with queue position
- Partial fills
- Realistic latency simulation
- Fill probability estimation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import numpy as np
import uuid


logger = logging.getLogger(__name__)


class FillMode(Enum):
    """Fill simulation modes"""
    INSTANT = "instant"  # Instant fills (unrealistic)
    REALISTIC = "realistic"  # Realistic fills with latency
    CONSERVATIVE = "conservative"  # Conservative fills (harder to fill)


@dataclass
class SimulatedOrder:
    """Order being simulated"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "day"
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Simulation state
    queue_position: Optional[int] = None
    queue_size: Optional[int] = None
    fill_probability: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.quantity = Decimal(str(self.quantity))
        self.filled_quantity = Decimal(str(self.filled_quantity))
        if self.limit_price:
            self.limit_price = Decimal(str(self.limit_price))
        if self.stop_price:
            self.stop_price = Decimal(str(self.stop_price))

    @property
    def remaining_quantity(self) -> Decimal:
        """Remaining quantity to fill"""
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.filled_quantity >= self.quantity


@dataclass
class SimulatedFill:
    """Simulated fill event"""
    fill_id: str
    order_id: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    commission: Decimal = Decimal("0")
    liquidity_flag: str = "removed"  # "added" or "removed"

    def __post_init__(self):
        self.price = Decimal(str(self.price))
        self.quantity = Decimal(str(self.quantity))
        self.commission = Decimal(str(self.commission))


@dataclass
class MarketSnapshot:
    """Market data snapshot for simulation"""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    last_price: Decimal
    volume: Decimal

    def __post_init__(self):
        self.bid = Decimal(str(self.bid))
        self.ask = Decimal(str(self.ask))
        self.bid_size = Decimal(str(self.bid_size))
        self.ask_size = Decimal(str(self.ask_size))
        self.last_price = Decimal(str(self.last_price))
        self.volume = Decimal(str(self.volume))

    @property
    def spread(self) -> Decimal:
        """Bid-ask spread"""
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal:
        """Mid price"""
        return (self.bid + self.ask) / 2


class FillSimulator:
    """
    Order fill simulator for paper trading

    Provides realistic fill simulation including:
    - Market impact
    - Queue position for limit orders
    - Partial fills
    - Fill latency
    - Fill probability
    """

    def __init__(
        self,
        mode: FillMode = FillMode.REALISTIC,
        base_latency_ms: float = 50.0,
        latency_std_ms: float = 20.0,
        slippage_model=None,
        commission_per_share: Decimal = Decimal("0"),
        commission_rate: Decimal = Decimal("0"),
        partial_fill_probability: float = 0.1,
        min_partial_fill_ratio: float = 0.3
    ):
        """
        Initialize fill simulator

        Args:
            mode: Fill simulation mode
            base_latency_ms: Base latency in milliseconds
            latency_std_ms: Latency standard deviation
            slippage_model: Slippage estimation model
            commission_per_share: Commission per share
            commission_rate: Commission as percentage of notional
            partial_fill_probability: Probability of partial fill
            min_partial_fill_ratio: Minimum ratio for partial fills
        """
        self.mode = mode
        self.base_latency_ms = base_latency_ms
        self.latency_std_ms = latency_std_ms
        self.slippage_model = slippage_model
        self.commission_per_share = Decimal(str(commission_per_share))
        self.commission_rate = Decimal(str(commission_rate))
        self.partial_fill_probability = partial_fill_probability
        self.min_partial_fill_ratio = min_partial_fill_ratio

        # Active orders
        self._active_orders: Dict[str, SimulatedOrder] = {}

        # Market data provider (can be set externally)
        self.market_data_provider: Optional[Callable] = None

        # Fill callbacks
        self._fill_callbacks: List[Callable] = []

        logger.info(f"FillSimulator initialized with mode={mode.value}")

    async def submit_order(self, order: Any) -> str:
        """
        Submit an order for simulation

        Args:
            order: Order object from order manager

        Returns:
            Simulated broker order ID
        """
        sim_order_id = f"SIM-{uuid.uuid4().hex[:12].upper()}"

        sim_order = SimulatedOrder(
            order_id=sim_order_id,
            symbol=order.symbol,
            side=order.side.value if hasattr(order.side, 'value') else order.side,
            order_type=order.order_type.value if hasattr(order.order_type, 'value') else order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force.value if hasattr(order.time_in_force, 'value') else order.time_in_force,
            metadata={"original_order_id": order.order_id}
        )

        self._active_orders[sim_order_id] = sim_order

        # Start simulation for this order
        asyncio.create_task(self._simulate_order(sim_order))

        logger.info(f"Submitted simulated order: {sim_order_id}")
        return sim_order_id

    async def cancel_order(self, sim_order_id: str) -> bool:
        """
        Cancel a simulated order

        Args:
            sim_order_id: Simulated order ID

        Returns:
            True if cancelled
        """
        if sim_order_id in self._active_orders:
            del self._active_orders[sim_order_id]
            logger.info(f"Cancelled simulated order: {sim_order_id}")
            return True
        return False

    async def _simulate_order(self, order: SimulatedOrder):
        """
        Simulate order execution

        Args:
            order: Simulated order
        """
        try:
            # Simulate latency
            await self._simulate_latency()

            # Get market data
            market = await self._get_market_data(order.symbol)
            if not market:
                logger.warning(f"No market data for {order.symbol}, using fallback")
                market = self._generate_fallback_market_data(order.symbol)

            # Simulate based on order type
            if order.order_type.lower() == "market":
                await self._simulate_market_order(order, market)
            elif order.order_type.lower() in ["limit", "stop_limit"]:
                await self._simulate_limit_order(order, market)
            elif order.order_type.lower() == "stop":
                await self._simulate_stop_order(order, market)
            else:
                logger.warning(f"Unsupported order type: {order.order_type}")

        except Exception as e:
            logger.error(f"Error simulating order {order.order_id}: {e}")

    async def _simulate_market_order(
        self,
        order: SimulatedOrder,
        market: MarketSnapshot
    ):
        """Simulate market order fill"""
        # Market orders fill immediately (with slippage)

        # Determine fill price
        if order.side.lower() == "buy":
            base_price = market.ask
            available_liquidity = market.ask_size
        else:
            base_price = market.bid
            available_liquidity = market.bid_size

        # Apply slippage
        if self.slippage_model:
            from ..slippage.slippage_model import MarketState
            market_state = MarketState(
                symbol=order.symbol,
                timestamp=market.timestamp,
                price=market.last_price,
                bid=market.bid,
                ask=market.ask,
                volume=market.volume,
                avg_daily_volume=market.volume * 390,
                volatility=0.02,
                spread_bps=float(market.spread / market.mid_price * 10000)
            )
            slippage_estimate = self.slippage_model.estimate(
                market_state,
                order.side,
                order.quantity,
                urgency=1.0  # Market orders are aggressive
            )
            fill_price = slippage_estimate.estimated_price
        else:
            fill_price = base_price

        # Check for partial fill
        fill_quantity = order.quantity
        if self.mode == FillMode.CONSERVATIVE:
            # Conservative mode: may not fill entire order immediately
            if order.quantity > available_liquidity * Decimal("0.5"):
                if np.random.random() < self.partial_fill_probability:
                    fill_ratio = np.random.uniform(
                        self.min_partial_fill_ratio,
                        1.0
                    )
                    fill_quantity = order.quantity * Decimal(str(fill_ratio))

        # Create fill
        await self._create_fill(order, fill_price, fill_quantity, market.timestamp)

        # If not fully filled, schedule another fill attempt
        if not order.is_filled and self.mode == FillMode.CONSERVATIVE:
            asyncio.create_task(self._schedule_additional_fill(order, market))

    async def _simulate_limit_order(
        self,
        order: SimulatedOrder,
        market: MarketSnapshot
    ):
        """Simulate limit order fill with queue position"""
        if not order.limit_price:
            logger.warning(f"Limit order {order.order_id} has no limit price")
            return

        # Check if limit price is marketable
        if order.side.lower() == "buy":
            is_marketable = order.limit_price >= market.ask
            reference_size = market.ask_size
        else:
            is_marketable = order.limit_price <= market.bid
            reference_size = market.bid_size

        if is_marketable:
            # Marketable limit order - treat as aggressive market order
            if self.mode == FillMode.INSTANT:
                fill_price = order.limit_price
            else:
                # Still might not fill at limit in realistic mode
                if order.side.lower() == "buy":
                    fill_price = min(order.limit_price, market.ask)
                else:
                    fill_price = max(order.limit_price, market.bid)

            # Calculate fill probability based on how deep in the book
            if self.mode == FillMode.CONSERVATIVE:
                # May not fill entire order
                available_at_limit = reference_size * Decimal("0.7")
                if order.quantity > available_at_limit:
                    fill_quantity = min(order.quantity, available_at_limit)
                else:
                    fill_quantity = order.quantity
            else:
                fill_quantity = order.quantity

            await self._create_fill(order, fill_price, fill_quantity, market.timestamp)

            # Schedule additional fills if needed
            if not order.is_filled:
                asyncio.create_task(self._schedule_additional_fill(order, market))
        else:
            # Non-marketable limit order - need to wait in queue
            await self._simulate_limit_queue(order, market)

    async def _simulate_limit_queue(
        self,
        order: SimulatedOrder,
        market: MarketSnapshot
    ):
        """
        Simulate limit order waiting in queue

        Models queue position and fill probability based on market flow
        """
        # Estimate queue position
        if order.side.lower() == "buy":
            depth_at_price = market.bid_size * Decimal("2")
        else:
            depth_at_price = market.ask_size * Decimal("2")

        # Random queue position
        queue_size = int(depth_at_price * Decimal(np.random.uniform(0.5, 1.5)))
        queue_position = np.random.randint(0, max(1, queue_size))

        order.queue_size = queue_size
        order.queue_position = queue_position

        # Calculate fill probability based on queue position
        if queue_size > 0:
            position_ratio = queue_position / queue_size
            # Better position = higher fill probability
            base_fill_prob = 1.0 - position_ratio
        else:
            base_fill_prob = 0.5

        # Adjust for mode
        if self.mode == FillMode.INSTANT:
            fill_prob = 1.0
        elif self.mode == FillMode.CONSERVATIVE:
            fill_prob = base_fill_prob * 0.5
        else:
            fill_prob = base_fill_prob * 0.8

        order.fill_probability = fill_prob

        # Simulate waiting and periodic fill checks
        asyncio.create_task(self._monitor_limit_order(order, market))

    async def _monitor_limit_order(
        self,
        order: SimulatedOrder,
        initial_market: MarketSnapshot
    ):
        """
        Monitor limit order and simulate fills as market moves

        Args:
            order: Simulated order
            initial_market: Initial market snapshot
        """
        check_interval = 1.0 if self.mode == FillMode.REALISTIC else 5.0

        while order.order_id in self._active_orders and not order.is_filled:
            await asyncio.sleep(check_interval)

            # Get updated market data
            market = await self._get_market_data(order.symbol)
            if not market:
                continue

            # Check if price has reached limit
            if order.side.lower() == "buy":
                price_touched = market.ask <= order.limit_price
            else:
                price_touched = market.bid >= order.limit_price

            if price_touched:
                # Simulate fill probability
                fill_chance = np.random.random()

                if fill_chance < order.fill_probability:
                    # Fill order
                    fill_quantity = order.remaining_quantity

                    # Might be partial fill
                    if self.mode == FillMode.CONSERVATIVE and np.random.random() < 0.3:
                        fill_ratio = np.random.uniform(0.3, 0.8)
                        fill_quantity = order.remaining_quantity * Decimal(str(fill_ratio))

                    await self._create_fill(
                        order,
                        order.limit_price,
                        fill_quantity,
                        market.timestamp
                    )

                    if not order.is_filled:
                        # Update queue position (moved up in queue)
                        if order.queue_position and order.queue_position > 0:
                            order.queue_position = max(0, order.queue_position - 10)
                            if order.queue_size:
                                order.fill_probability = 1.0 - (order.queue_position / order.queue_size)

    async def _simulate_stop_order(
        self,
        order: SimulatedOrder,
        market: MarketSnapshot
    ):
        """Simulate stop order"""
        if not order.stop_price:
            logger.warning(f"Stop order {order.order_id} has no stop price")
            return

        # Monitor until stop is triggered
        asyncio.create_task(self._monitor_stop_order(order, market))

    async def _monitor_stop_order(
        self,
        order: SimulatedOrder,
        initial_market: MarketSnapshot
    ):
        """Monitor stop order for trigger"""
        check_interval = 0.5

        while order.order_id in self._active_orders and not order.is_filled:
            await asyncio.sleep(check_interval)

            market = await self._get_market_data(order.symbol)
            if not market:
                continue

            # Check if stop is triggered
            triggered = False
            if order.side.lower() == "buy":
                # Buy stop: trigger when price rises above stop
                if market.last_price >= order.stop_price:
                    triggered = True
            else:
                # Sell stop: trigger when price falls below stop
                if market.last_price <= order.stop_price:
                    triggered = True

            if triggered:
                # Convert to market order
                logger.info(f"Stop triggered for order {order.order_id}")

                # Fill at market price (with slippage)
                if order.side.lower() == "buy":
                    fill_price = market.ask * Decimal("1.001")  # Slippage
                else:
                    fill_price = market.bid * Decimal("0.999")  # Slippage

                await self._create_fill(
                    order,
                    fill_price,
                    order.quantity,
                    market.timestamp
                )
                break

    async def _schedule_additional_fill(
        self,
        order: SimulatedOrder,
        market: MarketSnapshot
    ):
        """Schedule an additional fill attempt for partial fills"""
        if order.is_filled or order.order_id not in self._active_orders:
            return

        # Wait a bit
        await asyncio.sleep(np.random.uniform(0.5, 2.0))

        # Try to fill remaining
        remaining = order.remaining_quantity
        fill_ratio = np.random.uniform(0.5, 1.0)
        fill_quantity = remaining * Decimal(str(fill_ratio))

        if order.side.lower() == "buy":
            fill_price = market.ask
        else:
            fill_price = market.bid

        await self._create_fill(order, fill_price, fill_quantity, datetime.now(timezone.utc))

    async def _create_fill(
        self,
        order: SimulatedOrder,
        price: Decimal,
        quantity: Decimal,
        timestamp: datetime
    ):
        """Create a fill for an order"""
        # Round quantity to reasonable precision
        quantity = Decimal(str(quantity)).quantize(Decimal("0.000001"))

        # Don't overfill
        quantity = min(quantity, order.remaining_quantity)

        if quantity <= 0:
            return

        # Calculate commission
        notional = price * quantity
        commission = (
            quantity * self.commission_per_share +
            notional * self.commission_rate
        )

        # Create fill
        fill = SimulatedFill(
            fill_id=f"FILL-{uuid.uuid4().hex[:12].upper()}",
            order_id=order.order_id,
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            commission=commission,
            liquidity_flag="removed" if order.order_type.lower() == "market" else "added"
        )

        # Update order
        order.filled_quantity += quantity

        logger.info(
            f"Simulated fill: {order.order_id} {quantity} @ {price} "
            f"({order.filled_quantity}/{order.quantity})"
        )

        # Trigger callbacks
        await self._trigger_fill_callbacks(order, fill)

        # Remove from active orders if fully filled
        if order.is_filled:
            if order.order_id in self._active_orders:
                del self._active_orders[order.order_id]

    async def _simulate_latency(self):
        """Simulate execution latency"""
        if self.mode == FillMode.INSTANT:
            return

        # Generate random latency
        latency = max(
            0,
            np.random.normal(self.base_latency_ms, self.latency_std_ms)
        )

        await asyncio.sleep(latency / 1000.0)

    async def _get_market_data(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get market data for a symbol"""
        if self.market_data_provider:
            try:
                data = await self.market_data_provider(symbol)
                if data:
                    return MarketSnapshot(**data)
            except Exception as e:
                logger.error(f"Error getting market data: {e}")

        return None

    def _generate_fallback_market_data(self, symbol: str) -> MarketSnapshot:
        """Generate fallback market data when real data unavailable"""
        # Use a reasonable default
        base_price = Decimal("100.0")
        spread = base_price * Decimal("0.001")  # 10 bps spread

        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bid=base_price - spread / 2,
            ask=base_price + spread / 2,
            bid_size=Decimal("1000"),
            ask_size=Decimal("1000"),
            last_price=base_price,
            volume=Decimal("100000")
        )

    def register_fill_callback(self, callback: Callable):
        """Register callback for fill events"""
        self._fill_callbacks.append(callback)

    async def _trigger_fill_callbacks(
        self,
        order: SimulatedOrder,
        fill: SimulatedFill
    ):
        """Trigger fill callbacks"""
        for callback in self._fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, fill)
                else:
                    callback(order, fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")

    def set_market_data_provider(self, provider: Callable):
        """Set market data provider function"""
        self.market_data_provider = provider

    def get_active_orders(self) -> List[SimulatedOrder]:
        """Get list of active orders"""
        return list(self._active_orders.values())

    def get_order(self, sim_order_id: str) -> Optional[SimulatedOrder]:
        """Get simulated order by ID"""
        return self._active_orders.get(sim_order_id)


class RealisticFillSimulator(FillSimulator):
    """
    Realistic fill simulator with advanced features

    Includes:
    - Order book simulation
    - Queue position modeling
    - Adverse selection
    - Information leakage
    """

    def __init__(
        self,
        adverse_selection_factor: float = 0.1,
        information_leakage_factor: float = 0.05,
        **kwargs
    ):
        """
        Initialize realistic fill simulator

        Args:
            adverse_selection_factor: Factor for adverse selection
            information_leakage_factor: Factor for information leakage
            **kwargs: Arguments for base FillSimulator
        """
        super().__init__(mode=FillMode.REALISTIC, **kwargs)
        self.adverse_selection_factor = adverse_selection_factor
        self.information_leakage_factor = information_leakage_factor

    async def _simulate_market_order(
        self,
        order: SimulatedOrder,
        market: MarketSnapshot
    ):
        """Enhanced market order simulation with adverse selection"""
        # Apply adverse selection
        # Market orders are more likely to fill when price is moving against us

        if order.side.lower() == "buy":
            # When buying, price may already be moving up
            adverse_move = market.spread * Decimal(str(self.adverse_selection_factor))
            fill_price = market.ask + adverse_move
        else:
            # When selling, price may already be moving down
            adverse_move = market.spread * Decimal(str(self.adverse_selection_factor))
            fill_price = market.bid - adverse_move

        # Apply slippage model if available
        if self.slippage_model:
            from ..slippage.slippage_model import MarketState
            market_state = MarketState(
                symbol=order.symbol,
                timestamp=market.timestamp,
                price=market.last_price,
                bid=market.bid,
                ask=market.ask,
                volume=market.volume,
                avg_daily_volume=market.volume * 390,
                volatility=0.02,
                spread_bps=float(market.spread / market.mid_price * 10000)
            )
            slippage_estimate = self.slippage_model.estimate(
                market_state,
                order.side,
                order.quantity,
                urgency=1.0
            )
            fill_price = slippage_estimate.estimated_price

        await self._create_fill(order, fill_price, order.quantity, market.timestamp)
