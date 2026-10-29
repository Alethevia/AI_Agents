"""
Smart Order Splitting Algorithms

This module provides sophisticated order splitting strategies including:
- Time-Weighted Average Price (TWAP)
- Volume-Weighted Average Price (VWAP)
- Implementation Shortfall
- Percentage of Volume (POV)
- Adaptive algorithms
"""

import asyncio
import logging
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from decimal import Decimal
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum


logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    """Order splitting strategies"""
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"


@dataclass
class ChildOrder:
    """Represents a child order created by splitting"""
    parent_order_id: str
    sequence_number: int
    symbol: str
    side: str
    quantity: Decimal
    scheduled_time: datetime
    limit_price: Optional[Decimal] = None
    urgency: float = 0.5  # 0 = passive, 1 = aggressive
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.quantity = Decimal(str(self.quantity))
        if self.limit_price is not None:
            self.limit_price = Decimal(str(self.limit_price))


@dataclass
class MarketState:
    """Current market state for adaptive algorithms"""
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    spread: Decimal
    volatility: float
    participation_rate: float = 0.0


class OrderSplitter(ABC):
    """Base class for order splitting algorithms"""

    def __init__(self, max_participation_rate: float = 0.3):
        """
        Initialize order splitter

        Args:
            max_participation_rate: Maximum participation in market volume
        """
        self.max_participation_rate = max_participation_rate

    @abstractmethod
    async def split_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """
        Split an order into child orders

        Args:
            order_id: Parent order ID
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to split
            start_time: Start time for execution
            end_time: End time for execution
            market_data: Market data for informed splitting

        Returns:
            List of child orders
        """
        pass


class TWAPSplitter(OrderSplitter):
    """
    Time-Weighted Average Price (TWAP) splitter

    Splits orders evenly across time periods
    """

    def __init__(
        self,
        slice_interval_seconds: int = 300,  # 5 minutes
        randomize_size: bool = True,
        randomize_timing: bool = True,
        **kwargs
    ):
        """
        Initialize TWAP splitter

        Args:
            slice_interval_seconds: Time interval between slices
            randomize_size: Add randomness to slice sizes
            randomize_timing: Add randomness to timing
        """
        super().__init__(**kwargs)
        self.slice_interval_seconds = slice_interval_seconds
        self.randomize_size = randomize_size
        self.randomize_timing = randomize_timing

    async def split_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """Split order using TWAP strategy"""
        total_duration = (end_time - start_time).total_seconds()
        num_slices = max(1, int(total_duration / self.slice_interval_seconds))

        child_orders = []

        if self.randomize_size:
            # Generate random weights that sum to 1
            weights = np.random.dirichlet(np.ones(num_slices))
        else:
            # Equal weights
            weights = np.ones(num_slices) / num_slices

        cumulative_quantity = Decimal("0")
        for i in range(num_slices):
            # Calculate quantity for this slice
            if i == num_slices - 1:
                # Last slice gets remaining quantity
                slice_quantity = total_quantity - cumulative_quantity
            else:
                slice_quantity = Decimal(str(float(total_quantity) * weights[i]))
                cumulative_quantity += slice_quantity

            # Calculate scheduled time
            if self.randomize_timing:
                # Add random jitter (±20% of interval)
                jitter = np.random.uniform(-0.2, 0.2) * self.slice_interval_seconds
                scheduled_seconds = i * self.slice_interval_seconds + jitter
            else:
                scheduled_seconds = i * self.slice_interval_seconds

            scheduled_time = start_time + timedelta(seconds=scheduled_seconds)

            child_orders.append(ChildOrder(
                parent_order_id=order_id,
                sequence_number=i,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                scheduled_time=scheduled_time,
                urgency=0.5,  # Neutral urgency for TWAP
                metadata={"strategy": "TWAP", "total_slices": num_slices}
            ))

        logger.info(f"TWAP split order {order_id} into {num_slices} slices")
        return child_orders


class VWAPSplitter(OrderSplitter):
    """
    Volume-Weighted Average Price (VWAP) splitter

    Splits orders based on historical volume profile
    """

    def __init__(
        self,
        historical_volume_profile: Optional[Dict[int, float]] = None,
        slice_interval_seconds: int = 300,
        **kwargs
    ):
        """
        Initialize VWAP splitter

        Args:
            historical_volume_profile: Historical volume by time bucket (minute -> volume %)
            slice_interval_seconds: Time interval between slices
        """
        super().__init__(**kwargs)
        self.slice_interval_seconds = slice_interval_seconds

        # Default intraday volume profile (U-shaped: high at open/close)
        if historical_volume_profile is None:
            self.volume_profile = self._default_volume_profile()
        else:
            self.volume_profile = historical_volume_profile

    def _default_volume_profile(self) -> Dict[int, float]:
        """
        Generate default U-shaped intraday volume profile

        Returns higher volume at market open and close
        """
        # Market hours: 9:30 AM - 4:00 PM (390 minutes)
        profile = {}
        total_minutes = 390

        for minute in range(total_minutes):
            # U-shaped curve: high at open (minute 0) and close (minute 389)
            normalized_minute = minute / total_minutes
            # Quadratic U-shape
            volume_factor = 1.0 + 2.0 * (0.5 - abs(normalized_minute - 0.5)) ** 2
            profile[minute] = volume_factor

        # Normalize to sum to 1
        total = sum(profile.values())
        profile = {k: v / total for k, v in profile.items()}

        return profile

    async def split_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """Split order using VWAP strategy"""
        total_duration = (end_time - start_time).total_seconds()
        num_slices = max(1, int(total_duration / self.slice_interval_seconds))

        # Get volume weights for each slice based on time of day
        weights = []
        for i in range(num_slices):
            slice_time = start_time + timedelta(seconds=i * self.slice_interval_seconds)
            # Convert to minutes since market open (9:30 AM)
            market_open = slice_time.replace(hour=9, minute=30, second=0)
            minutes_since_open = int((slice_time - market_open).total_seconds() / 60)
            minutes_since_open = max(0, min(389, minutes_since_open))  # Clamp to market hours

            weight = self.volume_profile.get(minutes_since_open, 1.0 / num_slices)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Create child orders
        child_orders = []
        cumulative_quantity = Decimal("0")

        for i in range(num_slices):
            # Calculate quantity for this slice
            if i == num_slices - 1:
                slice_quantity = total_quantity - cumulative_quantity
            else:
                slice_quantity = Decimal(str(float(total_quantity) * weights[i]))
                cumulative_quantity += slice_quantity

            scheduled_time = start_time + timedelta(seconds=i * self.slice_interval_seconds)

            # Higher urgency during high-volume periods
            urgency = min(1.0, weights[i] * num_slices)

            child_orders.append(ChildOrder(
                parent_order_id=order_id,
                sequence_number=i,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                scheduled_time=scheduled_time,
                urgency=urgency,
                metadata={"strategy": "VWAP", "volume_weight": weights[i]}
            ))

        logger.info(f"VWAP split order {order_id} into {num_slices} slices")
        return child_orders


class POVSplitter(OrderSplitter):
    """
    Percentage of Volume (POV) splitter

    Splits orders to maintain a target participation rate
    """

    def __init__(
        self,
        target_participation_rate: float = 0.1,
        slice_interval_seconds: int = 60,
        **kwargs
    ):
        """
        Initialize POV splitter

        Args:
            target_participation_rate: Target % of market volume (e.g., 0.1 = 10%)
            slice_interval_seconds: Time interval for volume sampling
        """
        super().__init__(**kwargs)
        self.target_participation_rate = min(
            target_participation_rate,
            self.max_participation_rate
        )
        self.slice_interval_seconds = slice_interval_seconds

    async def split_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """
        Split order using POV strategy

        Note: POV requires real-time volume data. This implementation
        uses estimated volume based on historical patterns.
        """
        total_duration = (end_time - start_time).total_seconds()
        num_slices = max(1, int(total_duration / self.slice_interval_seconds))

        # Estimate market volume per slice
        # In production, this would come from real-time market data
        estimated_daily_volume = market_data.get("avg_daily_volume", 1000000) if market_data else 1000000
        estimated_volume_per_minute = estimated_daily_volume / 390  # Trading minutes per day
        estimated_slice_volume = estimated_volume_per_minute * (self.slice_interval_seconds / 60)

        # Calculate target quantity per slice
        target_slice_quantity = Decimal(str(estimated_slice_volume * self.target_participation_rate))

        child_orders = []
        cumulative_quantity = Decimal("0")

        for i in range(num_slices):
            # Don't exceed remaining quantity
            remaining_quantity = total_quantity - cumulative_quantity
            if remaining_quantity <= 0:
                break

            slice_quantity = min(target_slice_quantity, remaining_quantity)
            cumulative_quantity += slice_quantity

            scheduled_time = start_time + timedelta(seconds=i * self.slice_interval_seconds)

            child_orders.append(ChildOrder(
                parent_order_id=order_id,
                sequence_number=i,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                scheduled_time=scheduled_time,
                urgency=0.6,  # Slightly aggressive to maintain participation
                metadata={
                    "strategy": "POV",
                    "target_participation": self.target_participation_rate
                }
            ))

            # Stop if we've allocated all quantity
            if cumulative_quantity >= total_quantity:
                break

        logger.info(
            f"POV split order {order_id} into {len(child_orders)} slices "
            f"(target participation: {self.target_participation_rate:.1%})"
        )
        return child_orders


class ImplementationShortfallSplitter(OrderSplitter):
    """
    Implementation Shortfall splitter

    Minimizes the difference between decision price and execution price
    Balances market impact and timing risk
    """

    def __init__(
        self,
        risk_aversion: float = 0.5,
        volatility_scaling: bool = True,
        **kwargs
    ):
        """
        Initialize Implementation Shortfall splitter

        Args:
            risk_aversion: Risk aversion parameter (0 = aggressive, 1 = passive)
            volatility_scaling: Adjust urgency based on volatility
        """
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        self.volatility_scaling = volatility_scaling

    async def split_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """Split order using Implementation Shortfall strategy"""
        total_duration = (end_time - start_time).total_seconds()

        # Get market parameters
        volatility = market_data.get("volatility", 0.02) if market_data else 0.02
        avg_daily_volume = market_data.get("avg_daily_volume", 1000000) if market_data else 1000000

        # Calculate optimal execution trajectory
        # Front-load if low risk aversion (aggressive)
        # Back-load if high risk aversion (passive)

        num_slices = max(5, int(total_duration / 300))  # 5-minute intervals

        # Generate execution schedule based on risk aversion
        # Lower risk aversion = more aggressive = front-loaded
        schedule = self._generate_optimal_schedule(
            num_slices,
            self.risk_aversion,
            volatility if self.volatility_scaling else 0.02
        )

        child_orders = []
        cumulative_quantity = Decimal("0")

        for i in range(num_slices):
            if i == num_slices - 1:
                slice_quantity = total_quantity - cumulative_quantity
            else:
                slice_quantity = Decimal(str(float(total_quantity) * schedule[i]))
                cumulative_quantity += slice_quantity

            scheduled_time = start_time + timedelta(seconds=i * (total_duration / num_slices))

            # Urgency decreases over time for low risk aversion (front-loaded)
            # Urgency increases over time for high risk aversion (back-loaded)
            urgency = schedule[i] * num_slices

            child_orders.append(ChildOrder(
                parent_order_id=order_id,
                sequence_number=i,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                scheduled_time=scheduled_time,
                urgency=urgency,
                metadata={
                    "strategy": "ImplementationShortfall",
                    "risk_aversion": self.risk_aversion
                }
            ))

        logger.info(
            f"Implementation Shortfall split order {order_id} into {num_slices} slices "
            f"(risk aversion: {self.risk_aversion:.2f})"
        )
        return child_orders

    def _generate_optimal_schedule(
        self,
        num_slices: int,
        risk_aversion: float,
        volatility: float
    ) -> List[float]:
        """
        Generate optimal execution schedule

        Low risk aversion → exponentially decreasing (front-loaded)
        High risk aversion → exponentially increasing (back-loaded)
        """
        # Generate weights based on risk aversion
        # α controls front-loading (α < 0) vs back-loading (α > 0)
        alpha = (risk_aversion - 0.5) * 4  # Maps [0, 1] to [-2, 2]

        weights = []
        for i in range(num_slices):
            t = i / num_slices  # Time from 0 to 1
            if alpha == 0:
                weight = 1.0  # Uniform
            elif alpha > 0:
                # Back-loaded: weight increases over time
                weight = np.exp(alpha * t)
            else:
                # Front-loaded: weight decreases over time
                weight = np.exp(-abs(alpha) * (1 - t))

            weights.append(weight)

        # Normalize to sum to 1
        total = sum(weights)
        return [w / total for w in weights]


class AdaptiveSplitter(OrderSplitter):
    """
    Adaptive order splitter

    Dynamically adjusts execution based on:
    - Real-time market conditions
    - Execution progress
    - Market impact
    - Volatility changes
    """

    def __init__(
        self,
        base_strategy: SplitStrategy = SplitStrategy.VWAP,
        adaptation_rate: float = 0.3,
        min_slice_interval: int = 60,
        max_slice_interval: int = 600,
        **kwargs
    ):
        """
        Initialize Adaptive splitter

        Args:
            base_strategy: Base splitting strategy
            adaptation_rate: How quickly to adapt (0 = no adaptation, 1 = full)
            min_slice_interval: Minimum time between slices
            max_slice_interval: Maximum time between slices
        """
        super().__init__(**kwargs)
        self.base_strategy = base_strategy
        self.adaptation_rate = adaptation_rate
        self.min_slice_interval = min_slice_interval
        self.max_slice_interval = max_slice_interval

        # Initialize base splitter
        if base_strategy == SplitStrategy.TWAP:
            self.base_splitter = TWAPSplitter(**kwargs)
        elif base_strategy == SplitStrategy.VWAP:
            self.base_splitter = VWAPSplitter(**kwargs)
        elif base_strategy == SplitStrategy.POV:
            self.base_splitter = POVSplitter(**kwargs)
        else:
            self.base_splitter = ImplementationShortfallSplitter(**kwargs)

    async def split_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """Split order using adaptive strategy"""
        # Start with base strategy
        child_orders = await self.base_splitter.split_order(
            order_id, symbol, side, total_quantity,
            start_time, end_time, market_data
        )

        # Apply adaptive adjustments based on market conditions
        if market_data:
            child_orders = self._apply_adaptive_adjustments(
                child_orders,
                market_data
            )

        logger.info(
            f"Adaptive split order {order_id} into {len(child_orders)} slices "
            f"(base: {self.base_strategy.value})"
        )
        return child_orders

    def _apply_adaptive_adjustments(
        self,
        child_orders: List[ChildOrder],
        market_data: Dict[str, Any]
    ) -> List[ChildOrder]:
        """Apply adaptive adjustments to child orders"""
        # Get market conditions
        volatility = market_data.get("volatility", 0.02)
        spread = market_data.get("spread", 0.01)
        volume = market_data.get("current_volume", 100000)

        # Adjust based on market conditions
        for order in child_orders:
            # Increase urgency in high volatility
            if volatility > 0.03:
                adjustment = (volatility - 0.03) / 0.03 * self.adaptation_rate
                order.urgency = min(1.0, order.urgency + adjustment)

            # Decrease urgency in wide spreads (to reduce cost)
            if spread > 0.02:
                adjustment = (spread - 0.02) / 0.02 * self.adaptation_rate
                order.urgency = max(0.0, order.urgency - adjustment * 0.5)

            # Adjust based on current volume
            avg_volume = market_data.get("avg_volume", 100000)
            if volume < avg_volume * 0.5:
                # Low volume - be more passive
                order.urgency = max(0.0, order.urgency - 0.2 * self.adaptation_rate)

        return child_orders


class OrderSplitterManager:
    """
    Manager for order splitting strategies

    Provides easy interface to select and use different splitting strategies
    """

    def __init__(self):
        """Initialize order splitter manager"""
        self.splitters = {
            SplitStrategy.TWAP: TWAPSplitter(),
            SplitStrategy.VWAP: VWAPSplitter(),
            SplitStrategy.POV: POVSplitter(),
            SplitStrategy.IMPLEMENTATION_SHORTFALL: ImplementationShortfallSplitter(),
            SplitStrategy.ADAPTIVE: AdaptiveSplitter()
        }

    def register_splitter(self, strategy: SplitStrategy, splitter: OrderSplitter):
        """Register a custom splitter"""
        self.splitters[strategy] = splitter

    async def split_order(
        self,
        strategy: SplitStrategy,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[ChildOrder]:
        """
        Split order using specified strategy

        Args:
            strategy: Splitting strategy to use
            order_id: Parent order ID
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to split
            start_time: Start time for execution
            end_time: End time for execution
            market_data: Market data for informed splitting

        Returns:
            List of child orders
        """
        if strategy not in self.splitters:
            raise ValueError(f"Unknown splitting strategy: {strategy}")

        splitter = self.splitters[strategy]
        return await splitter.split_order(
            order_id, symbol, side, total_quantity,
            start_time, end_time, market_data
        )
