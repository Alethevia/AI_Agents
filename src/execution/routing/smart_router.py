"""
Smart Order Routing System

This module provides intelligent order routing across multiple brokers
with optimization for execution quality, latency, fees, and liquidity.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Order routing strategies"""
    BEST_PRICE = "best_price"  # Route to best price
    LOWEST_FEE = "lowest_fee"  # Route to lowest fee
    LOWEST_LATENCY = "lowest_latency"  # Route to fastest execution
    BEST_LIQUIDITY = "best_liquidity"  # Route to best liquidity
    SMART = "smart"  # Balanced optimization
    LOAD_BALANCE = "load_balance"  # Distribute across brokers


@dataclass
class BrokerQuote:
    """Quote from a broker"""
    broker_id: str
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp: datetime
    latency_ms: float = 0.0
    fee_rate: Decimal = Decimal("0.001")  # 0.1% default
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.bid = Decimal(str(self.bid))
        self.ask = Decimal(str(self.ask))
        self.bid_size = Decimal(str(self.bid_size))
        self.ask_size = Decimal(str(self.ask_size))
        self.fee_rate = Decimal(str(self.fee_rate))

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread"""
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2

    def execution_price(self, side: str, quantity: Decimal) -> Optional[Decimal]:
        """
        Estimate execution price for a given quantity

        Args:
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            Expected execution price or None if insufficient liquidity
        """
        if side.lower() == "buy":
            if quantity > self.ask_size:
                return None
            return self.ask
        else:  # sell
            if quantity > self.bid_size:
                return None
            return self.bid

    def total_cost(self, side: str, quantity: Decimal) -> Optional[Decimal]:
        """
        Calculate total cost including fees

        Args:
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            Total cost or None if insufficient liquidity
        """
        price = self.execution_price(side, quantity)
        if price is None:
            return None

        notional = price * quantity
        fees = notional * self.fee_rate
        return notional + fees


@dataclass
class RoutingDecision:
    """Routing decision for an order"""
    broker_id: str
    symbol: str
    quantity: Decimal
    expected_price: Decimal
    expected_fee: Decimal
    expected_latency_ms: float
    confidence: float  # 0-1 confidence in this routing
    reasoning: str
    quote: Optional[BrokerQuote] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.quantity = Decimal(str(self.quantity))
        self.expected_price = Decimal(str(self.expected_price))
        self.expected_fee = Decimal(str(self.expected_fee))

    @property
    def expected_total_cost(self) -> Decimal:
        """Calculate expected total cost"""
        return self.expected_price * self.quantity + self.expected_fee


@dataclass
class BrokerPerformance:
    """Track broker performance metrics"""
    broker_id: str
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    avg_latency_ms: float = 0.0
    avg_slippage: Decimal = Decimal("0")
    avg_fill_rate: float = 1.0
    reliability_score: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.avg_slippage = Decimal(str(self.avg_slippage))

    def update_success(
        self,
        latency_ms: float,
        slippage: Decimal,
        fill_rate: float
    ):
        """Update metrics after successful order"""
        self.total_orders += 1
        self.successful_orders += 1

        # Exponential moving average
        alpha = 0.1
        self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
        self.avg_slippage = (1 - Decimal(str(alpha))) * self.avg_slippage + Decimal(str(alpha)) * slippage
        self.avg_fill_rate = (1 - alpha) * self.avg_fill_rate + alpha * fill_rate

        # Update reliability score
        self.reliability_score = self.successful_orders / self.total_orders
        self.last_updated = datetime.now(timezone.utc)

    def update_failure(self):
        """Update metrics after failed order"""
        self.total_orders += 1
        self.failed_orders += 1
        self.reliability_score = self.successful_orders / self.total_orders
        self.last_updated = datetime.now(timezone.utc)


class SmartRouter:
    """
    Smart order routing system

    Routes orders across multiple brokers optimizing for:
    - Best execution price
    - Lowest fees
    - Lowest latency
    - Best liquidity
    - Broker reliability
    """

    def __init__(
        self,
        broker_adapters: Dict[str, Any],
        default_strategy: RoutingStrategy = RoutingStrategy.SMART,
        enable_performance_tracking: bool = True,
        quote_timeout_seconds: float = 1.0
    ):
        """
        Initialize smart router

        Args:
            broker_adapters: Dictionary of broker_id -> broker adapter
            default_strategy: Default routing strategy
            enable_performance_tracking: Track broker performance
            quote_timeout_seconds: Timeout for quote requests
        """
        self.broker_adapters = broker_adapters
        self.default_strategy = default_strategy
        self.enable_performance_tracking = enable_performance_tracking
        self.quote_timeout_seconds = quote_timeout_seconds

        # Performance tracking
        self.broker_performance: Dict[str, BrokerPerformance] = {}
        for broker_id in broker_adapters.keys():
            self.broker_performance[broker_id] = BrokerPerformance(broker_id=broker_id)

        # Routing history
        self.routing_history: List[RoutingDecision] = []

        logger.info(f"SmartRouter initialized with {len(broker_adapters)} brokers")

    async def route_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "market",
        strategy: Optional[RoutingStrategy] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route an order to the best broker

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            order_type: Order type ("market", "limit", etc.)
            strategy: Routing strategy (uses default if None)
            constraints: Additional constraints (max_latency, max_fee, etc.)

        Returns:
            RoutingDecision with broker selection and reasoning

        Raises:
            ValueError: If no suitable broker found
        """
        strategy = strategy or self.default_strategy
        constraints = constraints or {}

        # Get quotes from all brokers
        quotes = await self._get_broker_quotes(symbol, side, quantity)

        if not quotes:
            raise ValueError(f"No quotes available for {symbol}")

        # Filter quotes based on constraints
        filtered_quotes = self._apply_constraints(quotes, side, quantity, constraints)

        if not filtered_quotes:
            raise ValueError(f"No brokers meet constraints for {symbol}")

        # Select best broker based on strategy
        decision = await self._select_broker(
            filtered_quotes,
            side,
            quantity,
            strategy
        )

        # Store routing decision
        self.routing_history.append(decision)

        logger.info(
            f"Routed {side} {quantity} {symbol} to {decision.broker_id} "
            f"(price: {decision.expected_price}, fee: {decision.expected_fee})"
        )

        return decision

    async def route_split_orders(
        self,
        symbol: str,
        side: str,
        quantities: List[Decimal],
        strategy: Optional[RoutingStrategy] = None
    ) -> List[RoutingDecision]:
        """
        Route multiple split orders

        Can distribute across multiple brokers for load balancing

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantities: List of order quantities
            strategy: Routing strategy

        Returns:
            List of routing decisions
        """
        strategy = strategy or self.default_strategy

        if strategy == RoutingStrategy.LOAD_BALANCE:
            # Distribute across multiple brokers
            return await self._route_load_balanced(symbol, side, quantities)
        else:
            # Route each order independently
            decisions = []
            for qty in quantities:
                decision = await self.route_order(symbol, side, qty, strategy=strategy)
                decisions.append(decision)
            return decisions

    async def _get_broker_quotes(
        self,
        symbol: str,
        side: str,
        quantity: Decimal
    ) -> List[BrokerQuote]:
        """
        Get quotes from all available brokers

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            List of broker quotes
        """
        tasks = []
        for broker_id, adapter in self.broker_adapters.items():
            task = self._get_quote_with_timeout(broker_id, adapter, symbol, side, quantity)
            tasks.append(task)

        # Get all quotes concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and None results
        quotes = []
        for result in results:
            if isinstance(result, BrokerQuote):
                quotes.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Error getting quote: {result}")

        return quotes

    async def _get_quote_with_timeout(
        self,
        broker_id: str,
        adapter: Any,
        symbol: str,
        side: str,
        quantity: Decimal
    ) -> Optional[BrokerQuote]:
        """Get quote from broker with timeout"""
        try:
            start_time = datetime.now(timezone.utc)

            # Get quote from broker
            quote_task = adapter.get_quote(symbol, side, quantity)
            quote_data = await asyncio.wait_for(
                quote_task,
                timeout=self.quote_timeout_seconds
            )

            # Calculate latency
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Create quote object
            quote = BrokerQuote(
                broker_id=broker_id,
                symbol=symbol,
                bid=quote_data["bid"],
                ask=quote_data["ask"],
                bid_size=quote_data.get("bid_size", Decimal("1000")),
                ask_size=quote_data.get("ask_size", Decimal("1000")),
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                fee_rate=quote_data.get("fee_rate", Decimal("0.001"))
            )

            return quote

        except asyncio.TimeoutError:
            logger.warning(f"Quote timeout for broker {broker_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting quote from {broker_id}: {e}")
            return None

    def _apply_constraints(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal,
        constraints: Dict[str, Any]
    ) -> List[BrokerQuote]:
        """Filter quotes based on constraints"""
        filtered = []

        for quote in quotes:
            # Check liquidity
            available_size = quote.ask_size if side.lower() == "buy" else quote.bid_size
            if available_size < quantity:
                continue

            # Check max latency
            if "max_latency_ms" in constraints:
                if quote.latency_ms > constraints["max_latency_ms"]:
                    continue

            # Check max fee
            if "max_fee_rate" in constraints:
                if quote.fee_rate > constraints["max_fee_rate"]:
                    continue

            # Check broker whitelist
            if "allowed_brokers" in constraints:
                if quote.broker_id not in constraints["allowed_brokers"]:
                    continue

            # Check broker blacklist
            if "blocked_brokers" in constraints:
                if quote.broker_id in constraints["blocked_brokers"]:
                    continue

            # Check minimum reliability
            if "min_reliability" in constraints:
                perf = self.broker_performance.get(quote.broker_id)
                if perf and perf.reliability_score < constraints["min_reliability"]:
                    continue

            filtered.append(quote)

        return filtered

    async def _select_broker(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal,
        strategy: RoutingStrategy
    ) -> RoutingDecision:
        """Select best broker based on strategy"""
        if strategy == RoutingStrategy.BEST_PRICE:
            return self._select_best_price(quotes, side, quantity)
        elif strategy == RoutingStrategy.LOWEST_FEE:
            return self._select_lowest_fee(quotes, side, quantity)
        elif strategy == RoutingStrategy.LOWEST_LATENCY:
            return self._select_lowest_latency(quotes, side, quantity)
        elif strategy == RoutingStrategy.BEST_LIQUIDITY:
            return self._select_best_liquidity(quotes, side, quantity)
        elif strategy == RoutingStrategy.SMART:
            return self._select_smart(quotes, side, quantity)
        else:
            # Default to smart routing
            return self._select_smart(quotes, side, quantity)

    def _select_best_price(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal
    ) -> RoutingDecision:
        """Select broker with best execution price"""
        best_quote = None
        best_price = None

        for quote in quotes:
            price = quote.execution_price(side, quantity)
            if price is None:
                continue

            if best_price is None or (
                (side.lower() == "buy" and price < best_price) or
                (side.lower() == "sell" and price > best_price)
            ):
                best_price = price
                best_quote = quote

        if best_quote is None:
            raise ValueError("No broker has sufficient liquidity")

        notional = best_price * quantity
        fee = notional * best_quote.fee_rate

        return RoutingDecision(
            broker_id=best_quote.broker_id,
            symbol=best_quote.symbol,
            quantity=quantity,
            expected_price=best_price,
            expected_fee=fee,
            expected_latency_ms=best_quote.latency_ms,
            confidence=0.9,
            reasoning=f"Best price: {best_price}",
            quote=best_quote
        )

    def _select_lowest_fee(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal
    ) -> RoutingDecision:
        """Select broker with lowest fees"""
        best_quote = None
        best_total_cost = None

        for quote in quotes:
            total_cost = quote.total_cost(side, quantity)
            if total_cost is None:
                continue

            if best_total_cost is None or total_cost < best_total_cost:
                best_total_cost = total_cost
                best_quote = quote

        if best_quote is None:
            raise ValueError("No broker has sufficient liquidity")

        price = best_quote.execution_price(side, quantity)
        notional = price * quantity
        fee = notional * best_quote.fee_rate

        return RoutingDecision(
            broker_id=best_quote.broker_id,
            symbol=best_quote.symbol,
            quantity=quantity,
            expected_price=price,
            expected_fee=fee,
            expected_latency_ms=best_quote.latency_ms,
            confidence=0.9,
            reasoning=f"Lowest total cost: {best_total_cost}",
            quote=best_quote
        )

    def _select_lowest_latency(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal
    ) -> RoutingDecision:
        """Select broker with lowest latency"""
        # Filter quotes with sufficient liquidity
        valid_quotes = [
            q for q in quotes
            if q.execution_price(side, quantity) is not None
        ]

        if not valid_quotes:
            raise ValueError("No broker has sufficient liquidity")

        # Select lowest latency
        best_quote = min(valid_quotes, key=lambda q: q.latency_ms)

        price = best_quote.execution_price(side, quantity)
        notional = price * quantity
        fee = notional * best_quote.fee_rate

        return RoutingDecision(
            broker_id=best_quote.broker_id,
            symbol=best_quote.symbol,
            quantity=quantity,
            expected_price=price,
            expected_fee=fee,
            expected_latency_ms=best_quote.latency_ms,
            confidence=0.9,
            reasoning=f"Lowest latency: {best_quote.latency_ms:.2f}ms",
            quote=best_quote
        )

    def _select_best_liquidity(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal
    ) -> RoutingDecision:
        """Select broker with best liquidity"""
        best_quote = None
        best_liquidity = Decimal("0")

        for quote in quotes:
            liquidity = quote.ask_size if side.lower() == "buy" else quote.bid_size

            if liquidity > best_liquidity:
                best_liquidity = liquidity
                best_quote = quote

        if best_quote is None:
            raise ValueError("No liquidity available")

        price = best_quote.execution_price(side, quantity)
        notional = price * quantity
        fee = notional * best_quote.fee_rate

        return RoutingDecision(
            broker_id=best_quote.broker_id,
            symbol=best_quote.symbol,
            quantity=quantity,
            expected_price=price,
            expected_fee=fee,
            expected_latency_ms=best_quote.latency_ms,
            confidence=0.9,
            reasoning=f"Best liquidity: {best_liquidity}",
            quote=best_quote
        )

    def _select_smart(
        self,
        quotes: List[BrokerQuote],
        side: str,
        quantity: Decimal
    ) -> RoutingDecision:
        """
        Smart broker selection using multi-criteria optimization

        Balances:
        - Execution price (40%)
        - Fees (20%)
        - Latency (15%)
        - Liquidity (10%)
        - Broker reliability (15%)
        """
        scores = []

        for quote in quotes:
            price = quote.execution_price(side, quantity)
            if price is None:
                continue

            total_cost = quote.total_cost(side, quantity)
            liquidity = quote.ask_size if side.lower() == "buy" else quote.bid_size
            perf = self.broker_performance.get(quote.broker_id)

            # Normalize metrics to [0, 1] for comparison
            # Lower is better for price (buy), higher for price (sell)
            # Lower is better for fees and latency
            # Higher is better for liquidity and reliability

            score_data = {
                "price": float(price),
                "total_cost": float(total_cost),
                "latency": quote.latency_ms,
                "liquidity": float(liquidity),
                "reliability": perf.reliability_score if perf else 1.0
            }

            scores.append((quote, score_data))

        if not scores:
            raise ValueError("No broker has sufficient liquidity")

        # Normalize scores
        price_values = [s[1]["price"] for s in scores]
        cost_values = [s[1]["total_cost"] for s in scores]
        latency_values = [s[1]["latency"] for s in scores]
        liquidity_values = [s[1]["liquidity"] for s in scores]

        # Calculate composite scores
        composite_scores = []
        for quote, score_data in scores:
            # Normalize each metric
            if side.lower() == "buy":
                # For buy, lower price is better
                price_score = 1 - (score_data["price"] - min(price_values)) / (max(price_values) - min(price_values) + 1e-10)
            else:
                # For sell, higher price is better
                price_score = (score_data["price"] - min(price_values)) / (max(price_values) - min(price_values) + 1e-10)

            cost_score = 1 - (score_data["total_cost"] - min(cost_values)) / (max(cost_values) - min(cost_values) + 1e-10)
            latency_score = 1 - (score_data["latency"] - min(latency_values)) / (max(latency_values) - min(latency_values) + 1e-10)
            liquidity_score = (score_data["liquidity"] - min(liquidity_values)) / (max(liquidity_values) - min(liquidity_values) + 1e-10)
            reliability_score = score_data["reliability"]

            # Weighted composite score
            composite = (
                0.40 * price_score +
                0.20 * cost_score +
                0.15 * latency_score +
                0.10 * liquidity_score +
                0.15 * reliability_score
            )

            composite_scores.append((quote, composite, score_data))

        # Select best composite score
        best_quote, best_score, score_data = max(composite_scores, key=lambda x: x[1])

        price = best_quote.execution_price(side, quantity)
        notional = price * quantity
        fee = notional * best_quote.fee_rate

        return RoutingDecision(
            broker_id=best_quote.broker_id,
            symbol=best_quote.symbol,
            quantity=quantity,
            expected_price=price,
            expected_fee=fee,
            expected_latency_ms=best_quote.latency_ms,
            confidence=best_score,
            reasoning=(
                f"Smart routing: score={best_score:.3f} "
                f"(price={price}, latency={best_quote.latency_ms:.1f}ms, "
                f"reliability={score_data['reliability']:.2f})"
            ),
            quote=best_quote
        )

    async def _route_load_balanced(
        self,
        symbol: str,
        side: str,
        quantities: List[Decimal]
    ) -> List[RoutingDecision]:
        """Route orders with load balancing across brokers"""
        # Get quotes
        total_quantity = sum(quantities)
        quotes = await self._get_broker_quotes(symbol, side, total_quantity)

        if not quotes:
            raise ValueError(f"No quotes available for {symbol}")

        # Distribute orders across brokers
        broker_ids = [q.broker_id for q in quotes]
        decisions = []

        for i, qty in enumerate(quantities):
            broker_id = broker_ids[i % len(broker_ids)]
            quote = next(q for q in quotes if q.broker_id == broker_id)

            price = quote.execution_price(side, qty)
            if price is None:
                # Fall back to another broker
                for q in quotes:
                    price = q.execution_price(side, qty)
                    if price:
                        quote = q
                        break

            notional = price * qty
            fee = notional * quote.fee_rate

            decision = RoutingDecision(
                broker_id=quote.broker_id,
                symbol=symbol,
                quantity=qty,
                expected_price=price,
                expected_fee=fee,
                expected_latency_ms=quote.latency_ms,
                confidence=0.8,
                reasoning=f"Load balanced to {quote.broker_id}",
                quote=quote
            )
            decisions.append(decision)

        return decisions

    async def update_performance(
        self,
        broker_id: str,
        success: bool,
        latency_ms: float = 0.0,
        slippage: Decimal = Decimal("0"),
        fill_rate: float = 1.0
    ):
        """
        Update broker performance metrics

        Args:
            broker_id: Broker identifier
            success: Whether order was successful
            latency_ms: Execution latency
            slippage: Execution slippage
            fill_rate: Fill rate (0-1)
        """
        if broker_id not in self.broker_performance:
            self.broker_performance[broker_id] = BrokerPerformance(broker_id=broker_id)

        perf = self.broker_performance[broker_id]

        if success:
            perf.update_success(latency_ms, slippage, fill_rate)
        else:
            perf.update_failure()

    def get_broker_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all brokers"""
        return {
            broker_id: {
                "total_orders": perf.total_orders,
                "successful_orders": perf.successful_orders,
                "failed_orders": perf.failed_orders,
                "success_rate": perf.reliability_score,
                "avg_latency_ms": perf.avg_latency_ms,
                "avg_slippage": float(perf.avg_slippage),
                "avg_fill_rate": perf.avg_fill_rate
            }
            for broker_id, perf in self.broker_performance.items()
        }
