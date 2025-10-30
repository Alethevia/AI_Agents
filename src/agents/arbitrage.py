"""
Arbitrage agent for exploiting price discrepancies.

Implements cross-exchange arbitrage, statistical arbitrage,
triangular arbitrage, and latency arbitrage detection.
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentMode
from src.core.types.signals import Signal, SignalType, SignalStrength
from src.core.logging.logger import get_logger


class ArbitrageConfig(BaseModel):
    """Arbitrage configuration."""

    # Cross-exchange arbitrage
    min_spread_bps: Decimal = Field(
        default=Decimal("10"),
        description="Minimum profitable spread in basis points",
        gt=0
    )
    exchange_fee_bps: Decimal = Field(
        default=Decimal("10"),
        description="Exchange fee in basis points",
        ge=0
    )
    transfer_cost_bps: Decimal = Field(
        default=Decimal("5"),
        description="Transfer cost in basis points",
        ge=0
    )

    # Triangular arbitrage
    triangular_min_profit_bps: Decimal = Field(
        default=Decimal("5"),
        description="Minimum profit for triangular arbitrage",
        gt=0
    )
    triangular_max_hops: int = Field(
        default=3,
        description="Maximum hops for triangular arbitrage",
        ge=2,
        le=5
    )

    # Statistical arbitrage
    stat_arb_zscore_threshold: Decimal = Field(
        default=Decimal("2.0"),
        description="Z-score threshold for stat arb",
        gt=0
    )
    stat_arb_lookback: int = Field(
        default=60,
        description="Lookback period for stat arb",
        ge=10
    )
    stat_arb_half_life: int = Field(
        default=20,
        description="Expected half-life of mean reversion",
        ge=1
    )

    # Latency arbitrage
    latency_threshold_ms: int = Field(
        default=100,
        description="Maximum latency for arbitrage in ms",
        ge=1
    )
    price_staleness_ms: int = Field(
        default=1000,
        description="Maximum price staleness in ms",
        ge=10
    )

    # Position sizing
    min_position_size: Decimal = Field(
        default=Decimal("100"),
        description="Minimum position size in USD",
        gt=0
    )
    max_position_size: Decimal = Field(
        default=Decimal("50000"),
        description="Maximum position size in USD",
        gt=0
    )
    position_scale_factor: Decimal = Field(
        default=Decimal("0.5"),
        description="Position sizing based on opportunity size",
        gt=0,
        le=1
    )

    # Risk management
    max_holding_time_seconds: int = Field(
        default=30,
        description="Maximum holding time for arbitrage",
        ge=1
    )
    execution_confidence_threshold: Decimal = Field(
        default=Decimal("0.9"),
        description="Minimum execution confidence",
        ge=0,
        le=1
    )

    # Filters
    min_confidence: Decimal = Field(
        default=Decimal("0.7"),
        description="Minimum confidence for signal",
        ge=0,
        le=1
    )
    min_volume_usd: Decimal = Field(
        default=Decimal("10000"),
        description="Minimum volume in USD",
        gt=0
    )


class ArbitrageAgent(BaseAgent):
    """
    Arbitrage agent that exploits price discrepancies.

    Implements multiple arbitrage strategies including cross-exchange,
    triangular, statistical, and latency-based arbitrage.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Arbitrage",
        mode: AgentMode = AgentMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
        arb_config: Optional[ArbitrageConfig] = None,
    ):
        """
        Initialize arbitrage agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            mode: Operating mode
            config: General agent configuration
            arb_config: Arbitrage specific configuration
        """
        super().__init__(agent_id, name, mode, config)

        # Arbitrage configuration
        self.arb_config = arb_config or ArbitrageConfig()

        # Market data by exchange
        self._prices: Dict[str, Dict[str, Decimal]] = {}  # exchange -> symbol -> price
        self._volumes: Dict[str, Dict[str, Decimal]] = {}  # exchange -> symbol -> volume
        self._timestamps: Dict[str, Dict[str, datetime]] = {}  # exchange -> symbol -> timestamp
        self._orderbooks: Dict[str, Dict[str, Dict]] = {}  # exchange -> symbol -> orderbook

        # Price history for statistical arbitrage
        self._price_history: Dict[str, List[Decimal]] = {}  # symbol -> prices
        self._spread_history: Dict[Tuple[str, str], List[Decimal]] = {}  # (sym1, sym2) -> spreads

        # Exchange graph for triangular arbitrage
        self._exchange_pairs: Set[Tuple[str, str]] = set()  # Available trading pairs
        self._exchange_rates: Dict[Tuple[str, str, str], Decimal] = {}  # (exch, base, quote) -> rate

        # Opportunity tracking
        self._opportunities: List[Dict[str, Any]] = []
        self._executed_arbitrages: List[Dict[str, Any]] = []

        self.logger = get_logger(f"{__name__}.{self.name}")

    async def _initialize(self) -> None:
        """Initialize arbitrage agent."""
        self.logger.info(
            "Initializing arbitrage agent",
            agent_id=self.agent_id,
            config=self.arb_config.model_dump()
        )

        # Clear caches
        self._prices.clear()
        self._volumes.clear()
        self._timestamps.clear()
        self._opportunities.clear()

        self.logger.info("Arbitrage agent initialized")

    async def _cleanup(self) -> None:
        """Cleanup arbitrage resources."""
        self.logger.info("Cleaning up arbitrage agent")
        self._executed_arbitrages.clear()
        self.logger.info("Arbitrage agent cleaned up")

    async def _update(self, market_data: Dict[str, Any]) -> None:
        """
        Update arbitrage agent with new market data.

        Args:
            market_data: Market data dictionary with exchange info
        """
        symbol = market_data.get('symbol')
        exchange = market_data.get('exchange', 'default')

        if not symbol:
            return

        # Initialize exchange data structures
        if exchange not in self._prices:
            self._prices[exchange] = {}
            self._volumes[exchange] = {}
            self._timestamps[exchange] = {}
            self._orderbooks[exchange] = {}

        # Update price
        price = market_data.get('price')
        if price:
            self._prices[exchange][symbol] = Decimal(str(price))
            self._timestamps[exchange][symbol] = market_data.get('timestamp', datetime.utcnow())

            # Update global price history
            price_key = f"{exchange}:{symbol}"
            if price_key not in self._price_history:
                self._price_history[price_key] = []
            self._price_history[price_key].append(Decimal(str(price)))

            # Keep limited history
            if len(self._price_history[price_key]) > self.arb_config.stat_arb_lookback * 2:
                self._price_history[price_key] = self._price_history[price_key][-self.arb_config.stat_arb_lookback * 2:]

        # Update volume
        volume = market_data.get('volume')
        if volume:
            self._volumes[exchange][symbol] = Decimal(str(volume))

        # Update orderbook if available
        orderbook = market_data.get('orderbook')
        if orderbook:
            self._orderbooks[exchange][symbol] = orderbook

        # Update exchange pairs for triangular arbitrage
        base = market_data.get('base_currency')
        quote = market_data.get('quote_currency')
        if base and quote and price:
            self._exchange_pairs.add((base, quote))
            self._exchange_rates[(exchange, base, quote)] = Decimal(str(price))

    async def _decide(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate arbitrage decision.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            Trading signal or None
        """
        # Try different arbitrage strategies
        opportunities = []

        # Cross-exchange arbitrage
        cross_exchange_opp = await self._find_cross_exchange_arbitrage(symbol)
        if cross_exchange_opp:
            opportunities.append(cross_exchange_opp)

        # Triangular arbitrage
        triangular_opp = await self._find_triangular_arbitrage(symbol)
        if triangular_opp:
            opportunities.append(triangular_opp)

        # Statistical arbitrage
        stat_arb_opp = await self._find_statistical_arbitrage(symbol)
        if stat_arb_opp:
            opportunities.append(stat_arb_opp)

        # Select best opportunity
        if not opportunities:
            return None

        best_opp = max(opportunities, key=lambda x: x['expected_profit'])

        # Validate opportunity
        if not await self._validate_opportunity(best_opp):
            return None

        # Calculate confidence
        confidence = await self._calculate_confidence(best_opp)

        if confidence < self.arb_config.min_confidence:
            return None

        # Create signal
        signal_type = SignalType.BUY if best_opp['action'] == 'buy' else SignalType.SELL
        strength = self._determine_strength(best_opp['expected_profit_bps'])

        position_size = await self._calculate_position_size(
            best_opp['expected_profit_bps'],
            best_opp.get('price', Decimal("0"))
        )

        signal = Signal(
            signal_id=f"arb_{self.agent_id}_{symbol}_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source=self.name,
            strategy_id=self.agent_id,
            market_price=best_opp.get('price', Decimal("0")),
            position_size=position_size,
            expiry=datetime.utcnow() + timedelta(seconds=self.arb_config.max_holding_time_seconds),
            indicators={
                'expected_profit_bps': best_opp['expected_profit_bps'],
                'arbitrage_type': Decimal(str(hash(best_opp['type']) % 1000)),  # Encode type
                'latency_ms': Decimal(str(best_opp.get('latency_ms', 0)))
            },
            metadata={
                'agent_type': 'arbitrage',
                'arbitrage_type': best_opp['type'],
                'expected_profit_bps': str(best_opp['expected_profit_bps']),
                'exchanges': best_opp.get('exchanges', []),
                'execution_steps': best_opp.get('steps', [])
            }
        )

        # Track opportunity
        self._opportunities.append({
            'signal_id': signal.signal_id,
            'opportunity': best_opp,
            'timestamp': datetime.utcnow()
        })

        self.logger.info(
            "Arbitrage signal generated",
            symbol=symbol,
            type=best_opp['type'],
            expected_profit_bps=float(best_opp['expected_profit_bps']),
            confidence=float(confidence)
        )

        return signal

    async def _find_cross_exchange_arbitrage(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find cross-exchange arbitrage opportunities.

        Args:
            symbol: Trading symbol

        Returns:
            Opportunity dict or None
        """
        # Need at least 2 exchanges
        exchanges_with_symbol = [
            exch for exch in self._prices.keys()
            if symbol in self._prices[exch]
        ]

        if len(exchanges_with_symbol) < 2:
            return None

        # Find best bid and ask across exchanges
        best_bid_exchange = None
        best_bid_price = Decimal("0")
        best_ask_exchange = None
        best_ask_price = Decimal("inf")

        for exchange in exchanges_with_symbol:
            price = self._prices[exchange][symbol]

            # Check if price is fresh
            if not await self._is_price_fresh(exchange, symbol):
                continue

            # In practice, you'd get actual bid/ask from orderbook
            # For now, approximate using mid price
            spread_estimate = price * Decimal("0.001")  # Assume 10 bps spread
            bid = price - spread_estimate
            ask = price + spread_estimate

            if bid > best_bid_price:
                best_bid_price = bid
                best_bid_exchange = exchange

            if ask < best_ask_price:
                best_ask_price = ask
                best_ask_exchange = exchange

        # Check if there's an arbitrage opportunity
        if not best_bid_exchange or not best_ask_exchange:
            return None

        if best_bid_exchange == best_ask_exchange:
            return None

        # Calculate spread
        spread_bps = (best_bid_price - best_ask_price) / best_ask_price * Decimal("10000")

        # Account for costs
        total_cost_bps = (
            self.arb_config.exchange_fee_bps * Decimal("2") +  # Buy and sell fees
            self.arb_config.transfer_cost_bps
        )

        net_profit_bps = spread_bps - total_cost_bps

        if net_profit_bps < self.arb_config.min_spread_bps:
            return None

        # Calculate expected profit
        expected_profit = net_profit_bps / Decimal("10000")

        return {
            'type': 'cross_exchange',
            'symbol': symbol,
            'action': 'buy',
            'price': best_ask_price,
            'buy_exchange': best_ask_exchange,
            'sell_exchange': best_bid_exchange,
            'buy_price': best_ask_price,
            'sell_price': best_bid_price,
            'spread_bps': spread_bps,
            'expected_profit_bps': net_profit_bps,
            'expected_profit': expected_profit,
            'exchanges': [best_ask_exchange, best_bid_exchange],
            'steps': [
                f"Buy {symbol} on {best_ask_exchange} at {best_ask_price}",
                f"Transfer {symbol} to {best_bid_exchange}",
                f"Sell {symbol} on {best_bid_exchange} at {best_bid_price}"
            ]
        }

    async def _find_triangular_arbitrage(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find triangular arbitrage opportunities.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')

        Returns:
            Opportunity dict or None
        """
        if '/' not in symbol:
            return None

        base, quote = symbol.split('/')

        # Look for triangular paths: base -> intermediate -> quote
        best_opportunity = None
        best_profit = Decimal("0")

        for intermediate in self._get_available_currencies():
            if intermediate == base or intermediate == quote:
                continue

            # Try path: quote -> base -> intermediate -> quote
            profit = await self._calculate_triangular_profit(
                quote, base, intermediate, quote
            )

            if profit and profit > best_profit:
                best_profit = profit
                best_opportunity = {
                    'type': 'triangular',
                    'symbol': symbol,
                    'action': 'buy',
                    'price': self._get_exchange_rate(base, quote),
                    'path': [quote, base, intermediate, quote],
                    'expected_profit_bps': profit * Decimal("10000"),
                    'expected_profit': profit,
                    'exchanges': ['default'],  # Single exchange triangular
                    'steps': [
                        f"Convert {quote} to {base}",
                        f"Convert {base} to {intermediate}",
                        f"Convert {intermediate} to {quote}"
                    ]
                }

        if best_opportunity and best_profit >= self.arb_config.triangular_min_profit_bps / Decimal("10000"):
            return best_opportunity

        return None

    async def _find_statistical_arbitrage(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find statistical arbitrage opportunities.

        Args:
            symbol: Trading symbol

        Returns:
            Opportunity dict or None
        """
        # Look for correlated pairs with price divergence
        for other_symbol in self._price_history.keys():
            if other_symbol == symbol:
                continue

            # Check if we have enough history
            if (symbol not in self._price_history or
                len(self._price_history[symbol]) < self.arb_config.stat_arb_lookback):
                continue

            # Calculate spread
            spread_key = (symbol, other_symbol)
            if spread_key not in self._spread_history:
                continue

            spreads = self._spread_history[spread_key]
            if len(spreads) < self.arb_config.stat_arb_lookback:
                continue

            # Calculate Z-score of current spread
            spread_array = np.array([float(s) for s in spreads])
            mean_spread = Decimal(str(np.mean(spread_array)))
            std_spread = Decimal(str(np.std(spread_array)))

            if std_spread == 0:
                continue

            current_spread = spreads[-1]
            zscore = (current_spread - mean_spread) / std_spread

            # Check if Z-score is significant
            if abs(zscore) < self.arb_config.stat_arb_zscore_threshold:
                continue

            # Calculate expected profit based on mean reversion
            expected_reversion = current_spread - mean_spread
            current_price = self._price_history[symbol][-1]

            expected_profit_bps = abs(expected_reversion) / current_price * Decimal("10000")

            return {
                'type': 'statistical',
                'symbol': symbol,
                'action': 'buy' if zscore < 0 else 'sell',
                'price': current_price,
                'pair_symbol': other_symbol,
                'zscore': float(zscore),
                'expected_profit_bps': expected_profit_bps,
                'expected_profit': expected_profit_bps / Decimal("10000"),
                'exchanges': ['default'],
                'steps': [
                    f"{'Buy' if zscore < 0 else 'Sell'} {symbol}",
                    f"Hedge with {other_symbol}",
                    f"Wait for mean reversion"
                ]
            }

        return None

    async def _calculate_triangular_profit(
        self,
        curr1: str,
        curr2: str,
        curr3: str,
        curr4: str
    ) -> Optional[Decimal]:
        """Calculate profit from triangular arbitrage path."""
        # Get exchange rates
        rate12 = self._get_exchange_rate(curr1, curr2)
        rate23 = self._get_exchange_rate(curr2, curr3)
        rate34 = self._get_exchange_rate(curr3, curr4)

        if not rate12 or not rate23 or not rate34:
            return None

        # Calculate final amount after conversions
        initial = Decimal("1")
        final = initial * rate12 * rate23 * rate34

        # Account for fees
        fee_per_trade = self.arb_config.exchange_fee_bps / Decimal("10000")
        final_after_fees = final * ((Decimal("1") - fee_per_trade) ** 3)

        profit = final_after_fees - initial

        return profit if profit > 0 else None

    def _get_exchange_rate(
        self,
        from_curr: str,
        to_curr: str,
        exchange: str = 'default'
    ) -> Optional[Decimal]:
        """Get exchange rate between two currencies."""
        # Direct rate
        key = (exchange, from_curr, to_curr)
        if key in self._exchange_rates:
            return self._exchange_rates[key]

        # Inverse rate
        inverse_key = (exchange, to_curr, from_curr)
        if inverse_key in self._exchange_rates:
            return Decimal("1") / self._exchange_rates[inverse_key]

        return None

    def _get_available_currencies(self) -> Set[str]:
        """Get all available currencies."""
        currencies = set()
        for base, quote in self._exchange_pairs:
            currencies.add(base)
            currencies.add(quote)
        return currencies

    async def _is_price_fresh(self, exchange: str, symbol: str) -> bool:
        """Check if price is fresh enough for arbitrage."""
        if exchange not in self._timestamps or symbol not in self._timestamps[exchange]:
            return False

        timestamp = self._timestamps[exchange][symbol]
        age_ms = (datetime.utcnow() - timestamp).total_seconds() * 1000

        return age_ms <= self.arb_config.price_staleness_ms

    async def _validate_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Validate arbitrage opportunity."""
        # Check volume
        symbol = opportunity['symbol']
        exchanges = opportunity.get('exchanges', [])

        for exchange in exchanges:
            if exchange in self._volumes and symbol in self._volumes[exchange]:
                volume_usd = self._volumes[exchange][symbol] * opportunity.get('price', Decimal("0"))
                if volume_usd < self.arb_config.min_volume_usd:
                    return False

        # Check latency for cross-exchange arbitrage
        if opportunity['type'] == 'cross_exchange':
            # In practice, you'd measure actual latency
            estimated_latency_ms = 50  # Assume 50ms
            if estimated_latency_ms > self.arb_config.latency_threshold_ms:
                return False

        return True

    async def _calculate_confidence(self, opportunity: Dict[str, Any]) -> Decimal:
        """Calculate confidence in arbitrage opportunity."""
        confidence = Decimal("0.8")  # Base confidence for arbitrage

        # Boost for higher profit
        profit_bps = opportunity['expected_profit_bps']
        if profit_bps > Decimal("50"):
            confidence += Decimal("0.1")

        # Reduce for cross-exchange (higher execution risk)
        if opportunity['type'] == 'cross_exchange':
            confidence -= Decimal("0.1")

        # Reduce for statistical (mean reversion uncertainty)
        if opportunity['type'] == 'statistical':
            confidence -= Decimal("0.15")

        return min(max(confidence, Decimal("0")), Decimal("1"))

    def _determine_strength(self, profit_bps: Decimal) -> SignalStrength:
        """Determine signal strength based on profit."""
        if profit_bps >= Decimal("100"):
            return SignalStrength.VERY_STRONG
        elif profit_bps >= Decimal("50"):
            return SignalStrength.STRONG
        elif profit_bps >= Decimal("20"):
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    async def _calculate_position_size(
        self,
        profit_bps: Decimal,
        price: Decimal
    ) -> Decimal:
        """Calculate position size for arbitrage."""
        # Base size
        position_size_usd = self.arb_config.min_position_size

        # Scale by profit opportunity
        if profit_bps > Decimal("20"):
            multiplier = min(profit_bps / Decimal("20"), Decimal("5"))
            position_size_usd *= multiplier * self.arb_config.position_scale_factor

        # Clamp to max
        position_size_usd = min(position_size_usd, self.arb_config.max_position_size)

        # Convert to shares
        if price > 0:
            shares = position_size_usd / price
        else:
            shares = Decimal("0")

        return shares

    def get_arbitrage_summary(self) -> Dict[str, Any]:
        """Get arbitrage opportunity summary."""
        return {
            'opportunities_found': len(self._opportunities),
            'arbitrages_executed': len(self._executed_arbitrages),
            'exchanges_monitored': len(self._prices),
            'symbols_tracked': sum(len(symbols) for symbols in self._prices.values()),
            'active_opportunities': [
                {
                    'type': opp['opportunity']['type'],
                    'symbol': opp['opportunity']['symbol'],
                    'expected_profit_bps': float(opp['opportunity']['expected_profit_bps']),
                    'age_seconds': (datetime.utcnow() - opp['timestamp']).total_seconds()
                }
                for opp in self._opportunities[-10:]  # Last 10
            ]
        }
