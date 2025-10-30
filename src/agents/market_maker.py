"""
Market making agent for providing liquidity.

Implements bid-ask spread optimization, inventory management,
quote placement strategy, and risk limits.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentMode
from src.core.types.signals import Signal, SignalType, SignalStrength
from src.core.logging.logger import get_logger


class MarketMakerConfig(BaseModel):
    """Market maker configuration."""

    # Spread parameters
    min_spread_bps: Decimal = Field(
        default=Decimal("5"),
        description="Minimum spread in basis points",
        gt=0
    )
    max_spread_bps: Decimal = Field(
        default=Decimal("50"),
        description="Maximum spread in basis points",
        gt=0
    )
    spread_adjustment_factor: Decimal = Field(
        default=Decimal("0.1"),
        description="Spread adjustment factor based on volatility"
    )

    # Inventory management
    max_inventory_value: Decimal = Field(
        default=Decimal("100000"),
        description="Maximum inventory value in USD",
        gt=0
    )
    target_inventory_pct: Decimal = Field(
        default=Decimal("0.5"),
        description="Target inventory as % of max (0-1)",
        ge=0,
        le=1
    )
    inventory_skew_factor: Decimal = Field(
        default=Decimal("0.2"),
        description="How much to skew quotes based on inventory"
    )

    # Quote sizing
    min_quote_size: Decimal = Field(
        default=Decimal("100"),
        description="Minimum quote size in units",
        gt=0
    )
    max_quote_size: Decimal = Field(
        default=Decimal("1000"),
        description="Maximum quote size in units",
        gt=0
    )
    quote_layers: int = Field(
        default=3,
        description="Number of quote layers to place",
        ge=1,
        le=10
    )

    # Risk limits
    max_position_size: Decimal = Field(
        default=Decimal("50000"),
        description="Maximum position size in USD",
        gt=0
    )
    max_loss_per_symbol: Decimal = Field(
        default=Decimal("5000"),
        description="Maximum loss per symbol",
        gt=0
    )
    min_profit_bps: Decimal = Field(
        default=Decimal("2"),
        description="Minimum profit in basis points",
        gt=0
    )

    # Market conditions
    volatility_window: int = Field(
        default=20,
        description="Window for volatility calculation",
        ge=5
    )
    volume_threshold: Decimal = Field(
        default=Decimal("0.01"),
        description="Minimum volume threshold as % of ADV",
        gt=0
    )

    # Update frequency
    quote_update_interval_ms: int = Field(
        default=100,
        description="Quote update interval in milliseconds",
        ge=10
    )


class MarketMakerAgent(BaseAgent):
    """
    Market making agent that provides liquidity.

    This agent places both bid and ask orders to capture the spread while
    managing inventory risk and adjusting quotes based on market conditions.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "MarketMaker",
        mode: AgentMode = AgentMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
        mm_config: Optional[MarketMakerConfig] = None,
    ):
        """
        Initialize market maker agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            mode: Operating mode
            config: General agent configuration
            mm_config: Market maker specific configuration
        """
        super().__init__(agent_id, name, mode, config)

        # Market maker configuration
        self.mm_config = mm_config or MarketMakerConfig()

        # Inventory tracking
        self._inventory: Dict[str, Decimal] = {}  # symbol -> quantity
        self._inventory_value: Dict[str, Decimal] = {}  # symbol -> USD value
        self._average_cost: Dict[str, Decimal] = {}  # symbol -> avg cost

        # Quote management
        self._active_quotes: Dict[str, List[Dict]] = {}  # symbol -> [quotes]
        self._quote_history: List[Dict] = []

        # Market data cache
        self._price_history: Dict[str, List[Decimal]] = {}
        self._volume_history: Dict[str, List[Decimal]] = {}
        self._volatility: Dict[str, Decimal] = {}

        # Performance tracking
        self._realized_pnl: Dict[str, Decimal] = {}
        self._unrealized_pnl: Dict[str, Decimal] = {}

        self.logger = get_logger(f"{__name__}.{self.name}")

    async def _initialize(self) -> None:
        """Initialize market maker agent."""
        self.logger.info(
            "Initializing market maker agent",
            agent_id=self.agent_id,
            config=self.mm_config.model_dump()
        )

        # Initialize state
        self._inventory.clear()
        self._inventory_value.clear()
        self._average_cost.clear()
        self._active_quotes.clear()

        self.logger.info("Market maker agent initialized")

    async def _cleanup(self) -> None:
        """Cleanup market maker resources."""
        self.logger.info("Cleaning up market maker agent")

        # Cancel all active quotes
        for symbol in list(self._active_quotes.keys()):
            await self._cancel_quotes(symbol)

        self.logger.info("Market maker agent cleaned up")

    async def _update(self, market_data: Dict[str, Any]) -> None:
        """
        Update market maker with new market data.

        Args:
            market_data: Market data dictionary
        """
        symbol = market_data.get('symbol')
        if not symbol:
            return

        # Update price history
        price = market_data.get('price')
        if price:
            price = Decimal(str(price))
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            self._price_history[symbol].append(price)

            # Keep only recent history
            max_len = self.mm_config.volatility_window * 2
            if len(self._price_history[symbol]) > max_len:
                self._price_history[symbol] = self._price_history[symbol][-max_len:]

        # Update volume history
        volume = market_data.get('volume')
        if volume:
            volume = Decimal(str(volume))
            if symbol not in self._volume_history:
                self._volume_history[symbol] = []
            self._volume_history[symbol].append(volume)

            if len(self._volume_history[symbol]) > max_len:
                self._volume_history[symbol] = self._volume_history[symbol][-max_len:]

        # Calculate volatility
        await self._calculate_volatility(symbol)

        # Update unrealized P&L
        await self._update_unrealized_pnl(symbol, price)

    async def _decide(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate market making decision.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            Trading signal or None
        """
        # Check if we should participate in this market
        if not await self._should_make_market(symbol, market_data):
            return None

        # Get current market price
        mid_price = Decimal(str(market_data.get('price', 0)))
        if mid_price <= 0:
            return None

        # Calculate optimal spread
        spread = await self._calculate_optimal_spread(symbol, mid_price)

        # Calculate inventory skew
        inventory_skew = await self._calculate_inventory_skew(symbol)

        # Determine bid/ask prices
        bid_price = mid_price * (Decimal("1") - spread / Decimal("2") - inventory_skew)
        ask_price = mid_price * (Decimal("1") + spread / Decimal("2") + inventory_skew)

        # Determine quote sizes
        bid_size, ask_size = await self._calculate_quote_sizes(symbol, mid_price)

        # Generate signal (we'll use both bid and ask in metadata)
        signal = Signal(
            signal_id=f"mm_{self.agent_id}_{symbol}_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            signal_type=SignalType.HOLD,  # Market maker is neutral
            strength=SignalStrength.MODERATE,
            confidence=Decimal("0.7"),
            source=self.name,
            strategy_id=self.agent_id,
            market_price=mid_price,
            indicators={
                'spread_bps': spread * Decimal("10000"),
                'bid_price': bid_price,
                'ask_price': ask_price,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'inventory_skew': inventory_skew,
                'volatility': self._volatility.get(symbol, Decimal("0"))
            },
            metadata={
                'agent_type': 'market_maker',
                'bid_price': str(bid_price),
                'ask_price': str(ask_price),
                'bid_size': str(bid_size),
                'ask_size': str(ask_size),
                'spread_bps': str(spread * Decimal("10000")),
                'inventory': str(self._inventory.get(symbol, Decimal("0"))),
                'inventory_value': str(self._inventory_value.get(symbol, Decimal("0")))
            }
        )

        # Store active quote
        if symbol not in self._active_quotes:
            self._active_quotes[symbol] = []

        self._active_quotes[symbol].append({
            'timestamp': datetime.utcnow(),
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'signal_id': signal.signal_id
        })

        return signal

    async def _should_make_market(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Determine if we should make market for this symbol.

        Args:
            symbol: Trading symbol
            market_data: Market data

        Returns:
            True if should make market
        """
        # Check position limits
        current_position = self._inventory_value.get(symbol, Decimal("0"))
        if abs(current_position) >= self.mm_config.max_position_size:
            self.logger.debug(
                "Position limit reached",
                symbol=symbol,
                position=float(current_position)
            )
            return False

        # Check loss limits
        realized_loss = self._realized_pnl.get(symbol, Decimal("0"))
        if realized_loss < -self.mm_config.max_loss_per_symbol:
            self.logger.debug(
                "Loss limit reached",
                symbol=symbol,
                loss=float(realized_loss)
            )
            return False

        # Check volume
        avg_volume = self._get_average_volume(symbol)
        if avg_volume <= 0:
            return False

        current_volume = Decimal(str(market_data.get('volume', 0)))
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else Decimal("0")

        if volume_ratio < self.mm_config.volume_threshold:
            return False

        return True

    async def _calculate_optimal_spread(
        self,
        symbol: str,
        mid_price: Decimal
    ) -> Decimal:
        """
        Calculate optimal bid-ask spread.

        Args:
            symbol: Trading symbol
            mid_price: Current mid price

        Returns:
            Optimal spread as a decimal (e.g., 0.001 for 10 bps)
        """
        # Base spread
        base_spread = self.mm_config.min_spread_bps / Decimal("10000")

        # Adjust for volatility
        volatility = self._volatility.get(symbol, Decimal("0"))
        volatility_adjustment = volatility * self.mm_config.spread_adjustment_factor

        # Calculate total spread
        spread = base_spread + volatility_adjustment

        # Clamp to min/max
        min_spread = self.mm_config.min_spread_bps / Decimal("10000")
        max_spread = self.mm_config.max_spread_bps / Decimal("10000")

        return max(min_spread, min(max_spread, spread))

    async def _calculate_inventory_skew(self, symbol: str) -> Decimal:
        """
        Calculate inventory skew to adjust quotes.

        Args:
            symbol: Trading symbol

        Returns:
            Inventory skew adjustment
        """
        current_inventory = self._inventory.get(symbol, Decimal("0"))
        max_inventory = self.mm_config.max_inventory_value

        if max_inventory == 0:
            return Decimal("0")

        # Calculate skew based on deviation from target
        inventory_ratio = current_inventory / max_inventory
        target_ratio = self.mm_config.target_inventory_pct

        skew = (inventory_ratio - target_ratio) * self.mm_config.inventory_skew_factor

        return skew

    async def _calculate_quote_sizes(
        self,
        symbol: str,
        mid_price: Decimal
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate bid and ask quote sizes.

        Args:
            symbol: Trading symbol
            mid_price: Current mid price

        Returns:
            Tuple of (bid_size, ask_size)
        """
        # Base size
        base_size = self.mm_config.min_quote_size

        # Adjust based on inventory
        current_inventory = self._inventory.get(symbol, Decimal("0"))
        max_inventory = self.mm_config.max_inventory_value

        inventory_ratio = abs(current_inventory) / max_inventory if max_inventory > 0 else Decimal("0")

        # If long inventory, increase bid size to sell more
        # If short inventory, increase ask size to buy more
        if current_inventory > 0:
            bid_size = base_size * (Decimal("1") + inventory_ratio)
            ask_size = base_size * (Decimal("1") - inventory_ratio / Decimal("2"))
        else:
            bid_size = base_size * (Decimal("1") - abs(inventory_ratio) / Decimal("2"))
            ask_size = base_size * (Decimal("1") + abs(inventory_ratio))

        # Clamp to limits
        bid_size = max(
            self.mm_config.min_quote_size,
            min(self.mm_config.max_quote_size, bid_size)
        )
        ask_size = max(
            self.mm_config.min_quote_size,
            min(self.mm_config.max_quote_size, ask_size)
        )

        return bid_size, ask_size

    async def _calculate_volatility(self, symbol: str) -> None:
        """Calculate realized volatility."""
        if symbol not in self._price_history:
            return

        prices = self._price_history[symbol]
        if len(prices) < self.mm_config.volatility_window:
            return

        # Use recent prices
        recent_prices = prices[-self.mm_config.volatility_window:]
        price_array = np.array([float(p) for p in recent_prices])

        # Calculate log returns
        returns = np.diff(np.log(price_array))

        # Calculate volatility
        volatility = Decimal(str(np.std(returns)))
        self._volatility[symbol] = volatility

    async def _update_unrealized_pnl(
        self,
        symbol: str,
        current_price: Decimal
    ) -> None:
        """Update unrealized P&L for a symbol."""
        if symbol not in self._inventory:
            return

        quantity = self._inventory[symbol]
        avg_cost = self._average_cost.get(symbol, Decimal("0"))

        unrealized = (current_price - avg_cost) * quantity
        self._unrealized_pnl[symbol] = unrealized

    async def _cancel_quotes(self, symbol: str) -> None:
        """Cancel all active quotes for a symbol."""
        if symbol in self._active_quotes:
            self.logger.info(
                "Cancelling quotes",
                symbol=symbol,
                count=len(self._active_quotes[symbol])
            )
            self._active_quotes[symbol].clear()

    def _get_average_volume(self, symbol: str) -> Decimal:
        """Get average volume for a symbol."""
        if symbol not in self._volume_history or not self._volume_history[symbol]:
            return Decimal("0")

        volumes = self._volume_history[symbol]
        return sum(volumes) / len(volumes)

    def update_inventory(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        is_buy: bool
    ) -> None:
        """
        Update inventory after a trade.

        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            is_buy: True if buy, False if sell
        """
        if symbol not in self._inventory:
            self._inventory[symbol] = Decimal("0")
            self._average_cost[symbol] = Decimal("0")

        current_qty = self._inventory[symbol]
        current_avg = self._average_cost[symbol]

        if is_buy:
            new_qty = current_qty + quantity
            new_avg = (current_avg * current_qty + price * quantity) / new_qty
        else:
            new_qty = current_qty - quantity
            # Realize P&L on sale
            realized_pnl = (price - current_avg) * quantity
            if symbol not in self._realized_pnl:
                self._realized_pnl[symbol] = Decimal("0")
            self._realized_pnl[symbol] += realized_pnl
            self.record_pnl(realized_pnl)

            new_avg = current_avg if new_qty != 0 else Decimal("0")

        self._inventory[symbol] = new_qty
        self._average_cost[symbol] = new_avg
        self._inventory_value[symbol] = new_qty * price

        self.logger.info(
            "Inventory updated",
            symbol=symbol,
            quantity=float(quantity),
            price=float(price),
            is_buy=is_buy,
            new_inventory=float(new_qty),
            inventory_value=float(self._inventory_value[symbol])
        )

    def get_inventory_summary(self) -> Dict[str, Any]:
        """Get inventory summary."""
        total_value = sum(self._inventory_value.values())
        total_realized = sum(self._realized_pnl.values())
        total_unrealized = sum(self._unrealized_pnl.values())

        return {
            'total_inventory_value': float(total_value),
            'total_realized_pnl': float(total_realized),
            'total_unrealized_pnl': float(total_unrealized),
            'total_pnl': float(total_realized + total_unrealized),
            'positions': {
                symbol: {
                    'quantity': float(qty),
                    'avg_cost': float(self._average_cost.get(symbol, Decimal("0"))),
                    'value': float(self._inventory_value.get(symbol, Decimal("0"))),
                    'realized_pnl': float(self._realized_pnl.get(symbol, Decimal("0"))),
                    'unrealized_pnl': float(self._unrealized_pnl.get(symbol, Decimal("0")))
                }
                for symbol, qty in self._inventory.items()
            }
        }
