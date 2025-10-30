"""
Momentum agent for capturing price momentum.

Implements relative strength strategies, momentum factor calculation,
sector rotation, and cross-sectional momentum.
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


class MomentumConfig(BaseModel):
    """Momentum configuration."""

    # Momentum calculation
    momentum_lookback_days: int = Field(
        default=20,
        description="Lookback period for momentum calculation",
        ge=1
    )
    short_momentum_days: int = Field(
        default=5,
        description="Short-term momentum period",
        ge=1
    )
    long_momentum_days: int = Field(
        default=60,
        description="Long-term momentum period",
        ge=10
    )

    # Relative strength
    rs_percentile_threshold: Decimal = Field(
        default=Decimal("0.8"),
        description="Percentile threshold for relative strength",
        ge=0,
        le=1
    )
    rs_universe_size: int = Field(
        default=100,
        description="Number of symbols in RS universe",
        ge=10
    )

    # RSI parameters
    rsi_period: int = Field(default=14, description="RSI period", ge=2)
    rsi_oversold: Decimal = Field(
        default=Decimal("30"),
        description="RSI oversold level",
        ge=0,
        le=100
    )
    rsi_overbought: Decimal = Field(
        default=Decimal("70"),
        description="RSI overbought level",
        ge=0,
        le=100
    )

    # Sector rotation
    enable_sector_rotation: bool = Field(
        default=True,
        description="Enable sector rotation strategy"
    )
    sector_momentum_period: int = Field(
        default=30,
        description="Period for sector momentum",
        ge=5
    )
    min_sector_strength: Decimal = Field(
        default=Decimal("0.6"),
        description="Minimum sector strength score",
        ge=0,
        le=1
    )

    # Cross-sectional momentum
    rank_top_n: int = Field(
        default=10,
        description="Top N symbols to consider",
        ge=1
    )
    rank_bottom_n: int = Field(
        default=10,
        description="Bottom N symbols to avoid/short",
        ge=1
    )

    # Position sizing
    base_position_size: Decimal = Field(
        default=Decimal("2000"),
        description="Base position size in USD",
        gt=0
    )
    max_position_size: Decimal = Field(
        default=Decimal("10000"),
        description="Maximum position size in USD",
        gt=0
    )
    momentum_size_multiplier: Decimal = Field(
        default=Decimal("2.0"),
        description="Position size multiplier based on momentum strength"
    )

    # Risk management
    stop_loss_pct: Decimal = Field(
        default=Decimal("0.05"),
        description="Stop loss percentage",
        gt=0,
        le=1
    )
    take_profit_pct: Decimal = Field(
        default=Decimal("0.15"),
        description="Take profit percentage",
        gt=0
    )
    trailing_stop_pct: Decimal = Field(
        default=Decimal("0.03"),
        description="Trailing stop percentage",
        gt=0
    )

    # Filters
    min_confidence: Decimal = Field(
        default=Decimal("0.65"),
        description="Minimum confidence for signal",
        ge=0,
        le=1
    )
    min_volume_ratio: Decimal = Field(
        default=Decimal("1.0"),
        description="Minimum volume ratio vs average",
        gt=0
    )


class MomentumAgent(BaseAgent):
    """
    Momentum agent that captures price momentum.

    Uses relative strength, momentum factors, sector rotation, and
    cross-sectional momentum to identify strong trending opportunities.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Momentum",
        mode: AgentMode = AgentMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
        mom_config: Optional[MomentumConfig] = None,
    ):
        """
        Initialize momentum agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            mode: Operating mode
            config: General agent configuration
            mom_config: Momentum specific configuration
        """
        super().__init__(agent_id, name, mode, config)

        # Momentum configuration
        self.mom_config = mom_config or MomentumConfig()

        # Price data storage
        self._price_data: Dict[str, List[Decimal]] = {}
        self._volume_data: Dict[str, List[Decimal]] = {}
        self._timestamp_data: Dict[str, List[datetime]] = {}

        # Momentum indicators
        self._momentum: Dict[str, Decimal] = {}
        self._short_momentum: Dict[str, Decimal] = {}
        self._long_momentum: Dict[str, Decimal] = {}
        self._rsi: Dict[str, Decimal] = {}
        self._relative_strength: Dict[str, Decimal] = {}

        # Rankings
        self._momentum_ranks: Dict[str, int] = {}
        self._sector_rankings: Dict[str, Decimal] = {}

        # Sector mapping (symbol -> sector)
        self._symbol_sectors: Dict[str, str] = {}
        self._sector_performance: Dict[str, List[Decimal]] = {}

        # Position tracking
        self._positions: Dict[str, Dict[str, Any]] = {}

        self.logger = get_logger(f"{__name__}.{self.name}")

    async def _initialize(self) -> None:
        """Initialize momentum agent."""
        self.logger.info(
            "Initializing momentum agent",
            agent_id=self.agent_id,
            config=self.mom_config.model_dump()
        )

        # Clear caches
        self._price_data.clear()
        self._momentum.clear()
        self._rsi.clear()
        self._relative_strength.clear()

        self.logger.info("Momentum agent initialized")

    async def _cleanup(self) -> None:
        """Cleanup momentum resources."""
        self.logger.info("Cleaning up momentum agent")
        self._positions.clear()
        self.logger.info("Momentum agent cleaned up")

    async def _update(self, market_data: Dict[str, Any]) -> None:
        """
        Update momentum agent with new market data.

        Args:
            market_data: Market data dictionary
        """
        symbol = market_data.get('symbol')
        if not symbol:
            return

        # Update price data
        price = market_data.get('close') or market_data.get('price')
        if price:
            price = Decimal(str(price))
            if symbol not in self._price_data:
                self._price_data[symbol] = []
            self._price_data[symbol].append(price)

        # Update volume data
        volume = market_data.get('volume')
        if volume:
            if symbol not in self._volume_data:
                self._volume_data[symbol] = []
            self._volume_data[symbol].append(Decimal(str(volume)))

        # Update timestamp
        timestamp = market_data.get('timestamp', datetime.utcnow())
        if symbol not in self._timestamp_data:
            self._timestamp_data[symbol] = []
        self._timestamp_data[symbol].append(timestamp)

        # Update sector mapping
        sector = market_data.get('sector')
        if sector:
            self._symbol_sectors[symbol] = sector

        # Keep only necessary history
        max_period = max(
            self.mom_config.long_momentum_days,
            self.mom_config.rsi_period * 2
        )

        for data_dict in [self._price_data, self._volume_data, self._timestamp_data]:
            if symbol in data_dict and len(data_dict[symbol]) > max_period:
                data_dict[symbol] = data_dict[symbol][-max_period:]

        # Calculate indicators
        await self._calculate_indicators(symbol)

        # Update rankings across all symbols
        await self._update_rankings()

        # Update sector performance
        if self.mom_config.enable_sector_rotation:
            await self._update_sector_performance()

    async def _decide(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate momentum decision.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            Trading signal or None
        """
        # Need sufficient data
        if symbol not in self._price_data:
            return None

        if len(self._price_data[symbol]) < self.mom_config.momentum_lookback_days:
            return None

        current_price = self._price_data[symbol][-1]

        # Get momentum indicators
        momentum = self._momentum.get(symbol, Decimal("0"))
        short_mom = self._short_momentum.get(symbol, Decimal("0"))
        long_mom = self._long_momentum.get(symbol, Decimal("0"))
        rsi = self._rsi.get(symbol, Decimal("50"))
        rs = self._relative_strength.get(symbol, Decimal("0.5"))

        # Check momentum signals
        momentum_signal = await self._check_momentum_signal(
            symbol, momentum, short_mom, long_mom
        )

        if momentum_signal == 0:
            return None

        # Check RSI for confirmation
        rsi_signal = await self._check_rsi_signal(rsi, momentum_signal)

        # Check relative strength ranking
        rs_signal = await self._check_relative_strength(symbol, rs)

        # Check sector rotation if enabled
        sector_signal = 0
        if self.mom_config.enable_sector_rotation and symbol in self._symbol_sectors:
            sector_signal = await self._check_sector_rotation(symbol)

        # Combine signals
        signals = [momentum_signal, rsi_signal, rs_signal]
        if sector_signal != 0:
            signals.append(sector_signal)

        # Need majority agreement
        signal_sum = sum(signals)
        if abs(signal_sum) < len(signals) / 2:
            return None

        # Determine final signal
        signal_direction = np.sign(signal_sum)
        signal_type = SignalType.BUY if signal_direction > 0 else SignalType.SELL

        # Calculate confidence
        confidence = await self._calculate_confidence(
            symbol,
            momentum,
            rs,
            rsi,
            signals
        )

        if confidence < self.mom_config.min_confidence:
            return None

        # Determine strength
        strength = await self._determine_strength(momentum, rs)

        # Calculate position size
        position_size = await self._calculate_position_size(
            symbol,
            current_price,
            momentum,
            rs
        )

        # Calculate stop loss and take profit
        stop_loss, take_profit = await self._calculate_stop_take(
            current_price,
            signal_type,
            momentum
        )

        # Create signal
        signal = Signal(
            signal_id=f"mom_{self.agent_id}_{symbol}_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source=self.name,
            strategy_id=self.agent_id,
            market_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            indicators={
                'momentum': momentum,
                'short_momentum': short_mom,
                'long_momentum': long_mom,
                'rsi': rsi,
                'relative_strength': rs,
                'rank': Decimal(str(self._momentum_ranks.get(symbol, 0)))
            },
            metadata={
                'agent_type': 'momentum',
                'momentum_score': str(momentum),
                'rs_score': str(rs),
                'sector': self._symbol_sectors.get(symbol, 'unknown'),
                'rank': self._momentum_ranks.get(symbol, 0)
            }
        )

        self.logger.info(
            "Momentum signal generated",
            symbol=symbol,
            signal_type=signal_type.value,
            strength=strength.value,
            confidence=float(confidence),
            momentum=float(momentum),
            rs=float(rs)
        )

        return signal

    async def _calculate_indicators(self, symbol: str) -> None:
        """Calculate momentum indicators for a symbol."""
        if symbol not in self._price_data:
            return

        prices = self._price_data[symbol]

        # Calculate momentum (rate of change)
        if len(prices) >= self.mom_config.momentum_lookback_days:
            lookback = self.mom_config.momentum_lookback_days
            momentum = (prices[-1] - prices[-lookback]) / prices[-lookback]
            self._momentum[symbol] = momentum

        # Calculate short-term momentum
        if len(prices) >= self.mom_config.short_momentum_days:
            lookback = self.mom_config.short_momentum_days
            short_mom = (prices[-1] - prices[-lookback]) / prices[-lookback]
            self._short_momentum[symbol] = short_mom

        # Calculate long-term momentum
        if len(prices) >= self.mom_config.long_momentum_days:
            lookback = self.mom_config.long_momentum_days
            long_mom = (prices[-1] - prices[-lookback]) / prices[-lookback]
            self._long_momentum[symbol] = long_mom

        # Calculate RSI
        if len(prices) >= self.mom_config.rsi_period + 1:
            self._rsi[symbol] = self._calculate_rsi(prices, self.mom_config.rsi_period)

    def _calculate_rsi(self, prices: List[Decimal], period: int) -> Decimal:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return Decimal("50")

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(c, Decimal("0")) for c in changes[-period:]]
        losses = [abs(min(c, Decimal("0"))) for c in changes[-period:]]

        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return Decimal("100")

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    async def _update_rankings(self) -> None:
        """Update momentum rankings across all symbols."""
        if len(self._momentum) < 2:
            return

        # Sort symbols by momentum
        sorted_symbols = sorted(
            self._momentum.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Assign ranks
        for rank, (symbol, momentum) in enumerate(sorted_symbols, 1):
            self._momentum_ranks[symbol] = rank

        # Calculate relative strength (percentile rank)
        n = len(sorted_symbols)
        for rank, (symbol, momentum) in enumerate(sorted_symbols, 1):
            percentile = Decimal(str((n - rank) / n))
            self._relative_strength[symbol] = percentile

    async def _update_sector_performance(self) -> None:
        """Update sector performance metrics."""
        # Group symbols by sector
        sector_returns: Dict[str, List[Decimal]] = {}

        for symbol, sector in self._symbol_sectors.items():
            if symbol not in self._momentum:
                continue

            if sector not in sector_returns:
                sector_returns[sector] = []

            sector_returns[sector].append(self._momentum[symbol])

        # Calculate average return per sector
        for sector, returns in sector_returns.items():
            avg_return = sum(returns) / len(returns) if returns else Decimal("0")

            if sector not in self._sector_performance:
                self._sector_performance[sector] = []

            self._sector_performance[sector].append(avg_return)

            # Keep limited history
            max_len = self.mom_config.sector_momentum_period * 2
            if len(self._sector_performance[sector]) > max_len:
                self._sector_performance[sector] = self._sector_performance[sector][-max_len:]

        # Rank sectors
        if sector_returns:
            sorted_sectors = sorted(
                [(s, sum(r) / len(r)) for s, r in sector_returns.items()],
                key=lambda x: x[1],
                reverse=True
            )

            n = len(sorted_sectors)
            for rank, (sector, avg_return) in enumerate(sorted_sectors, 1):
                percentile = Decimal(str((n - rank) / n))
                self._sector_rankings[sector] = percentile

    async def _check_momentum_signal(
        self,
        symbol: str,
        momentum: Decimal,
        short_mom: Decimal,
        long_mom: Decimal
    ) -> int:
        """
        Check momentum for signal.

        Returns:
            1 for buy, -1 for sell, 0 for no signal
        """
        # Positive momentum and short > long = buy
        if momentum > 0 and short_mom > long_mom and short_mom > Decimal("0.01"):
            return 1

        # Negative momentum and short < long = sell
        if momentum < 0 and short_mom < long_mom and short_mom < Decimal("-0.01"):
            return -1

        return 0

    async def _check_rsi_signal(self, rsi: Decimal, momentum_signal: int) -> int:
        """
        Check RSI for confirmation.

        Returns:
            1 for buy, -1 for sell, 0 for neutral
        """
        # Don't buy if overbought
        if momentum_signal > 0 and rsi >= self.mom_config.rsi_overbought:
            return 0

        # Don't sell if oversold
        if momentum_signal < 0 and rsi <= self.mom_config.rsi_oversold:
            return 0

        # Confirm momentum signal
        if momentum_signal > 0 and rsi > Decimal("50"):
            return 1

        if momentum_signal < 0 and rsi < Decimal("50"):
            return -1

        return 0

    async def _check_relative_strength(self, symbol: str, rs: Decimal) -> int:
        """
        Check relative strength ranking.

        Returns:
            1 for buy, -1 for sell, 0 for neutral
        """
        # Buy if in top percentile
        if rs >= self.mom_config.rs_percentile_threshold:
            return 1

        # Sell/avoid if in bottom percentile
        if rs <= (Decimal("1") - self.mom_config.rs_percentile_threshold):
            return -1

        return 0

    async def _check_sector_rotation(self, symbol: str) -> int:
        """
        Check sector rotation signal.

        Returns:
            1 for buy, -1 for sell, 0 for neutral
        """
        sector = self._symbol_sectors.get(symbol)
        if not sector or sector not in self._sector_rankings:
            return 0

        sector_strength = self._sector_rankings[sector]

        # Buy if sector is strong
        if sector_strength >= self.mom_config.min_sector_strength:
            return 1

        # Sell if sector is weak
        if sector_strength <= (Decimal("1") - self.mom_config.min_sector_strength):
            return -1

        return 0

    async def _calculate_confidence(
        self,
        symbol: str,
        momentum: Decimal,
        rs: Decimal,
        rsi: Decimal,
        signals: List[int]
    ) -> Decimal:
        """Calculate signal confidence."""
        # Base confidence
        confidence = Decimal("0.5")

        # Signal agreement
        if signals:
            agreement = sum(1 for s in signals if s == signals[0])
            agreement_score = Decimal(str(agreement)) / Decimal(str(len(signals)))
            confidence += agreement_score * Decimal("0.3")

        # Momentum strength
        momentum_strength = min(abs(momentum), Decimal("0.2")) / Decimal("0.2")
        confidence += momentum_strength * Decimal("0.2")

        # Relative strength
        if rs > Decimal("0.8"):
            confidence += Decimal("0.1")
        elif rs < Decimal("0.2"):
            confidence += Decimal("0.1")

        # Volume confirmation
        if symbol in self._volume_data and len(self._volume_data[symbol]) > 20:
            volumes = self._volume_data[symbol]
            avg_volume = sum(volumes[-20:-1]) / Decimal("19")
            current_volume = volumes[-1]

            if current_volume > avg_volume * self.mom_config.min_volume_ratio:
                confidence += Decimal("0.1")

        return min(confidence, Decimal("1"))

    async def _determine_strength(
        self,
        momentum: Decimal,
        rs: Decimal
    ) -> SignalStrength:
        """Determine signal strength."""
        momentum_abs = abs(momentum)

        # Very strong if high momentum and high RS
        if momentum_abs > Decimal("0.15") and rs > Decimal("0.9"):
            return SignalStrength.VERY_STRONG

        # Strong if good momentum or high RS
        if momentum_abs > Decimal("0.10") or rs > Decimal("0.85"):
            return SignalStrength.STRONG

        # Moderate if decent momentum
        if momentum_abs > Decimal("0.05"):
            return SignalStrength.MODERATE

        return SignalStrength.WEAK

    async def _calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        momentum: Decimal,
        rs: Decimal
    ) -> Decimal:
        """Calculate position size based on momentum strength."""
        # Base size
        position_size_usd = self.mom_config.base_position_size

        # Scale by momentum and RS
        momentum_strength = min(abs(momentum), Decimal("0.2")) / Decimal("0.2")
        rs_strength = abs(rs - Decimal("0.5")) * Decimal("2")  # 0-1 scale

        combined_strength = (momentum_strength + rs_strength) / Decimal("2")
        multiplier = Decimal("1") + (combined_strength * self.mom_config.momentum_size_multiplier)

        position_size_usd *= multiplier

        # Clamp to max
        position_size_usd = min(position_size_usd, self.mom_config.max_position_size)

        # Convert to shares
        shares = position_size_usd / price

        return shares

    async def _calculate_stop_take(
        self,
        price: Decimal,
        signal_type: SignalType,
        momentum: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Calculate stop loss and take profit levels."""
        # Adjust based on momentum strength
        momentum_abs = min(abs(momentum), Decimal("0.2"))
        momentum_factor = Decimal("1") + momentum_abs

        if signal_type in [SignalType.BUY, SignalType.INCREASE]:
            stop_loss = price * (Decimal("1") - self.mom_config.stop_loss_pct)
            take_profit = price * (Decimal("1") + self.mom_config.take_profit_pct * momentum_factor)
        else:
            stop_loss = price * (Decimal("1") + self.mom_config.stop_loss_pct)
            take_profit = price * (Decimal("1") - self.mom_config.take_profit_pct * momentum_factor)

        return stop_loss, take_profit

    def get_momentum_summary(self) -> Dict[str, Any]:
        """Get momentum summary across all symbols."""
        # Get top momentum stocks
        top_symbols = sorted(
            self._momentum.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.mom_config.rank_top_n]

        # Get bottom momentum stocks
        bottom_symbols = sorted(
            self._momentum.items(),
            key=lambda x: x[1]
        )[:self.mom_config.rank_bottom_n]

        return {
            'symbols_tracked': len(self._price_data),
            'sectors_tracked': len(self._sector_rankings),
            'top_momentum': [
                {
                    'symbol': symbol,
                    'momentum': float(momentum),
                    'rs': float(self._relative_strength.get(symbol, Decimal("0"))),
                    'rank': self._momentum_ranks.get(symbol, 0)
                }
                for symbol, momentum in top_symbols
            ],
            'bottom_momentum': [
                {
                    'symbol': symbol,
                    'momentum': float(momentum),
                    'rs': float(self._relative_strength.get(symbol, Decimal("0"))),
                    'rank': self._momentum_ranks.get(symbol, 0)
                }
                for symbol, momentum in bottom_symbols
            ],
            'sector_rankings': {
                sector: float(score)
                for sector, score in sorted(
                    self._sector_rankings.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        }
