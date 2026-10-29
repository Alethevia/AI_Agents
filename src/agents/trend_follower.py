"""
Trend following agent for capturing sustained price movements.

Implements trend detection using moving averages, ADX, breakout detection,
and multiple timeframe analysis.
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentMode
from src.core.types.signals import Signal, SignalType, SignalStrength
from src.core.logging.logger import get_logger


class TrendFollowerConfig(BaseModel):
    """Trend follower configuration."""

    # Moving average parameters
    fast_ma_period: int = Field(default=10, description="Fast MA period", ge=2)
    slow_ma_period: int = Field(default=30, description="Slow MA period", ge=5)
    ma_type: str = Field(default="EMA", description="MA type (SMA, EMA, WMA)")

    # ADX parameters
    adx_period: int = Field(default=14, description="ADX period", ge=5)
    adx_threshold: Decimal = Field(
        default=Decimal("25"),
        description="ADX threshold for strong trend",
        ge=0,
        le=100
    )

    # Breakout parameters
    breakout_lookback: int = Field(
        default=20,
        description="Lookback period for breakout detection",
        ge=5
    )
    breakout_threshold: Decimal = Field(
        default=Decimal("0.02"),
        description="Breakout threshold as % of price",
        gt=0
    )

    # Position sizing
    base_position_size: Decimal = Field(
        default=Decimal("1000"),
        description="Base position size in USD",
        gt=0
    )
    max_position_size: Decimal = Field(
        default=Decimal("10000"),
        description="Maximum position size in USD",
        gt=0
    )
    trend_strength_multiplier: Decimal = Field(
        default=Decimal("2.0"),
        description="Position size multiplier based on trend strength"
    )

    # Risk management
    stop_loss_atr_multiplier: Decimal = Field(
        default=Decimal("2.0"),
        description="Stop loss as multiple of ATR",
        gt=0
    )
    take_profit_atr_multiplier: Decimal = Field(
        default=Decimal("4.0"),
        description="Take profit as multiple of ATR",
        gt=0
    )
    trailing_stop_atr_multiplier: Decimal = Field(
        default=Decimal("1.5"),
        description="Trailing stop as multiple of ATR",
        gt=0
    )

    # Timeframes
    primary_timeframe: str = Field(default="1h", description="Primary timeframe")
    confirmation_timeframes: List[str] = Field(
        default=["4h", "1d"],
        description="Confirmation timeframes"
    )

    # Filters
    min_confidence: Decimal = Field(
        default=Decimal("0.6"),
        description="Minimum confidence for signal",
        ge=0,
        le=1
    )
    min_volume_ratio: Decimal = Field(
        default=Decimal("1.2"),
        description="Minimum volume ratio vs average",
        gt=0
    )


class TrendFollowerAgent(BaseAgent):
    """
    Trend following agent that captures sustained price movements.

    Uses moving average crossovers, ADX for trend strength, breakout detection,
    and multiple timeframe analysis to identify and follow trends.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "TrendFollower",
        mode: AgentMode = AgentMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
        tf_config: Optional[TrendFollowerConfig] = None,
    ):
        """
        Initialize trend follower agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            mode: Operating mode
            config: General agent configuration
            tf_config: Trend follower specific configuration
        """
        super().__init__(agent_id, name, mode, config)

        # Trend follower configuration
        self.tf_config = tf_config or TrendFollowerConfig()

        # Market data storage
        self._price_data: Dict[str, List[Decimal]] = {}
        self._high_data: Dict[str, List[Decimal]] = {}
        self._low_data: Dict[str, List[Decimal]] = {}
        self._volume_data: Dict[str, List[Decimal]] = {}

        # Indicator cache
        self._fast_ma: Dict[str, Decimal] = {}
        self._slow_ma: Dict[str, Decimal] = {}
        self._adx: Dict[str, Decimal] = {}
        self._atr: Dict[str, Decimal] = {}
        self._trend_direction: Dict[str, int] = {}  # 1=up, -1=down, 0=neutral

        # Position tracking
        self._positions: Dict[str, Dict[str, Any]] = {}

        self.logger = get_logger(f"{__name__}.{self.name}")

    async def _initialize(self) -> None:
        """Initialize trend follower agent."""
        self.logger.info(
            "Initializing trend follower agent",
            agent_id=self.agent_id,
            config=self.tf_config.model_dump()
        )

        # Clear caches
        self._price_data.clear()
        self._fast_ma.clear()
        self._slow_ma.clear()
        self._adx.clear()
        self._atr.clear()
        self._trend_direction.clear()

        self.logger.info("Trend follower agent initialized")

    async def _cleanup(self) -> None:
        """Cleanup trend follower resources."""
        self.logger.info("Cleaning up trend follower agent")
        self._positions.clear()
        self.logger.info("Trend follower agent cleaned up")

    async def _update(self, market_data: Dict[str, Any]) -> None:
        """
        Update trend follower with new market data.

        Args:
            market_data: Market data dictionary
        """
        symbol = market_data.get('symbol')
        if not symbol:
            return

        # Update price data
        close = market_data.get('close') or market_data.get('price')
        if close:
            close = Decimal(str(close))
            if symbol not in self._price_data:
                self._price_data[symbol] = []
            self._price_data[symbol].append(close)

        # Update high/low data
        high = market_data.get('high')
        low = market_data.get('low')
        if high:
            if symbol not in self._high_data:
                self._high_data[symbol] = []
            self._high_data[symbol].append(Decimal(str(high)))
        if low:
            if symbol not in self._low_data:
                self._low_data[symbol] = []
            self._low_data[symbol].append(Decimal(str(low)))

        # Update volume data
        volume = market_data.get('volume')
        if volume:
            if symbol not in self._volume_data:
                self._volume_data[symbol] = []
            self._volume_data[symbol].append(Decimal(str(volume)))

        # Keep only necessary history
        max_period = max(
            self.tf_config.slow_ma_period,
            self.tf_config.adx_period,
            self.tf_config.breakout_lookback
        ) * 2

        for data_dict in [self._price_data, self._high_data, self._low_data, self._volume_data]:
            if symbol in data_dict and len(data_dict[symbol]) > max_period:
                data_dict[symbol] = data_dict[symbol][-max_period:]

        # Calculate indicators
        await self._calculate_indicators(symbol)

    async def _decide(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate trend following decision.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            Trading signal or None
        """
        # Need sufficient data
        if symbol not in self._price_data:
            return None

        if len(self._price_data[symbol]) < self.tf_config.slow_ma_period:
            return None

        current_price = self._price_data[symbol][-1]

        # Detect trend
        trend_signal, trend_strength = await self._detect_trend(symbol)

        if trend_signal == 0:
            return None

        # Check for breakout
        breakout_signal = await self._detect_breakout(symbol)

        # Get ADX value
        adx = self._adx.get(symbol, Decimal("0"))

        # Calculate confidence
        confidence = await self._calculate_confidence(
            symbol,
            trend_signal,
            trend_strength,
            breakout_signal,
            adx
        )

        if confidence < self.tf_config.min_confidence:
            return None

        # Determine signal type and strength
        signal_type = SignalType.BUY if trend_signal > 0 else SignalType.SELL

        if adx >= self.tf_config.adx_threshold and trend_strength > Decimal("0.7"):
            strength = SignalStrength.VERY_STRONG
        elif adx >= self.tf_config.adx_threshold or trend_strength > Decimal("0.6"):
            strength = SignalStrength.STRONG
        elif trend_strength > Decimal("0.4"):
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Calculate position size
        position_size = await self._calculate_position_size(
            symbol,
            current_price,
            trend_strength
        )

        # Calculate stop loss and take profit
        atr = self._atr.get(symbol, current_price * Decimal("0.02"))
        stop_loss, take_profit = await self._calculate_stop_take(
            current_price,
            atr,
            signal_type
        )

        # Create signal
        signal = Signal(
            signal_id=f"tf_{self.agent_id}_{symbol}_{uuid.uuid4().hex[:8]}",
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
                'fast_ma': self._fast_ma.get(symbol, Decimal("0")),
                'slow_ma': self._slow_ma.get(symbol, Decimal("0")),
                'adx': adx,
                'atr': atr,
                'trend_strength': trend_strength,
                'breakout_signal': Decimal(str(breakout_signal))
            },
            metadata={
                'agent_type': 'trend_follower',
                'trend_direction': trend_signal,
                'trend_strength': str(trend_strength),
                'adx': str(adx),
                'breakout_detected': breakout_signal != 0
            }
        )

        self.logger.info(
            "Trend signal generated",
            symbol=symbol,
            signal_type=signal_type.value,
            strength=strength.value,
            confidence=float(confidence),
            adx=float(adx),
            trend_strength=float(trend_strength)
        )

        return signal

    async def _calculate_indicators(self, symbol: str) -> None:
        """Calculate technical indicators for a symbol."""
        if symbol not in self._price_data:
            return

        prices = self._price_data[symbol]
        if len(prices) < max(self.tf_config.fast_ma_period, self.tf_config.slow_ma_period):
            return

        # Calculate moving averages
        self._fast_ma[symbol] = self._calculate_ma(
            prices,
            self.tf_config.fast_ma_period,
            self.tf_config.ma_type
        )
        self._slow_ma[symbol] = self._calculate_ma(
            prices,
            self.tf_config.slow_ma_period,
            self.tf_config.ma_type
        )

        # Calculate ATR
        if symbol in self._high_data and symbol in self._low_data:
            self._atr[symbol] = self._calculate_atr(symbol)

        # Calculate ADX
        if symbol in self._high_data and symbol in self._low_data:
            self._adx[symbol] = self._calculate_adx(symbol)

        # Determine trend direction
        if self._fast_ma[symbol] > self._slow_ma[symbol]:
            self._trend_direction[symbol] = 1
        elif self._fast_ma[symbol] < self._slow_ma[symbol]:
            self._trend_direction[symbol] = -1
        else:
            self._trend_direction[symbol] = 0

    def _calculate_ma(
        self,
        prices: List[Decimal],
        period: int,
        ma_type: str
    ) -> Decimal:
        """Calculate moving average."""
        if len(prices) < period:
            return Decimal("0")

        recent_prices = prices[-period:]

        if ma_type == "SMA":
            return sum(recent_prices) / len(recent_prices)
        elif ma_type == "EMA":
            return self._calculate_ema(prices, period)
        elif ma_type == "WMA":
            return self._calculate_wma(recent_prices)
        else:
            return sum(recent_prices) / len(recent_prices)

    def _calculate_ema(self, prices: List[Decimal], period: int) -> Decimal:
        """Calculate exponential moving average."""
        if len(prices) < period:
            return Decimal("0")

        multiplier = Decimal("2") / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_wma(self, prices: List[Decimal]) -> Decimal:
        """Calculate weighted moving average."""
        n = len(prices)
        weights = [Decimal(str(i + 1)) for i in range(n)]
        weighted_sum = sum(p * w for p, w in zip(prices, weights))
        return weighted_sum / sum(weights)

    def _calculate_atr(self, symbol: str, period: int = 14) -> Decimal:
        """Calculate Average True Range."""
        if symbol not in self._high_data or symbol not in self._low_data:
            return Decimal("0")

        highs = self._high_data[symbol]
        lows = self._low_data[symbol]
        closes = self._price_data[symbol]

        if len(highs) < period or len(lows) < period or len(closes) < period:
            return Decimal("0")

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return Decimal("0")

        return sum(true_ranges[-period:]) / period

    def _calculate_adx(self, symbol: str) -> Decimal:
        """Calculate Average Directional Index."""
        period = self.tf_config.adx_period

        if symbol not in self._high_data or symbol not in self._low_data:
            return Decimal("0")

        highs = self._high_data[symbol]
        lows = self._low_data[symbol]
        closes = self._price_data[symbol]

        if len(highs) < period + 1 or len(lows) < period + 1:
            return Decimal("0")

        # Calculate +DM and -DM
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i - 1]
            low_diff = lows[i - 1] - lows[i]

            plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else Decimal("0")
            minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else Decimal("0")

            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        if len(plus_dm_list) < period:
            return Decimal("0")

        # Calculate smoothed DM
        atr = self._atr.get(symbol, Decimal("0.01"))
        if atr == 0:
            return Decimal("0")

        plus_di = (sum(plus_dm_list[-period:]) / period) / atr * Decimal("100")
        minus_di = (sum(minus_dm_list[-period:]) / period) / atr * Decimal("100")

        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return Decimal("0")

        dx = abs(plus_di - minus_di) / di_sum * Decimal("100")

        return dx  # Simplified - full ADX would smooth this further

    async def _detect_trend(self, symbol: str) -> Tuple[int, Decimal]:
        """
        Detect trend direction and strength.

        Returns:
            Tuple of (direction, strength) where direction is 1, -1, or 0
            and strength is 0-1
        """
        if symbol not in self._fast_ma or symbol not in self._slow_ma:
            return 0, Decimal("0")

        fast_ma = self._fast_ma[symbol]
        slow_ma = self._slow_ma[symbol]

        if fast_ma == 0 or slow_ma == 0:
            return 0, Decimal("0")

        # Direction
        direction = 1 if fast_ma > slow_ma else -1

        # Strength based on MA separation
        separation = abs(fast_ma - slow_ma) / slow_ma
        strength = min(separation * Decimal("10"), Decimal("1"))  # Scale to 0-1

        return direction, strength

    async def _detect_breakout(self, symbol: str) -> int:
        """
        Detect price breakouts.

        Returns:
            1 for upward breakout, -1 for downward, 0 for none
        """
        if symbol not in self._high_data or symbol not in self._low_data:
            return 0

        lookback = self.tf_config.breakout_lookback
        highs = self._high_data[symbol]
        lows = self._low_data[symbol]
        prices = self._price_data[symbol]

        if len(highs) < lookback + 1 or len(lows) < lookback + 1:
            return 0

        current_price = prices[-1]
        recent_high = max(highs[-lookback - 1:-1])
        recent_low = min(lows[-lookback - 1:-1])

        threshold = self.tf_config.breakout_threshold

        # Check for upward breakout
        if current_price > recent_high * (Decimal("1") + threshold):
            return 1

        # Check for downward breakout
        if current_price < recent_low * (Decimal("1") - threshold):
            return -1

        return 0

    async def _calculate_confidence(
        self,
        symbol: str,
        trend_signal: int,
        trend_strength: Decimal,
        breakout_signal: int,
        adx: Decimal
    ) -> Decimal:
        """Calculate signal confidence."""
        confidence = Decimal("0.5")  # Base confidence

        # Boost if trend and breakout align
        if trend_signal == breakout_signal and breakout_signal != 0:
            confidence += Decimal("0.2")

        # Boost based on trend strength
        confidence += trend_strength * Decimal("0.2")

        # Boost based on ADX
        adx_score = min(adx / Decimal("50"), Decimal("1"))
        confidence += adx_score * Decimal("0.1")

        # Check volume confirmation
        if symbol in self._volume_data and len(self._volume_data[symbol]) > 20:
            volumes = self._volume_data[symbol]
            avg_volume = sum(volumes[-20:-1]) / Decimal("19")
            current_volume = volumes[-1]

            if current_volume > avg_volume * self.tf_config.min_volume_ratio:
                confidence += Decimal("0.1")

        return min(confidence, Decimal("1"))

    async def _calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        trend_strength: Decimal
    ) -> Decimal:
        """Calculate position size based on trend strength."""
        base_size = self.tf_config.base_position_size

        # Scale by trend strength
        multiplier = Decimal("1") + (trend_strength * self.tf_config.trend_strength_multiplier)
        position_size = base_size * multiplier

        # Clamp to max
        position_size = min(position_size, self.tf_config.max_position_size)

        # Convert to shares
        shares = position_size / price

        return shares

    async def _calculate_stop_take(
        self,
        price: Decimal,
        atr: Decimal,
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal]:
        """Calculate stop loss and take profit levels."""
        if signal_type in [SignalType.BUY, SignalType.INCREASE]:
            stop_loss = price - (atr * self.tf_config.stop_loss_atr_multiplier)
            take_profit = price + (atr * self.tf_config.take_profit_atr_multiplier)
        else:
            stop_loss = price + (atr * self.tf_config.stop_loss_atr_multiplier)
            take_profit = price - (atr * self.tf_config.take_profit_atr_multiplier)

        return stop_loss, take_profit

    def get_trend_summary(self) -> Dict[str, Any]:
        """Get trend summary across all symbols."""
        return {
            'symbols_tracked': len(self._price_data),
            'trends': {
                symbol: {
                    'direction': self._trend_direction.get(symbol, 0),
                    'fast_ma': float(self._fast_ma.get(symbol, Decimal("0"))),
                    'slow_ma': float(self._slow_ma.get(symbol, Decimal("0"))),
                    'adx': float(self._adx.get(symbol, Decimal("0"))),
                    'atr': float(self._atr.get(symbol, Decimal("0")))
                }
                for symbol in self._price_data.keys()
            }
        }
