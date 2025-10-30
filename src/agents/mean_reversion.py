"""
Mean reversion agent for trading price deviations.

Implements Bollinger Bands strategy, statistical arbitrage,
Z-score based trading, and pairs trading logic.
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from .base_agent import BaseAgent, AgentMode
from src.core.types.signals import Signal, SignalType, SignalStrength
from src.core.logging.logger import get_logger


class MeanReversionConfig(BaseModel):
    """Mean reversion configuration."""

    # Bollinger Bands parameters
    bb_period: int = Field(default=20, description="Bollinger Bands period", ge=5)
    bb_std_dev: Decimal = Field(
        default=Decimal("2.0"),
        description="Standard deviations for bands",
        gt=0
    )
    bb_entry_threshold: Decimal = Field(
        default=Decimal("0.95"),
        description="Entry threshold (0-1, % to band)",
        ge=0,
        le=1
    )

    # Z-score parameters
    zscore_period: int = Field(default=20, description="Z-score lookback period", ge=5)
    zscore_entry_threshold: Decimal = Field(
        default=Decimal("2.0"),
        description="Z-score entry threshold",
        gt=0
    )
    zscore_exit_threshold: Decimal = Field(
        default=Decimal("0.5"),
        description="Z-score exit threshold",
        gt=0
    )

    # Statistical parameters
    min_observations: int = Field(
        default=30,
        description="Minimum observations for statistics",
        ge=10
    )
    significance_level: Decimal = Field(
        default=Decimal("0.05"),
        description="Significance level for tests",
        gt=0,
        lt=1
    )

    # Pairs trading parameters
    correlation_threshold: Decimal = Field(
        default=Decimal("0.8"),
        description="Minimum correlation for pairs",
        ge=-1,
        le=1
    )
    cointegration_pvalue: Decimal = Field(
        default=Decimal("0.05"),
        description="P-value threshold for cointegration",
        gt=0,
        lt=1
    )
    hedge_ratio_lookback: int = Field(
        default=30,
        description="Lookback for hedge ratio calculation",
        ge=10
    )

    # Position sizing
    base_position_size: Decimal = Field(
        default=Decimal("1000"),
        description="Base position size in USD",
        gt=0
    )
    max_position_size: Decimal = Field(
        default=Decimal("5000"),
        description="Maximum position size in USD",
        gt=0
    )

    # Risk management
    max_holding_period_hours: int = Field(
        default=48,
        description="Maximum holding period in hours",
        ge=1
    )
    stop_loss_pct: Decimal = Field(
        default=Decimal("0.03"),
        description="Stop loss percentage",
        gt=0,
        le=1
    )
    take_profit_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Take profit percentage",
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
        default=Decimal("0.8"),
        description="Minimum volume ratio vs average",
        gt=0
    )


class MeanReversionAgent(BaseAgent):
    """
    Mean reversion agent that trades price deviations from mean.

    Uses Bollinger Bands, Z-scores, statistical tests, and pairs trading
    to identify and exploit mean reversion opportunities.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "MeanReversion",
        mode: AgentMode = AgentMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
        mr_config: Optional[MeanReversionConfig] = None,
    ):
        """
        Initialize mean reversion agent.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            mode: Operating mode
            config: General agent configuration
            mr_config: Mean reversion specific configuration
        """
        super().__init__(agent_id, name, mode, config)

        # Mean reversion configuration
        self.mr_config = mr_config or MeanReversionConfig()

        # Price data storage
        self._price_data: Dict[str, List[Decimal]] = {}
        self._volume_data: Dict[str, List[Decimal]] = {}
        self._timestamp_data: Dict[str, List[datetime]] = {}

        # Statistical cache
        self._mean: Dict[str, Decimal] = {}
        self._std: Dict[str, Decimal] = {}
        self._zscore: Dict[str, Decimal] = {}
        self._bb_upper: Dict[str, Decimal] = {}
        self._bb_lower: Dict[str, Decimal] = {}
        self._bb_middle: Dict[str, Decimal] = {}

        # Pairs tracking
        self._pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._correlation_matrix: Dict[str, Dict[str, Decimal]] = {}

        # Position tracking
        self._open_positions: Dict[str, Dict[str, Any]] = {}

        self.logger = get_logger(f"{__name__}.{self.name}")

    async def _initialize(self) -> None:
        """Initialize mean reversion agent."""
        self.logger.info(
            "Initializing mean reversion agent",
            agent_id=self.agent_id,
            config=self.mr_config.model_dump()
        )

        # Clear caches
        self._price_data.clear()
        self._mean.clear()
        self._std.clear()
        self._zscore.clear()
        self._pairs.clear()

        self.logger.info("Mean reversion agent initialized")

    async def _cleanup(self) -> None:
        """Cleanup mean reversion resources."""
        self.logger.info("Cleaning up mean reversion agent")
        self._open_positions.clear()
        self.logger.info("Mean reversion agent cleaned up")

    async def _update(self, market_data: Dict[str, Any]) -> None:
        """
        Update mean reversion agent with new market data.

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

        # Keep only necessary history
        max_period = max(
            self.mr_config.bb_period,
            self.mr_config.zscore_period,
            self.mr_config.hedge_ratio_lookback
        ) * 2

        for data_dict in [self._price_data, self._volume_data, self._timestamp_data]:
            if symbol in data_dict and len(data_dict[symbol]) > max_period:
                data_dict[symbol] = data_dict[symbol][-max_period:]

        # Calculate statistics
        await self._calculate_statistics(symbol)

        # Update pairs if we have multiple symbols
        if len(self._price_data) >= 2:
            await self._update_pairs()

    async def _decide(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate mean reversion decision.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            Trading signal or None
        """
        # Need sufficient data
        if symbol not in self._price_data:
            return None

        if len(self._price_data[symbol]) < self.mr_config.min_observations:
            return None

        current_price = self._price_data[symbol][-1]

        # Check Bollinger Bands signal
        bb_signal = await self._check_bollinger_bands(symbol, current_price)

        # Check Z-score signal
        zscore_signal = await self._check_zscore(symbol)

        # Check for pairs trading opportunity
        pairs_signal = await self._check_pairs_trading(symbol, current_price)

        # Combine signals
        if bb_signal == 0 and zscore_signal == 0 and pairs_signal is None:
            return None

        # Determine final signal
        signal_votes = [bb_signal, zscore_signal]
        if pairs_signal:
            signal_votes.append(1 if pairs_signal == SignalType.BUY else -1)

        # Majority vote
        signal_direction = np.sign(sum(signal_votes))
        if signal_direction == 0:
            return None

        # Calculate confidence
        confidence = await self._calculate_confidence(
            symbol,
            bb_signal,
            zscore_signal,
            pairs_signal
        )

        if confidence < self.mr_config.min_confidence:
            return None

        # Determine signal type and strength
        signal_type = SignalType.BUY if signal_direction > 0 else SignalType.SELL

        zscore_abs = abs(self._zscore.get(symbol, Decimal("0")))
        if zscore_abs >= Decimal("3.0"):
            strength = SignalStrength.VERY_STRONG
        elif zscore_abs >= Decimal("2.5"):
            strength = SignalStrength.STRONG
        elif zscore_abs >= Decimal("2.0"):
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Calculate position size
        position_size = await self._calculate_position_size(symbol, current_price)

        # Calculate stop loss and take profit
        stop_loss, take_profit = await self._calculate_stop_take(
            current_price,
            signal_type
        )

        # Create signal
        signal = Signal(
            signal_id=f"mr_{self.agent_id}_{symbol}_{uuid.uuid4().hex[:8]}",
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
            expiry=datetime.utcnow() + timedelta(hours=self.mr_config.max_holding_period_hours),
            indicators={
                'zscore': self._zscore.get(symbol, Decimal("0")),
                'bb_upper': self._bb_upper.get(symbol, Decimal("0")),
                'bb_middle': self._bb_middle.get(symbol, Decimal("0")),
                'bb_lower': self._bb_lower.get(symbol, Decimal("0")),
                'mean': self._mean.get(symbol, Decimal("0")),
                'std': self._std.get(symbol, Decimal("0"))
            },
            metadata={
                'agent_type': 'mean_reversion',
                'bb_signal': bb_signal,
                'zscore_signal': zscore_signal,
                'pairs_trading': pairs_signal is not None
            }
        )

        self.logger.info(
            "Mean reversion signal generated",
            symbol=symbol,
            signal_type=signal_type.value,
            strength=strength.value,
            confidence=float(confidence),
            zscore=float(self._zscore.get(symbol, Decimal("0")))
        )

        return signal

    async def _calculate_statistics(self, symbol: str) -> None:
        """Calculate statistical measures for a symbol."""
        if symbol not in self._price_data:
            return

        prices = self._price_data[symbol]
        if len(prices) < self.mr_config.min_observations:
            return

        # Calculate mean and standard deviation
        price_array = np.array([float(p) for p in prices[-self.mr_config.bb_period:]])
        self._mean[symbol] = Decimal(str(np.mean(price_array)))
        self._std[symbol] = Decimal(str(np.std(price_array)))

        # Calculate Bollinger Bands
        std_multiplier = self.mr_config.bb_std_dev
        self._bb_middle[symbol] = self._mean[symbol]
        self._bb_upper[symbol] = self._mean[symbol] + (self._std[symbol] * std_multiplier)
        self._bb_lower[symbol] = self._mean[symbol] - (self._std[symbol] * std_multiplier)

        # Calculate Z-score
        current_price = prices[-1]
        if self._std[symbol] > 0:
            self._zscore[symbol] = (current_price - self._mean[symbol]) / self._std[symbol]
        else:
            self._zscore[symbol] = Decimal("0")

    async def _check_bollinger_bands(
        self,
        symbol: str,
        current_price: Decimal
    ) -> int:
        """
        Check Bollinger Bands for signal.

        Returns:
            1 for buy, -1 for sell, 0 for no signal
        """
        if symbol not in self._bb_upper or symbol not in self._bb_lower:
            return 0

        upper = self._bb_upper[symbol]
        lower = self._bb_lower[symbol]
        middle = self._bb_middle[symbol]

        if upper == lower:
            return 0

        # Calculate position within bands
        band_width = upper - lower
        distance_from_lower = current_price - lower
        position_in_band = distance_from_lower / band_width

        threshold = self.mr_config.bb_entry_threshold

        # Buy if near lower band (oversold)
        if position_in_band <= (Decimal("1") - threshold):
            return 1

        # Sell if near upper band (overbought)
        if position_in_band >= threshold:
            return -1

        return 0

    async def _check_zscore(self, symbol: str) -> int:
        """
        Check Z-score for signal.

        Returns:
            1 for buy, -1 for sell, 0 for no signal
        """
        if symbol not in self._zscore:
            return 0

        zscore = self._zscore[symbol]
        threshold = self.mr_config.zscore_entry_threshold

        # Buy if Z-score is very negative (oversold)
        if zscore <= -threshold:
            return 1

        # Sell if Z-score is very positive (overbought)
        if zscore >= threshold:
            return -1

        return 0

    async def _check_pairs_trading(
        self,
        symbol: str,
        current_price: Decimal
    ) -> Optional[SignalType]:
        """
        Check for pairs trading opportunities.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            Signal type or None
        """
        # Find best pair for this symbol
        best_pair = None
        best_spread_zscore = Decimal("0")

        for (sym1, sym2), pair_data in self._pairs.items():
            if sym1 != symbol and sym2 != symbol:
                continue

            spread_zscore = pair_data.get('spread_zscore', Decimal("0"))

            if abs(spread_zscore) > abs(best_spread_zscore):
                best_spread_zscore = spread_zscore
                best_pair = (sym1, sym2)

        if not best_pair:
            return None

        # Check if spread is significant
        if abs(best_spread_zscore) < self.mr_config.zscore_entry_threshold:
            return None

        # Determine signal based on which symbol and spread direction
        sym1, sym2 = best_pair
        if symbol == sym1:
            # If spread is high, short sym1 (sell)
            # If spread is low, long sym1 (buy)
            return SignalType.SELL if best_spread_zscore > 0 else SignalType.BUY
        else:
            # Opposite for sym2
            return SignalType.BUY if best_spread_zscore > 0 else SignalType.SELL

    async def _update_pairs(self) -> None:
        """Update pairs trading statistics."""
        symbols = list(self._price_data.keys())

        if len(symbols) < 2:
            return

        # Only update if we have enough data for all symbols
        min_len = min(len(self._price_data[s]) for s in symbols)
        if min_len < self.mr_config.hedge_ratio_lookback:
            return

        # Update correlations and cointegration
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                await self._analyze_pair(sym1, sym2)

    async def _analyze_pair(self, sym1: str, sym2: str) -> None:
        """Analyze a pair of symbols for cointegration."""
        lookback = self.mr_config.hedge_ratio_lookback

        prices1 = self._price_data[sym1][-lookback:]
        prices2 = self._price_data[sym2][-lookback:]

        if len(prices1) < lookback or len(prices2) < lookback:
            return

        # Convert to arrays
        p1 = np.array([float(p) for p in prices1])
        p2 = np.array([float(p) for p in prices2])

        # Calculate correlation
        correlation = Decimal(str(np.corrcoef(p1, p2)[0, 1]))

        # Check correlation threshold
        if abs(correlation) < self.mr_config.correlation_threshold:
            return

        # Calculate hedge ratio using linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(p2, p1)
        hedge_ratio = Decimal(str(slope))

        # Calculate spread
        spread = p1 - (float(hedge_ratio) * p2)
        spread_mean = Decimal(str(np.mean(spread)))
        spread_std = Decimal(str(np.std(spread)))

        # Calculate spread Z-score
        current_spread = Decimal(str(spread[-1]))
        spread_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else Decimal("0")

        # Store pair data
        pair_key = (sym1, sym2)
        self._pairs[pair_key] = {
            'correlation': correlation,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_zscore': spread_zscore,
            'cointegration_pvalue': Decimal(str(p_value)),
            'last_updated': datetime.utcnow()
        }

        self.logger.debug(
            "Pair analyzed",
            sym1=sym1,
            sym2=sym2,
            correlation=float(correlation),
            spread_zscore=float(spread_zscore)
        )

    async def _calculate_confidence(
        self,
        symbol: str,
        bb_signal: int,
        zscore_signal: int,
        pairs_signal: Optional[SignalType]
    ) -> Decimal:
        """Calculate signal confidence."""
        confidence = Decimal("0.5")  # Base confidence

        # Count agreeing signals
        signals = [bb_signal, zscore_signal]
        if pairs_signal:
            signals.append(1 if pairs_signal == SignalType.BUY else -1)

        # Agreement boosts confidence
        if len(signals) >= 2:
            agreement = sum(1 for s in signals if s == signals[0])
            confidence += Decimal(str(agreement)) / Decimal(str(len(signals))) * Decimal("0.3")

        # Z-score magnitude boosts confidence
        zscore_abs = abs(self._zscore.get(symbol, Decimal("0")))
        zscore_boost = min(zscore_abs / Decimal("3"), Decimal("1")) * Decimal("0.2")
        confidence += zscore_boost

        # Volume confirmation
        if symbol in self._volume_data and len(self._volume_data[symbol]) > 20:
            volumes = self._volume_data[symbol]
            avg_volume = sum(volumes[-20:-1]) / Decimal("19")
            current_volume = volumes[-1]

            if current_volume > avg_volume * self.mr_config.min_volume_ratio:
                confidence += Decimal("0.1")

        return min(confidence, Decimal("1"))

    async def _calculate_position_size(
        self,
        symbol: str,
        price: Decimal
    ) -> Decimal:
        """Calculate position size."""
        # Base size
        position_size_usd = self.mr_config.base_position_size

        # Scale by Z-score magnitude (higher deviation = larger position)
        zscore_abs = abs(self._zscore.get(symbol, Decimal("0")))
        if zscore_abs > Decimal("2"):
            multiplier = min(zscore_abs / Decimal("2"), Decimal("2"))
            position_size_usd *= multiplier

        # Clamp to max
        position_size_usd = min(position_size_usd, self.mr_config.max_position_size)

        # Convert to shares
        shares = position_size_usd / price

        return shares

    async def _calculate_stop_take(
        self,
        price: Decimal,
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal]:
        """Calculate stop loss and take profit levels."""
        if signal_type in [SignalType.BUY, SignalType.INCREASE]:
            stop_loss = price * (Decimal("1") - self.mr_config.stop_loss_pct)
            take_profit = price * (Decimal("1") + self.mr_config.take_profit_pct)
        else:
            stop_loss = price * (Decimal("1") + self.mr_config.stop_loss_pct)
            take_profit = price * (Decimal("1") - self.mr_config.take_profit_pct)

        return stop_loss, take_profit

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get statistical summary across all symbols."""
        return {
            'symbols_tracked': len(self._price_data),
            'pairs_identified': len(self._pairs),
            'statistics': {
                symbol: {
                    'mean': float(self._mean.get(symbol, Decimal("0"))),
                    'std': float(self._std.get(symbol, Decimal("0"))),
                    'zscore': float(self._zscore.get(symbol, Decimal("0"))),
                    'bb_upper': float(self._bb_upper.get(symbol, Decimal("0"))),
                    'bb_lower': float(self._bb_lower.get(symbol, Decimal("0")))
                }
                for symbol in self._price_data.keys()
            },
            'pairs': {
                f"{sym1}_{sym2}": {
                    'correlation': float(data['correlation']),
                    'hedge_ratio': float(data['hedge_ratio']),
                    'spread_zscore': float(data['spread_zscore'])
                }
                for (sym1, sym2), data in self._pairs.items()
            }
        }
