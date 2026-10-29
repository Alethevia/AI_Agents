"""
Signal combiner module for combining trading signals from multiple sources.

Provides various methods to aggregate signals including weighted average,
voting systems, ensemble methods, and conflict resolution strategies.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.core.logging.logger import get_logger
from src.core.types.signals import Signal, SignalType, SignalStrength


logger = get_logger(__name__)


class CombinationMethod(str, Enum):
    """Signal combination methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS_VOTE = "unanimous_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RANK_BASED = "rank_based"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"
    MEDIAN = "median"
    MAX_CONFIDENCE = "max_confidence"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""

    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_RECENT = "most_recent"
    WEIGHTED_AVERAGE = "weighted_average"
    DISCARD = "discard"
    HOLD = "hold"


class SignalCombinerConfig(BaseModel):
    """Configuration for signal combiner."""

    combination_method: CombinationMethod = Field(
        default=CombinationMethod.WEIGHTED_AVERAGE,
        description="Primary combination method"
    )
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.HIGHEST_CONFIDENCE,
        description="Conflict resolution strategy"
    )
    min_signals: int = Field(
        default=2,
        ge=1,
        description="Minimum number of signals required"
    )
    min_confidence: Decimal = Field(
        default=Decimal("0.5"),
        ge=0,
        le=1,
        description="Minimum confidence threshold"
    )
    min_agreement: Decimal = Field(
        default=Decimal("0.6"),
        ge=0,
        le=1,
        description="Minimum agreement threshold for voting"
    )
    source_weights: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Source-specific weights"
    )
    enable_filters: bool = Field(
        default=True,
        description="Enable signal filtering"
    )
    max_signal_age_seconds: int = Field(
        default=300,
        ge=0,
        description="Maximum signal age in seconds"
    )


class CombinedSignal(BaseModel):
    """Combined signal result with metadata."""

    signal: Signal = Field(..., description="Combined signal")
    source_signals: List[Signal] = Field(..., description="Source signals")
    method: CombinationMethod = Field(..., description="Combination method used")
    agreement_score: Decimal = Field(
        ...,
        ge=0,
        le=1,
        description="Agreement score among sources"
    )
    conflict_detected: bool = Field(
        default=False,
        description="Whether conflicts were detected"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class SignalCombiner:
    """
    Combine trading signals from multiple sources.

    Supports various combination methods, conflict resolution strategies,
    and signal filtering to produce high-quality aggregated signals.
    """

    def __init__(self, config: Optional[SignalCombinerConfig] = None):
        """
        Initialize signal combiner.

        Args:
            config: Combiner configuration
        """
        self.config = config or SignalCombinerConfig()
        self.logger = get_logger(f"{__name__}.SignalCombiner")

        # Statistics
        self._stats = {
            'total_combinations': 0,
            'conflicts_detected': 0,
            'signals_filtered': 0,
            'successful_combinations': 0,
        }

    async def combine_signals(
        self,
        signals: List[Signal],
        symbol: str,
        method: Optional[CombinationMethod] = None,
    ) -> Optional[CombinedSignal]:
        """
        Combine multiple signals into a single signal.

        Args:
            signals: List of signals to combine
            symbol: Trading symbol
            method: Override combination method

        Returns:
            Combined signal or None if combination fails
        """
        try:
            self._stats['total_combinations'] += 1

            # Filter signals
            filtered_signals = await self._filter_signals(signals, symbol)

            if len(filtered_signals) < self.config.min_signals:
                self.logger.warning(
                    "Insufficient signals after filtering",
                    symbol=symbol,
                    original_count=len(signals),
                    filtered_count=len(filtered_signals),
                    min_required=self.config.min_signals
                )
                return None

            # Use specified method or default
            combination_method = method or self.config.combination_method

            # Detect conflicts
            has_conflict, conflict_type = self._detect_conflicts(filtered_signals)

            if has_conflict:
                self._stats['conflicts_detected'] += 1
                self.logger.info(
                    "Conflict detected in signals",
                    symbol=symbol,
                    conflict_type=conflict_type,
                    signal_count=len(filtered_signals)
                )

            # Combine signals using specified method
            combined_signal = await self._combine_by_method(
                filtered_signals,
                symbol,
                combination_method
            )

            if not combined_signal:
                return None

            # Calculate agreement score
            agreement = self._calculate_agreement(filtered_signals)

            # Create combined signal result
            result = CombinedSignal(
                signal=combined_signal,
                source_signals=filtered_signals,
                method=combination_method,
                agreement_score=agreement,
                conflict_detected=has_conflict,
                metadata={
                    'original_count': len(signals),
                    'filtered_count': len(filtered_signals),
                    'conflict_type': conflict_type,
                }
            )

            self._stats['successful_combinations'] += 1

            self.logger.info(
                "Signals combined successfully",
                symbol=symbol,
                method=combination_method.value,
                signal_count=len(filtered_signals),
                final_type=combined_signal.signal_type.value,
                confidence=float(combined_signal.confidence),
                agreement=float(agreement)
            )

            return result

        except Exception as e:
            self.logger.error(
                "Error combining signals",
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return None

    async def _filter_signals(
        self,
        signals: List[Signal],
        symbol: str
    ) -> List[Signal]:
        """Filter signals based on various criteria."""
        if not self.config.enable_filters:
            return signals

        filtered = []
        now = datetime.utcnow()

        for signal in signals:
            # Filter by symbol
            if signal.symbol != symbol:
                self._stats['signals_filtered'] += 1
                continue

            # Filter by age
            age_seconds = (now - signal.timestamp).total_seconds()
            if age_seconds > self.config.max_signal_age_seconds:
                self._stats['signals_filtered'] += 1
                self.logger.debug(
                    "Signal filtered due to age",
                    symbol=symbol,
                    age_seconds=age_seconds,
                    max_age=self.config.max_signal_age_seconds
                )
                continue

            # Filter by expiry
            if signal.is_expired():
                self._stats['signals_filtered'] += 1
                continue

            # Filter by confidence
            if signal.confidence < self.config.min_confidence:
                self._stats['signals_filtered'] += 1
                continue

            filtered.append(signal)

        return filtered

    def _detect_conflicts(self, signals: List[Signal]) -> Tuple[bool, Optional[str]]:
        """Detect conflicts among signals."""
        if len(signals) < 2:
            return False, None

        signal_types = [s.signal_type for s in signals]

        # Check for buy/sell conflicts
        has_buy = any(st in [SignalType.BUY, SignalType.INCREASE] for st in signal_types)
        has_sell = any(st in [SignalType.SELL, SignalType.REDUCE] for st in signal_types)

        if has_buy and has_sell:
            return True, "buy_sell_conflict"

        # Check for strength conflicts
        strengths = [s.strength for s in signals]
        if len(set(strengths)) > 1:
            return True, "strength_conflict"

        return False, None

    async def _combine_by_method(
        self,
        signals: List[Signal],
        symbol: str,
        method: CombinationMethod
    ) -> Optional[Signal]:
        """Combine signals using specified method."""
        combiners = {
            CombinationMethod.WEIGHTED_AVERAGE: self._combine_weighted_average,
            CombinationMethod.MAJORITY_VOTE: self._combine_majority_vote,
            CombinationMethod.UNANIMOUS_VOTE: self._combine_unanimous_vote,
            CombinationMethod.CONFIDENCE_WEIGHTED: self._combine_confidence_weighted,
            CombinationMethod.RANK_BASED: self._combine_rank_based,
            CombinationMethod.BAYESIAN: self._combine_bayesian,
            CombinationMethod.ENSEMBLE: self._combine_ensemble,
            CombinationMethod.MEDIAN: self._combine_median,
            CombinationMethod.MAX_CONFIDENCE: self._combine_max_confidence,
        }

        combiner = combiners.get(method)
        if not combiner:
            self.logger.error(f"Unknown combination method: {method}")
            return None

        return await combiner(signals, symbol)

    async def _combine_weighted_average(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals using weighted average."""
        # Calculate weights
        weights = []
        for signal in signals:
            # Use source weight if available, else use confidence
            source_weight = self.config.source_weights.get(
                signal.source,
                Decimal("1.0")
            )
            weight = source_weight * signal.confidence
            weights.append(float(weight))

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return None

        weights = [w / total_weight for w in weights]

        # Calculate weighted signal type
        signal_scores = {}
        for signal, weight in zip(signals, weights):
            if signal.signal_type not in signal_scores:
                signal_scores[signal.signal_type] = 0
            signal_scores[signal.signal_type] += weight

        # Select signal type with highest score
        signal_type = max(signal_scores.items(), key=lambda x: x[1])[0]

        # Calculate weighted confidence
        confidence = Decimal(str(sum(
            float(s.confidence) * w
            for s, w in zip(signals, weights)
        )))

        # Determine strength based on confidence
        strength = self._confidence_to_strength(confidence)

        # Create combined signal
        return await self._create_combined_signal(
            signals=signals,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            method="weighted_average"
        )

    async def _combine_majority_vote(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals using majority voting."""
        # Count votes for each signal type
        votes = {}
        for signal in signals:
            if signal.signal_type not in votes:
                votes[signal.signal_type] = []
            votes[signal.signal_type].append(signal)

        # Find majority
        max_votes = max(len(v) for v in votes.values())
        majority_type = [k for k, v in votes.items() if len(v) == max_votes][0]

        # Check if majority meets threshold
        agreement = Decimal(str(max_votes / len(signals)))
        if agreement < self.config.min_agreement:
            self.logger.debug(
                "Majority vote failed to meet agreement threshold",
                symbol=symbol,
                agreement=float(agreement),
                threshold=float(self.config.min_agreement)
            )
            return None

        # Calculate average confidence of majority
        majority_signals = votes[majority_type]
        avg_confidence = sum(s.confidence for s in majority_signals) / len(majority_signals)

        strength = self._confidence_to_strength(avg_confidence)

        return await self._create_combined_signal(
            signals=majority_signals,
            symbol=symbol,
            signal_type=majority_type,
            strength=strength,
            confidence=avg_confidence,
            method="majority_vote"
        )

    async def _combine_unanimous_vote(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals requiring unanimous agreement."""
        # Check if all signals agree on type
        signal_types = set(s.signal_type for s in signals)

        if len(signal_types) > 1:
            self.logger.debug(
                "Unanimous vote failed - signals disagree",
                symbol=symbol,
                signal_types=[st.value for st in signal_types]
            )
            return None

        signal_type = list(signal_types)[0]
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        strength = self._confidence_to_strength(avg_confidence)

        return await self._create_combined_signal(
            signals=signals,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            method="unanimous_vote"
        )

    async def _combine_confidence_weighted(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals weighted purely by confidence."""
        # Calculate confidence weights
        confidences = [float(s.confidence) for s in signals]
        total_confidence = sum(confidences)

        if total_confidence == 0:
            return None

        weights = [c / total_confidence for c in confidences]

        # Calculate weighted signal type
        signal_scores = {}
        for signal, weight in zip(signals, weights):
            if signal.signal_type not in signal_scores:
                signal_scores[signal.signal_type] = 0
            signal_scores[signal.signal_type] += weight

        signal_type = max(signal_scores.items(), key=lambda x: x[1])[0]

        # Weight confidence by agreement
        type_signals = [s for s in signals if s.signal_type == signal_type]
        avg_confidence = sum(s.confidence for s in type_signals) / len(type_signals)

        strength = self._confidence_to_strength(avg_confidence)

        return await self._create_combined_signal(
            signals=type_signals,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            method="confidence_weighted"
        )

    async def _combine_rank_based(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals using rank-based method."""
        # Rank signals by confidence
        ranked = sorted(signals, key=lambda s: s.confidence, reverse=True)

        # Use top 50% of signals
        top_n = max(1, len(ranked) // 2)
        top_signals = ranked[:top_n]

        # Use majority vote on top signals
        return await self._combine_majority_vote(top_signals, symbol)

    async def _combine_bayesian(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals using Bayesian approach."""
        # Group by signal type
        signal_groups = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal)

        # Calculate posterior probability for each signal type
        posteriors = {}
        for signal_type, group in signal_groups.items():
            # Prior based on frequency
            prior = Decimal(str(len(group) / len(signals)))

            # Likelihood based on average confidence
            likelihood = sum(s.confidence for s in group) / len(group)

            # Posterior (unnormalized)
            posteriors[signal_type] = prior * likelihood

        # Normalize posteriors
        total = sum(posteriors.values())
        if total == 0:
            return None

        posteriors = {k: v / total for k, v in posteriors.items()}

        # Select type with highest posterior
        signal_type = max(posteriors.items(), key=lambda x: x[1])[0]
        confidence = posteriors[signal_type]
        strength = self._confidence_to_strength(confidence)

        return await self._create_combined_signal(
            signals=signal_groups[signal_type],
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            method="bayesian"
        )

    async def _combine_ensemble(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals using ensemble of methods."""
        # Try multiple methods and vote on results
        methods = [
            CombinationMethod.WEIGHTED_AVERAGE,
            CombinationMethod.MAJORITY_VOTE,
            CombinationMethod.CONFIDENCE_WEIGHTED,
        ]

        results = []
        for method in methods:
            result = await self._combine_by_method(signals, symbol, method)
            if result:
                results.append(result)

        if not results:
            return None

        # Vote on ensemble results
        return await self._combine_majority_vote(results, symbol)

    async def _combine_median(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Combine signals using median confidence."""
        # Group by signal type
        signal_groups = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal)

        # Find group with most signals
        largest_group_type = max(
            signal_groups.items(),
            key=lambda x: len(x[1])
        )[0]

        group = signal_groups[largest_group_type]

        # Calculate median confidence
        confidences = sorted([float(s.confidence) for s in group])
        median_confidence = Decimal(str(np.median(confidences)))

        strength = self._confidence_to_strength(median_confidence)

        return await self._create_combined_signal(
            signals=group,
            symbol=symbol,
            signal_type=largest_group_type,
            strength=strength,
            confidence=median_confidence,
            method="median"
        )

    async def _combine_max_confidence(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[Signal]:
        """Select signal with maximum confidence."""
        max_signal = max(signals, key=lambda s: s.confidence)

        # Return a copy with updated metadata
        return Signal(
            signal_id=f"combined_{max_signal.signal_id}",
            symbol=symbol,
            signal_type=max_signal.signal_type,
            strength=max_signal.strength,
            confidence=max_signal.confidence,
            source="signal_combiner",
            strategy_id=max_signal.strategy_id,
            features=max_signal.features,
            indicators=max_signal.indicators,
            target_price=max_signal.target_price,
            stop_loss=max_signal.stop_loss,
            take_profit=max_signal.take_profit,
            position_size=max_signal.position_size,
            market_price=max_signal.market_price,
            market_regime=max_signal.market_regime,
            notes=f"Combined from {len(signals)} signals using max_confidence",
            metadata={
                'original_source': max_signal.source,
                'combination_method': 'max_confidence',
                'source_count': len(signals),
            }
        )

    def _calculate_agreement(self, signals: List[Signal]) -> Decimal:
        """Calculate agreement score among signals."""
        if len(signals) < 2:
            return Decimal("1.0")

        # Count signals by type
        type_counts = {}
        for signal in signals:
            if signal.signal_type not in type_counts:
                type_counts[signal.signal_type] = 0
            type_counts[signal.signal_type] += 1

        # Agreement is proportion of majority
        max_count = max(type_counts.values())
        agreement = Decimal(str(max_count / len(signals)))

        return agreement

    def _confidence_to_strength(self, confidence: Decimal) -> SignalStrength:
        """Convert confidence to strength level."""
        if confidence >= Decimal("0.8"):
            return SignalStrength.VERY_STRONG
        elif confidence >= Decimal("0.65"):
            return SignalStrength.STRONG
        elif confidence >= Decimal("0.5"):
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    async def _create_combined_signal(
        self,
        signals: List[Signal],
        symbol: str,
        signal_type: SignalType,
        strength: SignalStrength,
        confidence: Decimal,
        method: str
    ) -> Signal:
        """Create a combined signal from multiple source signals."""
        # Aggregate features and indicators
        combined_features = {}
        combined_indicators = {}

        for signal in signals:
            combined_features.update(signal.features)
            combined_indicators.update(signal.indicators)

        # Calculate average prices if available
        prices_with_values = [
            s for s in signals
            if s.market_price is not None
        ]

        market_price = None
        if prices_with_values:
            market_price = sum(s.market_price for s in prices_with_values) / len(prices_with_values)

        # Aggregate target/stop/profit
        target_prices = [s.target_price for s in signals if s.target_price]
        stop_losses = [s.stop_loss for s in signals if s.stop_loss]
        take_profits = [s.take_profit for s in signals if s.take_profit]

        target_price = sum(target_prices) / len(target_prices) if target_prices else None
        stop_loss = sum(stop_losses) / len(stop_losses) if stop_losses else None
        take_profit = sum(take_profits) / len(take_profits) if take_profits else None

        return Signal(
            signal_id=f"combined_{datetime.utcnow().timestamp()}",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source="signal_combiner",
            features=combined_features,
            indicators=combined_indicators,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_price=market_price,
            notes=f"Combined from {len(signals)} signals using {method}",
            metadata={
                'combination_method': method,
                'source_count': len(signals),
                'sources': [s.source for s in signals],
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get combiner statistics."""
        return {
            **self._stats,
            'success_rate': (
                self._stats['successful_combinations'] / self._stats['total_combinations']
                if self._stats['total_combinations'] > 0 else 0
            ),
            'conflict_rate': (
                self._stats['conflicts_detected'] / self._stats['total_combinations']
                if self._stats['total_combinations'] > 0 else 0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset combiner statistics."""
        self._stats = {
            'total_combinations': 0,
            'conflicts_detected': 0,
            'signals_filtered': 0,
            'successful_combinations': 0,
        }
