"""
Signal generator module for generating trading signals from multiple sources.

Combines signals from agents, models, technical indicators, and other sources
with confidence weighting, filtering, and validation.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Callable

import numpy as np
from pydantic import BaseModel, Field

from src.core.logging.logger import get_logger, log_async_execution_time
from src.core.types.signals import Signal, SignalType, SignalStrength, SignalCollection
from src.strategy.signals.signal_combiner import (
    SignalCombiner,
    SignalCombinerConfig,
    CombinationMethod,
)


logger = get_logger(__name__)


class SignalSource(BaseModel):
    """Configuration for a signal source."""

    name: str = Field(..., description="Source name")
    source_type: str = Field(..., description="Source type (agent/model/indicator)")
    weight: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        description="Source weight in combination"
    )
    enabled: bool = Field(default=True, description="Whether source is enabled")
    min_confidence: Decimal = Field(
        default=Decimal("0.0"),
        ge=0,
        le=1,
        description="Minimum confidence threshold"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class SignalGeneratorConfig(BaseModel):
    """Configuration for signal generator."""

    # Signal sources
    sources: List[SignalSource] = Field(
        default_factory=list,
        description="Configured signal sources"
    )

    # Aggregation settings
    combination_method: CombinationMethod = Field(
        default=CombinationMethod.WEIGHTED_AVERAGE,
        description="Default combination method"
    )
    min_sources: int = Field(
        default=1,
        ge=1,
        description="Minimum sources required for signal"
    )
    require_agreement: bool = Field(
        default=False,
        description="Require sources to agree on direction"
    )
    min_agreement_pct: Decimal = Field(
        default=Decimal("0.6"),
        ge=0,
        le=1,
        description="Minimum agreement percentage"
    )

    # Filtering
    enable_filtering: bool = Field(
        default=True,
        description="Enable signal filtering"
    )
    min_confidence: Decimal = Field(
        default=Decimal("0.5"),
        ge=0,
        le=1,
        description="Minimum signal confidence"
    )
    max_signal_age_seconds: int = Field(
        default=300,
        ge=0,
        description="Maximum signal age in seconds"
    )
    filter_duplicates: bool = Field(
        default=True,
        description="Filter duplicate signals"
    )

    # Validation
    enable_validation: bool = Field(
        default=True,
        description="Enable signal validation"
    )
    require_market_data: bool = Field(
        default=True,
        description="Require market data for validation"
    )
    validate_price_levels: bool = Field(
        default=True,
        description="Validate stop loss/take profit levels"
    )

    # Performance
    cache_signals: bool = Field(
        default=True,
        description="Cache generated signals"
    )
    cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        description="Cache TTL in seconds"
    )


class SignalGenerator:
    """
    Generate and aggregate trading signals from multiple sources.

    Combines signals from agents, models, indicators, and other sources
    using configurable aggregation strategies, filtering, and validation.
    """

    def __init__(self, config: Optional[SignalGeneratorConfig] = None):
        """
        Initialize signal generator.

        Args:
            config: Generator configuration
        """
        self.config = config or SignalGeneratorConfig()
        self.logger = get_logger(f"{__name__}.SignalGenerator")

        # Initialize signal combiner
        combiner_config = SignalCombinerConfig(
            combination_method=self.config.combination_method,
            min_signals=self.config.min_sources,
            min_confidence=self.config.min_confidence,
            min_agreement=self.config.min_agreement_pct,
            enable_filters=self.config.enable_filtering,
            max_signal_age_seconds=self.config.max_signal_age_seconds,
            source_weights={
                source.name: source.weight
                for source in self.config.sources
            }
        )
        self.combiner = SignalCombiner(combiner_config)

        # Signal collection and cache
        self.signal_collection = SignalCollection()
        self._signal_cache: Dict[str, tuple] = {}  # (signal, timestamp)

        # Source registry
        self._sources: Dict[str, SignalSource] = {
            source.name: source
            for source in self.config.sources
        }

        # Signal handlers (for async signal generation)
        self._signal_handlers: Dict[str, Callable] = {}

        # Statistics
        self._stats = {
            'total_generated': 0,
            'filtered': 0,
            'validated': 0,
            'combined': 0,
            'cached': 0,
            'errors': 0,
        }

    @log_async_execution_time()
    async def generate_signal(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        sources: Optional[List[str]] = None,
    ) -> Optional[Signal]:
        """
        Generate a trading signal for a symbol.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol
            sources: Optional list of source names to use

        Returns:
            Generated signal or None
        """
        try:
            # Check cache
            if self.config.cache_signals:
                cached_signal = self._get_cached_signal(symbol)
                if cached_signal:
                    self._stats['cached'] += 1
                    return cached_signal

            # Collect signals from sources
            source_signals = await self._collect_signals(
                symbol,
                market_data,
                sources
            )

            if not source_signals:
                self.logger.debug(
                    "No signals collected from sources",
                    symbol=symbol,
                    requested_sources=sources
                )
                return None

            # Filter signals
            if self.config.enable_filtering:
                source_signals = await self._filter_signals(
                    source_signals,
                    symbol,
                    market_data
                )

            if not source_signals:
                self.logger.debug(
                    "All signals filtered out",
                    symbol=symbol
                )
                return None

            # Validate signals
            if self.config.enable_validation:
                source_signals = await self._validate_signals(
                    source_signals,
                    symbol,
                    market_data
                )

            if not source_signals:
                self.logger.debug(
                    "All signals failed validation",
                    symbol=symbol
                )
                return None

            # Combine signals
            combined = await self.combiner.combine_signals(
                source_signals,
                symbol
            )

            if not combined:
                return None

            signal = combined.signal
            self._stats['combined'] += 1

            # Cache signal
            if self.config.cache_signals:
                self._cache_signal(symbol, signal)

            # Add to collection
            self.signal_collection.add_signal(signal)

            self._stats['total_generated'] += 1

            self.logger.info(
                "Signal generated successfully",
                symbol=symbol,
                signal_type=signal.signal_type.value,
                confidence=float(signal.confidence),
                source_count=len(source_signals),
                agreement=float(combined.agreement_score)
            )

            return signal

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error generating signal",
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return None

    async def _collect_signals(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]],
        source_names: Optional[List[str]] = None
    ) -> List[Signal]:
        """Collect signals from configured sources."""
        signals = []

        # Determine which sources to query
        sources_to_query = source_names or list(self._sources.keys())

        # Filter by enabled sources
        sources_to_query = [
            name for name in sources_to_query
            if name in self._sources and self._sources[name].enabled
        ]

        # Collect signals from each source
        tasks = []
        for source_name in sources_to_query:
            if source_name in self._signal_handlers:
                task = self._signal_handlers[source_name](symbol, market_data)
                tasks.append((source_name, task))

        if tasks:
            # Execute all tasks concurrently
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            for (source_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Error collecting signal from source",
                        source=source_name,
                        error=str(result)
                    )
                    continue

                if result and isinstance(result, Signal):
                    signals.append(result)
                elif result and isinstance(result, list):
                    signals.extend(result)

        return signals

    async def _filter_signals(
        self,
        signals: List[Signal],
        symbol: str,
        market_data: Optional[Dict[str, Any]]
    ) -> List[Signal]:
        """Filter signals based on various criteria."""
        filtered = []
        now = datetime.utcnow()

        for signal in signals:
            # Check symbol
            if signal.symbol != symbol:
                self._stats['filtered'] += 1
                continue

            # Check age
            age = (now - signal.timestamp).total_seconds()
            if age > self.config.max_signal_age_seconds:
                self._stats['filtered'] += 1
                self.logger.debug(
                    "Signal filtered due to age",
                    symbol=symbol,
                    age_seconds=age,
                    source=signal.source
                )
                continue

            # Check expiry
            if signal.is_expired():
                self._stats['filtered'] += 1
                continue

            # Check source-specific confidence threshold
            source = self._sources.get(signal.source)
            if source:
                if signal.confidence < source.min_confidence:
                    self._stats['filtered'] += 1
                    continue

            # Check global confidence threshold
            if signal.confidence < self.config.min_confidence:
                self._stats['filtered'] += 1
                continue

            filtered.append(signal)

        # Filter duplicates
        if self.config.filter_duplicates and len(filtered) > 1:
            filtered = self._remove_duplicates(filtered)

        return filtered

    def _remove_duplicates(self, signals: List[Signal]) -> List[Signal]:
        """Remove duplicate signals, keeping the one with highest confidence."""
        seen = {}

        for signal in signals:
            key = (signal.symbol, signal.signal_type, signal.source)

            if key not in seen:
                seen[key] = signal
            else:
                # Keep signal with higher confidence
                if signal.confidence > seen[key].confidence:
                    seen[key] = signal

        return list(seen.values())

    async def _validate_signals(
        self,
        signals: List[Signal],
        symbol: str,
        market_data: Optional[Dict[str, Any]]
    ) -> List[Signal]:
        """Validate signals."""
        validated = []

        for signal in signals:
            # Validate market data requirement
            if self.config.require_market_data and not market_data:
                self._stats['filtered'] += 1
                continue

            # Validate price levels
            if self.config.validate_price_levels:
                if not self._validate_price_levels(signal, market_data):
                    self._stats['filtered'] += 1
                    self.logger.debug(
                        "Signal failed price level validation",
                        symbol=symbol,
                        source=signal.source
                    )
                    continue

            validated.append(signal)
            self._stats['validated'] += 1

        return validated

    def _validate_price_levels(
        self,
        signal: Signal,
        market_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Validate stop loss and take profit levels."""
        if not signal.market_price:
            return True

        market_price = signal.market_price

        # Validate stop loss
        if signal.stop_loss:
            if signal.is_buy():
                if signal.stop_loss >= market_price:
                    return False
            elif signal.is_sell():
                if signal.stop_loss <= market_price:
                    return False

        # Validate take profit
        if signal.take_profit:
            if signal.is_buy():
                if signal.take_profit <= market_price:
                    return False
            elif signal.is_sell():
                if signal.take_profit >= market_price:
                    return False

        # Validate risk/reward ratio
        risk_reward = signal.risk_reward_ratio()
        if risk_reward and risk_reward < Decimal("1.0"):
            # Risk/reward is unfavorable
            return False

        return True

    def _get_cached_signal(self, symbol: str) -> Optional[Signal]:
        """Get cached signal if available and not expired."""
        if symbol not in self._signal_cache:
            return None

        signal, timestamp = self._signal_cache[symbol]

        # Check if cache is still valid
        age = (datetime.utcnow() - timestamp).total_seconds()
        if age > self.config.cache_ttl_seconds:
            del self._signal_cache[symbol]
            return None

        return signal

    def _cache_signal(self, symbol: str, signal: Signal) -> None:
        """Cache a signal."""
        self._signal_cache[symbol] = (signal, datetime.utcnow())

    def register_source(
        self,
        name: str,
        source_type: str,
        handler: Callable,
        weight: Decimal = Decimal("1.0"),
        min_confidence: Decimal = Decimal("0.0"),
        **kwargs
    ) -> None:
        """
        Register a signal source.

        Args:
            name: Source name
            source_type: Source type (agent/model/indicator)
            handler: Async callable that generates signals
            weight: Source weight
            min_confidence: Minimum confidence threshold
            **kwargs: Additional metadata
        """
        source = SignalSource(
            name=name,
            source_type=source_type,
            weight=weight,
            min_confidence=min_confidence,
            metadata=kwargs
        )

        self._sources[name] = source
        self._signal_handlers[name] = handler

        # Update combiner weights
        self.combiner.config.source_weights[name] = weight

        self.logger.info(
            "Signal source registered",
            name=name,
            source_type=source_type,
            weight=float(weight)
        )

    def unregister_source(self, name: str) -> None:
        """Unregister a signal source."""
        if name in self._sources:
            del self._sources[name]
        if name in self._signal_handlers:
            del self._signal_handlers[name]
        if name in self.combiner.config.source_weights:
            del self.combiner.config.source_weights[name]

        self.logger.info("Signal source unregistered", name=name)

    def enable_source(self, name: str) -> None:
        """Enable a signal source."""
        if name in self._sources:
            self._sources[name].enabled = True
            self.logger.info("Signal source enabled", name=name)

    def disable_source(self, name: str) -> None:
        """Disable a signal source."""
        if name in self._sources:
            self._sources[name].enabled = False
            self.logger.info("Signal source disabled", name=name)

    def update_source_weight(self, name: str, weight: Decimal) -> None:
        """Update source weight."""
        if name in self._sources:
            self._sources[name].weight = weight
            self.combiner.config.source_weights[name] = weight
            self.logger.info(
                "Source weight updated",
                name=name,
                weight=float(weight)
            )

    def get_source_config(self, name: str) -> Optional[SignalSource]:
        """Get source configuration."""
        return self._sources.get(name)

    def list_sources(self) -> List[SignalSource]:
        """List all registered sources."""
        return list(self._sources.values())

    def get_enabled_sources(self) -> List[str]:
        """Get list of enabled source names."""
        return [
            name for name, source in self._sources.items()
            if source.enabled
        ]

    def clear_cache(self) -> None:
        """Clear signal cache."""
        self._signal_cache.clear()
        self.logger.info("Signal cache cleared")

    def get_signal_collection(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> SignalCollection:
        """Get signal collection with optional filtering."""
        collection = self.signal_collection

        if symbol:
            filtered_signals = collection.filter_by_symbol(symbol)
            collection = SignalCollection(signals=filtered_signals)

        if since:
            filtered_signals = [
                s for s in collection.signals
                if s.timestamp >= since
            ]
            collection = SignalCollection(signals=filtered_signals)

        if limit:
            filtered_signals = collection.signals[-limit:]
            collection = SignalCollection(signals=filtered_signals)

        return collection

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        combiner_stats = self.combiner.get_statistics()

        return {
            **self._stats,
            'combiner': combiner_stats,
            'sources': {
                'total': len(self._sources),
                'enabled': len(self.get_enabled_sources()),
                'disabled': len(self._sources) - len(self.get_enabled_sources()),
            },
            'cache': {
                'size': len(self._signal_cache),
                'hit_rate': (
                    self._stats['cached'] / self._stats['total_generated']
                    if self._stats['total_generated'] > 0 else 0
                ),
            },
            'success_rate': (
                self._stats['total_generated'] /
                (self._stats['total_generated'] + self._stats['errors'])
                if (self._stats['total_generated'] + self._stats['errors']) > 0 else 0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset generator statistics."""
        self._stats = {
            'total_generated': 0,
            'filtered': 0,
            'validated': 0,
            'combined': 0,
            'cached': 0,
            'errors': 0,
        }
        self.combiner.reset_statistics()
