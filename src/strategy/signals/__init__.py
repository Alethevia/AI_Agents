"""Signal combination and aggregation modules."""

from src.strategy.signals.signal_combiner import (
    SignalCombiner,
    SignalCombinerConfig,
    CombinationMethod,
    ConflictResolution,
    CombinedSignal,
)

__all__ = [
    'SignalCombiner',
    'SignalCombinerConfig',
    'CombinationMethod',
    'ConflictResolution',
    'CombinedSignal',
]
