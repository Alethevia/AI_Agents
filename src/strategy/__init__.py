"""
Trading strategy generation, optimization, and evaluation modules.

This package provides comprehensive tools for:
- Signal generation and aggregation from multiple sources
- Strategy generation using templates and ML
- Parameter optimization using various methods
- Portfolio optimization
- Strategy performance evaluation
"""

# Signal modules
from src.strategy.signals import (
    SignalCombiner,
    SignalCombinerConfig,
    CombinationMethod,
    ConflictResolution,
    CombinedSignal,
)

# Generator modules
from src.strategy.generators import (
    SignalGenerator,
    SignalGeneratorConfig,
    SignalSource,
    StrategyGenerator,
    StrategyGeneratorConfig,
    StrategyType,
    StrategyTemplate,
    StrategyDefinition,
    StrategyRule,
    ParameterDefinition,
    ParameterType,
)

# Optimizer modules
from src.strategy.optimizers import (
    ParameterOptimizer,
    ParameterOptimizerConfig,
    OptimizationMethod,
    ParameterSpace,
    OptimizationResult,
    PortfolioOptimizer,
    PortfolioOptimizerConfig,
    OptimizationObjective,
    PortfolioConstraints,
    PortfolioAllocation,
)

# Evaluator modules
from src.strategy.evaluators import (
    StrategyEvaluator,
    StrategyEvaluatorConfig,
    PerformanceMetrics,
    DrawdownPeriod,
)

__all__ = [
    # Signal modules
    'SignalCombiner',
    'SignalCombinerConfig',
    'CombinationMethod',
    'ConflictResolution',
    'CombinedSignal',

    # Signal generator
    'SignalGenerator',
    'SignalGeneratorConfig',
    'SignalSource',

    # Strategy generator
    'StrategyGenerator',
    'StrategyGeneratorConfig',
    'StrategyType',
    'StrategyTemplate',
    'StrategyDefinition',
    'StrategyRule',
    'ParameterDefinition',
    'ParameterType',

    # Parameter optimizer
    'ParameterOptimizer',
    'ParameterOptimizerConfig',
    'OptimizationMethod',
    'ParameterSpace',
    'OptimizationResult',

    # Portfolio optimizer
    'PortfolioOptimizer',
    'PortfolioOptimizerConfig',
    'OptimizationObjective',
    'PortfolioConstraints',
    'PortfolioAllocation',

    # Strategy evaluator
    'StrategyEvaluator',
    'StrategyEvaluatorConfig',
    'PerformanceMetrics',
    'DrawdownPeriod',
]

# Version
__version__ = '1.0.0'
