"""
Portfolio Management Module

Comprehensive portfolio management system with:
- Advanced portfolio optimization (Mean-variance, Black-Litterman, Risk Parity, HRP, etc.)
- Portfolio rebalancing strategies (Threshold, Periodic, Tax-aware, Cost-optimized)
- Performance analytics (Risk-adjusted metrics, Rolling metrics, Benchmarking)
- Attribution analysis (Brinson, Factor, Sector, Style)
- Real-time portfolio tracking (P&L, Positions, Corporate actions)
"""

# Optimizer
from src.portfolio.optimizer.portfolio_optimizer import (
    PortfolioOptimizer,
    MultiPeriodOptimizer,
    OptimizationMethod,
    OptimizationConstraints,
    OptimizationResult
)

# Rebalancer
from src.portfolio.rebalancer.rebalancer import (
    PortfolioRebalancer,
    RebalanceMethod,
    RebalanceConfig,
    RebalanceResult,
    Position as RebalancerPosition
)

# Performance Analytics
from src.portfolio.analytics.performance_analytics import (
    PerformanceAnalyzer,
    BenchmarkComparison,
    PerformanceMetrics,
    RollingMetrics
)

# Attribution
from src.portfolio.analytics.attribution import (
    AttributionAnalyzer,
    MultiPeriodAttribution,
    StyleAttribution,
    AttributionMethod,
    BrinsonAttribution,
    FactorAttribution,
    SectorAttribution
)

# Tracker
from src.portfolio.tracker.portfolio_tracker import (
    PortfolioTracker,
    Transaction,
    TransactionType,
    Position,
    PortfolioSnapshot,
    CorporateAction,
    CorporateActionType
)

__all__ = [
    # Optimizer
    'PortfolioOptimizer',
    'MultiPeriodOptimizer',
    'OptimizationMethod',
    'OptimizationConstraints',
    'OptimizationResult',

    # Rebalancer
    'PortfolioRebalancer',
    'RebalanceMethod',
    'RebalanceConfig',
    'RebalanceResult',
    'RebalancerPosition',

    # Performance Analytics
    'PerformanceAnalyzer',
    'BenchmarkComparison',
    'PerformanceMetrics',
    'RollingMetrics',

    # Attribution
    'AttributionAnalyzer',
    'MultiPeriodAttribution',
    'StyleAttribution',
    'AttributionMethod',
    'BrinsonAttribution',
    'FactorAttribution',
    'SectorAttribution',

    # Tracker
    'PortfolioTracker',
    'Transaction',
    'TransactionType',
    'Position',
    'PortfolioSnapshot',
    'CorporateAction',
    'CorporateActionType'
]

__version__ = '1.0.0'
