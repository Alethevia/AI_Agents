"""
Risk management system.

This module provides comprehensive risk management functionality including:
- Value at Risk (VaR) calculations
- Risk metrics (volatility, Sharpe ratio, drawdown, etc.)
- Position and risk limit enforcement
- Real-time risk monitoring
- Stress testing and scenario analysis
"""

# Models
from .models import (
    # VaR models
    VaRResult,
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    CornishFisherVaR,
    VaRCalculator,

    # Risk metrics
    RiskMetrics,
    VolatilityCalculator,
    PerformanceMetrics,
    DrawdownCalculator,
    BetaCalculator,
    RiskMetricsCalculator,
)

# Limits
from .limits import (
    # Position limits
    LimitViolation,
    PositionLimitCheck,
    PositionSizeEnforcer,
    ExposureLimitEnforcer,
    CorrelationLimitEnforcer,
    LeverageLimitEnforcer,
    PositionLimitManager,

    # Risk limits
    RiskLimitCheck,
    VaRLimitEnforcer,
    VolatilityLimitEnforcer,
    DrawdownLimitEnforcer,
    LossLimitEnforcer,
    CircuitBreaker,
    RiskLimitManager,
)

# Monitoring
from .monitoring import (
    RiskAlert,
    RiskSnapshot,
    AlertManager,
    RiskMonitor,
)

# Analytics
from .analytics import (
    StressTestResult,
    StressScenario,
    HistoricalScenarioAnalyzer,
    HypotheticalScenarioAnalyzer,
    FactorShockAnalyzer,
    ExtremeValueAnalyzer,
    StressTestManager,
)

__all__ = [
    # VaR models
    'VaRResult',
    'HistoricalVaR',
    'ParametricVaR',
    'MonteCarloVaR',
    'CornishFisherVaR',
    'VaRCalculator',

    # Risk metrics
    'RiskMetrics',
    'VolatilityCalculator',
    'PerformanceMetrics',
    'DrawdownCalculator',
    'BetaCalculator',
    'RiskMetricsCalculator',

    # Position limits
    'LimitViolation',
    'PositionLimitCheck',
    'PositionSizeEnforcer',
    'ExposureLimitEnforcer',
    'CorrelationLimitEnforcer',
    'LeverageLimitEnforcer',
    'PositionLimitManager',

    # Risk limits
    'RiskLimitCheck',
    'VaRLimitEnforcer',
    'VolatilityLimitEnforcer',
    'DrawdownLimitEnforcer',
    'LossLimitEnforcer',
    'CircuitBreaker',
    'RiskLimitManager',

    # Monitoring
    'RiskAlert',
    'RiskSnapshot',
    'AlertManager',
    'RiskMonitor',

    # Analytics
    'StressTestResult',
    'StressScenario',
    'HistoricalScenarioAnalyzer',
    'HypotheticalScenarioAnalyzer',
    'FactorShockAnalyzer',
    'ExtremeValueAnalyzer',
    'StressTestManager',
]

__version__ = '1.0.0'
