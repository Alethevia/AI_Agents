"""
Risk management configuration.
"""

from typing import Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class RiskModel(str, Enum):
    """Risk modeling approaches."""
    VARIANCE = "variance"
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"


class PortfolioOptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"  # Hierarchical Risk Parity
    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"


class RiskConfig(BaseModel):
    """Risk management configuration."""

    # Risk models
    primary_risk_model: RiskModel = Field(
        default=RiskModel.CVAR,
        description="Primary risk model"
    )
    secondary_risk_models: List[RiskModel] = Field(
        default=[RiskModel.VAR, RiskModel.MAX_DRAWDOWN],
        description="Secondary risk models"
    )

    # Portfolio limits
    max_portfolio_var: float = Field(
        default=0.02,
        description="Maximum portfolio VaR",
        gt=0,
        le=1.0
    )
    max_portfolio_cvar: float = Field(
        default=0.03,
        description="Maximum portfolio CVaR",
        gt=0,
        le=1.0
    )
    max_drawdown: float = Field(
        default=0.15,
        description="Maximum allowed drawdown",
        gt=0,
        le=1.0
    )

    # Position limits
    max_position_var: float = Field(
        default=0.005,
        description="Maximum position VaR",
        gt=0,
        le=1.0
    )
    max_position_size_pct: float = Field(
        default=0.1,
        description="Maximum position size as % of portfolio",
        gt=0,
        le=1.0
    )
    max_position_concentration: float = Field(
        default=0.3,
        description="Maximum concentration in top N positions",
        gt=0,
        le=1.0
    )

    # Sector/Industry limits
    max_sector_exposure: float = Field(
        default=0.3,
        description="Maximum sector exposure",
        gt=0,
        le=1.0
    )
    max_industry_exposure: float = Field(
        default=0.2,
        description="Maximum industry exposure",
        gt=0,
        le=1.0
    )

    # Correlation limits
    max_portfolio_correlation: float = Field(
        default=0.7,
        description="Maximum average portfolio correlation",
        ge=-1.0,
        le=1.0
    )
    max_position_correlation: float = Field(
        default=0.8,
        description="Maximum correlation between two positions",
        ge=-1.0,
        le=1.0
    )

    # Volatility limits
    max_portfolio_volatility: float = Field(
        default=0.20,
        description="Maximum portfolio volatility (annualized)",
        gt=0
    )
    max_position_volatility: float = Field(
        default=0.40,
        description="Maximum position volatility (annualized)",
        gt=0
    )

    # Beta limits
    portfolio_beta_min: float = Field(
        default=0.5,
        description="Minimum portfolio beta"
    )
    portfolio_beta_max: float = Field(
        default=1.5,
        description="Maximum portfolio beta"
    )

    # Leverage limits
    max_gross_leverage: float = Field(
        default=2.0,
        description="Maximum gross leverage",
        ge=1.0
    )
    max_net_leverage: float = Field(
        default=1.5,
        description="Maximum net leverage",
        ge=0.0
    )

    # Liquidity requirements
    min_liquidity_ratio: float = Field(
        default=0.05,
        description="Minimum cash/liquid assets ratio",
        ge=0,
        le=1.0
    )
    max_illiquid_exposure: float = Field(
        default=0.2,
        description="Maximum exposure to illiquid assets",
        ge=0,
        le=1.0
    )

    # Stop loss parameters
    use_stop_loss: bool = Field(
        default=True,
        description="Enable stop loss"
    )
    stop_loss_type: str = Field(
        default="trailing",
        description="Stop loss type (fixed/trailing)"
    )
    default_stop_loss_pct: float = Field(
        default=0.02,
        description="Default stop loss percentage",
        gt=0,
        le=1.0
    )
    trailing_stop_distance: float = Field(
        default=0.015,
        description="Trailing stop distance",
        gt=0
    )

    # Take profit parameters
    use_take_profit: bool = Field(
        default=True,
        description="Enable take profit"
    )
    default_take_profit_pct: float = Field(
        default=0.05,
        description="Default take profit percentage",
        gt=0
    )
    partial_profit_levels: List[float] = Field(
        default=[0.03, 0.05, 0.08],
        description="Partial profit taking levels"
    )
    partial_profit_pcts: List[float] = Field(
        default=[0.33, 0.33, 0.34],
        description="Percentage to close at each level"
    )

    # Portfolio optimization
    optimization_method: PortfolioOptimizationMethod = Field(
        default=PortfolioOptimizationMethod.BLACK_LITTERMAN,
        description="Portfolio optimization method"
    )
    rebalance_frequency_hours: int = Field(
        default=24,
        description="Rebalancing frequency",
        ge=1
    )
    rebalance_threshold: float = Field(
        default=0.05,
        description="Rebalancing threshold",
        gt=0
    )

    # Risk-adjusted returns
    target_sharpe_ratio: float = Field(
        default=1.5,
        description="Target Sharpe ratio",
        ge=0
    )
    target_sortino_ratio: float = Field(
        default=2.0,
        description="Target Sortino ratio",
        ge=0
    )
    target_calmar_ratio: float = Field(
        default=2.0,
        description="Target Calmar ratio",
        ge=0
    )

    # Risk-free rate
    risk_free_rate: float = Field(
        default=0.045,
        description="Risk-free rate (annualized)",
        ge=0
    )

    # VaR/CVaR parameters
    var_confidence_level: float = Field(
        default=0.95,
        description="VaR confidence level",
        gt=0,
        lt=1.0
    )
    cvar_confidence_level: float = Field(
        default=0.95,
        description="CVaR confidence level",
        gt=0,
        lt=1.0
    )
    var_method: str = Field(
        default="historical",
        description="VaR calculation method"
    )
    var_lookback_days: int = Field(
        default=252,
        description="VaR lookback period in days",
        ge=30
    )

    # Stress testing
    enable_stress_tests: bool = Field(
        default=True,
        description="Enable stress testing"
    )
    stress_test_scenarios: List[str] = Field(
        default=["market_crash", "flash_crash", "sector_rotation", "volatility_spike"],
        description="Stress test scenarios"
    )
    max_stress_loss: float = Field(
        default=0.10,
        description="Maximum loss in stress scenarios",
        gt=0,
        le=1.0
    )

    # Circuit breakers
    enable_circuit_breakers: bool = Field(
        default=True,
        description="Enable circuit breakers"
    )
    daily_loss_limit: float = Field(
        default=0.03,
        description="Daily loss limit (circuit breaker)",
        gt=0,
        le=1.0
    )
    weekly_loss_limit: float = Field(
        default=0.05,
        description="Weekly loss limit",
        gt=0,
        le=1.0
    )
    monthly_loss_limit: float = Field(
        default=0.08,
        description="Monthly loss limit",
        gt=0,
        le=1.0
    )

    # Monitoring
    risk_monitoring_interval_seconds: int = Field(
        default=60,
        description="Risk monitoring interval",
        ge=1
    )
    alert_on_limit_breach: bool = Field(
        default=True,
        description="Send alerts on limit breaches"
    )
    auto_reduce_on_breach: bool = Field(
        default=True,
        description="Automatically reduce positions on breach"
    )
    risk_reduction_pct: float = Field(
        default=0.5,
        description="Position reduction percentage on breach",
        gt=0,
        le=1.0
    )

    # Margin requirements
    initial_margin: float = Field(
        default=0.50,
        description="Initial margin requirement",
        gt=0,
        le=1.0
    )
    maintenance_margin: float = Field(
        default=0.30,
        description="Maintenance margin requirement",
        gt=0,
        le=1.0
    )
    margin_call_threshold: float = Field(
        default=0.35,
        description="Margin call threshold",
        gt=0,
        le=1.0
    )

    # Exposure limits by asset class
    max_equity_exposure: float = Field(
        default=1.0,
        description="Maximum equity exposure",
        ge=0
    )
    max_fixed_income_exposure: float = Field(
        default=0.5,
        description="Maximum fixed income exposure",
        ge=0
    )
    max_commodity_exposure: float = Field(
        default=0.3,
        description="Maximum commodity exposure",
        ge=0
    )
    max_crypto_exposure: float = Field(
        default=0.1,
        description="Maximum crypto exposure",
        ge=0
    )

    class Config:
        use_enum_values = True
