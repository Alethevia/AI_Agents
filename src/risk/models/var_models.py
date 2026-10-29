"""
Value at Risk (VaR) models for portfolio risk assessment.

This module implements various VaR calculation methods including:
- Historical VaR
- Parametric VaR (variance-covariance)
- Monte Carlo VaR
- CVaR (Conditional VaR / Expected Shortfall)
- Cornish-Fisher VaR (accounts for skewness and kurtosis)
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field


class VaRResult(BaseModel):
    """Value at Risk calculation result."""

    var: float = Field(..., description="VaR value")
    cvar: Optional[float] = Field(None, description="Conditional VaR (Expected Shortfall)")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence level")
    method: str = Field(..., description="VaR calculation method")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio_value: float = Field(..., gt=0, description="Portfolio value")

    # Additional metrics
    var_percentage: float = Field(..., description="VaR as percentage of portfolio")
    cvar_percentage: Optional[float] = Field(None, description="CVaR as percentage")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HistoricalVaR:
    """Historical VaR calculation using empirical distribution of returns."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Historical VaR calculator.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
        """
        self.confidence_level = confidence_level

    async def calculate(
        self,
        returns: pd.Series,
        portfolio_value: float,
        lookback_days: int = 252
    ) -> VaRResult:
        """
        Calculate historical VaR.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            lookback_days: Number of days to look back

        Returns:
            VaRResult with VaR and CVaR
        """
        # Use recent returns based on lookback
        recent_returns = returns.tail(lookback_days)

        # Calculate VaR as the percentile of returns
        alpha = 1 - self.confidence_level
        var_percentile = np.percentile(recent_returns, alpha * 100)
        var = -var_percentile * portfolio_value

        # Calculate CVaR (Expected Shortfall) - average of returns worse than VaR
        tail_returns = recent_returns[recent_returns <= var_percentile]
        if len(tail_returns) > 0:
            cvar_percentile = tail_returns.mean()
            cvar = -cvar_percentile * portfolio_value
        else:
            cvar = var

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method="historical",
            portfolio_value=portfolio_value,
            var_percentage=var / portfolio_value,
            cvar_percentage=cvar / portfolio_value if cvar else None
        )

    async def calculate_multiple_horizons(
        self,
        returns: pd.Series,
        portfolio_value: float,
        horizons: List[int] = [1, 5, 10, 20]
    ) -> Dict[int, VaRResult]:
        """
        Calculate VaR for multiple time horizons.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            horizons: List of horizons in days

        Returns:
            Dictionary mapping horizon to VaRResult
        """
        results = {}
        for horizon in horizons:
            # Scale returns for horizon (assuming IID returns)
            scaled_returns = returns * np.sqrt(horizon)
            result = await self.calculate(scaled_returns, portfolio_value)
            results[horizon] = result

        return results


class ParametricVaR:
    """Parametric VaR using variance-covariance method."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Parametric VaR calculator.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
        """
        self.confidence_level = confidence_level

    async def calculate(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> VaRResult:
        """
        Calculate parametric VaR assuming normal distribution.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value

        Returns:
            VaRResult with VaR and CVaR
        """
        # Calculate mean and standard deviation
        mu = returns.mean()
        sigma = returns.std()

        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # Calculate VaR: VaR = -(μ + z*σ) * portfolio_value
        var_return = -(mu + z_score * sigma)
        var = var_return * portfolio_value

        # Calculate CVaR for normal distribution
        # CVaR = σ * φ(z) / (1-c) - μ
        phi_z = stats.norm.pdf(z_score)
        cvar_return = -(sigma * phi_z / (1 - self.confidence_level) + mu)
        cvar = cvar_return * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method="parametric",
            portfolio_value=portfolio_value,
            var_percentage=var / portfolio_value,
            cvar_percentage=cvar / portfolio_value
        )

    async def calculate_portfolio_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        portfolio_value: float
    ) -> VaRResult:
        """
        Calculate portfolio VaR using covariance matrix.

        Args:
            weights: Portfolio weights
            returns: DataFrame of asset returns
            portfolio_value: Current portfolio value

        Returns:
            VaRResult
        """
        # Calculate portfolio statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Portfolio mean and variance
        portfolio_mean = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Calculate VaR
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var_return = -(portfolio_mean + z_score * portfolio_std)
        var = var_return * portfolio_value

        # Calculate CVaR
        phi_z = stats.norm.pdf(z_score)
        cvar_return = -(portfolio_std * phi_z / (1 - self.confidence_level) + portfolio_mean)
        cvar = cvar_return * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method="parametric_portfolio",
            portfolio_value=portfolio_value,
            var_percentage=var / portfolio_value,
            cvar_percentage=cvar / portfolio_value
        )


class MonteCarloVaR:
    """Monte Carlo VaR using simulated returns."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo VaR calculator.

        Args:
            confidence_level: Confidence level
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.n_simulations = n_simulations
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    async def calculate(
        self,
        returns: pd.Series,
        portfolio_value: float,
        horizon: int = 1
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR using simulated returns.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            horizon: Time horizon in days

        Returns:
            VaRResult with VaR and CVaR
        """
        # Estimate parameters from historical data
        mu = returns.mean()
        sigma = returns.std()

        # Generate random returns
        simulated_returns = np.random.normal(
            mu * horizon,
            sigma * np.sqrt(horizon),
            self.n_simulations
        )

        # Calculate VaR
        alpha = 1 - self.confidence_level
        var_percentile = np.percentile(simulated_returns, alpha * 100)
        var = -var_percentile * portfolio_value

        # Calculate CVaR
        tail_returns = simulated_returns[simulated_returns <= var_percentile]
        cvar_percentile = tail_returns.mean()
        cvar = -cvar_percentile * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method="monte_carlo",
            portfolio_value=portfolio_value,
            var_percentage=var / portfolio_value,
            cvar_percentage=cvar / portfolio_value
        )

    async def calculate_portfolio_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        portfolio_value: float,
        horizon: int = 1
    ) -> VaRResult:
        """
        Calculate portfolio VaR using Monte Carlo simulation.

        Args:
            weights: Portfolio weights
            returns: DataFrame of asset returns
            portfolio_value: Current portfolio value
            horizon: Time horizon in days

        Returns:
            VaRResult
        """
        # Calculate parameters
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        # Generate correlated random returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns * horizon,
            cov_matrix * horizon,
            self.n_simulations
        )

        # Calculate portfolio returns
        portfolio_returns = np.dot(simulated_returns, weights)

        # Calculate VaR
        alpha = 1 - self.confidence_level
        var_percentile = np.percentile(portfolio_returns, alpha * 100)
        var = -var_percentile * portfolio_value

        # Calculate CVaR
        tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
        cvar_percentile = tail_returns.mean()
        cvar = -cvar_percentile * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method="monte_carlo_portfolio",
            portfolio_value=portfolio_value,
            var_percentage=var / portfolio_value,
            cvar_percentage=cvar / portfolio_value
        )


class CornishFisherVaR:
    """Cornish-Fisher VaR accounting for skewness and kurtosis."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Cornish-Fisher VaR calculator.

        Args:
            confidence_level: Confidence level
        """
        self.confidence_level = confidence_level

    def _cornish_fisher_adjustment(
        self,
        z: float,
        skewness: float,
        kurtosis: float
    ) -> float:
        """
        Apply Cornish-Fisher expansion to adjust for skewness and kurtosis.

        Args:
            z: Standard normal quantile
            skewness: Return distribution skewness
            kurtosis: Return distribution excess kurtosis

        Returns:
            Adjusted quantile
        """
        z_cf = (
            z +
            (z**2 - 1) * skewness / 6 +
            (z**3 - 3*z) * kurtosis / 24 -
            (2*z**3 - 5*z) * (skewness**2) / 36
        )
        return z_cf

    async def calculate(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> VaRResult:
        """
        Calculate Cornish-Fisher VaR.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value

        Returns:
            VaRResult with modified VaR
        """
        # Calculate moments
        mu = returns.mean()
        sigma = returns.std()
        skewness = returns.skew()
        excess_kurtosis = returns.kurtosis()

        # Get standard normal quantile
        z = stats.norm.ppf(1 - self.confidence_level)

        # Apply Cornish-Fisher adjustment
        z_cf = self._cornish_fisher_adjustment(z, skewness, excess_kurtosis)

        # Calculate VaR
        var_return = -(mu + z_cf * sigma)
        var = var_return * portfolio_value

        # For CVaR, use a simple approximation
        # In practice, this requires numerical integration
        cvar = var * 1.2  # Rough approximation

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method="cornish_fisher",
            portfolio_value=portfolio_value,
            var_percentage=var / portfolio_value,
            cvar_percentage=cvar / portfolio_value
        )


class VaRCalculator:
    """Unified VaR calculator supporting multiple methods."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: Confidence level for VaR calculations
        """
        self.confidence_level = confidence_level
        self.historical = HistoricalVaR(confidence_level)
        self.parametric = ParametricVaR(confidence_level)
        self.monte_carlo = MonteCarloVaR(confidence_level)
        self.cornish_fisher = CornishFisherVaR(confidence_level)

    async def calculate_all_methods(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> Dict[str, VaRResult]:
        """
        Calculate VaR using all available methods.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value

        Returns:
            Dictionary mapping method name to VaRResult
        """
        results = {}

        # Calculate using all methods concurrently
        historical_task = self.historical.calculate(returns, portfolio_value)
        parametric_task = self.parametric.calculate(returns, portfolio_value)
        monte_carlo_task = self.monte_carlo.calculate(returns, portfolio_value)
        cornish_fisher_task = self.cornish_fisher.calculate(returns, portfolio_value)

        # Gather results
        (
            results['historical'],
            results['parametric'],
            results['monte_carlo'],
            results['cornish_fisher']
        ) = await asyncio.gather(
            historical_task,
            parametric_task,
            monte_carlo_task,
            cornish_fisher_task
        )

        return results

    async def calculate(
        self,
        returns: pd.Series,
        portfolio_value: float,
        method: str = "historical"
    ) -> VaRResult:
        """
        Calculate VaR using specified method.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            method: VaR calculation method

        Returns:
            VaRResult
        """
        method_map = {
            "historical": self.historical.calculate,
            "parametric": self.parametric.calculate,
            "monte_carlo": self.monte_carlo.calculate,
            "cornish_fisher": self.cornish_fisher.calculate
        }

        if method not in method_map:
            raise ValueError(f"Unknown VaR method: {method}")

        return await method_map[method](returns, portfolio_value)

    def get_aggregate_var(self, results: Dict[str, VaRResult]) -> VaRResult:
        """
        Get aggregate VaR across multiple methods.

        Uses the average of all methods as a robust estimate.

        Args:
            results: Dictionary of VaRResults from different methods

        Returns:
            Aggregated VaRResult
        """
        var_values = [r.var for r in results.values()]
        cvar_values = [r.cvar for r in results.values() if r.cvar is not None]

        avg_var = np.mean(var_values)
        avg_cvar = np.mean(cvar_values) if cvar_values else None

        # Use portfolio value from first result
        first_result = next(iter(results.values()))

        return VaRResult(
            var=avg_var,
            cvar=avg_cvar,
            confidence_level=self.confidence_level,
            method="aggregate",
            portfolio_value=first_result.portfolio_value,
            var_percentage=avg_var / first_result.portfolio_value,
            cvar_percentage=avg_cvar / first_result.portfolio_value if avg_cvar else None
        )
