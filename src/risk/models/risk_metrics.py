"""
Risk metrics calculation module.

This module provides various risk metrics including:
- Volatility (historical, EWMA, GARCH)
- Beta and correlation
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown
- Ulcer index
- Downside deviation
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from pydantic import BaseModel, Field
from scipy import stats


class RiskMetrics(BaseModel):
    """Container for risk metrics."""

    # Volatility metrics
    volatility: float = Field(..., description="Annualized volatility")
    downside_deviation: float = Field(..., description="Downside deviation")

    # Return metrics
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")

    # Drawdown metrics
    max_drawdown: float = Field(..., description="Maximum drawdown")
    current_drawdown: float = Field(..., description="Current drawdown")
    ulcer_index: float = Field(..., description="Ulcer index")

    # Correlation metrics
    beta: Optional[float] = Field(None, description="Portfolio beta")
    correlation: Optional[float] = Field(None, description="Correlation with benchmark")

    # Additional metrics
    var_95: Optional[float] = Field(None, description="95% VaR")
    cvar_95: Optional[float] = Field(None, description="95% CVaR")

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VolatilityCalculator:
    """Calculate various volatility metrics."""

    def __init__(self, annualization_factor: int = 252):
        """
        Initialize volatility calculator.

        Args:
            annualization_factor: Trading days per year for annualization
        """
        self.annualization_factor = annualization_factor

    async def historical_volatility(
        self,
        returns: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """
        Calculate historical volatility.

        Args:
            returns: Returns series
            window: Rolling window size (None for full sample)

        Returns:
            Annualized volatility
        """
        if window:
            vol = returns.rolling(window=window).std().iloc[-1]
        else:
            vol = returns.std()

        return vol * np.sqrt(self.annualization_factor)

    async def ewma_volatility(
        self,
        returns: pd.Series,
        span: int = 60
    ) -> float:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.

        Args:
            returns: Returns series
            span: Span for exponential weighting

        Returns:
            Annualized EWMA volatility
        """
        ewma_var = returns.ewm(span=span).var().iloc[-1]
        ewma_vol = np.sqrt(ewma_var)

        return ewma_vol * np.sqrt(self.annualization_factor)

    async def garch_volatility(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1
    ) -> Tuple[float, Dict]:
        """
        Calculate GARCH volatility forecast.

        Args:
            returns: Returns series (in percentage terms)
            p: GARCH lag order
            q: ARCH lag order

        Returns:
            Tuple of (forecasted volatility, model results)
        """
        # Convert to percentage returns for GARCH
        returns_pct = returns * 100

        # Fit GARCH model
        model = arch_model(returns_pct, vol='Garch', p=p, q=q)
        model_fit = model.fit(disp='off', show_warning=False)

        # Get forecast
        forecast = model_fit.forecast(horizon=1)
        forecast_variance = forecast.variance.values[-1, 0]
        forecast_vol = np.sqrt(forecast_variance) / 100  # Convert back to decimal

        # Annualize
        annual_vol = forecast_vol * np.sqrt(self.annualization_factor)

        results = {
            'omega': float(model_fit.params['omega']),
            'alpha': [float(model_fit.params[f'alpha[{i}]']) for i in range(1, q+1)],
            'beta': [float(model_fit.params[f'beta[{i}]']) for i in range(1, p+1)],
            'log_likelihood': float(model_fit.loglikelihood),
            'aic': float(model_fit.aic),
            'bic': float(model_fit.bic)
        }

        return annual_vol, results

    async def realized_volatility(
        self,
        prices: pd.Series,
        frequency: str = '1D'
    ) -> float:
        """
        Calculate realized volatility from high-frequency data.

        Args:
            prices: Price series
            frequency: Resampling frequency

        Returns:
            Annualized realized volatility
        """
        # Calculate returns at given frequency
        resampled = prices.resample(frequency).last()
        returns = resampled.pct_change().dropna()

        # Sum of squared returns
        realized_var = (returns ** 2).sum()
        realized_vol = np.sqrt(realized_var)

        # Annualize based on frequency
        periods_per_year = {
            '1T': 252 * 390,  # Minutes (6.5 hours per day)
            '5T': 252 * 78,
            '15T': 252 * 26,
            '1H': 252 * 6.5,
            '1D': 252
        }

        factor = periods_per_year.get(frequency, 252)
        return realized_vol * np.sqrt(factor)

    async def downside_deviation(
        self,
        returns: pd.Series,
        mar: float = 0.0
    ) -> float:
        """
        Calculate downside deviation (semi-deviation).

        Args:
            returns: Returns series
            mar: Minimum Acceptable Return (threshold)

        Returns:
            Annualized downside deviation
        """
        downside_returns = returns[returns < mar]
        downside_var = ((downside_returns - mar) ** 2).mean()
        downside_std = np.sqrt(downside_var)

        return downside_std * np.sqrt(self.annualization_factor)


class PerformanceMetrics:
    """Calculate risk-adjusted performance metrics."""

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        annualization_factor: int = 252
    ):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.vol_calculator = VolatilityCalculator(annualization_factor)

    async def sharpe_ratio(
        self,
        returns: pd.Series,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Returns series
            volatility: Pre-calculated volatility (optional)

        Returns:
            Sharpe ratio
        """
        # Annualize returns
        annual_return = returns.mean() * self.annualization_factor

        # Calculate volatility if not provided
        if volatility is None:
            volatility = await self.vol_calculator.historical_volatility(returns)

        if volatility == 0:
            return 0.0

        # Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / volatility

        return sharpe

    async def sortino_ratio(
        self,
        returns: pd.Series,
        mar: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Returns series
            mar: Minimum Acceptable Return

        Returns:
            Sortino ratio
        """
        # Annualize returns
        annual_return = returns.mean() * self.annualization_factor

        # Calculate downside deviation
        downside_dev = await self.vol_calculator.downside_deviation(returns, mar)

        if downside_dev == 0:
            return 0.0

        # Sortino ratio
        sortino = (annual_return - mar) / downside_dev

        return sortino

    async def calmar_ratio(
        self,
        returns: pd.Series,
        max_drawdown: Optional[float] = None
    ) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns: Returns series
            max_drawdown: Pre-calculated max drawdown (optional)

        Returns:
            Calmar ratio
        """
        # Annualize returns
        annual_return = returns.mean() * self.annualization_factor

        # Calculate max drawdown if not provided
        if max_drawdown is None:
            drawdown_calc = DrawdownCalculator()
            max_drawdown = await drawdown_calc.max_drawdown(returns)

        if max_drawdown == 0:
            return 0.0

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown)

        return calmar

    async def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information ratio.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Information ratio
        """
        # Calculate excess returns
        excess_returns = returns - benchmark_returns

        # Annualize
        annual_excess = excess_returns.mean() * self.annualization_factor
        tracking_error = excess_returns.std() * np.sqrt(self.annualization_factor)

        if tracking_error == 0:
            return 0.0

        return annual_excess / tracking_error


class DrawdownCalculator:
    """Calculate drawdown-related metrics."""

    async def calculate_drawdown_series(
        self,
        returns: pd.Series
    ) -> pd.Series:
        """
        Calculate drawdown series from returns.

        Args:
            returns: Returns series

        Returns:
            Drawdown series
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cum_returns.expanding().max()

        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max

        return drawdown

    async def max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Returns series

        Returns:
            Maximum drawdown (negative value)
        """
        drawdown = await self.calculate_drawdown_series(returns)
        return drawdown.min()

    async def current_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate current drawdown.

        Args:
            returns: Returns series

        Returns:
            Current drawdown
        """
        drawdown = await self.calculate_drawdown_series(returns)
        return drawdown.iloc[-1]

    async def drawdown_duration(
        self,
        returns: pd.Series
    ) -> Dict[str, int]:
        """
        Calculate drawdown duration statistics.

        Args:
            returns: Returns series

        Returns:
            Dictionary with duration statistics
        """
        drawdown = await self.calculate_drawdown_series(returns)

        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_duration = 0

        for dd in is_drawdown:
            if dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0

        # Add final period if still in drawdown
        if current_duration > 0:
            drawdown_periods.append(current_duration)

        if not drawdown_periods:
            return {
                'max_duration': 0,
                'avg_duration': 0,
                'current_duration': 0
            }

        return {
            'max_duration': max(drawdown_periods),
            'avg_duration': np.mean(drawdown_periods),
            'current_duration': current_duration
        }

    async def ulcer_index(self, returns: pd.Series) -> float:
        """
        Calculate Ulcer Index (downside risk measure).

        Args:
            returns: Returns series

        Returns:
            Ulcer index
        """
        drawdown = await self.calculate_drawdown_series(returns)

        # Square the drawdowns
        squared_dd = drawdown ** 2

        # Mean of squared drawdowns
        mean_squared_dd = squared_dd.mean()

        # Ulcer index is square root of mean
        ulcer = np.sqrt(mean_squared_dd)

        return abs(ulcer)


class BetaCalculator:
    """Calculate beta and correlation metrics."""

    async def calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate portfolio beta.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Beta coefficient
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )

        # Calculate covariance and variance
        covariance = aligned_returns.cov(aligned_benchmark)
        benchmark_variance = aligned_benchmark.var()

        if benchmark_variance == 0:
            return 0.0

        beta = covariance / benchmark_variance

        return beta

    async def calculate_correlation(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate correlation with benchmark.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Correlation coefficient
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )

        correlation = aligned_returns.corr(aligned_benchmark)

        return correlation

    async def rolling_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling beta.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            window: Rolling window size

        Returns:
            Rolling beta series
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )

        # Calculate rolling covariance and variance
        rolling_cov = aligned_returns.rolling(window).cov(aligned_benchmark)
        rolling_var = aligned_benchmark.rolling(window).var()

        rolling_beta = rolling_cov / rolling_var

        return rolling_beta


class RiskMetricsCalculator:
    """Unified risk metrics calculator."""

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        annualization_factor: int = 252
    ):
        """
        Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

        self.vol_calculator = VolatilityCalculator(annualization_factor)
        self.performance = PerformanceMetrics(risk_free_rate, annualization_factor)
        self.drawdown = DrawdownCalculator()
        self.beta_calc = BetaCalculator()

    async def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        var_95: Optional[float] = None,
        cvar_95: Optional[float] = None
    ) -> RiskMetrics:
        """
        Calculate all risk metrics.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns (optional)
            var_95: Pre-calculated 95% VaR (optional)
            cvar_95: Pre-calculated 95% CVaR (optional)

        Returns:
            RiskMetrics object with all metrics
        """
        # Calculate volatility metrics
        volatility = await self.vol_calculator.historical_volatility(returns)
        downside_dev = await self.vol_calculator.downside_deviation(returns)

        # Calculate performance ratios
        sharpe = await self.performance.sharpe_ratio(returns, volatility)
        sortino = await self.performance.sortino_ratio(returns)

        # Calculate drawdown metrics
        max_dd = await self.drawdown.max_drawdown(returns)
        current_dd = await self.drawdown.current_drawdown(returns)
        ulcer = await self.drawdown.ulcer_index(returns)

        # Calculate Calmar ratio
        calmar = await self.performance.calmar_ratio(returns, max_dd)

        # Calculate beta and correlation if benchmark provided
        beta = None
        correlation = None
        if benchmark_returns is not None:
            beta = await self.beta_calc.calculate_beta(returns, benchmark_returns)
            correlation = await self.beta_calc.calculate_correlation(
                returns, benchmark_returns
            )

        return RiskMetrics(
            volatility=volatility,
            downside_deviation=downside_dev,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            ulcer_index=ulcer,
            beta=beta,
            correlation=correlation,
            var_95=var_95,
            cvar_95=cvar_95
        )

    async def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.

        Args:
            returns: Returns series
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(self.annualization_factor)

        # Calculate rolling Sharpe
        rolling_mean = returns.rolling(window).mean() * self.annualization_factor
        rolling_sharpe = (rolling_mean - self.risk_free_rate) / rolling_vol

        # Calculate rolling max drawdown
        rolling_max_dd = pd.Series(index=returns.index, dtype=float)
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            dd = await self.drawdown.max_drawdown(window_returns)
            rolling_max_dd.iloc[i] = dd

        return pd.DataFrame({
            'volatility': rolling_vol,
            'sharpe_ratio': rolling_sharpe,
            'max_drawdown': rolling_max_dd
        })
