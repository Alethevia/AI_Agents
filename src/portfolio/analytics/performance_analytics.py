"""
Performance Analytics Module

Implements comprehensive performance analysis for portfolios:
- Returns analysis (daily, monthly, annual)
- Risk-adjusted performance (Sharpe, Sortino, Calmar, Omega)
- Attribution analysis (factor, security)
- Contribution analysis
- Rolling metrics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float
    daily_return_mean: float
    daily_return_std: float

    # Risk metrics
    volatility: float  # Annualized
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int  # Days

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: Optional[float] = None

    # Statistical properties
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)

    # Win/loss metrics
    win_rate: float
    profit_factor: float
    best_day: float
    worst_day: float
    consecutive_wins: int
    consecutive_losses: int

    # Period-specific
    ytd_return: Optional[float] = None
    mtd_return: Optional[float] = None
    qtd_return: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'total_return': float(self.total_return),
            'annualized_return': float(self.annualized_return),
            'cumulative_return': float(self.cumulative_return),
            'volatility': float(self.volatility),
            'downside_deviation': float(self.downside_deviation),
            'max_drawdown': float(self.max_drawdown),
            'max_drawdown_duration': int(self.max_drawdown_duration),
            'sharpe_ratio': float(self.sharpe_ratio),
            'sortino_ratio': float(self.sortino_ratio),
            'calmar_ratio': float(self.calmar_ratio),
            'omega_ratio': float(self.omega_ratio),
            'information_ratio': float(self.information_ratio) if self.information_ratio else None,
            'skewness': float(self.skewness),
            'kurtosis': float(self.kurtosis),
            'var_95': float(self.var_95),
            'cvar_95': float(self.cvar_95),
            'win_rate': float(self.win_rate),
            'profit_factor': float(self.profit_factor),
            'best_day': float(self.best_day),
            'worst_day': float(self.worst_day)
        }


@dataclass
class RollingMetrics:
    """Rolling performance metrics"""
    dates: List[datetime]
    returns: List[float]
    volatility: List[float]
    sharpe_ratio: List[float]
    max_drawdown: List[float]
    beta: Optional[List[float]] = None
    alpha: Optional[List[float]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        data = {
            'date': self.dates,
            'return': self.returns,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }

        if self.beta is not None:
            data['beta'] = self.beta
        if self.alpha is not None:
            data['alpha'] = self.alpha

        return pd.DataFrame(data)


class PerformanceAnalyzer:
    """
    Advanced performance analyzer for portfolios
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        frequency: int = 252,  # Trading days per year
        benchmark_returns: Optional[pd.Series] = None
    ):
        """
        Initialize performance analyzer

        Args:
            risk_free_rate: Risk-free rate (annualized)
            frequency: Number of trading periods per year
            benchmark_returns: Benchmark returns for comparison
        """
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.benchmark_returns = benchmark_returns

        logger.info(
            f"Initialized PerformanceAnalyzer with risk_free_rate={risk_free_rate:.4f}, "
            f"frequency={frequency}"
        )

    def analyze(
        self,
        returns: Union[pd.Series, np.ndarray],
        values: Optional[Union[pd.Series, np.ndarray]] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> PerformanceMetrics:
        """
        Comprehensive performance analysis

        Args:
            returns: Portfolio returns (daily)
            values: Portfolio values (optional)
            dates: Dates corresponding to returns (optional)

        Returns:
            PerformanceMetrics with comprehensive metrics
        """
        if isinstance(returns, pd.Series):
            returns_array = returns.values
            dates = dates or returns.index
        else:
            returns_array = np.asarray(returns)

        # Basic return metrics
        total_return = self._calculate_total_return(returns_array)
        annualized_return = self._calculate_annualized_return(returns_array)
        cumulative_return = self._calculate_cumulative_return(returns_array)

        # Risk metrics
        volatility = self._calculate_volatility(returns_array)
        downside_deviation = self._calculate_downside_deviation(returns_array)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(returns_array)

        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array, volatility)
        sortino_ratio = self._calculate_sortino_ratio(returns_array, downside_deviation)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        omega_ratio = self._calculate_omega_ratio(returns_array)

        # Information ratio (if benchmark provided)
        information_ratio = None
        if self.benchmark_returns is not None:
            information_ratio = self._calculate_information_ratio(
                returns_array, self.benchmark_returns.values
            )

        # Statistical properties
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        var_95 = self._calculate_var(returns_array, 0.95)
        cvar_95 = self._calculate_cvar(returns_array, 0.95)

        # Win/loss metrics
        win_rate = self._calculate_win_rate(returns_array)
        profit_factor = self._calculate_profit_factor(returns_array)
        best_day = np.max(returns_array)
        worst_day = np.min(returns_array)
        consecutive_wins = self._calculate_consecutive_streak(returns_array, positive=True)
        consecutive_losses = self._calculate_consecutive_streak(returns_array, positive=False)

        # Period-specific returns
        ytd_return = None
        mtd_return = None
        qtd_return = None

        if dates is not None and len(dates) > 0:
            ytd_return = self._calculate_period_return(returns_array, dates, 'YTD')
            mtd_return = self._calculate_period_return(returns_array, dates, 'MTD')
            qtd_return = self._calculate_period_return(returns_array, dates, 'QTD')

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            daily_return_mean=np.mean(returns_array),
            daily_return_std=np.std(returns_array),
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            information_ratio=information_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            best_day=best_day,
            worst_day=worst_day,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            ytd_return=ytd_return,
            mtd_return=mtd_return,
            qtd_return=qtd_return
        )

    async def analyze_async(
        self,
        returns: Union[pd.Series, np.ndarray],
        values: Optional[Union[pd.Series, np.ndarray]] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> PerformanceMetrics:
        """Async version of analyze"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyze,
            returns,
            values,
            dates
        )

    def calculate_rolling_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: int = 60,
        min_periods: Optional[int] = None
    ) -> RollingMetrics:
        """
        Calculate rolling performance metrics

        Args:
            returns: Portfolio returns
            window: Rolling window size (days)
            min_periods: Minimum periods for calculation

        Returns:
            RollingMetrics with time series of metrics
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        min_periods = min_periods or window // 2

        # Rolling returns
        rolling_returns = returns.rolling(window=window, min_periods=min_periods).mean()

        # Rolling volatility
        rolling_vol = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(self.frequency)

        # Rolling Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / self.frequency)
        rolling_sharpe = (
            excess_returns.rolling(window=window, min_periods=min_periods).mean() /
            returns.rolling(window=window, min_periods=min_periods).std() *
            np.sqrt(self.frequency)
        )

        # Rolling max drawdown
        rolling_max_dd = returns.rolling(window=window, min_periods=min_periods).apply(
            lambda x: self._calculate_max_drawdown(x.values)[0],
            raw=False
        )

        # Rolling beta and alpha (if benchmark provided)
        rolling_beta = None
        rolling_alpha = None

        if self.benchmark_returns is not None:
            rolling_beta = returns.rolling(window=window, min_periods=min_periods).cov(
                self.benchmark_returns
            ) / self.benchmark_returns.rolling(window=window, min_periods=min_periods).var()

            rolling_alpha = (
                rolling_returns -
                (self.risk_free_rate / self.frequency + rolling_beta * (
                    self.benchmark_returns.rolling(window=window, min_periods=min_periods).mean() -
                    self.risk_free_rate / self.frequency
                ))
            ) * self.frequency

        # Remove NaN values
        valid_mask = ~rolling_returns.isna()

        return RollingMetrics(
            dates=returns.index[valid_mask].tolist() if hasattr(returns, 'index') else list(range(len(rolling_returns[valid_mask]))),
            returns=rolling_returns[valid_mask].tolist(),
            volatility=rolling_vol[valid_mask].tolist(),
            sharpe_ratio=rolling_sharpe[valid_mask].tolist(),
            max_drawdown=rolling_max_dd[valid_mask].tolist(),
            beta=rolling_beta[valid_mask].tolist() if rolling_beta is not None else None,
            alpha=rolling_alpha[valid_mask].tolist() if rolling_alpha is not None else None
        )

    def monthly_returns_table(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Generate monthly returns table

        Args:
            returns: Portfolio returns with datetime index

        Returns:
            DataFrame with monthly returns by year
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have DatetimeIndex")

        # Calculate monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1

        # Create pivot table
        monthly_table = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })

        # Pivot to get years as rows and months as columns
        pivot_table = monthly_table.pivot(index='Year', columns='Month', values='Return')

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]

        # Add annual returns
        annual_returns = (1 + returns).resample('Y').prod() - 1
        pivot_table['Annual'] = annual_returns.values

        return pivot_table

    def annual_returns(
        self,
        returns: pd.Series
    ) -> pd.Series:
        """
        Calculate annual returns

        Args:
            returns: Portfolio returns with datetime index

        Returns:
            Series of annual returns
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have DatetimeIndex")

        annual = (1 + returns).resample('Y').prod() - 1
        annual.index = annual.index.year

        return annual

    def drawdown_series(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> pd.Series:
        """
        Calculate drawdown series

        Args:
            returns: Portfolio returns

        Returns:
            Series of drawdowns over time
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown

    def contribution_analysis(
        self,
        asset_returns: pd.DataFrame,
        weights: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Analyze contribution of each asset to portfolio returns

        Args:
            asset_returns: DataFrame of individual asset returns
            weights: Portfolio weights (matching asset_returns columns)

        Returns:
            DataFrame with contribution analysis
        """
        if isinstance(weights, np.ndarray):
            weights = pd.DataFrame(
                weights,
                index=asset_returns.index,
                columns=asset_returns.columns
            )

        # Calculate weighted returns
        weighted_returns = asset_returns * weights

        # Portfolio return
        portfolio_return = weighted_returns.sum(axis=1)

        # Contribution to return
        contribution = weighted_returns.div(portfolio_return, axis=0)

        # Summary statistics
        summary = pd.DataFrame({
            'Total Contribution': weighted_returns.sum(),
            'Avg Contribution': weighted_returns.mean(),
            'Contribution Volatility': weighted_returns.std(),
            'Contribution %': contribution.mean() * 100
        })

        return summary

    def risk_decomposition(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Decompose portfolio risk by asset

        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix of returns
            asset_names: Asset names (optional)

        Returns:
            DataFrame with risk decomposition
        """
        # Portfolio variance
        portfolio_var = weights @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)

        # Marginal contribution to risk
        marginal_contrib = covariance_matrix @ weights / portfolio_vol

        # Total contribution to risk
        risk_contrib = weights * marginal_contrib

        # Percentage contribution
        risk_contrib_pct = risk_contrib / portfolio_vol * 100

        # Create DataFrame
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(len(weights))]

        risk_decomp = pd.DataFrame({
            'Weight': weights,
            'Marginal Risk': marginal_contrib,
            'Risk Contribution': risk_contrib,
            'Risk Contribution %': risk_contrib_pct
        }, index=asset_names)

        return risk_decomp

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return"""
        return np.prod(1 + returns) - 1

    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return"""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        return (1 + total_return) ** (self.frequency / n_periods) - 1

    def _calculate_cumulative_return(self, returns: np.ndarray) -> float:
        """Calculate cumulative return"""
        return np.prod(1 + returns) - 1

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        return np.std(returns) * np.sqrt(self.frequency)

    def _calculate_downside_deviation(
        self,
        returns: np.ndarray,
        target: float = 0.0
    ) -> float:
        """Calculate downside deviation (semi-deviation)"""
        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt(np.mean((downside_returns - target) ** 2)) * np.sqrt(self.frequency)

    def _calculate_max_drawdown(
        self,
        returns: np.ndarray
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration

        Returns:
            Tuple of (max_drawdown, duration_in_days)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        max_dd = np.min(drawdown)

        # Calculate duration
        max_dd_duration = 0
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_dd_duration

    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        volatility: float
    ) -> float:
        """Calculate Sharpe ratio"""
        if volatility == 0:
            return 0.0

        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        return excess_return / volatility

    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        downside_deviation: float
    ) -> float:
        """Calculate Sortino ratio"""
        if downside_deviation == 0:
            return 0.0

        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        return excess_return / downside_deviation

    def _calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0.0

        return annualized_return / abs(max_drawdown)

    def _calculate_omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - threshold

        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            return np.inf if gains > 0 else 0.0

        return gains / losses

    def _calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate Information ratio"""
        # Ensure same length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Active returns
        active_returns = returns - benchmark_returns

        # Tracking error
        tracking_error = np.std(active_returns) * np.sqrt(self.frequency)

        if tracking_error == 0:
            return 0.0

        # Annualized active return
        active_return = np.mean(active_returns) * self.frequency

        return active_return / tracking_error

    def _calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = -np.sum(returns[returns < 0])

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_consecutive_streak(
        self,
        returns: np.ndarray,
        positive: bool = True
    ) -> int:
        """Calculate maximum consecutive wins or losses"""
        condition = returns > 0 if positive else returns < 0

        max_streak = 0
        current_streak = 0

        for val in condition:
            if val:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_period_return(
        self,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        period: str
    ) -> Optional[float]:
        """Calculate period-specific return (YTD, MTD, QTD)"""
        if len(dates) == 0:
            return None

        current_date = dates[-1]

        if period == 'YTD':
            start_date = pd.Timestamp(year=current_date.year, month=1, day=1)
        elif period == 'MTD':
            start_date = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)
        elif period == 'QTD':
            quarter_start_month = ((current_date.month - 1) // 3) * 3 + 1
            start_date = pd.Timestamp(year=current_date.year, month=quarter_start_month, day=1)
        else:
            return None

        # Filter returns for the period
        mask = dates >= start_date
        period_returns = returns[mask]

        if len(period_returns) == 0:
            return None

        return np.prod(1 + period_returns) - 1


class BenchmarkComparison:
    """
    Compare portfolio performance against benchmarks
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Dict[str, pd.Series],
        risk_free_rate: float = 0.0
    ):
        """
        Initialize benchmark comparison

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Dictionary of benchmark returns {name: series}
            risk_free_rate: Risk-free rate
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

    def compare(self) -> pd.DataFrame:
        """
        Compare portfolio against all benchmarks

        Returns:
            DataFrame with comparison metrics
        """
        analyzer = PerformanceAnalyzer(risk_free_rate=self.risk_free_rate)

        # Analyze portfolio
        portfolio_metrics = analyzer.analyze(self.portfolio_returns)

        # Analyze each benchmark
        results = {
            'Portfolio': portfolio_metrics.to_dict()
        }

        for name, returns in self.benchmark_returns.items():
            bench_metrics = analyzer.analyze(returns)
            results[name] = bench_metrics.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame(results).T

        return df

    def relative_performance(self) -> pd.DataFrame:
        """
        Calculate relative performance vs benchmarks

        Returns:
            DataFrame with relative metrics
        """
        results = []

        for name, bench_returns in self.benchmark_returns.items():
            # Align dates
            aligned_portfolio, aligned_bench = self.portfolio_returns.align(
                bench_returns, join='inner'
            )

            # Active returns
            active_returns = aligned_portfolio - aligned_bench

            # Metrics
            excess_return = active_returns.mean() * 252
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0

            # Beta and alpha
            covariance = np.cov(aligned_portfolio, aligned_bench)[0, 1]
            bench_variance = np.var(aligned_bench)
            beta = covariance / bench_variance if bench_variance > 0 else 0.0

            portfolio_return = aligned_portfolio.mean() * 252
            benchmark_return = aligned_bench.mean() * 252
            alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

            results.append({
                'Benchmark': name,
                'Excess Return': excess_return,
                'Tracking Error': tracking_error,
                'Information Ratio': information_ratio,
                'Beta': beta,
                'Alpha': alpha
            })

        return pd.DataFrame(results)
