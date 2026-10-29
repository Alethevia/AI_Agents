"""
Performance Metrics Calculations

This module provides comprehensive performance metrics for backtesting analysis.
Includes standard metrics, risk-adjusted returns, and benchmark comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0
    monthly_return_mean: float = 0.0
    monthly_return_std: float = 0.0

    # Risk metrics
    volatility: float = 0.0  # Annualized
    downside_volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # In days

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Exposure
    avg_exposure: float = 0.0
    max_exposure: float = 0.0

    # Other metrics
    total_fees: float = 0.0
    total_slippage: float = 0.0
    expectancy: float = 0.0

    # Benchmark comparison
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None

    # Additional statistics
    skewness: float = 0.0
    kurtosis: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'returns': {
                'total_return': self.total_return,
                'cagr': self.cagr,
                'daily_return_mean': self.daily_return_mean,
                'daily_return_std': self.daily_return_std,
                'monthly_return_mean': self.monthly_return_mean,
                'monthly_return_std': self.monthly_return_std,
            },
            'risk': {
                'volatility': self.volatility,
                'downside_volatility': self.downside_volatility,
                'var_95': self.var_95,
                'cvar_95': self.cvar_95,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'omega_ratio': self.omega_ratio,
            },
            'trades': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'avg_win_loss_ratio': self.avg_win_loss_ratio,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
            },
            'exposure': {
                'avg_exposure': self.avg_exposure,
                'max_exposure': self.max_exposure,
            },
            'costs': {
                'total_fees': self.total_fees,
                'total_slippage': self.total_slippage,
            },
            'benchmark': {
                'alpha': self.alpha,
                'beta': self.beta,
                'information_ratio': self.information_ratio,
                'tracking_error': self.tracking_error,
            },
            'statistics': {
                'skewness': self.skewness,
                'kurtosis': self.kurtosis,
                'expectancy': self.expectancy,
            }
        }


class MetricsCalculator:
    """
    Performance metrics calculator

    Calculates comprehensive performance metrics from equity curve and trades.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        initial_capital: float = 100000.0
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            equity_curve: Time series of portfolio value
            trades: DataFrame with trade information
            benchmark: Benchmark returns for comparison
            initial_capital: Initial portfolio value

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if len(equity_curve) == 0:
            return metrics

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Returns metrics
        metrics.total_return = self._calculate_total_return(equity_curve, initial_capital)
        metrics.cagr = self._calculate_cagr(equity_curve, initial_capital)
        metrics.daily_return_mean = returns.mean()
        metrics.daily_return_std = returns.std()

        # Monthly returns
        monthly_returns = equity_curve.resample('ME').last().pct_change().dropna()
        if len(monthly_returns) > 0:
            metrics.monthly_return_mean = monthly_returns.mean()
            metrics.monthly_return_std = monthly_returns.std()

        # Risk metrics
        metrics.volatility = self._calculate_volatility(returns)
        metrics.downside_volatility = self._calculate_downside_volatility(returns)
        metrics.var_95 = self._calculate_var(returns, confidence=0.95)
        metrics.cvar_95 = self._calculate_cvar(returns, confidence=0.95)

        # Drawdown metrics
        dd_info = self._calculate_drawdown_metrics(equity_curve)
        metrics.max_drawdown = dd_info['max_drawdown']
        metrics.max_drawdown_duration = dd_info['max_duration']

        # Risk-adjusted returns
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self._calculate_calmar_ratio(
            metrics.cagr, metrics.max_drawdown
        )
        metrics.omega_ratio = self._calculate_omega_ratio(returns)

        # Statistical properties
        metrics.skewness = self._calculate_skewness(returns)
        metrics.kurtosis = self._calculate_kurtosis(returns)

        # Trade metrics
        if trades is not None and len(trades) > 0:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.total_trades = trade_metrics['total_trades']
            metrics.winning_trades = trade_metrics['winning_trades']
            metrics.losing_trades = trade_metrics['losing_trades']
            metrics.win_rate = trade_metrics['win_rate']
            metrics.profit_factor = trade_metrics['profit_factor']
            metrics.avg_win = trade_metrics['avg_win']
            metrics.avg_loss = trade_metrics['avg_loss']
            metrics.avg_win_loss_ratio = trade_metrics['avg_win_loss_ratio']
            metrics.largest_win = trade_metrics['largest_win']
            metrics.largest_loss = trade_metrics['largest_loss']
            metrics.expectancy = trade_metrics['expectancy']
            metrics.total_fees = trade_metrics.get('total_fees', 0.0)
            metrics.total_slippage = trade_metrics.get('total_slippage', 0.0)

        # Benchmark comparison
        if benchmark is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark)
            metrics.alpha = benchmark_metrics['alpha']
            metrics.beta = benchmark_metrics['beta']
            metrics.information_ratio = benchmark_metrics['information_ratio']
            metrics.tracking_error = benchmark_metrics['tracking_error']

        return metrics

    def _calculate_total_return(
        self,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> float:
        """Calculate total return"""
        if len(equity_curve) == 0:
            return 0.0

        final_value = equity_curve.iloc[-1]
        return (final_value - initial_capital) / initial_capital

    def _calculate_cagr(
        self,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(equity_curve) < 2:
            return 0.0

        final_value = equity_curve.iloc[-1]

        # Calculate number of years
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25

        if years <= 0:
            return 0.0

        cagr = (final_value / initial_capital) ** (1 / years) - 1
        return cagr

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0

        return returns.std() * np.sqrt(252)  # Assuming daily returns

    def _calculate_downside_volatility(
        self,
        returns: pd.Series,
        target: float = 0.0
    ) -> float:
        """Calculate downside volatility (semi-deviation)"""
        if len(returns) == 0:
            return 0.0

        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0.0

        return downside_returns.std() * np.sqrt(252)

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0

        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def _calculate_drawdown_metrics(
        self,
        equity_curve: pd.Series
    ) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        if len(equity_curve) == 0:
            return {
                'max_drawdown': 0.0,
                'max_duration': 0,
                'drawdown_series': pd.Series()
            }

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Max drawdown
        max_drawdown = drawdown.min()

        # Max drawdown duration
        max_duration = 0
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return {
            'max_drawdown': abs(max_drawdown),
            'max_duration': max_duration,
            'drawdown_series': drawdown
        }

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize returns and risk-free rate
        excess_returns = returns - (self.risk_free_rate / 252)

        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        target: float = 0.0
    ) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0

        downside_vol = self._calculate_downside_volatility(returns, target)

        if downside_vol == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        return (excess_returns.mean() * 252) / downside_vol

    def _calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0.0

        return cagr / max_drawdown

    def _calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0.0

        return gains.sum() / losses.sum()

    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness"""
        if len(returns) < 3:
            return 0.0

        return returns.skew()

    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate excess kurtosis"""
        if len(returns) < 4:
            return 0.0

        return returns.kurtosis()

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-level metrics"""
        if 'pnl' not in trades.columns:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_win_loss_ratio': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'expectancy': 0.0,
            }

        total_trades = len(trades)
        winning_trades = (trades['pnl'] > 0).sum()
        losing_trades = (trades['pnl'] < 0).sum()

        wins = trades[trades['pnl'] > 0]['pnl']
        losses = trades[trades['pnl'] < 0]['pnl']

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        profit_factor = (
            abs(wins.sum() / losses.sum())
            if len(losses) > 0 and losses.sum() != 0
            else 0.0
        )

        avg_win_loss_ratio = (
            abs(avg_win / avg_loss)
            if avg_loss != 0
            else 0.0
        )

        largest_win = wins.max() if len(wins) > 0 else 0.0
        largest_loss = losses.min() if len(losses) > 0 else 0.0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Fees and slippage
        total_fees = 0.0
        total_slippage = 0.0

        if 'commission' in trades.columns:
            total_fees = trades['commission'].sum()

        if 'slippage' in trades.columns:
            total_slippage = trades['slippage'].sum()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'expectancy': expectancy,
            'total_fees': total_fees,
            'total_slippage': total_slippage,
        }

    def _calculate_benchmark_metrics(
        self,
        returns: pd.Series,
        benchmark: pd.Series
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        # Align returns and benchmark
        aligned = pd.DataFrame({
            'returns': returns,
            'benchmark': benchmark
        }).dropna()

        if len(aligned) < 2:
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'information_ratio': 0.0,
                'tracking_error': 0.0,
            }

        # Beta and Alpha (using simple linear regression)
        covariance = aligned['returns'].cov(aligned['benchmark'])
        benchmark_variance = aligned['benchmark'].var()

        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0

        alpha = aligned['returns'].mean() - (beta * aligned['benchmark'].mean())

        # Information Ratio
        excess_returns = aligned['returns'] - aligned['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252)

        information_ratio = (
            (excess_returns.mean() * 252) / tracking_error
            if tracking_error != 0
            else 0.0
        )

        return {
            'alpha': alpha * 252,  # Annualized
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
        }

    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            equity_curve: Time series of portfolio value
            window: Rolling window size (default 252 = 1 year)

        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change().dropna()

        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling return
        rolling_metrics['return'] = equity_curve.pct_change(window)

        # Rolling volatility
        rolling_metrics['volatility'] = (
            returns.rolling(window).std() * np.sqrt(252)
        )

        # Rolling Sharpe ratio
        rolling_metrics['sharpe'] = (
            (returns.rolling(window).mean() - self.risk_free_rate / 252) /
            returns.rolling(window).std() * np.sqrt(252)
        )

        # Rolling max drawdown
        rolling_max = equity_curve.rolling(window).max()
        rolling_dd = (equity_curve - rolling_max) / rolling_max
        rolling_metrics['max_drawdown'] = rolling_dd.rolling(window).min().abs()

        return rolling_metrics.dropna()

    def calculate_monthly_returns(
        self,
        equity_curve: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate monthly returns table

        Args:
            equity_curve: Time series of portfolio value

        Returns:
            DataFrame with monthly returns by year and month
        """
        monthly = equity_curve.resample('ME').last()
        monthly_returns = monthly.pct_change().dropna()

        # Create pivot table
        returns_table = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        pivot = returns_table.pivot(
            index='year',
            columns='month',
            values='return'
        )

        # Add yearly returns
        yearly_equity = equity_curve.resample('Y').last()
        yearly_returns = yearly_equity.pct_change().dropna()

        pivot['Year'] = yearly_returns.values

        return pivot

    def calculate_monte_carlo_confidence(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        n_periods: int = 252
    ) -> Dict[str, Any]:
        """
        Calculate Monte Carlo confidence intervals

        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            n_periods: Number of periods to simulate

        Returns:
            Dictionary with confidence intervals
        """
        if len(returns) < 2:
            return {}

        # Calculate return statistics
        mean_return = returns.mean()
        std_return = returns.std()

        # Run simulations
        simulations = []

        for _ in range(n_simulations):
            # Generate random returns
            sim_returns = np.random.normal(mean_return, std_return, n_periods)

            # Calculate cumulative return
            cum_return = (1 + sim_returns).prod() - 1
            simulations.append(cum_return)

        simulations = np.array(simulations)

        return {
            'mean': np.mean(simulations),
            'std': np.std(simulations),
            'median': np.median(simulations),
            'percentile_5': np.percentile(simulations, 5),
            'percentile_25': np.percentile(simulations, 25),
            'percentile_75': np.percentile(simulations, 75),
            'percentile_95': np.percentile(simulations, 95),
            'worst_case': np.min(simulations),
            'best_case': np.max(simulations),
        }


def compare_strategies(
    strategy_results: Dict[str, pd.Series],
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Compare multiple strategies

    Args:
        strategy_results: Dictionary of strategy name -> equity curve
        initial_capital: Initial capital for each strategy

    Returns:
        DataFrame comparing strategy metrics
    """
    calculator = MetricsCalculator()

    comparison = {}

    for name, equity_curve in strategy_results.items():
        metrics = calculator.calculate_metrics(equity_curve, initial_capital=initial_capital)
        comparison[name] = metrics.to_dict()

    # Flatten nested dictionaries
    flattened = {}
    for strategy_name, metrics_dict in comparison.items():
        flat_metrics = {}
        for category, values in metrics_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_metrics[f"{category}.{key}"] = value
            else:
                flat_metrics[category] = values
        flattened[strategy_name] = flat_metrics

    return pd.DataFrame(flattened).T
