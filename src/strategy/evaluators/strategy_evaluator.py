"""
Strategy evaluator module for evaluating trading strategy performance.

Provides comprehensive performance metrics including return metrics (CAGR, total return),
risk metrics (volatility, VaR, CVaR), risk-adjusted returns (Sharpe, Sortino, Calmar),
drawdown analysis, win rate, profit factor, and Monte Carlo simulation.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from src.core.logging.logger import get_logger, log_async_execution_time


logger = get_logger(__name__)


class PerformanceMetrics(BaseModel):
    """Comprehensive strategy performance metrics."""

    # Return metrics
    total_return: float = Field(..., description="Total return")
    cagr: float = Field(..., description="Compound Annual Growth Rate")
    avg_return: float = Field(..., description="Average return")

    # Risk metrics
    volatility: float = Field(..., description="Annualized volatility")
    downside_deviation: float = Field(..., description="Downside deviation")
    var_95: float = Field(..., description="Value at Risk (95%)")
    cvar_95: float = Field(..., description="Conditional VaR (95%)")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    avg_drawdown: float = Field(..., description="Average drawdown")
    max_drawdown_duration: int = Field(..., description="Max drawdown duration in days")

    # Risk-adjusted returns
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    omega_ratio: float = Field(..., description="Omega ratio")

    # Trade statistics
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate")
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")
    avg_trade: float = Field(..., description="Average trade return")
    profit_factor: float = Field(..., description="Profit factor")

    # Additional metrics
    skewness: float = Field(..., description="Return skewness")
    kurtosis: float = Field(..., description="Return kurtosis")
    best_day: float = Field(..., description="Best daily return")
    worst_day: float = Field(..., description="Worst daily return")

    # Time periods
    start_date: datetime = Field(..., description="Strategy start date")
    end_date: datetime = Field(..., description="Strategy end date")
    trading_days: int = Field(..., description="Number of trading days")

    # Monte Carlo results (optional)
    monte_carlo_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Monte Carlo simulation results"
    )


class DrawdownPeriod(BaseModel):
    """Drawdown period information."""

    start_date: datetime = Field(..., description="Drawdown start date")
    end_date: datetime = Field(..., description="Drawdown end date")
    recovery_date: Optional[datetime] = Field(None, description="Recovery date")
    duration_days: int = Field(..., description="Drawdown duration in days")
    max_drawdown: float = Field(..., description="Maximum drawdown in period")


class StrategyEvaluatorConfig(BaseModel):
    """Configuration for strategy evaluator."""

    # Risk-free rate
    risk_free_rate: float = Field(
        default=0.02,
        ge=0,
        description="Annual risk-free rate"
    )

    # Trading days per year
    trading_days_per_year: int = Field(
        default=252,
        ge=1,
        description="Number of trading days per year"
    )

    # VaR/CVaR confidence level
    var_confidence: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="VaR/CVaR confidence level"
    )

    # Monte Carlo settings
    mc_n_simulations: int = Field(
        default=10000,
        ge=100,
        description="Number of Monte Carlo simulations"
    )
    mc_n_periods: int = Field(
        default=252,
        ge=1,
        description="Number of periods to simulate"
    )
    mc_confidence_levels: List[float] = Field(
        default=[0.05, 0.25, 0.50, 0.75, 0.95],
        description="Confidence levels for MC results"
    )


class StrategyEvaluator:
    """
    Evaluate trading strategy performance.

    Calculates comprehensive performance metrics including returns, risks,
    risk-adjusted returns, drawdowns, trade statistics, and Monte Carlo simulations.
    """

    def __init__(self, config: Optional[StrategyEvaluatorConfig] = None):
        """
        Initialize strategy evaluator.

        Args:
            config: Evaluator configuration
        """
        self.config = config or StrategyEvaluatorConfig()
        self.logger = get_logger(f"{__name__}.StrategyEvaluator")

        # Evaluation history
        self._history: List[PerformanceMetrics] = []

        # Statistics
        self._stats = {
            'total_evaluations': 0,
            'errors': 0,
        }

    @log_async_execution_time()
    async def evaluate(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        run_monte_carlo: bool = False,
        **kwargs
    ) -> PerformanceMetrics:
        """
        Evaluate strategy performance.

        Args:
            returns: Time series of returns
            trades: Optional DataFrame of individual trades
            run_monte_carlo: Whether to run Monte Carlo simulation
            **kwargs: Additional parameters

        Returns:
            Performance metrics
        """
        self.logger.info(
            "Starting strategy evaluation",
            n_periods=len(returns),
            run_monte_carlo=run_monte_carlo
        )

        try:
            self._stats['total_evaluations'] += 1

            # Calculate return metrics
            return_metrics = self._calculate_return_metrics(returns)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)

            # Calculate risk-adjusted returns
            risk_adjusted = self._calculate_risk_adjusted_returns(returns)

            # Calculate drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics(returns)

            # Calculate trade statistics
            if trades is not None and len(trades) > 0:
                trade_stats = self._calculate_trade_statistics(trades)
            else:
                trade_stats = self._default_trade_statistics()

            # Additional metrics
            additional = self._calculate_additional_metrics(returns)

            # Monte Carlo simulation
            mc_results = None
            if run_monte_carlo:
                mc_results = await self._run_monte_carlo_simulation(returns)

            # Combine all metrics
            metrics = PerformanceMetrics(
                # Returns
                total_return=return_metrics['total_return'],
                cagr=return_metrics['cagr'],
                avg_return=return_metrics['avg_return'],

                # Risk
                volatility=risk_metrics['volatility'],
                downside_deviation=risk_metrics['downside_deviation'],
                var_95=risk_metrics['var_95'],
                cvar_95=risk_metrics['cvar_95'],
                max_drawdown=drawdown_metrics['max_drawdown'],
                avg_drawdown=drawdown_metrics['avg_drawdown'],
                max_drawdown_duration=drawdown_metrics['max_drawdown_duration'],

                # Risk-adjusted
                sharpe_ratio=risk_adjusted['sharpe_ratio'],
                sortino_ratio=risk_adjusted['sortino_ratio'],
                calmar_ratio=risk_adjusted['calmar_ratio'],
                omega_ratio=risk_adjusted['omega_ratio'],

                # Trades
                total_trades=trade_stats['total_trades'],
                winning_trades=trade_stats['winning_trades'],
                losing_trades=trade_stats['losing_trades'],
                win_rate=trade_stats['win_rate'],
                avg_win=trade_stats['avg_win'],
                avg_loss=trade_stats['avg_loss'],
                avg_trade=trade_stats['avg_trade'],
                profit_factor=trade_stats['profit_factor'],

                # Additional
                skewness=additional['skewness'],
                kurtosis=additional['kurtosis'],
                best_day=additional['best_day'],
                worst_day=additional['worst_day'],

                # Time
                start_date=returns.index[0] if isinstance(returns.index[0], datetime) else datetime.utcnow(),
                end_date=returns.index[-1] if isinstance(returns.index[-1], datetime) else datetime.utcnow(),
                trading_days=len(returns),

                # Monte Carlo
                monte_carlo_results=mc_results
            )

            # Store in history
            self._history.append(metrics)

            self.logger.info(
                "Strategy evaluation completed",
                total_return=metrics.total_return,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown
            )

            return metrics

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error during strategy evaluation",
                error=str(e),
                exc_info=True
            )
            raise

    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return metrics."""
        # Total return
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # CAGR
        n_periods = len(returns)
        n_years = n_periods / self.config.trading_days_per_year
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Average return
        avg_return = returns.mean()

        return {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'avg_return': float(avg_return),
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(self.config.trading_days_per_year)

        # Downside deviation (annualized)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(self.config.trading_days_per_year)
            if len(downside_returns) > 0 else 0
        )

        # VaR and CVaR
        var_95 = np.percentile(returns, (1 - self.config.var_confidence) * 100)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        return {
            'volatility': float(volatility),
            'downside_deviation': float(downside_deviation),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
        }

    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Running maximum
        running_max = cumulative_returns.expanding().max()

        # Drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Average drawdown
        drawdowns = drawdown[drawdown < 0]
        avg_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0

        # Maximum drawdown duration
        dd_periods = self._identify_drawdown_periods(drawdown)
        max_dd_duration = max([p.duration_days for p in dd_periods]) if dd_periods else 0

        return {
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'max_drawdown_duration': max_dd_duration,
            'drawdown_periods': dd_periods,
        }

    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[DrawdownPeriod]:
        """Identify distinct drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = None
        max_dd = 0

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                max_dd = dd
            elif dd < 0 and in_drawdown:
                # Continuing drawdown
                max_dd = min(max_dd, dd)
            elif dd == 0 and in_drawdown:
                # Recovery
                in_drawdown = False
                if start_idx is not None:
                    periods.append(DrawdownPeriod(
                        start_date=drawdown.index[start_idx] if isinstance(drawdown.index[start_idx], datetime) else datetime.utcnow(),
                        end_date=drawdown.index[i] if isinstance(drawdown.index[i], datetime) else datetime.utcnow(),
                        recovery_date=drawdown.index[i] if isinstance(drawdown.index[i], datetime) else datetime.utcnow(),
                        duration_days=i - start_idx,
                        max_drawdown=float(max_dd)
                    ))
                start_idx = None
                max_dd = 0

        # Handle ongoing drawdown
        if in_drawdown and start_idx is not None:
            periods.append(DrawdownPeriod(
                start_date=drawdown.index[start_idx] if isinstance(drawdown.index[start_idx], datetime) else datetime.utcnow(),
                end_date=drawdown.index[-1] if isinstance(drawdown.index[-1], datetime) else datetime.utcnow(),
                recovery_date=None,
                duration_days=len(drawdown) - start_idx,
                max_drawdown=float(max_dd)
            ))

        return periods

    def _calculate_risk_adjusted_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        # Annual risk-free rate converted to period rate
        rf_period = self.config.risk_free_rate / self.config.trading_days_per_year

        # Sharpe ratio
        excess_returns = returns - rf_period
        sharpe_ratio = (
            (excess_returns.mean() / returns.std()) * np.sqrt(self.config.trading_days_per_year)
            if returns.std() > 0 else 0
        )

        # Sortino ratio
        downside_returns = returns[returns < rf_period]
        downside_std = downside_returns.std()
        sortino_ratio = (
            (excess_returns.mean() / downside_std) * np.sqrt(self.config.trading_days_per_year)
            if downside_std > 0 else 0
        )

        # Calmar ratio
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        n_years = len(returns) / self.config.trading_days_per_year
        ann_return = ((1 + returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0

        calmar_ratio = ann_return / max_drawdown if max_drawdown > 0 else 0

        # Omega ratio (threshold = risk-free rate)
        threshold = rf_period
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')

        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'omega_ratio': float(omega_ratio) if omega_ratio != float('inf') else 999.0,
        }

    def _calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade statistics."""
        # Assume trades DataFrame has 'pnl' column
        if 'pnl' not in trades.columns:
            return self._default_trade_statistics()

        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_trade = trades['pnl'].mean() if total_trades > 0 else 0

        # Profit factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'avg_trade': float(avg_trade),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.0,
        }

    def _default_trade_statistics(self) -> Dict[str, Any]:
        """Return default trade statistics when no trades available."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'profit_factor': 0.0,
        }

    def _calculate_additional_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate additional metrics."""
        return {
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'best_day': float(returns.max()),
            'worst_day': float(returns.min()),
        }

    async def _run_monte_carlo_simulation(
        self,
        returns: pd.Series
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        self.logger.info(
            "Running Monte Carlo simulation",
            n_simulations=self.config.mc_n_simulations,
            n_periods=self.config.mc_n_periods
        )

        # Calculate return distribution parameters
        mean_return = returns.mean()
        std_return = returns.std()

        # Run simulations
        simulated_returns = np.random.normal(
            mean_return,
            std_return,
            size=(self.config.mc_n_simulations, self.config.mc_n_periods)
        )

        # Calculate cumulative returns for each simulation
        cumulative_returns = (1 + simulated_returns).cumprod(axis=1)
        final_returns = cumulative_returns[:, -1] - 1

        # Calculate percentiles
        percentiles = {
            f'p{int(p*100)}': float(np.percentile(final_returns, p * 100))
            for p in self.config.mc_confidence_levels
        }

        # Calculate probability of profit
        prob_profit = (final_returns > 0).sum() / len(final_returns)

        # Calculate expected final return
        expected_return = final_returns.mean()

        # Calculate worst-case scenario
        worst_case = final_returns.min()
        best_case = final_returns.max()

        return {
            'expected_return': float(expected_return),
            'prob_profit': float(prob_profit),
            'worst_case': float(worst_case),
            'best_case': float(best_case),
            'percentiles': percentiles,
            'n_simulations': self.config.mc_n_simulations,
            'n_periods': self.config.mc_n_periods,
        }

    def compare_strategies(
        self,
        strategies: Dict[str, PerformanceMetrics]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategies: Dict of strategy_name -> PerformanceMetrics

        Returns:
            Comparison DataFrame
        """
        comparison = {}

        for name, metrics in strategies.items():
            comparison[name] = {
                'Total Return': metrics.total_return,
                'CAGR': metrics.cagr,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Sortino Ratio': metrics.sortino_ratio,
                'Calmar Ratio': metrics.calmar_ratio,
                'Max Drawdown': metrics.max_drawdown,
                'Volatility': metrics.volatility,
                'Win Rate': metrics.win_rate,
                'Profit Factor': metrics.profit_factor,
            }

        df = pd.DataFrame(comparison).T

        self.logger.info(
            "Strategy comparison completed",
            n_strategies=len(strategies)
        )

        return df

    def get_evaluation_history(self) -> List[PerformanceMetrics]:
        """Get evaluation history."""
        return self._history

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return {
            **self._stats,
            'evaluations_completed': len(self._history),
            'success_rate': (
                (self._stats['total_evaluations'] - self._stats['errors']) /
                self._stats['total_evaluations']
                if self._stats['total_evaluations'] > 0 else 0
            ),
        }

    def reset_history(self) -> None:
        """Reset evaluation history."""
        self._history.clear()
        self.logger.info("Evaluation history reset")

    def reset_statistics(self) -> None:
        """Reset evaluator statistics."""
        self._stats = {
            'total_evaluations': 0,
            'errors': 0,
        }
