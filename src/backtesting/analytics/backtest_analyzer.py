"""
Backtest Performance Analyzer

This module provides comprehensive analysis of backtest results including:
- Returns analysis
- Risk metrics
- Drawdown analysis
- Monte Carlo simulations
- Benchmark comparisons
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .trade_analyzer import TradeAnalyzer, Trade, TradeAnalytics
from ..metrics.performance_metrics import MetricsCalculator, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Complete backtest results"""

    # Time period
    start_date: datetime
    end_date: datetime
    duration_days: int

    # Portfolio
    initial_capital: float
    final_capital: float
    equity_curve: pd.Series
    returns: pd.Series

    # Performance
    performance_metrics: PerformanceMetrics
    trade_analytics: TradeAnalytics

    # Trades
    trades: pd.DataFrame

    # Positions history
    positions_history: Optional[pd.DataFrame] = None

    # Benchmark
    benchmark: Optional[pd.Series] = None

    # Strategy metadata
    strategy_name: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None

    # Execution details
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'duration_days': self.duration_days,
            },
            'portfolio': {
                'initial_capital': self.initial_capital,
                'final_capital': self.final_capital,
            },
            'performance': self.performance_metrics.to_dict(),
            'trades': self.trade_analytics.to_dict(),
            'execution': {
                'total_orders': self.total_orders,
                'filled_orders': self.filled_orders,
                'rejected_orders': self.rejected_orders,
                'fill_rate': self.filled_orders / self.total_orders if self.total_orders > 0 else 0.0,
            },
            'strategy': {
                'name': self.strategy_name,
                'params': self.strategy_params,
            }
        }


@dataclass
class DrawdownPeriod:
    """Drawdown period information"""
    start_date: datetime
    end_date: datetime
    duration_days: int
    drawdown_pct: float
    peak_value: float
    trough_value: float
    recovered: bool
    recovery_date: Optional[datetime] = None


class BacktestAnalyzer:
    """
    Comprehensive backtest analyzer

    Analyzes backtest results and generates performance reports.
    """

    def __init__(self, initial_capital: float = 100000.0, risk_free_rate: float = 0.02):
        """
        Initialize backtest analyzer

        Args:
            initial_capital: Initial portfolio value
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculations
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.metrics_calculator = MetricsCalculator(risk_free_rate=risk_free_rate)
        self.trade_analyzer = TradeAnalyzer()

        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.trades_list: List[Trade] = []
        self.positions_history: List[Dict[str, Any]] = []
        self.orders_history: List[Dict[str, Any]] = []

        self.logger = logging.getLogger(__name__)

    def add_equity_point(self, timestamp: datetime, value: float) -> None:
        """
        Add a point to the equity curve

        Args:
            timestamp: Timestamp
            value: Portfolio value
        """
        self.equity_curve.loc[timestamp] = value

    def add_trade(self, trade: Trade) -> None:
        """
        Add a completed trade

        Args:
            trade: Trade object
        """
        self.trades_list.append(trade)
        self.trade_analyzer.add_trade(trade)

    def add_position_update(self, timestamp: datetime, positions: Dict[str, Any]) -> None:
        """
        Add position update

        Args:
            timestamp: Timestamp
            positions: Position information
        """
        self.positions_history.append({
            'timestamp': timestamp,
            **positions
        })

    def add_order(self, order_info: Dict[str, Any]) -> None:
        """
        Add order information

        Args:
            order_info: Order details
        """
        self.orders_history.append(order_info)

    def analyze(
        self,
        benchmark: Optional[pd.Series] = None,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResults:
        """
        Perform comprehensive backtest analysis

        Args:
            benchmark: Benchmark returns for comparison
            strategy_name: Name of the strategy
            strategy_params: Strategy parameters

        Returns:
            BacktestResults object
        """
        if len(self.equity_curve) == 0:
            raise ValueError("No equity data to analyze")

        # Sort equity curve by timestamp
        self.equity_curve = self.equity_curve.sort_index()

        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()

        # Calculate performance metrics
        trades_df = None
        if self.trades_list:
            trades_df = pd.DataFrame([t.to_dict() for t in self.trades_list])

        performance_metrics = self.metrics_calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            trades=trades_df,
            benchmark=benchmark,
            initial_capital=self.initial_capital
        )

        # Calculate trade analytics
        trade_analytics = self.trade_analyzer.analyze()

        # Calculate time period
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        duration_days = (end_date - start_date).days

        # Count orders
        total_orders = len(self.orders_history)
        filled_orders = sum(
            1 for order in self.orders_history
            if order.get('status') == 'FILLED'
        )
        rejected_orders = sum(
            1 for order in self.orders_history
            if order.get('status') == 'REJECTED'
        )

        # Create positions history DataFrame
        positions_df = None
        if self.positions_history:
            positions_df = pd.DataFrame(self.positions_history)

        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            initial_capital=self.initial_capital,
            final_capital=self.equity_curve.iloc[-1],
            equity_curve=self.equity_curve,
            returns=returns,
            performance_metrics=performance_metrics,
            trade_analytics=trade_analytics,
            trades=trades_df if trades_df is not None else pd.DataFrame(),
            positions_history=positions_df,
            benchmark=benchmark,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            total_orders=total_orders,
            filled_orders=filled_orders,
            rejected_orders=rejected_orders
        )

    def analyze_drawdowns(
        self,
        min_duration_days: int = 0
    ) -> List[DrawdownPeriod]:
        """
        Analyze all drawdown periods

        Args:
            min_duration_days: Minimum duration to include

        Returns:
            List of drawdown periods
        """
        if len(self.equity_curve) == 0:
            return []

        # Calculate running maximum
        running_max = self.equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (self.equity_curve - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        current_period = None

        for timestamp, dd_value in drawdown.items():
            if dd_value < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                current_period = {
                    'start_date': timestamp,
                    'peak_value': running_max.loc[timestamp],
                    'trough_value': self.equity_curve.loc[timestamp],
                    'min_dd': dd_value
                }

            elif dd_value < 0 and in_drawdown:
                # Continue drawdown
                if dd_value < current_period['min_dd']:
                    current_period['min_dd'] = dd_value
                    current_period['trough_value'] = self.equity_curve.loc[timestamp]
                    current_period['trough_date'] = timestamp

            elif dd_value >= 0 and in_drawdown:
                # End of drawdown (recovered)
                in_drawdown = False
                current_period['end_date'] = timestamp
                current_period['recovered'] = True
                current_period['recovery_date'] = timestamp

                duration = (current_period['end_date'] - current_period['start_date']).days

                if duration >= min_duration_days:
                    drawdown_periods.append(DrawdownPeriod(
                        start_date=current_period['start_date'],
                        end_date=current_period['end_date'],
                        duration_days=duration,
                        drawdown_pct=abs(current_period['min_dd']),
                        peak_value=current_period['peak_value'],
                        trough_value=current_period['trough_value'],
                        recovered=True,
                        recovery_date=current_period['recovery_date']
                    ))

        # If still in drawdown at the end
        if in_drawdown and current_period:
            current_period['end_date'] = self.equity_curve.index[-1]
            current_period['recovered'] = False
            duration = (current_period['end_date'] - current_period['start_date']).days

            if duration >= min_duration_days:
                drawdown_periods.append(DrawdownPeriod(
                    start_date=current_period['start_date'],
                    end_date=current_period['end_date'],
                    duration_days=duration,
                    drawdown_pct=abs(current_period['min_dd']),
                    peak_value=current_period['peak_value'],
                    trough_value=current_period['trough_value'],
                    recovered=False
                ))

        # Sort by drawdown magnitude
        drawdown_periods.sort(key=lambda x: x.drawdown_pct, reverse=True)

        return drawdown_periods

    def analyze_monthly_returns(self) -> pd.DataFrame:
        """
        Analyze monthly returns

        Returns:
            DataFrame with monthly returns by year and month
        """
        return self.metrics_calculator.calculate_monthly_returns(self.equity_curve)

    def analyze_rolling_metrics(self, window_days: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            window_days: Rolling window size in days

        Returns:
            DataFrame with rolling metrics
        """
        return self.metrics_calculator.calculate_rolling_metrics(
            self.equity_curve,
            window=window_days
        )

    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        n_periods: int = 252
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation

        Args:
            n_simulations: Number of simulations
            n_periods: Number of periods to simulate

        Returns:
            Dictionary with simulation results
        """
        returns = self.equity_curve.pct_change().dropna()

        return self.metrics_calculator.calculate_monte_carlo_confidence(
            returns,
            n_simulations=n_simulations,
            n_periods=n_periods
        )

    def get_summary_report(self, results: Optional[BacktestResults] = None) -> str:
        """
        Generate formatted summary report

        Args:
            results: BacktestResults object (if None, will analyze first)

        Returns:
            Formatted string report
        """
        if results is None:
            results = self.analyze()

        perf = results.performance_metrics
        trades = results.trade_analytics

        report = f"""
{'=' * 80}
BACKTEST SUMMARY REPORT
{'=' * 80}

Strategy: {results.strategy_name or 'Unnamed'}
Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}
Duration: {results.duration_days} days

{'=' * 80}
PORTFOLIO PERFORMANCE
{'=' * 80}

Initial Capital:         ${results.initial_capital:>15,.2f}
Final Capital:           ${results.final_capital:>15,.2f}
Total Return:            {perf.total_return:>15.2%}
CAGR:                    {perf.cagr:>15.2%}

{'=' * 80}
RISK METRICS
{'=' * 80}

Volatility (Annual):     {perf.volatility:>15.2%}
Downside Volatility:     {perf.downside_volatility:>15.2%}
Max Drawdown:            {perf.max_drawdown:>15.2%}
Max DD Duration:         {perf.max_drawdown_duration:>15} days
Value at Risk (95%):     {perf.var_95:>15.2%}
CVaR (95%):              {perf.cvar_95:>15.2%}

{'=' * 80}
RISK-ADJUSTED RETURNS
{'=' * 80}

Sharpe Ratio:            {perf.sharpe_ratio:>15.2f}
Sortino Ratio:           {perf.sortino_ratio:>15.2f}
Calmar Ratio:            {perf.calmar_ratio:>15.2f}
Omega Ratio:             {perf.omega_ratio:>15.2f}

{'=' * 80}
TRADE STATISTICS
{'=' * 80}

Total Trades:            {trades.total_trades:>15}
Winning Trades:          {trades.winning_trades:>15}
Losing Trades:           {trades.losing_trades:>15}
Win Rate:                {trades.win_rate:>15.2%}

Average Win:             ${trades.avg_win:>15,.2f}
Average Loss:            ${trades.avg_loss:>15,.2f}
Largest Win:             ${trades.largest_win:>15,.2f}
Largest Loss:            ${trades.largest_loss:>15,.2f}

Profit Factor:           {trades.profit_factor:>15.2f}
Expectancy:              ${trades.expectancy:>15,.2f}
Win/Loss Ratio:          {trades.avg_win_loss_ratio:>15.2f}

{'=' * 80}
TRADE DURATION
{'=' * 80}

Average Duration:        {trades.avg_duration_hours:>15.2f} hours
Average Win Duration:    {trades.avg_win_duration:>15.2f} hours
Average Loss Duration:   {trades.avg_loss_duration:>15.2f} hours

{'=' * 80}
COSTS & SLIPPAGE
{'=' * 80}

Total Commission:        ${trades.total_commission:>15,.2f}
Total Slippage:          ${trades.total_slippage:>15,.2f}
Total Costs:             ${trades.total_costs:>15,.2f}
Avg Cost per Trade:      ${trades.avg_commission_per_trade + trades.avg_slippage_per_trade:>15,.2f}

{'=' * 80}
EXECUTION STATISTICS
{'=' * 80}

Total Orders:            {results.total_orders:>15}
Filled Orders:           {results.filled_orders:>15}
Rejected Orders:         {results.rejected_orders:>15}
Fill Rate:               {results.filled_orders / results.total_orders if results.total_orders > 0 else 0:>15.2%}

{'=' * 80}
"""

        # Add benchmark comparison if available
        if perf.alpha is not None:
            report += f"""
BENCHMARK COMPARISON
{'=' * 80}

Alpha:                   {perf.alpha:>15.2%}
Beta:                    {perf.beta:>15.2f}
Information Ratio:       {perf.information_ratio:>15.2f}
Tracking Error:          {perf.tracking_error:>15.2%}

{'=' * 80}
"""

        return report

    def get_detailed_report(self, results: Optional[BacktestResults] = None) -> str:
        """
        Generate detailed report with additional analysis

        Args:
            results: BacktestResults object (if None, will analyze first)

        Returns:
            Formatted string report
        """
        if results is None:
            results = self.analyze()

        summary = self.get_summary_report(results)

        # Add drawdown analysis
        drawdowns = self.analyze_drawdowns(min_duration_days=1)

        detailed = summary + "\n"
        detailed += f"""
{'=' * 80}
TOP 5 DRAWDOWN PERIODS
{'=' * 80}

"""

        for i, dd in enumerate(drawdowns[:5], 1):
            detailed += f"""
{i}. Drawdown: {dd.drawdown_pct:.2%}
   Start: {dd.start_date.strftime('%Y-%m-%d')}
   End: {dd.end_date.strftime('%Y-%m-%d')}
   Duration: {dd.duration_days} days
   Peak Value: ${dd.peak_value:,.2f}
   Trough Value: ${dd.trough_value:,.2f}
   Recovered: {'Yes' if dd.recovered else 'No'}
"""

        # Add monthly returns
        monthly_returns = self.analyze_monthly_returns()

        if not monthly_returns.empty:
            detailed += f"""
{'=' * 80}
RECENT MONTHLY RETURNS
{'=' * 80}

{monthly_returns.tail(12).to_string()}

"""

        return detailed

    def compare_with_benchmark(
        self,
        benchmark: pd.Series,
        benchmark_name: str = "Benchmark"
    ) -> Dict[str, Any]:
        """
        Compare strategy performance with benchmark

        Args:
            benchmark: Benchmark returns series
            benchmark_name: Name of the benchmark

        Returns:
            Dictionary with comparison metrics
        """
        strategy_returns = self.equity_curve.pct_change().dropna()

        # Align returns
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark
        }).dropna()

        if len(aligned) == 0:
            return {}

        # Calculate metrics for both
        strategy_metrics = self.metrics_calculator.calculate_metrics(
            self.equity_curve,
            initial_capital=self.initial_capital
        )

        # Create equity curve for benchmark
        benchmark_equity = (1 + aligned['benchmark']).cumprod() * self.initial_capital

        benchmark_metrics = self.metrics_calculator.calculate_metrics(
            benchmark_equity,
            initial_capital=self.initial_capital
        )

        return {
            'strategy': {
                'name': 'Strategy',
                'total_return': strategy_metrics.total_return,
                'cagr': strategy_metrics.cagr,
                'volatility': strategy_metrics.volatility,
                'sharpe_ratio': strategy_metrics.sharpe_ratio,
                'max_drawdown': strategy_metrics.max_drawdown,
            },
            'benchmark': {
                'name': benchmark_name,
                'total_return': benchmark_metrics.total_return,
                'cagr': benchmark_metrics.cagr,
                'volatility': benchmark_metrics.volatility,
                'sharpe_ratio': benchmark_metrics.sharpe_ratio,
                'max_drawdown': benchmark_metrics.max_drawdown,
            },
            'relative': {
                'alpha': strategy_metrics.alpha,
                'beta': strategy_metrics.beta,
                'information_ratio': strategy_metrics.information_ratio,
                'tracking_error': strategy_metrics.tracking_error,
            }
        }

    def clear(self) -> None:
        """Clear all data"""
        self.equity_curve = pd.Series(dtype=float)
        self.trades_list.clear()
        self.positions_history.clear()
        self.orders_history.clear()
        self.trade_analyzer.clear()
