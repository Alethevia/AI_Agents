"""
Backtest Visualization

This module provides comprehensive visualization tools for backtesting results:
- Equity curves
- Drawdown charts
- Returns distributions
- Monthly returns heatmaps
- Trade visualizations
- Metrics dashboards
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ..analytics.backtest_analyzer import BacktestResults, BacktestAnalyzer
from ..analytics.trade_analyzer import TradeAnalyzer

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class BacktestPlotter:
    """
    Backtest visualization plotter

    Creates comprehensive visualizations of backtest results.
    """

    def __init__(self, results: Optional[BacktestResults] = None):
        """
        Initialize plotter

        Args:
            results: BacktestResults object
        """
        self.results = results
        self.logger = logging.getLogger(__name__)

    def set_results(self, results: BacktestResults) -> None:
        """Set results for plotting"""
        self.results = results

    def plot_equity_curve(
        self,
        benchmark: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot equity curve

        Args:
            benchmark: Optional benchmark series
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot equity curve
        equity = self.results.equity_curve
        ax.plot(equity.index, equity.values, label='Strategy', linewidth=2, color='#2E86DE')

        # Plot benchmark if provided
        if benchmark is not None or self.results.benchmark is not None:
            bench = benchmark if benchmark is not None else self.results.benchmark

            # Convert to equity curve if returns
            if bench.iloc[0] < 1:  # Likely returns
                bench_equity = (1 + bench).cumprod() * self.results.initial_capital
            else:
                bench_equity = bench

            ax.plot(
                bench_equity.index,
                bench_equity.values,
                label='Benchmark',
                linewidth=2,
                linestyle='--',
                color='#EE5A6F',
                alpha=0.7
            )

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Equity curve saved to {save_path}")

        return fig

    def plot_drawdown(
        self,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot drawdown chart

        Args:
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate drawdown
        equity = self.results.equity_curve
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100  # Convert to percentage

        # Plot
        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color='#EE5A6F',
            alpha=0.3,
            label='Drawdown'
        )
        ax.plot(drawdown.index, drawdown.values, color='#EE5A6F', linewidth=1.5)

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Drawdown chart saved to {save_path}")

        return fig

    def plot_returns_distribution(
        self,
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot returns distribution

        Args:
            bins: Number of histogram bins
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        returns = self.results.returns * 100  # Convert to percentage

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(returns, bins=bins, color='#2E86DE', alpha=0.7, edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')

        ax1.set_xlabel('Daily Returns (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Returns distribution saved to {save_path}")

        return fig

    def plot_monthly_returns_heatmap(
        self,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap

        Args:
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        # Get monthly returns
        monthly = self.results.equity_curve.resample('M').last()
        monthly_returns = monthly.pct_change().dropna() * 100  # Convert to percentage

        # Create pivot table
        returns_table = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        pivot = returns_table.pivot(index='year', columns='month', values='return')

        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[int(m)-1] for m in pivot.columns]

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return (%)'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Year', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Monthly returns heatmap saved to {save_path}")

        return fig

    def plot_trade_analysis(
        self,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive trade analysis

        Args:
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None or self.results.trades.empty:
            raise ValueError("No trades to plot")

        trades = self.results.trades

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. P&L per trade
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['green' if pnl > 0 else 'red' for pnl in trades['pnl']]
        ax1.bar(range(len(trades)), trades['pnl'], color=colors, alpha=0.6)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Trade Number', fontweight='bold')
        ax1.set_ylabel('P&L ($)', fontweight='bold')
        ax1.set_title('P&L per Trade', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative P&L
        ax2 = fig.add_subplot(gs[1, 0])
        cumulative_pnl = trades['pnl'].cumsum()
        ax2.plot(range(len(trades)), cumulative_pnl, linewidth=2, color='#2E86DE')
        ax2.fill_between(range(len(trades)), cumulative_pnl, alpha=0.3, color='#2E86DE')
        ax2.set_xlabel('Trade Number', fontweight='bold')
        ax2.set_ylabel('Cumulative P&L ($)', fontweight='bold')
        ax2.set_title('Cumulative P&L', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. Trade duration distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if 'duration_hours' in trades.columns:
            duration_days = trades['duration_hours'] / 24
            ax3.hist(duration_days, bins=30, color='#2E86DE', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Duration (days)', fontweight='bold')
            ax3.set_ylabel('Frequency', fontweight='bold')
            ax3.set_title('Trade Duration Distribution', fontweight='bold', fontsize=12)
            ax3.grid(True, alpha=0.3)

        # 4. Win/Loss distribution
        ax4 = fig.add_subplot(gs[2, 0])
        wins = trades[trades['pnl'] > 0]['pnl']
        losses = trades[trades['pnl'] < 0]['pnl']
        ax4.hist([wins, losses], bins=20, label=['Wins', 'Losses'],
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax4.set_xlabel('P&L ($)', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Win/Loss Distribution', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Returns distribution
        ax5 = fig.add_subplot(gs[2, 1])
        if 'return_pct' in trades.columns:
            returns_pct = trades['return_pct'] * 100
            ax5.hist(returns_pct, bins=30, color='#2E86DE', alpha=0.7, edgecolor='black')
            ax5.axvline(returns_pct.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {returns_pct.mean():.2f}%')
            ax5.set_xlabel('Return (%)', fontweight='bold')
            ax5.set_ylabel('Frequency', fontweight='bold')
            ax5.set_title('Trade Returns Distribution', fontweight='bold', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        plt.suptitle('Trade Analysis', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Trade analysis saved to {save_path}")

        return fig

    def plot_metrics_dashboard(
        self,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive metrics dashboard

        Args:
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        perf = self.results.performance_metrics
        trades = self.results.trade_analytics

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        # 1. Equity curve (top, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        equity = self.results.equity_curve
        ax1.plot(equity.index, equity.values, linewidth=2, color='#2E86DE')
        ax1.fill_between(equity.index, equity.values, alpha=0.3, color='#2E86DE')
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
        ax1.set_title('Equity Curve', fontweight='bold', fontsize=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)

        # 2. Key metrics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        metrics_text = f"""
        KEY METRICS

        Total Return: {perf.total_return:.2%}
        CAGR: {perf.cagr:.2%}
        Sharpe Ratio: {perf.sharpe_ratio:.2f}

        Max Drawdown: {perf.max_drawdown:.2%}
        Volatility: {perf.volatility:.2%}

        Win Rate: {trades.win_rate:.2%}
        Total Trades: {trades.total_trades}
        """
        ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, :2])
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        ax3.fill_between(drawdown.index, drawdown.values, 0, color='#EE5A6F', alpha=0.3)
        ax3.plot(drawdown.index, drawdown.values, color='#EE5A6F', linewidth=1.5)
        ax3.set_xlabel('Date', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontweight='bold')
        ax3.set_title('Drawdown', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 4. Monthly returns
        ax4 = fig.add_subplot(gs[1, 2])
        monthly = equity.resample('M').last()
        monthly_returns = monthly.pct_change().dropna() * 100
        colors = ['green' if r > 0 else 'red' for r in monthly_returns]
        ax4.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.6)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Month', fontweight='bold')
        ax4.set_ylabel('Return (%)', fontweight='bold')
        ax4.set_title('Monthly Returns', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3)

        # 5. Returns distribution
        ax5 = fig.add_subplot(gs[2, 0])
        returns = self.results.returns * 100
        ax5.hist(returns, bins=50, color='#2E86DE', alpha=0.7, edgecolor='black')
        ax5.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Daily Returns (%)', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Returns Distribution', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3)

        # 6. Trade P&L distribution
        ax6 = fig.add_subplot(gs[2, 1])
        if not self.results.trades.empty:
            trade_pnl = self.results.trades['pnl']
            wins = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            ax6.hist([wins, losses], bins=20, label=['Wins', 'Losses'],
                    color=['green', 'red'], alpha=0.6, edgecolor='black')
            ax6.set_xlabel('P&L ($)', fontweight='bold')
            ax6.set_ylabel('Frequency', fontweight='bold')
            ax6.set_title('Trade P&L Distribution', fontweight='bold', fontsize=12)
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Risk metrics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        risk_text = f"""
        RISK METRICS

        Volatility: {perf.volatility:.2%}
        Downside Vol: {perf.downside_volatility:.2%}

        Sortino: {perf.sortino_ratio:.2f}
        Calmar: {perf.calmar_ratio:.2f}

        VaR (95%): {perf.var_95:.2%}
        CVaR (95%): {perf.cvar_95:.2%}
        """
        ax7.text(0.1, 0.5, risk_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Backtest Dashboard', fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")

        return fig

    def plot_rolling_metrics(
        self,
        window: int = 252,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rolling performance metrics

        Args:
            window: Rolling window size (days)
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        returns = self.results.returns

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Rolling returns
        rolling_returns = returns.rolling(window).mean() * 252 * 100  # Annualized %
        axes[0].plot(rolling_returns.index, rolling_returns.values, linewidth=2, color='#2E86DE')
        axes[0].set_ylabel('Annualized Return (%)', fontweight='bold')
        axes[0].set_title(f'{window}-Day Rolling Returns', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100  # Annualized %
        axes[1].plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='#EE5A6F')
        axes[1].set_ylabel('Annualized Volatility (%)', fontweight='bold')
        axes[1].set_title(f'{window}-Day Rolling Volatility', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Rolling Sharpe ratio
        rolling_sharpe = (
            (returns.rolling(window).mean() * 252) /
            (returns.rolling(window).std() * np.sqrt(252))
        )
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='#10AC84')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].set_ylabel('Sharpe Ratio', fontweight='bold')
        axes[2].set_title(f'{window}-Day Rolling Sharpe Ratio', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Rolling metrics saved to {save_path}")

        return fig

    def plot_all(
        self,
        output_dir: Optional[str] = None,
        prefix: str = "backtest"
    ) -> Dict[str, plt.Figure]:
        """
        Generate all plots

        Args:
            output_dir: Directory to save plots (None to not save)
            prefix: Prefix for saved files

        Returns:
            Dictionary of plot name -> figure
        """
        if self.results is None:
            raise ValueError("No results to plot")

        plots = {}

        # Equity curve
        save_path = f"{output_dir}/{prefix}_equity.png" if output_dir else None
        plots['equity'] = self.plot_equity_curve(save_path=save_path)
        plt.close()

        # Drawdown
        save_path = f"{output_dir}/{prefix}_drawdown.png" if output_dir else None
        plots['drawdown'] = self.plot_drawdown(save_path=save_path)
        plt.close()

        # Returns distribution
        save_path = f"{output_dir}/{prefix}_returns_dist.png" if output_dir else None
        plots['returns_dist'] = self.plot_returns_distribution(save_path=save_path)
        plt.close()

        # Monthly returns heatmap
        save_path = f"{output_dir}/{prefix}_monthly_heatmap.png" if output_dir else None
        plots['monthly_heatmap'] = self.plot_monthly_returns_heatmap(save_path=save_path)
        plt.close()

        # Trade analysis (if trades exist)
        if not self.results.trades.empty:
            save_path = f"{output_dir}/{prefix}_trades.png" if output_dir else None
            plots['trades'] = self.plot_trade_analysis(save_path=save_path)
            plt.close()

        # Dashboard
        save_path = f"{output_dir}/{prefix}_dashboard.png" if output_dir else None
        plots['dashboard'] = self.plot_metrics_dashboard(save_path=save_path)
        plt.close()

        # Rolling metrics
        save_path = f"{output_dir}/{prefix}_rolling.png" if output_dir else None
        plots['rolling'] = self.plot_rolling_metrics(save_path=save_path)
        plt.close()

        self.logger.info(f"Generated {len(plots)} plots")

        return plots


def compare_strategies_plot(
    results_dict: Dict[str, BacktestResults],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple strategies

    Args:
        results_dict: Dictionary of strategy name -> BacktestResults
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Normalize equity curves to start at 100
    for name, results in results_dict.items():
        normalized = results.equity_curve / results.initial_capital * 100
        axes[0, 0].plot(normalized.index, normalized.values, label=name, linewidth=2)

    axes[0, 0].set_xlabel('Date', fontweight='bold')
    axes[0, 0].set_ylabel('Portfolio Value (Normalized)', fontweight='bold')
    axes[0, 0].set_title('Equity Curves Comparison', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdowns comparison
    for name, results in results_dict.items():
        equity = results.equity_curve
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        axes[0, 1].plot(drawdown.index, drawdown.values, label=name, linewidth=1.5, alpha=0.7)

    axes[0, 1].set_xlabel('Date', fontweight='bold')
    axes[0, 1].set_ylabel('Drawdown (%)', fontweight='bold')
    axes[0, 1].set_title('Drawdowns Comparison', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Returns comparison (bar chart)
    metrics_comparison = {
        name: {
            'Total Return': results.performance_metrics.total_return * 100,
            'CAGR': results.performance_metrics.cagr * 100,
            'Sharpe': results.performance_metrics.sharpe_ratio,
            'Max DD': results.performance_metrics.max_drawdown * 100
        }
        for name, results in results_dict.items()
    }

    df = pd.DataFrame(metrics_comparison).T
    df[['Total Return', 'CAGR']].plot(kind='bar', ax=axes[1, 0], color=['#2E86DE', '#10AC84'])
    axes[1, 0].set_xlabel('Strategy', fontweight='bold')
    axes[1, 0].set_ylabel('Return (%)', fontweight='bold')
    axes[1, 0].set_title('Returns Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Risk metrics comparison
    df[['Sharpe', 'Max DD']].plot(kind='bar', ax=axes[1, 1], color=['#EE5A6F', '#FFA502'])
    axes[1, 1].set_xlabel('Strategy', fontweight='bold')
    axes[1, 1].set_ylabel('Value', fontweight='bold')
    axes[1, 1].set_title('Risk Metrics Comparison', fontweight='bold', fontsize=12)
    axes[1, 1].legend(['Sharpe Ratio', 'Max Drawdown (%)'])
    axes[1, 1].grid(True, alpha=0.3)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle('Strategy Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy comparison saved to {save_path}")

    return fig
