"""
Trade-Level Analysis

This module provides detailed analysis of individual trades including:
- Trade statistics and distributions
- Entry/exit analysis
- Duration analysis
- Slippage and commission impact
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str  # "LONG" or "SHORT"
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    return_pct: float
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure Decimal types"""
        self.quantity = Decimal(str(self.quantity))
        self.entry_price = Decimal(str(self.entry_price))
        self.exit_price = Decimal(str(self.exit_price))
        self.pnl = Decimal(str(self.pnl))
        self.commission = Decimal(str(self.commission))
        self.slippage = Decimal(str(self.slippage))

    @property
    def duration(self) -> timedelta:
        """Trade duration"""
        return self.exit_time - self.entry_time

    @property
    def duration_hours(self) -> float:
        """Trade duration in hours"""
        return self.duration.total_seconds() / 3600

    @property
    def duration_days(self) -> float:
        """Trade duration in days"""
        return self.duration.total_seconds() / 86400

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0

    @property
    def gross_pnl(self) -> Decimal:
        """P&L before costs"""
        return self.pnl + self.commission + self.slippage

    @property
    def total_costs(self) -> Decimal:
        """Total transaction costs"""
        return self.commission + abs(self.slippage)

    @property
    def cost_percentage(self) -> float:
        """Costs as percentage of entry value"""
        entry_value = self.entry_price * self.quantity
        if entry_value == 0:
            return 0.0
        return float(self.total_costs / entry_value * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'side': self.side,
            'quantity': float(self.quantity),
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price),
            'pnl': float(self.pnl),
            'return_pct': self.return_pct,
            'commission': float(self.commission),
            'slippage': float(self.slippage),
            'duration_hours': self.duration_hours,
            'duration_days': self.duration_days,
            'is_winner': self.is_winner,
            'gross_pnl': float(self.gross_pnl),
            'total_costs': float(self.total_costs),
            'cost_percentage': self.cost_percentage,
            'strategy_id': self.strategy_id,
        }


@dataclass
class TradeAnalytics:
    """Trade analytics results"""

    # Basic statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0

    # Performance
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_costs: float = 0.0

    # Win/Loss statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0

    # Profit metrics
    profit_factor: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0

    # Duration statistics
    avg_duration_hours: float = 0.0
    avg_duration_days: float = 0.0
    avg_win_duration: float = 0.0
    avg_loss_duration: float = 0.0
    shortest_trade: float = 0.0
    longest_trade: float = 0.0

    # Streak analysis
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    current_streak_type: str = "none"  # "win", "loss", "none"

    # Cost analysis
    avg_commission_per_trade: float = 0.0
    avg_slippage_per_trade: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    avg_cost_percentage: float = 0.0

    # Return distribution
    return_mean: float = 0.0
    return_std: float = 0.0
    return_skewness: float = 0.0
    return_kurtosis: float = 0.0

    # By side
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'basic': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'break_even_trades': self.break_even_trades,
            },
            'performance': {
                'total_pnl': self.total_pnl,
                'gross_pnl': self.gross_pnl,
                'net_pnl': self.net_pnl,
                'total_costs': self.total_costs,
            },
            'win_loss': {
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
                'avg_win_loss_ratio': self.avg_win_loss_ratio,
            },
            'profit_metrics': {
                'profit_factor': self.profit_factor,
                'expectancy': self.expectancy,
                'kelly_criterion': self.kelly_criterion,
            },
            'duration': {
                'avg_duration_hours': self.avg_duration_hours,
                'avg_duration_days': self.avg_duration_days,
                'avg_win_duration': self.avg_win_duration,
                'avg_loss_duration': self.avg_loss_duration,
                'shortest_trade': self.shortest_trade,
                'longest_trade': self.longest_trade,
            },
            'streaks': {
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses,
                'current_streak': self.current_streak,
                'current_streak_type': self.current_streak_type,
            },
            'costs': {
                'avg_commission_per_trade': self.avg_commission_per_trade,
                'avg_slippage_per_trade': self.avg_slippage_per_trade,
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'avg_cost_percentage': self.avg_cost_percentage,
            },
            'returns': {
                'return_mean': self.return_mean,
                'return_std': self.return_std,
                'return_skewness': self.return_skewness,
                'return_kurtosis': self.return_kurtosis,
            },
            'by_side': {
                'long_trades': self.long_trades,
                'short_trades': self.short_trades,
                'long_win_rate': self.long_win_rate,
                'short_win_rate': self.short_win_rate,
                'long_pnl': self.long_pnl,
                'short_pnl': self.short_pnl,
            }
        }


class TradeAnalyzer:
    """
    Comprehensive trade-level analyzer

    Analyzes individual trades and provides detailed statistics.
    """

    def __init__(self):
        """Initialize trade analyzer"""
        self.trades: List[Trade] = []
        self.logger = logging.getLogger(__name__)

    def add_trade(self, trade: Trade) -> None:
        """
        Add a trade to the analyzer

        Args:
            trade: Trade object
        """
        self.trades.append(trade)

    def analyze(self) -> TradeAnalytics:
        """
        Perform comprehensive trade analysis

        Returns:
            TradeAnalytics object with all metrics
        """
        if not self.trades:
            return TradeAnalytics()

        analytics = TradeAnalytics()

        # Convert trades to DataFrame for easier analysis
        df = self._trades_to_dataframe()

        # Basic statistics
        analytics.total_trades = len(self.trades)
        analytics.winning_trades = (df['pnl'] > 0).sum()
        analytics.losing_trades = (df['pnl'] < 0).sum()
        analytics.break_even_trades = (df['pnl'] == 0).sum()

        # Performance
        analytics.total_pnl = df['pnl'].sum()
        analytics.gross_pnl = df['gross_pnl'].sum()
        analytics.net_pnl = analytics.total_pnl
        analytics.total_costs = df['total_costs'].sum()

        # Win/Loss statistics
        analytics.win_rate = (
            analytics.winning_trades / analytics.total_trades
            if analytics.total_trades > 0 else 0.0
        )

        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']

        analytics.avg_win = wins.mean() if len(wins) > 0 else 0.0
        analytics.avg_loss = losses.mean() if len(losses) > 0 else 0.0
        analytics.largest_win = wins.max() if len(wins) > 0 else 0.0
        analytics.largest_loss = losses.min() if len(losses) > 0 else 0.0

        analytics.avg_win_loss_ratio = (
            abs(analytics.avg_win / analytics.avg_loss)
            if analytics.avg_loss != 0 else 0.0
        )

        # Profit metrics
        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0

        analytics.profit_factor = (
            total_wins / total_losses
            if total_losses > 0 else 0.0
        )

        analytics.expectancy = (
            analytics.win_rate * analytics.avg_win +
            (1 - analytics.win_rate) * analytics.avg_loss
        )

        # Kelly Criterion
        if analytics.avg_loss != 0 and analytics.avg_win > 0:
            win_loss_ratio = abs(analytics.avg_win / analytics.avg_loss)
            analytics.kelly_criterion = (
                analytics.win_rate - ((1 - analytics.win_rate) / win_loss_ratio)
            )
        else:
            analytics.kelly_criterion = 0.0

        # Duration statistics
        analytics.avg_duration_hours = df['duration_hours'].mean()
        analytics.avg_duration_days = df['duration_days'].mean()

        win_durations = df[df['pnl'] > 0]['duration_hours']
        loss_durations = df[df['pnl'] < 0]['duration_hours']

        analytics.avg_win_duration = (
            win_durations.mean() if len(win_durations) > 0 else 0.0
        )
        analytics.avg_loss_duration = (
            loss_durations.mean() if len(loss_durations) > 0 else 0.0
        )

        analytics.shortest_trade = df['duration_hours'].min()
        analytics.longest_trade = df['duration_hours'].max()

        # Streak analysis
        streak_info = self._analyze_streaks(df)
        analytics.max_consecutive_wins = streak_info['max_wins']
        analytics.max_consecutive_losses = streak_info['max_losses']
        analytics.current_streak = streak_info['current_streak']
        analytics.current_streak_type = streak_info['current_type']

        # Cost analysis
        analytics.avg_commission_per_trade = df['commission'].mean()
        analytics.avg_slippage_per_trade = df['slippage'].abs().mean()
        analytics.total_commission = df['commission'].sum()
        analytics.total_slippage = df['slippage'].abs().sum()
        analytics.avg_cost_percentage = df['cost_percentage'].mean()

        # Return distribution
        returns = df['return_pct']
        analytics.return_mean = returns.mean()
        analytics.return_std = returns.std()

        if len(returns) >= 3:
            analytics.return_skewness = returns.skew()
        if len(returns) >= 4:
            analytics.return_kurtosis = returns.kurtosis()

        # By side
        long_trades = df[df['side'] == 'LONG']
        short_trades = df[df['side'] == 'SHORT']

        analytics.long_trades = len(long_trades)
        analytics.short_trades = len(short_trades)

        analytics.long_win_rate = (
            (long_trades['pnl'] > 0).sum() / len(long_trades)
            if len(long_trades) > 0 else 0.0
        )
        analytics.short_win_rate = (
            (short_trades['pnl'] > 0).sum() / len(short_trades)
            if len(short_trades) > 0 else 0.0
        )

        analytics.long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0.0
        analytics.short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0.0

        return analytics

    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        return pd.DataFrame([trade.to_dict() for trade in self.trades])

    def _analyze_streaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze win/loss streaks"""
        if len(df) == 0:
            return {
                'max_wins': 0,
                'max_losses': 0,
                'current_streak': 0,
                'current_type': 'none'
            }

        # Determine if each trade is a win
        is_win = df['pnl'] > 0

        max_wins = 0
        max_losses = 0
        current_streak = 0
        current_type = 'none'

        # Calculate streaks
        current_win_streak = 0
        current_loss_streak = 0

        for win in is_win:
            if win:
                current_win_streak += 1
                current_loss_streak = 0
            else:
                current_loss_streak += 1
                current_win_streak = 0

            max_wins = max(max_wins, current_win_streak)
            max_losses = max(max_losses, current_loss_streak)

        # Current streak
        if current_win_streak > 0:
            current_streak = current_win_streak
            current_type = 'win'
        elif current_loss_streak > 0:
            current_streak = current_loss_streak
            current_type = 'loss'

        return {
            'max_wins': max_wins,
            'max_losses': max_losses,
            'current_streak': current_streak,
            'current_type': current_type
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get trades as DataFrame

        Returns:
            DataFrame with all trades
        """
        return self._trades_to_dataframe()

    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """
        Get trades for a specific symbol

        Args:
            symbol: Symbol to filter by

        Returns:
            List of trades for the symbol
        """
        return [trade for trade in self.trades if trade.symbol == symbol]

    def get_trades_by_strategy(self, strategy_id: str) -> List[Trade]:
        """
        Get trades for a specific strategy

        Args:
            strategy_id: Strategy ID to filter by

        Returns:
            List of trades for the strategy
        """
        return [
            trade for trade in self.trades
            if trade.strategy_id == strategy_id
        ]

    def get_trades_by_timeframe(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Trade]:
        """
        Get trades within a timeframe

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of trades in the timeframe
        """
        return [
            trade for trade in self.trades
            if start_time <= trade.entry_time <= end_time
        ]

    def analyze_by_hour(self) -> pd.DataFrame:
        """
        Analyze performance by hour of day

        Returns:
            DataFrame with hourly statistics
        """
        df = self._trades_to_dataframe()

        if len(df) == 0:
            return pd.DataFrame()

        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour

        hourly = df.groupby('hour').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_winner': 'mean',
        })

        hourly.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']

        return hourly

    def analyze_by_day_of_week(self) -> pd.DataFrame:
        """
        Analyze performance by day of week

        Returns:
            DataFrame with daily statistics
        """
        df = self._trades_to_dataframe()

        if len(df) == 0:
            return pd.DataFrame()

        df['day_of_week'] = pd.to_datetime(df['entry_time']).dt.dayofweek

        daily = df.groupby('day_of_week').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_winner': 'mean',
        })

        daily.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']

        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily.index = daily.index.map(lambda x: day_names[x])

        return daily

    def analyze_by_duration(
        self,
        bins: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Analyze performance by trade duration

        Args:
            bins: Duration bins in hours (default: [0, 1, 4, 24, 168])

        Returns:
            DataFrame with duration statistics
        """
        df = self._trades_to_dataframe()

        if len(df) == 0:
            return pd.DataFrame()

        if bins is None:
            bins = [0, 1, 4, 24, 168, float('inf')]

        labels = [f'{bins[i]}-{bins[i+1]}h' for i in range(len(bins)-1)]
        df['duration_bin'] = pd.cut(df['duration_hours'], bins=bins, labels=labels)

        duration_analysis = df.groupby('duration_bin').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_winner': 'mean',
        })

        duration_analysis.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']

        return duration_analysis

    def get_best_trades(self, n: int = 10) -> List[Trade]:
        """
        Get best performing trades

        Args:
            n: Number of trades to return

        Returns:
            List of best trades by P&L
        """
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl, reverse=True)
        return sorted_trades[:n]

    def get_worst_trades(self, n: int = 10) -> List[Trade]:
        """
        Get worst performing trades

        Args:
            n: Number of trades to return

        Returns:
            List of worst trades by P&L
        """
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl)
        return sorted_trades[:n]

    def get_summary(self) -> str:
        """
        Get formatted summary of trade analysis

        Returns:
            Formatted string summary
        """
        analytics = self.analyze()

        summary = f"""
Trade Analysis Summary
{'=' * 50}

Basic Statistics:
  Total Trades: {analytics.total_trades}
  Winning Trades: {analytics.winning_trades}
  Losing Trades: {analytics.losing_trades}
  Win Rate: {analytics.win_rate:.2%}

Performance:
  Total P&L: ${analytics.total_pnl:,.2f}
  Average Win: ${analytics.avg_win:,.2f}
  Average Loss: ${analytics.avg_loss:,.2f}
  Largest Win: ${analytics.largest_win:,.2f}
  Largest Loss: ${analytics.largest_loss:,.2f}

Profit Metrics:
  Profit Factor: {analytics.profit_factor:.2f}
  Expectancy: ${analytics.expectancy:,.2f}
  Kelly Criterion: {analytics.kelly_criterion:.2%}

Duration:
  Average Duration: {analytics.avg_duration_hours:.2f} hours
  Average Win Duration: {analytics.avg_win_duration:.2f} hours
  Average Loss Duration: {analytics.avg_loss_duration:.2f} hours

Costs:
  Total Commission: ${analytics.total_commission:,.2f}
  Total Slippage: ${analytics.total_slippage:,.2f}
  Average Cost %: {analytics.avg_cost_percentage:.3f}%

Streaks:
  Max Consecutive Wins: {analytics.max_consecutive_wins}
  Max Consecutive Losses: {analytics.max_consecutive_losses}

By Side:
  Long Trades: {analytics.long_trades} (Win Rate: {analytics.long_win_rate:.2%})
  Short Trades: {analytics.short_trades} (Win Rate: {analytics.short_win_rate:.2%})
"""
        return summary

    def clear(self) -> None:
        """Clear all trades"""
        self.trades.clear()
