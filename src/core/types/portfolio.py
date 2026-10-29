"""Portfolio type definitions for trading system.

This module defines portfolio-related types including portfolio tracking,
metrics calculation, and historical snapshots.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .positions import Position, PositionSide


class Portfolio(BaseModel):
    """Portfolio class representing a trading portfolio.

    Tracks positions, cash balance, total value, and performance metrics.
    """

    # Identification
    portfolio_id: str = Field(..., description="Unique portfolio identifier")
    name: Optional[str] = Field(None, description="Portfolio name")

    # Positions
    positions: Dict[str, Position] = Field(
        default_factory=dict,
        description="Active positions keyed by symbol"
    )

    # Cash and value
    cash: Decimal = Field(..., ge=0, description="Available cash balance")
    initial_capital: Decimal = Field(..., gt=0, description="Initial capital")

    # Performance tracking
    total_realized_pnl: Decimal = Field(
        default=Decimal('0'),
        description="Total realized P&L across all positions"
    )
    total_unrealized_pnl: Decimal = Field(
        default=Decimal('0'),
        description="Total unrealized P&L across all positions"
    )

    # Cost tracking
    total_commission: Decimal = Field(
        default=Decimal('0'),
        ge=0,
        description="Total commissions paid"
    )
    total_fees: Decimal = Field(
        default=Decimal('0'),
        ge=0,
        description="Total fees paid"
    )

    # Timing
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Portfolio creation time"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update time"
    )

    @field_validator('portfolio_id')
    @classmethod
    def validate_portfolio_id(cls, v: str) -> str:
        """Validate portfolio ID."""
        if not v or not v.strip():
            raise ValueError("Portfolio ID cannot be empty")
        return v.strip()

    @model_validator(mode='after')
    def validate_portfolio(self) -> 'Portfolio':
        """Validate portfolio constraints."""
        # Cash cannot be negative
        if self.cash < 0:
            raise ValueError(f"Cash balance cannot be negative: {self.cash}")

        return self

    def total_position_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        return sum(pos.market_value() for pos in self.positions.values())

    def total_value(self) -> Decimal:
        """Calculate total portfolio value (cash + positions)."""
        return self.cash + self.total_position_value()

    def equity(self) -> Decimal:
        """Calculate equity (total value)."""
        return self.total_value()

    def net_liquidation_value(self) -> Decimal:
        """Calculate net liquidation value."""
        return self.equity()

    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    def total_costs(self) -> Decimal:
        """Calculate total costs (commissions + fees)."""
        return self.total_commission + self.total_fees

    def net_pnl(self) -> Decimal:
        """Calculate net P&L after costs."""
        return self.total_pnl() - self.total_costs()

    def total_return(self) -> Decimal:
        """Calculate total return as percentage of initial capital."""
        if self.initial_capital > 0:
            return (self.total_pnl() / self.initial_capital) * Decimal('100')
        return Decimal('0')

    def net_return(self) -> Decimal:
        """Calculate net return after costs as percentage."""
        if self.initial_capital > 0:
            return (self.net_pnl() / self.initial_capital) * Decimal('100')
        return Decimal('0')

    def leverage(self) -> Decimal:
        """Calculate leverage ratio."""
        equity = self.equity()
        if equity > 0:
            return self.total_position_value() / equity
        return Decimal('0')

    def position_count(self) -> int:
        """Count active positions."""
        return len([p for p in self.positions.values() if not p.is_flat()])

    def long_position_count(self) -> int:
        """Count long positions."""
        return len([p for p in self.positions.values() if p.is_long()])

    def short_position_count(self) -> int:
        """Count short positions."""
        return len([p for p in self.positions.values() if p.is_short()])

    def long_exposure(self) -> Decimal:
        """Calculate total long exposure."""
        return sum(
            p.market_value()
            for p in self.positions.values()
            if p.is_long()
        )

    def short_exposure(self) -> Decimal:
        """Calculate total short exposure."""
        return sum(
            p.market_value()
            for p in self.positions.values()
            if p.is_short()
        )

    def net_exposure(self) -> Decimal:
        """Calculate net exposure (long - short)."""
        return self.long_exposure() - self.short_exposure()

    def gross_exposure(self) -> Decimal:
        """Calculate gross exposure (long + short)."""
        return self.long_exposure() + self.short_exposure()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol.upper())

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol."""
        pos = self.get_position(symbol)
        return pos is not None and not pos.is_flat()

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self.positions[position.symbol] = position
        self.recalculate_metrics()

    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        symbol = symbol.upper()
        if symbol in self.positions:
            del self.positions[symbol]
        self.recalculate_metrics()

    def recalculate_metrics(self) -> None:
        """Recalculate portfolio-level metrics from positions."""
        self.total_unrealized_pnl = sum(
            p.unrealized_pnl for p in self.positions.values()
        )
        self.total_realized_pnl = sum(
            p.realized_pnl for p in self.positions.values()
        )
        self.total_commission = sum(
            p.total_commission for p in self.positions.values()
        )
        self.total_fees = sum(
            p.total_fees for p in self.positions.values()
        )
        self.last_updated = datetime.utcnow()

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update market prices for positions."""
        for symbol, price in prices.items():
            symbol = symbol.upper()
            if symbol in self.positions:
                self.positions[symbol].update_market_price(price)
        self.recalculate_metrics()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'positions': {
                symbol: pos.to_dict() for symbol, pos in self.positions.items()
            },
            'cash': str(self.cash),
            'initial_capital': str(self.initial_capital),
            'total_realized_pnl': str(self.total_realized_pnl),
            'total_unrealized_pnl': str(self.total_unrealized_pnl),
            'total_commission': str(self.total_commission),
            'total_fees': str(self.total_fees),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create Portfolio from dictionary."""
        data_copy = data.copy()

        # Convert datetime fields
        for field in ['created_at', 'last_updated']:
            if field in data_copy and data_copy[field] is not None:
                if isinstance(data_copy[field], str):
                    data_copy[field] = datetime.fromisoformat(data_copy[field])

        # Convert decimal fields
        decimal_fields = [
            'cash', 'initial_capital', 'total_realized_pnl',
            'total_unrealized_pnl', 'total_commission', 'total_fees'
        ]
        for field in decimal_fields:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        # Convert positions
        if 'positions' in data_copy:
            data_copy['positions'] = {
                symbol: Position.from_dict(pos) if isinstance(pos, dict) else pos
                for symbol, pos in data_copy['positions'].items()
            }

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics.

    Detailed performance and risk metrics for a portfolio.
    """

    # Returns
    total_return: Decimal = Field(..., description="Total return percentage")
    net_return: Decimal = Field(..., description="Net return after costs")
    annualized_return: Optional[Decimal] = Field(None, description="Annualized return")

    # Risk metrics
    max_drawdown: Decimal = Field(default=Decimal('0'), description="Maximum drawdown percentage")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[Decimal] = Field(None, description="Sortino ratio")
    volatility: Optional[Decimal] = Field(None, description="Return volatility")

    # Win/loss statistics
    total_trades: int = Field(default=0, ge=0, description="Total number of trades")
    winning_trades: int = Field(default=0, ge=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, ge=0, description="Number of losing trades")
    win_rate: Decimal = Field(default=Decimal('0'), ge=0, le=100, description="Win rate percentage")

    # Average metrics
    avg_win: Decimal = Field(default=Decimal('0'), description="Average winning trade")
    avg_loss: Decimal = Field(default=Decimal('0'), description="Average losing trade")
    avg_trade: Decimal = Field(default=Decimal('0'), description="Average trade P&L")

    # Profit factor
    profit_factor: Optional[Decimal] = Field(None, gt=0, description="Profit factor")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")

    @model_validator(mode='after')
    def validate_metrics(self) -> 'PortfolioMetrics':
        """Validate metrics consistency."""
        # Validate trade counts
        if self.winning_trades + self.losing_trades > self.total_trades:
            raise ValueError("Winning + losing trades cannot exceed total trades")

        return self

    def expectancy(self) -> Decimal:
        """Calculate expectancy (average expected profit per trade)."""
        if self.total_trades > 0:
            win_prob = self.win_rate / Decimal('100')
            loss_prob = Decimal('1') - win_prob
            return (win_prob * self.avg_win) - (loss_prob * abs(self.avg_loss))
        return Decimal('0')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_return': str(self.total_return),
            'net_return': str(self.net_return),
            'annualized_return': str(self.annualized_return) if self.annualized_return else None,
            'max_drawdown': str(self.max_drawdown),
            'sharpe_ratio': str(self.sharpe_ratio) if self.sharpe_ratio else None,
            'sortino_ratio': str(self.sortino_ratio) if self.sortino_ratio else None,
            'volatility': str(self.volatility) if self.volatility else None,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': str(self.win_rate),
            'avg_win': str(self.avg_win),
            'avg_loss': str(self.avg_loss),
            'avg_trade': str(self.avg_trade),
            'profit_factor': str(self.profit_factor) if self.profit_factor else None,
            'timestamp': self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioMetrics':
        """Create PortfolioMetrics from dictionary."""
        data_copy = data.copy()

        # Convert datetime
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert decimal fields
        decimal_fields = [
            'total_return', 'net_return', 'annualized_return', 'max_drawdown',
            'sharpe_ratio', 'sortino_ratio', 'volatility', 'win_rate',
            'avg_win', 'avg_loss', 'avg_trade', 'profit_factor'
        ]
        for field in decimal_fields:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot for historical tracking.

    Point-in-time snapshot of portfolio state for performance tracking
    and analytics.
    """

    # Timing
    timestamp: datetime = Field(..., description="Snapshot timestamp")

    # Portfolio reference
    portfolio_id: str = Field(..., description="Portfolio identifier")

    # Values
    cash: Decimal = Field(..., ge=0, description="Cash balance")
    position_value: Decimal = Field(..., ge=0, description="Total position value")
    total_value: Decimal = Field(..., gt=0, description="Total portfolio value")

    # P&L
    realized_pnl: Decimal = Field(..., description="Realized P&L")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    total_pnl: Decimal = Field(..., description="Total P&L")

    # Costs
    total_commission: Decimal = Field(default=Decimal('0'), ge=0, description="Total commissions")
    total_fees: Decimal = Field(default=Decimal('0'), ge=0, description="Total fees")

    # Position info
    position_count: int = Field(default=0, ge=0, description="Number of positions")
    long_exposure: Decimal = Field(default=Decimal('0'), ge=0, description="Long exposure")
    short_exposure: Decimal = Field(default=Decimal('0'), ge=0, description="Short exposure")

    # Metrics
    leverage: Decimal = Field(default=Decimal('0'), ge=0, description="Leverage ratio")
    return_pct: Decimal = Field(default=Decimal('0'), description="Return percentage")

    @field_validator('portfolio_id')
    @classmethod
    def validate_portfolio_id(cls, v: str) -> str:
        """Validate portfolio ID."""
        if not v or not v.strip():
            raise ValueError("Portfolio ID cannot be empty")
        return v.strip()

    @classmethod
    def from_portfolio(cls, portfolio: Portfolio) -> 'PortfolioSnapshot':
        """Create snapshot from current portfolio state."""
        return cls(
            timestamp=datetime.utcnow(),
            portfolio_id=portfolio.portfolio_id,
            cash=portfolio.cash,
            position_value=portfolio.total_position_value(),
            total_value=portfolio.total_value(),
            realized_pnl=portfolio.total_realized_pnl,
            unrealized_pnl=portfolio.total_unrealized_pnl,
            total_pnl=portfolio.total_pnl(),
            total_commission=portfolio.total_commission,
            total_fees=portfolio.total_fees,
            position_count=portfolio.position_count(),
            long_exposure=portfolio.long_exposure(),
            short_exposure=portfolio.short_exposure(),
            leverage=portfolio.leverage(),
            return_pct=portfolio.total_return(),
        )

    def net_exposure(self) -> Decimal:
        """Calculate net exposure."""
        return self.long_exposure - self.short_exposure

    def gross_exposure(self) -> Decimal:
        """Calculate gross exposure."""
        return self.long_exposure + self.short_exposure

    def net_pnl(self) -> Decimal:
        """Calculate net P&L after costs."""
        return self.total_pnl - self.total_commission - self.total_fees

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_id': self.portfolio_id,
            'cash': str(self.cash),
            'position_value': str(self.position_value),
            'total_value': str(self.total_value),
            'realized_pnl': str(self.realized_pnl),
            'unrealized_pnl': str(self.unrealized_pnl),
            'total_pnl': str(self.total_pnl),
            'total_commission': str(self.total_commission),
            'total_fees': str(self.total_fees),
            'position_count': self.position_count,
            'long_exposure': str(self.long_exposure),
            'short_exposure': str(self.short_exposure),
            'leverage': str(self.leverage),
            'return_pct': str(self.return_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioSnapshot':
        """Create PortfolioSnapshot from dictionary."""
        data_copy = data.copy()

        # Convert datetime
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])

        # Convert decimal fields
        decimal_fields = [
            'cash', 'position_value', 'total_value', 'realized_pnl',
            'unrealized_pnl', 'total_pnl', 'total_commission', 'total_fees',
            'long_exposure', 'short_exposure', 'leverage', 'return_pct'
        ]
        for field in decimal_fields:
            if field in data_copy and data_copy[field] is not None:
                if not isinstance(data_copy[field], Decimal):
                    data_copy[field] = Decimal(str(data_copy[field]))

        return cls(**data_copy)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }
