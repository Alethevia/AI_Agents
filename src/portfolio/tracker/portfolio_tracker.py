"""
Portfolio Tracker Module

Implements real-time portfolio tracking:
- Position tracking
- P&L calculation (realized, unrealized)
- Cash management
- Corporate actions handling
- Portfolio snapshot history
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
import json

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Transaction types"""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    FEE = "fee"
    INTEREST = "interest"


class CorporateActionType(Enum):
    """Corporate action types"""
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    MERGER = "merger"
    SPINOFF = "spinoff"


@dataclass
class Transaction:
    """Individual transaction"""
    transaction_id: str
    timestamp: datetime
    transaction_type: TransactionType
    symbol: Optional[str]
    quantity: float
    price: float
    commission: float = 0.0
    fees: float = 0.0
    notes: Optional[str] = None

    @property
    def total_cost(self) -> float:
        """Total cost including commissions and fees"""
        base_cost = self.quantity * self.price
        if self.transaction_type in [TransactionType.BUY, TransactionType.SELL]:
            return abs(base_cost) + self.commission + self.fees
        return base_cost

    @property
    def net_amount(self) -> float:
        """Net amount (positive = cash in, negative = cash out)"""
        if self.transaction_type == TransactionType.BUY:
            return -(self.quantity * self.price + self.commission + self.fees)
        elif self.transaction_type == TransactionType.SELL:
            return self.quantity * self.price - self.commission - self.fees
        elif self.transaction_type in [TransactionType.DIVIDEND, TransactionType.INTEREST]:
            return self.quantity * self.price
        elif self.transaction_type == TransactionType.DEPOSIT:
            return self.quantity  # quantity represents amount for deposits
        elif self.transaction_type == TransactionType.WITHDRAWAL:
            return -self.quantity
        elif self.transaction_type == TransactionType.FEE:
            return -self.quantity
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat(),
            'transaction_type': self.transaction_type.value,
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'commission': float(self.commission),
            'fees': float(self.fees),
            'notes': self.notes
        }


@dataclass
class Position:
    """Portfolio position with cost basis tracking"""
    symbol: str
    quantity: float
    cost_basis: float  # Average cost basis
    first_acquired: datetime
    last_updated: datetime
    realized_pnl: float = 0.0
    lots: List[Dict] = field(default_factory=list)  # For tax lot tracking

    @property
    def total_cost(self) -> float:
        """Total cost of position"""
        return self.quantity * self.cost_basis

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        return self.quantity * (current_price - self.cost_basis)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage"""
        if self.cost_basis == 0:
            return 0.0
        return (current_price - self.cost_basis) / self.cost_basis

    def to_dict(self, current_price: Optional[float] = None) -> Dict:
        """Convert to dictionary"""
        result = {
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'cost_basis': float(self.cost_basis),
            'total_cost': float(self.total_cost),
            'first_acquired': self.first_acquired.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'realized_pnl': float(self.realized_pnl)
        }
        if current_price is not None:
            result['current_price'] = float(current_price)
            result['market_value'] = float(self.quantity * current_price)
            result['unrealized_pnl'] = float(self.unrealized_pnl(current_price))
            result['unrealized_pnl_pct'] = float(self.unrealized_pnl_pct(current_price))
        return result


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time"""
    timestamp: datetime
    positions: Dict[str, Position]
    cash: float
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_deposits: float
    total_withdrawals: float
    total_fees: float
    metadata: Optional[Dict] = None

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_deposits(self) -> float:
        """Net deposits (deposits - withdrawals)"""
        return self.total_deposits - self.total_withdrawals

    @property
    def return_on_investment(self) -> float:
        """Return on investment"""
        if self.net_deposits == 0:
            return 0.0
        return self.total_pnl / self.net_deposits

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'positions': {
                symbol: pos.to_dict()
                for symbol, pos in self.positions.items()
            },
            'cash': float(self.cash),
            'total_value': float(self.total_value),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.total_pnl),
            'total_deposits': float(self.total_deposits),
            'total_withdrawals': float(self.total_withdrawals),
            'total_fees': float(self.total_fees),
            'net_deposits': float(self.net_deposits),
            'return_on_investment': float(self.return_on_investment),
            'metadata': self.metadata
        }


@dataclass
class CorporateAction:
    """Corporate action event"""
    action_id: str
    timestamp: datetime
    action_type: CorporateActionType
    symbol: str
    ratio: Optional[float] = None  # For splits
    dividend_amount: Optional[float] = None  # For dividends
    new_symbol: Optional[str] = None  # For mergers
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'action_id': self.action_id,
            'timestamp': self.timestamp.isoformat(),
            'action_type': self.action_type.value,
            'symbol': self.symbol,
            'ratio': float(self.ratio) if self.ratio else None,
            'dividend_amount': float(self.dividend_amount) if self.dividend_amount else None,
            'new_symbol': self.new_symbol,
            'metadata': self.metadata
        }


class PortfolioTracker:
    """
    Real-time portfolio tracker with comprehensive position and P&L tracking
    """

    def __init__(
        self,
        initial_cash: float = 0.0,
        enable_tax_lots: bool = True,
        snapshot_frequency: Optional[timedelta] = None
    ):
        """
        Initialize portfolio tracker

        Args:
            initial_cash: Initial cash balance
            enable_tax_lots: Enable tax lot tracking (FIFO, LIFO, etc.)
            snapshot_frequency: Frequency for automatic snapshots (None = manual only)
        """
        self.positions: Dict[str, Position] = {}
        self.cash = initial_cash
        self.transactions: List[Transaction] = []
        self.corporate_actions: List[CorporateAction] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.enable_tax_lots = enable_tax_lots
        self.snapshot_frequency = snapshot_frequency

        # Tracking metrics
        self.total_deposits = initial_cash
        self.total_withdrawals = 0.0
        self.total_fees = 0.0
        self.total_realized_pnl = 0.0

        # Last snapshot time
        self.last_snapshot_time: Optional[datetime] = None

        logger.info(
            f"Initialized PortfolioTracker with initial_cash={initial_cash:.2f}, "
            f"tax_lots={enable_tax_lots}"
        )

    def add_transaction(
        self,
        transaction: Transaction
    ) -> None:
        """
        Add a transaction to the portfolio

        Args:
            transaction: Transaction to add
        """
        self.transactions.append(transaction)

        # Update positions and cash based on transaction type
        if transaction.transaction_type == TransactionType.BUY:
            self._process_buy(transaction)
        elif transaction.transaction_type == TransactionType.SELL:
            self._process_sell(transaction)
        elif transaction.transaction_type == TransactionType.DIVIDEND:
            self._process_dividend(transaction)
        elif transaction.transaction_type == TransactionType.DEPOSIT:
            self._process_deposit(transaction)
        elif transaction.transaction_type == TransactionType.WITHDRAWAL:
            self._process_withdrawal(transaction)
        elif transaction.transaction_type == TransactionType.FEE:
            self._process_fee(transaction)
        elif transaction.transaction_type == TransactionType.INTEREST:
            self._process_interest(transaction)

        # Update cash
        self.cash += transaction.net_amount

        # Check if snapshot needed
        if self.snapshot_frequency is not None:
            if (self.last_snapshot_time is None or
                transaction.timestamp - self.last_snapshot_time >= self.snapshot_frequency):
                self.take_snapshot(transaction.timestamp)

        logger.debug(
            f"Added transaction: {transaction.transaction_type.value} "
            f"{transaction.symbol} {transaction.quantity}@{transaction.price}"
        )

    def _process_buy(self, transaction: Transaction) -> None:
        """Process buy transaction"""
        symbol = transaction.symbol

        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=transaction.quantity,
                cost_basis=transaction.price,
                first_acquired=transaction.timestamp,
                last_updated=transaction.timestamp,
                realized_pnl=0.0,
                lots=[]
            )

            if self.enable_tax_lots:
                self.positions[symbol].lots.append({
                    'quantity': transaction.quantity,
                    'cost_basis': transaction.price,
                    'acquisition_date': transaction.timestamp,
                    'transaction_id': transaction.transaction_id
                })

        else:
            # Add to existing position
            position = self.positions[symbol]
            total_cost = position.total_cost + (transaction.quantity * transaction.price)
            total_quantity = position.quantity + transaction.quantity

            position.cost_basis = total_cost / total_quantity if total_quantity > 0 else 0.0
            position.quantity = total_quantity
            position.last_updated = transaction.timestamp

            if self.enable_tax_lots:
                position.lots.append({
                    'quantity': transaction.quantity,
                    'cost_basis': transaction.price,
                    'acquisition_date': transaction.timestamp,
                    'transaction_id': transaction.transaction_id
                })

    def _process_sell(self, transaction: Transaction) -> None:
        """Process sell transaction"""
        symbol = transaction.symbol

        if symbol not in self.positions:
            logger.warning(f"Attempting to sell {symbol} with no position")
            return

        position = self.positions[symbol]

        if position.quantity < transaction.quantity:
            logger.warning(
                f"Attempting to sell {transaction.quantity} shares of {symbol}, "
                f"but only {position.quantity} available"
            )
            return

        # Calculate realized P&L
        if self.enable_tax_lots:
            realized_pnl = self._calculate_realized_pnl_fifo(position, transaction)
        else:
            realized_pnl = transaction.quantity * (transaction.price - position.cost_basis)

        position.realized_pnl += realized_pnl
        self.total_realized_pnl += realized_pnl

        # Update position
        position.quantity -= transaction.quantity
        position.last_updated = transaction.timestamp

        # Remove position if fully closed
        if position.quantity <= 1e-6:  # Allow for floating point errors
            del self.positions[symbol]

    def _calculate_realized_pnl_fifo(
        self,
        position: Position,
        transaction: Transaction
    ) -> float:
        """Calculate realized P&L using FIFO method"""
        remaining_quantity = transaction.quantity
        realized_pnl = 0.0
        lots_to_remove = []

        for i, lot in enumerate(position.lots):
            if remaining_quantity <= 0:
                break

            lot_quantity = lot['quantity']
            lot_cost = lot['cost_basis']

            if lot_quantity <= remaining_quantity:
                # Fully consume this lot
                realized_pnl += lot_quantity * (transaction.price - lot_cost)
                remaining_quantity -= lot_quantity
                lots_to_remove.append(i)
            else:
                # Partially consume this lot
                realized_pnl += remaining_quantity * (transaction.price - lot_cost)
                lot['quantity'] -= remaining_quantity
                remaining_quantity = 0

        # Remove fully consumed lots (in reverse order to maintain indices)
        for i in reversed(lots_to_remove):
            position.lots.pop(i)

        return realized_pnl

    def _process_dividend(self, transaction: Transaction) -> None:
        """Process dividend transaction"""
        # Dividend adds to cash (handled by net_amount)
        # Can also track dividend income separately if needed
        pass

    def _process_deposit(self, transaction: Transaction) -> None:
        """Process cash deposit"""
        self.total_deposits += transaction.quantity

    def _process_withdrawal(self, transaction: Transaction) -> None:
        """Process cash withdrawal"""
        self.total_withdrawals += transaction.quantity

    def _process_fee(self, transaction: Transaction) -> None:
        """Process fee transaction"""
        self.total_fees += transaction.quantity

    def _process_interest(self, transaction: Transaction) -> None:
        """Process interest transaction"""
        # Interest adds to cash (handled by net_amount)
        pass

    def process_corporate_action(
        self,
        action: CorporateAction
    ) -> None:
        """
        Process corporate action

        Args:
            action: Corporate action to process
        """
        self.corporate_actions.append(action)

        if action.action_type == CorporateActionType.SPLIT:
            self._process_split(action)
        elif action.action_type == CorporateActionType.REVERSE_SPLIT:
            self._process_reverse_split(action)
        elif action.action_type == CorporateActionType.DIVIDEND:
            self._process_dividend_action(action)
        elif action.action_type == CorporateActionType.MERGER:
            self._process_merger(action)
        elif action.action_type == CorporateActionType.SPINOFF:
            self._process_spinoff(action)

        logger.info(
            f"Processed corporate action: {action.action_type.value} for {action.symbol}"
        )

    def _process_split(self, action: CorporateAction) -> None:
        """Process stock split"""
        if action.symbol not in self.positions:
            return

        position = self.positions[action.symbol]
        ratio = action.ratio or 1.0

        # Adjust quantity and cost basis
        position.quantity *= ratio
        position.cost_basis /= ratio
        position.last_updated = action.timestamp

        # Adjust tax lots
        if self.enable_tax_lots:
            for lot in position.lots:
                lot['quantity'] *= ratio
                lot['cost_basis'] /= ratio

    def _process_reverse_split(self, action: CorporateAction) -> None:
        """Process reverse stock split"""
        if action.symbol not in self.positions:
            return

        position = self.positions[action.symbol]
        ratio = action.ratio or 1.0

        # Adjust quantity and cost basis (inverse of split)
        position.quantity /= ratio
        position.cost_basis *= ratio
        position.last_updated = action.timestamp

        # Adjust tax lots
        if self.enable_tax_lots:
            for lot in position.lots:
                lot['quantity'] /= ratio
                lot['cost_basis'] *= ratio

    def _process_dividend_action(self, action: CorporateAction) -> None:
        """Process dividend corporate action"""
        if action.symbol not in self.positions:
            return

        position = self.positions[action.symbol]
        dividend_per_share = action.dividend_amount or 0.0

        # Add dividend to cash
        total_dividend = position.quantity * dividend_per_share
        self.cash += total_dividend

        # Create transaction record
        transaction = Transaction(
            transaction_id=f"DIV_{action.action_id}",
            timestamp=action.timestamp,
            transaction_type=TransactionType.DIVIDEND,
            symbol=action.symbol,
            quantity=position.quantity,
            price=dividend_per_share
        )
        self.transactions.append(transaction)

    def _process_merger(self, action: CorporateAction) -> None:
        """Process merger corporate action"""
        if action.symbol not in self.positions:
            return

        position = self.positions[action.symbol]
        new_symbol = action.new_symbol

        if new_symbol:
            # Transfer position to new symbol
            position.symbol = new_symbol
            position.last_updated = action.timestamp

            self.positions[new_symbol] = position
            del self.positions[action.symbol]

    def _process_spinoff(self, action: CorporateAction) -> None:
        """Process spinoff corporate action"""
        # Spinoff typically creates a new position
        # Implementation depends on specific spinoff terms
        # This is a simplified version
        pass

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol

        Args:
            symbol: Symbol to lookup

        Returns:
            Position if exists, None otherwise
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return deepcopy(self.positions)

    def get_portfolio_value(
        self,
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate total portfolio value

        Args:
            prices: Current prices for all symbols

        Returns:
            Total portfolio value
        """
        positions_value = sum(
            pos.quantity * prices.get(pos.symbol, pos.cost_basis)
            for pos in self.positions.values()
        )

        return self.cash + positions_value

    def get_unrealized_pnl(
        self,
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate total unrealized P&L

        Args:
            prices: Current prices for all symbols

        Returns:
            Total unrealized P&L
        """
        return sum(
            pos.unrealized_pnl(prices.get(pos.symbol, pos.cost_basis))
            for pos in self.positions.values()
        )

    def get_position_weights(
        self,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate position weights

        Args:
            prices: Current prices for all symbols

        Returns:
            Dictionary of position weights
        """
        total_value = self.get_portfolio_value(prices)

        if total_value == 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            market_value = position.quantity * prices.get(symbol, position.cost_basis)
            weights[symbol] = market_value / total_value

        # Add cash weight
        weights['CASH'] = self.cash / total_value

        return weights

    def take_snapshot(
        self,
        timestamp: Optional[datetime] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> PortfolioSnapshot:
        """
        Take portfolio snapshot

        Args:
            timestamp: Snapshot timestamp (default: now)
            prices: Current prices (optional)

        Returns:
            PortfolioSnapshot
        """
        timestamp = timestamp or datetime.now()
        prices = prices or {}

        # Calculate metrics
        unrealized_pnl = self.get_unrealized_pnl(prices)
        total_value = self.get_portfolio_value(prices)

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            positions=deepcopy(self.positions),
            cash=self.cash,
            total_value=total_value,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_deposits=self.total_deposits,
            total_withdrawals=self.total_withdrawals,
            total_fees=self.total_fees
        )

        self.snapshots.append(snapshot)
        self.last_snapshot_time = timestamp

        logger.debug(f"Took portfolio snapshot at {timestamp}")

        return snapshot

    def get_snapshot_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PortfolioSnapshot]:
        """
        Get snapshot history

        Args:
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            List of snapshots
        """
        snapshots = self.snapshots

        if start_date:
            snapshots = [s for s in snapshots if s.timestamp >= start_date]

        if end_date:
            snapshots = [s for s in snapshots if s.timestamp <= end_date]

        return snapshots

    def get_transaction_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[TransactionType] = None
    ) -> List[Transaction]:
        """
        Get transaction history with filters

        Args:
            symbol: Filter by symbol (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            transaction_type: Filter by transaction type (optional)

        Returns:
            List of transactions
        """
        transactions = self.transactions

        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol]

        if start_date:
            transactions = [t for t in transactions if t.timestamp >= start_date]

        if end_date:
            transactions = [t for t in transactions if t.timestamp <= end_date]

        if transaction_type:
            transactions = [t for t in transactions if t.transaction_type == transaction_type]

        return transactions

    def get_performance_summary(
        self,
        prices: Dict[str, float]
    ) -> Dict:
        """
        Get comprehensive performance summary

        Args:
            prices: Current prices for all symbols

        Returns:
            Dictionary with performance metrics
        """
        total_value = self.get_portfolio_value(prices)
        unrealized_pnl = self.get_unrealized_pnl(prices)
        net_deposits = self.total_deposits - self.total_withdrawals

        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'realized_pnl': self.total_realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.total_realized_pnl + unrealized_pnl,
            'net_deposits': net_deposits,
            'total_deposits': self.total_deposits,
            'total_withdrawals': self.total_withdrawals,
            'total_fees': self.total_fees,
            'return_on_investment': (
                (self.total_realized_pnl + unrealized_pnl) / net_deposits
                if net_deposits > 0 else 0.0
            ),
            'num_positions': len(self.positions),
            'num_transactions': len(self.transactions)
        }

    def export_to_dict(self) -> Dict:
        """Export portfolio state to dictionary"""
        return {
            'positions': {
                symbol: pos.to_dict()
                for symbol, pos in self.positions.items()
            },
            'cash': float(self.cash),
            'total_deposits': float(self.total_deposits),
            'total_withdrawals': float(self.total_withdrawals),
            'total_fees': float(self.total_fees),
            'total_realized_pnl': float(self.total_realized_pnl),
            'transactions': [t.to_dict() for t in self.transactions],
            'corporate_actions': [a.to_dict() for a in self.corporate_actions]
        }

    def export_to_json(self, filepath: str) -> None:
        """Export portfolio state to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2)

        logger.info(f"Exported portfolio to {filepath}")

    async def add_transaction_async(self, transaction: Transaction) -> None:
        """Async version of add_transaction"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.add_transaction, transaction)

    async def take_snapshot_async(
        self,
        timestamp: Optional[datetime] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> PortfolioSnapshot:
        """Async version of take_snapshot"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.take_snapshot,
            timestamp,
            prices
        )
