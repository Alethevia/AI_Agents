"""
Core Backtesting Engine

Event-driven backtesting engine with:
- Historical data replay
- Order fill simulation
- Commission and slippage modeling
- Multi-strategy support
- Walk-forward analysis
- Concurrent backtesting
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from enum import Enum
import uuid

from .event_queue import (
    EventQueue,
    EventDispatcher,
    Event,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PositionUpdateEvent,
    PortfolioUpdateEvent,
    EventType
)
from ..analytics.backtest_analyzer import BacktestAnalyzer, BacktestResults
from ..analytics.trade_analyzer import Trade

logger = logging.getLogger(__name__)


class ExecutionModel(str, Enum):
    """Order execution models"""
    MARKET = "market"  # Fill at next bar open
    CLOSE = "close"  # Fill at current bar close
    REALISTIC = "realistic"  # Realistic fill with slippage


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    side: str  # "LONG" or "SHORT"
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None

    def __post_init__(self):
        """Ensure Decimal types"""
        self.quantity = Decimal(str(self.quantity))
        self.average_price = Decimal(str(self.average_price))
        self.current_price = Decimal(str(self.current_price))
        self.unrealized_pnl = Decimal(str(self.unrealized_pnl))
        self.realized_pnl = Decimal(str(self.realized_pnl))

    @property
    def market_value(self) -> Decimal:
        """Current market value of position"""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Cost basis of position"""
        return self.quantity * self.average_price

    def update_price(self, price: Decimal, timestamp: datetime) -> None:
        """Update current price and unrealized P&L"""
        self.current_price = Decimal(str(price))
        self.last_update = timestamp

        if self.side == "LONG":
            self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.average_price - self.current_price) * self.quantity


@dataclass
class Portfolio:
    """Portfolio state"""
    cash: Decimal
    initial_capital: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Ensure Decimal types"""
        self.cash = Decimal(str(self.cash))
        self.initial_capital = Decimal(str(self.initial_capital))

    @property
    def positions_value(self) -> Decimal:
        """Total value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_value(self) -> Decimal:
        """Total portfolio value"""
        return self.cash + self.positions_value

    @property
    def unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> Decimal:
        """Total realized P&L"""
        return sum(pos.realized_pnl for pos in self.positions.values())

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L"""
        return self.total_value - self.initial_capital


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    # Capital
    initial_capital: float = 100000.0

    # Execution
    execution_model: ExecutionModel = ExecutionModel.REALISTIC
    fill_delay_bars: int = 1  # Bars until fill

    # Costs
    commission_model: str = "fixed"  # "fixed", "percentage", "tiered"
    commission_rate: float = 0.001  # 0.1% or $1 per trade
    slippage_model: str = "fixed"  # "fixed", "volume", "spread"
    slippage_bps: float = 5.0  # 5 basis points

    # Risk management
    max_position_size: Optional[float] = None  # Max % of portfolio
    max_leverage: float = 1.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    # Data
    data_frequency: str = "1D"  # Data bar frequency

    # Walk-forward
    walk_forward: bool = False
    train_period_days: int = 252
    test_period_days: int = 63
    step_days: int = 21

    # Performance
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02


class BacktestEngine:
    """
    Event-driven backtesting engine

    Simulates trading strategies on historical data with realistic
    execution, costs, and risk management.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()

        # Event system
        self.event_queue = EventQueue()
        self.event_dispatcher = EventDispatcher()

        # Portfolio
        self.portfolio = Portfolio(
            cash=Decimal(str(self.config.initial_capital)),
            initial_capital=Decimal(str(self.config.initial_capital))
        )

        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_data: Dict[str, Dict[str, Any]] = {}

        # Strategy
        self.strategy: Optional[Callable] = None

        # Tracking
        self.analyzer = BacktestAnalyzer(
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate
        )

        # State
        self.current_time: Optional[datetime] = None
        self.is_running = False

        # Orders
        self.pending_orders: List[OrderEvent] = []
        self.order_history: List[Dict[str, Any]] = []

        # Completed trades
        self.completed_trades: List[Trade] = []

        self.logger = logging.getLogger(__name__)

    def load_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume"
    ) -> None:
        """
        Load market data for a symbol

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data (or at minimum price and volume)
            price_col: Name of price column
            volume_col: Name of volume column
        """
        # Validate data
        if data.empty:
            raise ValueError(f"Empty data for symbol {symbol}")

        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")

        # Store data
        self.market_data[symbol] = data.copy()
        self.logger.info(f"Loaded {len(data)} bars for {symbol}")

    def set_strategy(self, strategy: Callable) -> None:
        """
        Set trading strategy

        Args:
            strategy: Strategy function that takes (engine, symbol, data) and returns signals
        """
        self.strategy = strategy

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResults:
        """
        Run backtest

        Args:
            start_date: Start date (None for data start)
            end_date: End date (None for data end)

        Returns:
            BacktestResults object
        """
        if not self.market_data:
            raise ValueError("No market data loaded")

        if self.strategy is None:
            raise ValueError("No strategy set")

        self.logger.info("Starting backtest...")
        self.is_running = True

        try:
            # Initialize
            self._initialize_backtest(start_date, end_date)

            # Main event loop
            self._run_event_loop()

            # Finalize
            results = self._finalize_backtest()

            self.logger.info("Backtest completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            raise

        finally:
            self.is_running = False

    def _initialize_backtest(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> None:
        """Initialize backtest state"""
        # Find common date range
        all_dates = []
        for data in self.market_data.values():
            all_dates.extend(data.index.tolist())

        all_dates = sorted(set(all_dates))

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]

        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        self.trading_dates = all_dates

        if not self.trading_dates:
            raise ValueError("No trading dates in specified range")

        self.logger.info(
            f"Backtest period: {self.trading_dates[0]} to {self.trading_dates[-1]} "
            f"({len(self.trading_dates)} days)"
        )

        # Register event handlers
        self._register_event_handlers()

        # Generate initial market data events
        self._generate_market_data_events()

    def _register_event_handlers(self) -> None:
        """Register event handlers"""
        self.event_dispatcher.register_handler(
            EventType.MARKET_DATA,
            self._handle_market_data
        )
        self.event_dispatcher.register_handler(
            EventType.SIGNAL,
            self._handle_signal
        )
        self.event_dispatcher.register_handler(
            EventType.ORDER,
            self._handle_order
        )
        self.event_dispatcher.register_handler(
            EventType.FILL,
            self._handle_fill
        )

    def _generate_market_data_events(self) -> None:
        """Generate market data events from historical data"""
        for date in self.trading_dates:
            for symbol, data in self.market_data.items():
                if date in data.index:
                    bar = data.loc[date]

                    event = MarketDataEvent(
                        timestamp=date,
                        symbol=symbol,
                        price=Decimal(str(bar.get('close', bar.iloc[-1]))),
                        volume=Decimal(str(bar.get('volume', 0))),
                        open=Decimal(str(bar.get('open', 0))) if 'open' in bar else None,
                        high=Decimal(str(bar.get('high', 0))) if 'high' in bar else None,
                        low=Decimal(str(bar.get('low', 0))) if 'low' in bar else None,
                        close=Decimal(str(bar.get('close', 0))) if 'close' in bar else None,
                        bid=Decimal(str(bar.get('bid', 0))) if 'bid' in bar else None,
                        ask=Decimal(str(bar.get('ask', 0))) if 'ask' in bar else None,
                    )

                    self.event_queue.put(event)

    def _run_event_loop(self) -> None:
        """Main event loop"""
        while not self.event_queue.empty():
            event = self.event_queue.get()

            if event is None:
                break

            # Update current time
            self.current_time = event.timestamp

            # Dispatch event
            self.event_dispatcher.dispatch(event)

    def _handle_market_data(self, event: MarketDataEvent) -> None:
        """Handle market data event"""
        symbol = event.symbol

        # Update current data
        self.current_data[symbol] = {
            'price': event.price,
            'volume': event.volume,
            'open': event.open,
            'high': event.high,
            'low': event.low,
            'close': event.close,
            'bid': event.bid,
            'ask': event.ask,
            'timestamp': event.timestamp
        }

        # Update positions
        if symbol in self.portfolio.positions:
            self.portfolio.positions[symbol].update_price(event.price, event.timestamp)

        # Update portfolio timestamp
        self.portfolio.timestamp = event.timestamp

        # Record equity
        self.analyzer.add_equity_point(event.timestamp, float(self.portfolio.total_value))

        # Call strategy
        if self.strategy:
            try:
                # Get historical data up to current point
                hist_data = self.market_data[symbol].loc[:event.timestamp]

                # Call strategy
                signal = self.strategy(self, symbol, hist_data)

                if signal:
                    # Create signal event
                    signal_event = SignalEvent(
                        timestamp=event.timestamp,
                        strategy_id=signal.get('strategy_id', 'default'),
                        symbol=symbol,
                        signal_type=signal.get('type', 'BUY'),
                        strength=signal.get('strength', 1.0),
                        target_price=signal.get('target_price'),
                        stop_loss=signal.get('stop_loss'),
                        take_profit=signal.get('take_profit'),
                        rationale=signal.get('rationale')
                    )

                    self.event_queue.put(signal_event)

            except Exception as e:
                self.logger.error(f"Strategy error for {symbol}: {e}", exc_info=True)

        # Process pending orders
        self._process_pending_orders(event.timestamp)

    def _handle_signal(self, event: SignalEvent) -> None:
        """Handle trading signal"""
        symbol = event.symbol

        # Calculate position size
        position_size = self._calculate_position_size(
            symbol,
            event.signal_type,
            event.strength
        )

        if position_size == 0:
            return

        # Create order
        order = OrderEvent(
            timestamp=event.timestamp,
            symbol=symbol,
            order_type="MARKET",
            side=event.signal_type,
            quantity=Decimal(str(position_size)),
            strategy_id=event.strategy_id
        )

        # Add to pending orders
        self.pending_orders.append(order)
        self.order_history.append({
            'order_id': order.order_id,
            'timestamp': event.timestamp,
            'symbol': symbol,
            'side': order.side,
            'quantity': position_size,
            'status': 'PENDING'
        })

        self.logger.debug(
            f"Signal received: {symbol} {event.signal_type} "
            f"(strength={event.strength:.2f}, size={position_size})"
        )

    def _handle_order(self, event: OrderEvent) -> None:
        """Handle order event"""
        # Orders are processed via pending orders queue
        pass

    def _handle_fill(self, event: FillEvent) -> None:
        """Handle order fill"""
        symbol = event.symbol
        side = event.side
        quantity = event.quantity
        fill_price = event.fill_price

        # Update portfolio
        if side == "BUY":
            self._execute_buy(symbol, quantity, fill_price, event)
        else:  # SELL
            self._execute_sell(symbol, quantity, fill_price, event)

        # Update order status
        for order in self.order_history:
            if order.get('order_id') == event.order_id:
                order['status'] = 'FILLED'
                order['fill_price'] = float(fill_price)
                order['commission'] = float(event.commission)
                order['slippage'] = float(event.slippage)
                break

        self.logger.debug(
            f"Order filled: {symbol} {side} {quantity} @ {fill_price} "
            f"(commission={event.commission}, slippage={event.slippage})"
        )

    def _process_pending_orders(self, timestamp: datetime) -> None:
        """Process pending orders"""
        if not self.pending_orders:
            return

        filled_orders = []

        for order in self.pending_orders:
            # Check if order should be filled based on execution model
            should_fill, fill_price = self._should_fill_order(order, timestamp)

            if should_fill:
                # Calculate commission and slippage
                commission = self._calculate_commission(
                    order.symbol,
                    order.quantity,
                    fill_price
                )

                slippage = self._calculate_slippage(
                    order.symbol,
                    order.side,
                    order.quantity,
                    fill_price
                )

                # Adjust fill price for slippage
                if order.side == "BUY":
                    actual_fill_price = fill_price + slippage
                else:
                    actual_fill_price = fill_price - slippage

                # Create fill event
                fill = FillEvent(
                    timestamp=timestamp,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=actual_fill_price,
                    commission=commission,
                    slippage=abs(slippage * order.quantity),
                    strategy_id=order.strategy_id
                )

                # Process fill
                self._handle_fill(fill)

                filled_orders.append(order)

        # Remove filled orders
        for order in filled_orders:
            self.pending_orders.remove(order)

    def _should_fill_order(
        self,
        order: OrderEvent,
        timestamp: datetime
    ) -> Tuple[bool, Decimal]:
        """
        Determine if order should be filled

        Returns:
            (should_fill, fill_price)
        """
        if self.config.execution_model == ExecutionModel.MARKET:
            # Fill at next bar open
            # For simplicity, we'll fill at current price
            if order.symbol in self.current_data:
                return True, self.current_data[order.symbol]['price']

        elif self.config.execution_model == ExecutionModel.CLOSE:
            # Fill at current bar close
            if order.symbol in self.current_data:
                close_price = self.current_data[order.symbol].get('close')
                if close_price:
                    return True, close_price
                return True, self.current_data[order.symbol]['price']

        elif self.config.execution_model == ExecutionModel.REALISTIC:
            # Realistic fill with potential delays
            # Check if enough bars have passed
            bars_since_order = (timestamp - order.timestamp).days

            if bars_since_order >= self.config.fill_delay_bars:
                if order.symbol in self.current_data:
                    return True, self.current_data[order.symbol]['price']

        return False, Decimal("0")

    def _calculate_position_size(
        self,
        symbol: str,
        side: str,
        strength: float
    ) -> float:
        """Calculate position size for trade"""
        # Simple position sizing: use a percentage of portfolio value
        portfolio_value = float(self.portfolio.total_value)
        current_price = float(self.current_data[symbol]['price'])

        # Base position size (e.g., 10% of portfolio)
        base_pct = 0.1 * strength  # Scale by signal strength

        # Apply max position size limit
        if self.config.max_position_size:
            base_pct = min(base_pct, self.config.max_position_size)

        position_value = portfolio_value * base_pct
        quantity = position_value / current_price

        # Round to reasonable lot size
        quantity = round(quantity, 2)

        return quantity

    def _calculate_commission(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal
    ) -> Decimal:
        """Calculate commission"""
        if self.config.commission_model == "fixed":
            return Decimal(str(self.config.commission_rate))

        elif self.config.commission_model == "percentage":
            trade_value = quantity * price
            return trade_value * Decimal(str(self.config.commission_rate))

        elif self.config.commission_model == "tiered":
            # Simplified tiered model
            trade_value = float(quantity * price)
            if trade_value < 10000:
                rate = 0.001
            elif trade_value < 100000:
                rate = 0.0008
            else:
                rate = 0.0005

            return Decimal(str(trade_value * rate))

        return Decimal("0")

    def _calculate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal
    ) -> Decimal:
        """Calculate slippage"""
        if self.config.slippage_model == "fixed":
            # Fixed basis points
            slippage_pct = Decimal(str(self.config.slippage_bps / 10000))
            return price * slippage_pct

        elif self.config.slippage_model == "volume":
            # Volume-based slippage
            if symbol in self.current_data:
                volume = float(self.current_data[symbol].get('volume', 1000000))
                participation = float(quantity) / volume

                # Higher participation = more slippage
                slippage_pct = Decimal(str(participation * 0.01))  # 1% per 100% participation
                return price * slippage_pct

        elif self.config.slippage_model == "spread":
            # Spread-based slippage
            if symbol in self.current_data:
                bid = self.current_data[symbol].get('bid')
                ask = self.current_data[symbol].get('ask')

                if bid and ask:
                    spread = ask - bid
                    # Pay half spread as slippage
                    return spread / Decimal("2")

        # Default to fixed
        slippage_pct = Decimal(str(self.config.slippage_bps / 10000))
        return price * slippage_pct

    def _execute_buy(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        fill_event: FillEvent
    ) -> None:
        """Execute buy order"""
        # Calculate total cost
        trade_value = quantity * price
        total_cost = trade_value + fill_event.commission + fill_event.slippage

        # Check if enough cash
        if total_cost > self.portfolio.cash:
            self.logger.warning(
                f"Insufficient cash for buy order: need {total_cost}, have {self.portfolio.cash}"
            )
            return

        # Update cash
        self.portfolio.cash -= total_cost

        # Update or create position
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]

            if pos.side == "LONG":
                # Add to long position
                new_quantity = pos.quantity + quantity
                new_avg_price = (
                    (pos.average_price * pos.quantity + price * quantity) / new_quantity
                )
                pos.quantity = new_quantity
                pos.average_price = new_avg_price
            else:
                # Closing short position
                if quantity >= pos.quantity:
                    # Close and possibly reverse
                    realized_pnl = (pos.average_price - price) * pos.quantity
                    pos.realized_pnl += realized_pnl

                    # Record trade
                    self._record_trade(
                        symbol=symbol,
                        entry_time=pos.entry_time,
                        exit_time=fill_event.timestamp,
                        side="SHORT",
                        quantity=pos.quantity,
                        entry_price=pos.average_price,
                        exit_price=price,
                        pnl=realized_pnl,
                        commission=fill_event.commission,
                        slippage=fill_event.slippage
                    )

                    remaining = quantity - pos.quantity

                    if remaining > 0:
                        # Reverse to long
                        pos.side = "LONG"
                        pos.quantity = remaining
                        pos.average_price = price
                        pos.entry_time = fill_event.timestamp
                    else:
                        # Position closed
                        del self.portfolio.positions[symbol]
                else:
                    # Partial close
                    pos.quantity -= quantity
                    realized_pnl = (pos.average_price - price) * quantity
                    pos.realized_pnl += realized_pnl
        else:
            # New long position
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price,
                side="LONG",
                entry_time=fill_event.timestamp,
                last_update=fill_event.timestamp
            )

    def _execute_sell(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        fill_event: FillEvent
    ) -> None:
        """Execute sell order"""
        # Calculate proceeds
        trade_value = quantity * price
        proceeds = trade_value - fill_event.commission - fill_event.slippage

        # Update cash
        self.portfolio.cash += proceeds

        # Update or create position
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]

            if pos.side == "SHORT":
                # Add to short position
                new_quantity = pos.quantity + quantity
                new_avg_price = (
                    (pos.average_price * pos.quantity + price * quantity) / new_quantity
                )
                pos.quantity = new_quantity
                pos.average_price = new_avg_price
            else:
                # Closing long position
                if quantity >= pos.quantity:
                    # Close and possibly reverse
                    realized_pnl = (price - pos.average_price) * pos.quantity
                    pos.realized_pnl += realized_pnl

                    # Record trade
                    self._record_trade(
                        symbol=symbol,
                        entry_time=pos.entry_time,
                        exit_time=fill_event.timestamp,
                        side="LONG",
                        quantity=pos.quantity,
                        entry_price=pos.average_price,
                        exit_price=price,
                        pnl=realized_pnl,
                        commission=fill_event.commission,
                        slippage=fill_event.slippage
                    )

                    remaining = quantity - pos.quantity

                    if remaining > 0:
                        # Reverse to short
                        pos.side = "SHORT"
                        pos.quantity = remaining
                        pos.average_price = price
                        pos.entry_time = fill_event.timestamp
                    else:
                        # Position closed
                        del self.portfolio.positions[symbol]
                else:
                    # Partial close
                    pos.quantity -= quantity
                    realized_pnl = (price - pos.average_price) * quantity
                    pos.realized_pnl += realized_pnl
        else:
            # New short position
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price,
                side="SHORT",
                entry_time=fill_event.timestamp,
                last_update=fill_event.timestamp
            )

    def _record_trade(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        commission: Decimal,
        slippage: Decimal
    ) -> None:
        """Record completed trade"""
        # Calculate return percentage
        if side == "LONG":
            return_pct = float((exit_price - entry_price) / entry_price)
        else:  # SHORT
            return_pct = float((entry_price - exit_price) / entry_price)

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            return_pct=return_pct,
            commission=commission,
            slippage=slippage
        )

        self.completed_trades.append(trade)
        self.analyzer.add_trade(trade)

    def _finalize_backtest(self) -> BacktestResults:
        """Finalize backtest and generate results"""
        # Close any remaining positions
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol in self.current_data:
                exit_price = self.current_data[symbol]['price']

                # Record final trade
                if position.side == "LONG":
                    pnl = (exit_price - position.average_price) * position.quantity
                else:
                    pnl = (position.average_price - exit_price) * position.quantity

                self._record_trade(
                    symbol=symbol,
                    entry_time=position.entry_time,
                    exit_time=self.current_time,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=position.average_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    commission=Decimal("0"),
                    slippage=Decimal("0")
                )

        # Add final equity point
        self.analyzer.add_equity_point(
            self.current_time,
            float(self.portfolio.total_value)
        )

        # Add order history
        for order in self.order_history:
            self.analyzer.add_order(order)

        # Analyze results
        results = self.analyzer.analyze()

        return results

    def run_walk_forward(
        self,
        optimize_func: Optional[Callable] = None
    ) -> List[BacktestResults]:
        """
        Run walk-forward analysis

        Args:
            optimize_func: Function to optimize strategy parameters on training data

        Returns:
            List of BacktestResults for each test period
        """
        if not self.config.walk_forward:
            raise ValueError("Walk-forward not enabled in config")

        # Implementation would split data into train/test periods
        # and run optimization on each training period
        # This is a placeholder for the full implementation

        self.logger.warning("Walk-forward analysis not fully implemented yet")
        return []

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return {
            'timestamp': self.current_time,
            'cash': float(self.portfolio.cash),
            'positions_value': float(self.portfolio.positions_value),
            'total_value': float(self.portfolio.total_value),
            'unrealized_pnl': float(self.portfolio.unrealized_pnl),
            'realized_pnl': float(self.portfolio.realized_pnl),
            'total_pnl': float(self.portfolio.total_pnl),
            'positions': {
                symbol: {
                    'quantity': float(pos.quantity),
                    'average_price': float(pos.average_price),
                    'current_price': float(pos.current_price),
                    'side': pos.side,
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'market_value': float(pos.market_value)
                }
                for symbol, pos in self.portfolio.positions.items()
            }
        }


def run_backtest_parallel(
    configs: List[Tuple[BacktestConfig, Dict[str, pd.DataFrame], Callable]],
    max_workers: Optional[int] = None
) -> List[BacktestResults]:
    """
    Run multiple backtests in parallel

    Args:
        configs: List of (config, market_data, strategy) tuples
        max_workers: Maximum parallel workers

    Returns:
        List of BacktestResults
    """
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, (config, market_data, strategy) in enumerate(configs):
            future = executor.submit(_run_single_backtest, config, market_data, strategy)
            futures[future] = i

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Backtest {futures[future]} completed")
            except Exception as e:
                logger.error(f"Backtest {futures[future]} failed: {e}")

    return results


def _run_single_backtest(
    config: BacktestConfig,
    market_data: Dict[str, pd.DataFrame],
    strategy: Callable
) -> BacktestResults:
    """Helper function to run a single backtest (for parallel execution)"""
    engine = BacktestEngine(config)

    for symbol, data in market_data.items():
        engine.load_data(symbol, data)

    engine.set_strategy(strategy)

    return engine.run()
