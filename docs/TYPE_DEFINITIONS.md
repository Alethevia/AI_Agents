# Trading System Type Definitions

## Overview

Comprehensive type definition files for the trading system, providing robust data structures for market data, orders, positions, portfolios, and trading signals. All classes use Pydantic BaseModel for automatic validation, serialization, and type safety.

## Files Created

### 1. `/src/core/types/market_data.py` (23KB, 600+ lines)

**Market data structures for real-time and historical data.**

#### Classes:
- **`Bar`** - OHLCV bar data with timestamp and symbol
  - Fields: symbol, timestamp, open, high, low, close, volume, vwap, trade_count
  - Validation: OHLC relationships (high >= open/close, low <= open/close)
  - Methods: `to_dict()`, `from_dict()`

- **`Tick`** - Real-time price tick data
  - Fields: symbol, timestamp, price, size, bid, ask, bid_size, ask_size, exchange, conditions
  - Validation: bid <= ask
  - Methods: `spread()`, `mid_price()`, `to_dict()`, `from_dict()`

- **`Quote`** - Bid/ask quote data
  - Fields: symbol, timestamp, bid_price, bid_size, ask_price, ask_size, exchange
  - Validation: bid_price <= ask_price
  - Methods: `spread()`, `mid_price()`, `spread_bps()`, `to_dict()`, `from_dict()`

- **`Trade`** - Executed trade data
  - Fields: symbol, timestamp, trade_id, price, size, side, exchange, conditions
  - Enums: `TradeSide` (BUY, SELL)
  - Methods: `notional_value()`, `to_dict()`, `from_dict()`

- **`OrderBook`** - Full order book with bids and asks
  - Fields: symbol, timestamp, bids, asks, exchange, sequence
  - Validation: bid levels sorted descending, ask levels sorted ascending, no crossed book
  - Methods: `best_bid()`, `best_ask()`, `spread()`, `mid_price()`, `total_bid_volume()`, `total_ask_volume()`, `imbalance()`, `to_dict()`, `from_dict()`

- **`OrderBookLevel`** - Single price level in order book
  - Fields: price, size, order_count
  - Methods: `to_dict()`, `from_dict()`

- **`MarketDepth`** - Market depth aggregated by price levels
  - Fields: symbol, timestamp, bid_prices, bid_sizes, ask_prices, ask_sizes
  - Validation: prices sorted correctly, sizes non-decreasing (cumulative)
  - Methods: `best_bid()`, `best_ask()`, `spread()`, `mid_price()`, `to_dict()`, `from_dict()`

---

### 2. `/src/core/types/orders.py` (15KB, 400+ lines)

**Order management and execution tracking.**

#### Enums:
- **`OrderSide`** - BUY, SELL
- **`OrderType`** - MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP
- **`OrderStatus`** - PENDING, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED
- **`TimeInForce`** - DAY, GTC, IOC, FOK

#### Classes:
- **`Order`** - Complete order representation
  - Fields: order_id, client_order_id, broker_order_id, symbol, side, order_type, quantity, limit_price, stop_price, trailing_amount, trailing_percent, time_in_force, status, filled_quantity, remaining_quantity, average_fill_price, timestamps (created_at, submitted_at, updated_at, filled_at, cancelled_at), commission, notes, error_message
  - Validation:
    - Price parameters required based on order type
    - Filled quantity <= order quantity
    - Status consistency (FILLED orders fully filled, PARTIAL orders partially filled)
  - Methods: `is_active()`, `is_terminal()`, `is_buy()`, `is_sell()`, `fill_rate()`, `notional_value()`, `filled_notional_value()`, `to_dict()`, `from_dict()`

- **`Fill`** - Individual order execution
  - Fields: fill_id, order_id, execution_id, symbol, side, quantity, price, timestamp, commission, fee, exchange, liquidity, notes
  - Methods: `notional_value()`, `total_cost()`, `net_proceeds()`, `to_dict()`, `from_dict()`

---

### 3. `/src/core/types/positions.py` (11KB, 350+ lines)

**Position tracking with P&L calculations.**

#### Enums:
- **`PositionSide`** - LONG, SHORT, FLAT

#### Classes:
- **`Position`** - Trading position tracker
  - Fields: symbol, side, quantity, entry_price, entry_value, entry_time, current_price, last_updated, unrealized_pnl, realized_pnl, total_commission, total_fees, notes
  - Validation:
    - FLAT positions have zero quantity
    - Non-FLAT positions have positive quantity
    - Positions with quantity must have entry_price
  - Methods: `is_long()`, `is_short()`, `is_flat()`, `market_value()`, `cost_basis()`, `calculate_unrealized_pnl()`, `update_market_price()`, `total_pnl()`, `total_costs()`, `net_pnl()`, `pnl_percent()`, `return_on_investment()`, `duration_seconds()`, `to_dict()`, `from_dict()`

- **`PositionUpdate`** - Position change event
  - Fields: symbol, side, quantity_delta, price, timestamp, commission, fee
  - Methods: `is_opening()`, `is_closing()`, `to_dict()`, `from_dict()`

---

### 4. `/src/core/types/portfolio.py` (19KB, 550+ lines)

**Portfolio management and performance tracking.**

#### Classes:
- **`Portfolio`** - Complete portfolio tracker
  - Fields: portfolio_id, name, positions (dict), cash, initial_capital, total_realized_pnl, total_unrealized_pnl, total_commission, total_fees, created_at, last_updated
  - Validation: cash >= 0
  - Methods:
    - Value: `total_position_value()`, `total_value()`, `equity()`, `net_liquidation_value()`
    - P&L: `total_pnl()`, `total_costs()`, `net_pnl()`, `total_return()`, `net_return()`
    - Risk: `leverage()`
    - Positions: `position_count()`, `long_position_count()`, `short_position_count()`, `get_position()`, `has_position()`, `add_position()`, `remove_position()`
    - Exposure: `long_exposure()`, `short_exposure()`, `net_exposure()`, `gross_exposure()`
    - Utilities: `recalculate_metrics()`, `update_prices()`, `to_dict()`, `from_dict()`

- **`PortfolioMetrics`** - Performance and risk metrics
  - Fields: total_return, net_return, annualized_return, max_drawdown, sharpe_ratio, sortino_ratio, volatility, total_trades, winning_trades, losing_trades, win_rate, avg_win, avg_loss, avg_trade, profit_factor, timestamp
  - Methods: `expectancy()`, `to_dict()`, `from_dict()`

- **`PortfolioSnapshot`** - Point-in-time portfolio state
  - Fields: timestamp, portfolio_id, cash, position_value, total_value, realized_pnl, unrealized_pnl, total_pnl, total_commission, total_fees, position_count, long_exposure, short_exposure, leverage, return_pct
  - Methods: `from_portfolio()` (class method), `net_exposure()`, `gross_exposure()`, `net_pnl()`, `to_dict()`, `from_dict()`

---

### 5. `/src/core/types/signals.py` (15KB, 450+ lines)

**Trading signal generation and analysis.**

#### Enums:
- **`SignalType`** - BUY, SELL, HOLD, REDUCE, INCREASE
- **`SignalStrength`** - WEAK, MODERATE, STRONG, VERY_STRONG

#### Classes:
- **`Signal`** - Trading signal with metadata
  - Fields: signal_id, symbol, signal_type, strength, confidence, source, strategy_id, features (dict), indicators (dict), target_price, stop_loss, take_profit, position_size, timestamp, expiry, market_price, market_regime, notes, metadata (dict)
  - Validation:
    - For BUY signals: stop_loss < market_price < take_profit
    - For SELL signals: take_profit < market_price < stop_loss
    - Expiry must be after timestamp
  - Methods: `is_buy()`, `is_sell()`, `is_hold()`, `is_expired()`, `is_strong()`, `risk_reward_ratio()`, `expected_return()`, `get_indicator()`, `get_feature()`, `time_to_expiry()`, `to_dict()`, `from_dict()`

- **`SignalCollection`** - Collection of signals with analysis
  - Fields: signals (list), timestamp
  - Methods: `add_signal()`, `filter_by_symbol()`, `filter_by_type()`, `filter_by_strength()`, `filter_by_confidence()`, `filter_active()`, `get_buy_signals()`, `get_sell_signals()`, `get_strong_signals()`, `count()`, `average_confidence()`, `consensus_signal()`, `to_dict()`, `from_dict()`

---

## Key Features

### 1. **Pydantic BaseModel**
All classes inherit from Pydantic's BaseModel, providing:
- Automatic type validation
- JSON serialization/deserialization
- Schema generation
- IDE autocomplete support

### 2. **Decimal Precision**
All monetary and quantity values use Python's `Decimal` type for precise financial calculations, avoiding floating-point errors.

### 3. **Validation**
Comprehensive field validators and model validators ensure:
- Non-negative prices and quantities
- Logical relationships (e.g., high >= low in bars)
- Status consistency
- Price relationships in signals

### 4. **Serialization**
Every class includes:
- `to_dict()` - Convert to dictionary (ISO format for dates, strings for decimals)
- `from_dict()` - Create instance from dictionary

### 5. **Business Logic**
Rich methods for calculations:
- P&L calculations
- Risk metrics (leverage, exposure)
- Price calculations (spreads, mid prices)
- Performance metrics

### 6. **Type Safety**
Enums for categorical fields ensure type safety:
- OrderSide, OrderType, OrderStatus, TimeInForce
- PositionSide
- SignalType, SignalStrength
- TradeSide

---

## Statistics

- **Total Lines of Code**: ~2,200 lines
- **Total Classes**: 24 classes
- **Enums**: 8 enumerations
- **Files**: 5 type definition files

---

## Usage Example

See `/home/user/AI_Agents/examples/types_usage_example.py` for comprehensive usage examples.

```python
from decimal import Decimal
from datetime import datetime
from src.core.types.orders import Order, OrderSide, OrderType

# Create a limit buy order
order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=Decimal("100"),
    limit_price=Decimal("150.00"),
)

# Serialize to dictionary
order_dict = order.to_dict()

# Deserialize from dictionary
restored_order = Order.from_dict(order_dict)
```

---

## Dependencies

- **pydantic**: For BaseModel, Field, validators (v2.x recommended)
- **Python 3.8+**: For type hints and datetime support

---

## Integration

These types integrate with:
- Data feeds (market data types)
- Broker APIs (order types)
- Risk management (position and portfolio types)
- Strategy engines (signal types)
- Database storage (via to_dict/from_dict serialization)
- Event systems (all types are immutable by default)

---

## Best Practices

1. **Always use Decimal for financial values** - Never use float
2. **Validate before creating** - Let Pydantic validators catch errors early
3. **Use enums for categorical data** - Ensures type safety
4. **Serialize for storage** - Use to_dict() for database/file storage
5. **Update timestamps** - Keep last_updated fields current
6. **Check terminal states** - Use is_terminal(), is_active() methods
7. **Calculate metrics regularly** - Use recalculate_metrics() after updates

---

## Testing

All files have been validated for:
- ✓ Valid Python syntax
- ✓ Proper Pydantic structure
- ✓ Import compatibility
- ✓ Type consistency

---

## Future Enhancements

Potential additions:
- Risk metrics (VaR, CVaR)
- Advanced order types (iceberg, VWAP)
- Multi-asset portfolio support
- Performance attribution
- Transaction cost analysis
- Slippage modeling

---

## File Locations

```
/home/user/AI_Agents/src/core/types/
├── __init__.py                 # Module exports
├── market_data.py              # Market data types
├── orders.py                   # Order types
├── positions.py                # Position types
├── portfolio.py                # Portfolio types
└── signals.py                  # Signal types

/home/user/AI_Agents/examples/
└── types_usage_example.py      # Usage examples

/home/user/AI_Agents/docs/
└── TYPE_DEFINITIONS.md         # This file
```

---

## License

Part of the AI_Agents trading system.

---

*Last Updated: 2025-10-30*
