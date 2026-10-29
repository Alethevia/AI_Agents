# Trading System Types - Quick Reference

## Import Patterns

```python
# Import all types
from src.core.types import *

# Import specific modules
from src.core.types.market_data import Bar, Tick, Quote, OrderBook
from src.core.types.orders import Order, OrderSide, OrderType, OrderStatus
from src.core.types.positions import Position, PositionSide
from src.core.types.portfolio import Portfolio, PortfolioSnapshot
from src.core.types.signals import Signal, SignalType, SignalStrength
```

## Common Type Patterns

### Creating Market Data

```python
from decimal import Decimal
from datetime import datetime

# Bar
bar = Bar(
    symbol="AAPL",
    timestamp=datetime.utcnow(),
    open=Decimal("150.00"),
    high=Decimal("152.00"),
    low=Decimal("149.00"),
    close=Decimal("151.00"),
    volume=Decimal("1000000")
)

# Tick
tick = Tick(
    symbol="AAPL",
    timestamp=datetime.utcnow(),
    price=Decimal("151.00"),
    size=Decimal("100"),
    bid=Decimal("150.95"),
    ask=Decimal("151.05")
)
```

### Creating Orders

```python
# Market order
order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("100")
)

# Limit order
order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=Decimal("100"),
    limit_price=Decimal("150.00"),
    time_in_force=TimeInForce.GTC
)

# Stop-limit order
order = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.STOP_LIMIT,
    quantity=Decimal("100"),
    stop_price=Decimal("145.00"),
    limit_price=Decimal("144.50")
)
```

### Managing Positions

```python
# Create position
position = Position(
    symbol="AAPL",
    side=PositionSide.LONG,
    quantity=Decimal("100"),
    entry_price=Decimal("150.00")
)

# Update market price
position.update_market_price(Decimal("155.00"))

# Check P&L
print(f"Unrealized P&L: {position.unrealized_pnl}")
print(f"P&L %: {position.pnl_percent()}")
print(f"ROI: {position.return_on_investment()}")
```

### Portfolio Management

```python
# Create portfolio
portfolio = Portfolio(
    portfolio_id="main",
    cash=Decimal("100000"),
    initial_capital=Decimal("100000")
)

# Add position
portfolio.add_position(position)

# Get metrics
print(f"Total Value: {portfolio.total_value()}")
print(f"Total Return: {portfolio.total_return()}%")
print(f"Leverage: {portfolio.leverage()}")
print(f"Net Exposure: {portfolio.net_exposure()}")

# Update prices
portfolio.update_prices({
    "AAPL": Decimal("155.00"),
    "GOOGL": Decimal("2800.00")
})

# Take snapshot
snapshot = PortfolioSnapshot.from_portfolio(portfolio)
```

### Creating Signals

```python
# Buy signal
signal = Signal(
    signal_id="sig_001",
    symbol="AAPL",
    signal_type=SignalType.BUY,
    strength=SignalStrength.STRONG,
    confidence=Decimal("0.85"),
    source="momentum_strategy",
    market_price=Decimal("150.00"),
    target_price=Decimal("160.00"),
    stop_loss=Decimal("145.00"),
    indicators={
        "rsi": Decimal("65"),
        "macd": Decimal("2.5")
    }
)

# Check signal properties
print(f"Risk/Reward: {signal.risk_reward_ratio()}")
print(f"Expected Return: {signal.expected_return()}%")
print(f"Is Strong: {signal.is_strong()}")
print(f"Is Buy: {signal.is_buy()}")
```

## Serialization

```python
# All types support to_dict/from_dict

# Serialize
order_dict = order.to_dict()
position_dict = position.to_dict()
portfolio_dict = portfolio.to_dict()

# Deserialize
order = Order.from_dict(order_dict)
position = Position.from_dict(position_dict)
portfolio = Portfolio.from_dict(portfolio_dict)
```

## Enums Reference

### OrderSide
- `OrderSide.BUY`
- `OrderSide.SELL`

### OrderType
- `OrderType.MARKET`
- `OrderType.LIMIT`
- `OrderType.STOP`
- `OrderType.STOP_LIMIT`
- `OrderType.TRAILING_STOP`

### OrderStatus
- `OrderStatus.PENDING`
- `OrderStatus.SUBMITTED`
- `OrderStatus.PARTIAL`
- `OrderStatus.FILLED`
- `OrderStatus.CANCELLED`
- `OrderStatus.REJECTED`

### TimeInForce
- `TimeInForce.DAY` - Day order
- `TimeInForce.GTC` - Good till cancelled
- `TimeInForce.IOC` - Immediate or cancel
- `TimeInForce.FOK` - Fill or kill

### PositionSide
- `PositionSide.LONG`
- `PositionSide.SHORT`
- `PositionSide.FLAT`

### SignalType
- `SignalType.BUY`
- `SignalType.SELL`
- `SignalType.HOLD`
- `SignalType.REDUCE`
- `SignalType.INCREASE`

### SignalStrength
- `SignalStrength.WEAK`
- `SignalStrength.MODERATE`
- `SignalStrength.STRONG`
- `SignalStrength.VERY_STRONG`

## Validation Notes

### Always Use Decimal for Financial Values
```python
# ✓ Correct
price = Decimal("150.00")
quantity = Decimal("100")

# ✗ Wrong - float has precision issues
price = 150.00  # DON'T DO THIS
```

### Symbol Format
```python
# Symbols are automatically uppercased
bar = Bar(symbol="aapl", ...)  # Stored as "AAPL"
```

### OHLC Validation
```python
# These will raise validation errors:
Bar(high=100, low=110, ...)  # high < low ✗
Bar(high=100, open=110, ...)  # high < open ✗
```

### Order Validation
```python
# LIMIT orders need limit_price
Order(order_type=OrderType.LIMIT, limit_price=None)  # ✗

# STOP orders need stop_price
Order(order_type=OrderType.STOP, stop_price=None)  # ✗

# Filled quantity can't exceed order quantity
order.filled_quantity = Decimal("150")  # ✗ if quantity = 100
```

### Position Validation
```python
# FLAT positions must have zero quantity
Position(side=PositionSide.FLAT, quantity=Decimal("100"))  # ✗

# Non-FLAT positions need positive quantity
Position(side=PositionSide.LONG, quantity=Decimal("0"))  # ✗
```

### Signal Validation
```python
# For BUY signals:
# stop_loss < market_price < take_profit
Signal(
    signal_type=SignalType.BUY,
    market_price=Decimal("150"),
    stop_loss=Decimal("145"),  # ✓ below market
    take_profit=Decimal("160")  # ✓ above market
)

# For SELL signals:
# take_profit < market_price < stop_loss
Signal(
    signal_type=SignalType.SELL,
    market_price=Decimal("150"),
    take_profit=Decimal("145"),  # ✓ below market
    stop_loss=Decimal("155")     # ✓ above market
)
```

## Useful Methods

### Bar Methods
- `bar.to_dict()` - Serialize to dict
- `Bar.from_dict(data)` - Deserialize from dict

### Tick Methods
- `tick.spread()` - Calculate bid-ask spread
- `tick.mid_price()` - Calculate mid price

### Quote Methods
- `quote.spread()` - Bid-ask spread
- `quote.mid_price()` - Mid price
- `quote.spread_bps()` - Spread in basis points

### Order Methods
- `order.is_active()` - Check if active
- `order.is_terminal()` - Check if terminal state
- `order.fill_rate()` - Fill percentage
- `order.notional_value()` - Calculate notional value

### Position Methods
- `position.market_value()` - Current market value
- `position.cost_basis()` - Cost basis
- `position.calculate_unrealized_pnl()` - Calculate P&L
- `position.update_market_price(price)` - Update price and P&L
- `position.pnl_percent()` - P&L as percentage
- `position.return_on_investment()` - ROI

### Portfolio Methods
- `portfolio.total_value()` - Total portfolio value
- `portfolio.total_pnl()` - Total P&L
- `portfolio.leverage()` - Leverage ratio
- `portfolio.long_exposure()` - Long exposure
- `portfolio.short_exposure()` - Short exposure
- `portfolio.net_exposure()` - Net exposure
- `portfolio.add_position(position)` - Add position
- `portfolio.update_prices(prices_dict)` - Update all prices

### Signal Methods
- `signal.is_buy()` - Check if buy signal
- `signal.is_sell()` - Check if sell signal
- `signal.is_strong()` - Check if strong signal
- `signal.is_expired()` - Check if expired
- `signal.risk_reward_ratio()` - R/R ratio
- `signal.expected_return()` - Expected return

## Best Practices

1. **Always use Decimal** - Never use float for financial values
2. **Validate early** - Let Pydantic catch errors at creation time
3. **Use enums** - Type-safe categorical values
4. **Serialize for storage** - Use to_dict() for persistence
5. **Update timestamps** - Keep timing information current
6. **Check terminal states** - Use helper methods (is_active, is_terminal)
7. **Recalculate metrics** - Call recalculate_metrics() after updates
8. **Handle None values** - Check optional fields before using
9. **Use type hints** - Leverage IDE autocomplete
10. **Read docstrings** - Comprehensive documentation in each class

## Common Pitfalls

### Don't Use Float
```python
# ✗ Wrong
price = 150.00
quantity = 100.5

# ✓ Correct
price = Decimal("150.00")
quantity = Decimal("100.5")
```

### Don't Forget to Update P&L
```python
# ✗ Wrong - P&L not updated
position.current_price = new_price

# ✓ Correct - Updates P&L automatically
position.update_market_price(new_price)
```

### Don't Mix Order Types and Parameters
```python
# ✗ Wrong - MARKET order with limit_price
Order(order_type=OrderType.MARKET, limit_price=Decimal("150"))

# ✓ Correct
Order(order_type=OrderType.MARKET, quantity=Decimal("100"))
```

### Don't Forget to Recalculate Portfolio Metrics
```python
# ✗ Wrong - metrics not updated
portfolio.positions["AAPL"].unrealized_pnl = new_pnl

# ✓ Correct
portfolio.positions["AAPL"].update_market_price(new_price)
portfolio.recalculate_metrics()
```

## Testing

```python
# Test OHLC validation
try:
    bar = Bar(high=100, low=110, ...)
except ValueError as e:
    print(f"Validation caught: {e}")

# Test order status transitions
order = Order(...)
assert order.is_active()
order.status = OrderStatus.FILLED
assert order.is_terminal()

# Test P&L calculations
position = Position(
    side=PositionSide.LONG,
    entry_price=Decimal("100"),
    current_price=Decimal("110"),
    quantity=Decimal("10")
)
assert position.unrealized_pnl == Decimal("100")  # (110-100)*10
```

---

For complete documentation, see `/home/user/AI_Agents/docs/TYPE_DEFINITIONS.md`

For usage examples, see `/home/user/AI_Agents/examples/types_usage_example.py`
