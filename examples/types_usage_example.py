"""Example usage of trading system type definitions.

This example demonstrates how to use the comprehensive type definitions
for the trading system.

Note: Requires pydantic to be installed: pip install pydantic
"""

from datetime import datetime
from decimal import Decimal

# Import type definitions
from src.core.types.market_data import Bar, Tick, Quote, Trade, OrderBook, OrderBookLevel
from src.core.types.orders import Order, OrderSide, OrderType, OrderStatus, TimeInForce, Fill
from src.core.types.positions import Position, PositionSide
from src.core.types.portfolio import Portfolio, PortfolioSnapshot
from src.core.types.signals import Signal, SignalType, SignalStrength


def example_market_data():
    """Example: Creating market data objects."""
    print("=== Market Data Examples ===\n")

    # Create a Bar
    bar = Bar(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        open=Decimal("150.00"),
        high=Decimal("152.50"),
        low=Decimal("149.50"),
        close=Decimal("151.75"),
        volume=Decimal("1000000"),
    )
    print(f"Bar: {bar.symbol} - Close: ${bar.close}, Volume: {bar.volume}")

    # Create a Tick
    tick = Tick(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        price=Decimal("151.80"),
        size=Decimal("100"),
        bid=Decimal("151.75"),
        ask=Decimal("151.85"),
    )
    print(f"Tick: {tick.symbol} - Price: ${tick.price}, Spread: ${tick.spread()}")

    # Create a Quote
    quote = Quote(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        bid_price=Decimal("151.75"),
        bid_size=Decimal("500"),
        ask_price=Decimal("151.85"),
        ask_size=Decimal("300"),
    )
    print(f"Quote: {quote.symbol} - Mid: ${quote.mid_price()}, Spread: {quote.spread_bps()} bps\n")


def example_orders():
    """Example: Creating and managing orders."""
    print("=== Order Examples ===\n")

    # Create a limit order
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("150.00"),
        time_in_force=TimeInForce.GTC,
    )
    print(f"Order: {order.side.value} {order.quantity} {order.symbol} @ ${order.limit_price}")
    print(f"Status: {order.status.value}, Fill Rate: {order.fill_rate()}%")

    # Create a fill
    fill = Fill(
        order_id=order.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("50"),
        price=Decimal("150.00"),
        commission=Decimal("1.00"),
    )
    print(f"Fill: {fill.quantity} shares @ ${fill.price}, Notional: ${fill.notional_value()}\n")


def example_positions():
    """Example: Creating and tracking positions."""
    print("=== Position Examples ===\n")

    # Create a position
    position = Position(
        symbol="AAPL",
        side=PositionSide.LONG,
        quantity=Decimal("100"),
        entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
    )

    # Update market price
    position.update_market_price(Decimal("155.00"))

    print(f"Position: {position.side.value} {position.quantity} {position.symbol}")
    print(f"Entry: ${position.entry_price}, Current: ${position.current_price}")
    print(f"Unrealized P&L: ${position.unrealized_pnl}")
    print(f"P&L %: {position.pnl_percent()}%\n")


def example_portfolio():
    """Example: Creating and managing a portfolio."""
    print("=== Portfolio Examples ===\n")

    # Create a portfolio
    portfolio = Portfolio(
        portfolio_id="demo_portfolio",
        name="Demo Trading Portfolio",
        cash=Decimal("100000.00"),
        initial_capital=Decimal("100000.00"),
    )

    # Add a position
    position = Position(
        symbol="AAPL",
        side=PositionSide.LONG,
        quantity=Decimal("100"),
        entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
        entry_time=datetime.utcnow(),
    )
    position.update_market_price(Decimal("155.00"))
    portfolio.add_position(position)

    print(f"Portfolio: {portfolio.name}")
    print(f"Cash: ${portfolio.cash}")
    print(f"Position Value: ${portfolio.total_position_value()}")
    print(f"Total Value: ${portfolio.total_value()}")
    print(f"Total Return: {portfolio.total_return()}%")
    print(f"Position Count: {portfolio.position_count()}")

    # Create a snapshot
    snapshot = PortfolioSnapshot.from_portfolio(portfolio)
    print(f"\nSnapshot taken at {snapshot.timestamp}")
    print(f"Total Value: ${snapshot.total_value}\n")


def example_signals():
    """Example: Creating trading signals."""
    print("=== Signal Examples ===\n")

    # Create a buy signal
    signal = Signal(
        signal_id="signal_001",
        symbol="AAPL",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("0.85"),
        source="momentum_strategy",
        market_price=Decimal("150.00"),
        target_price=Decimal("160.00"),
        stop_loss=Decimal("145.00"),
        indicators={
            "rsi": Decimal("65.5"),
            "macd": Decimal("2.3"),
        },
        features={
            "trend": "bullish",
            "volatility": "low",
        },
    )

    print(f"Signal: {signal.signal_type.value} {signal.symbol}")
    print(f"Strength: {signal.strength.value}, Confidence: {signal.confidence}")
    print(f"Target: ${signal.target_price}, Stop Loss: ${signal.stop_loss}")
    print(f"Risk/Reward Ratio: {signal.risk_reward_ratio()}")
    print(f"Expected Return: {signal.expected_return()}%\n")


def example_serialization():
    """Example: Serializing and deserializing objects."""
    print("=== Serialization Examples ===\n")

    # Create an order
    original_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("150.00"),
    )

    # Serialize to dict
    order_dict = original_order.to_dict()
    print("Serialized to dict:")
    print(f"  Symbol: {order_dict['symbol']}")
    print(f"  Side: {order_dict['side']}")
    print(f"  Quantity: {order_dict['quantity']}")

    # Deserialize from dict
    restored_order = Order.from_dict(order_dict)
    print(f"\nRestored from dict: {restored_order.symbol} {restored_order.side.value}")
    print(f"Objects match: {original_order.order_id == restored_order.order_id}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Trading System Type Definitions - Usage Examples")
    print("=" * 60 + "\n")

    try:
        example_market_data()
        example_orders()
        example_positions()
        example_portfolio()
        example_signals()
        example_serialization()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
