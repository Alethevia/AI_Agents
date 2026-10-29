from datetime import datetime, timedelta
import pandas as pd
from src.backtesting.engine.event_queue import EventQueue, MarketDataEvent, SignalEvent
from src.backtesting import BacktestEngine, BacktestConfig, ExecutionModel

def test_future_market_data_cannot_overtake_signal():
    today = datetime(2024, 1, 1)
    queue = EventQueue()
    queue.put(MarketDataEvent(timestamp=today + timedelta(days=1), symbol='A', price=10))
    queue.put(SignalEvent(timestamp=today, symbol='A', signal_type='BUY'))
    assert queue.get().timestamp == today

def test_signals_fill_on_later_bars_and_sell_existing_shares():
    dates = pd.date_range('2024-01-01', periods=10)
    data = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=100000.), index=dates)
    engine = BacktestEngine(BacktestConfig(execution_model=ExecutionModel.REALISTIC))
    engine.load_data('A', data)
    def strategy(engine, symbol, history):
        if len(history) == 2: return {'type': 'BUY', 'strength': 1.0}
        if len(history) == 6: return {'type': 'SELL', 'strength': 1.0}
    engine.set_strategy(strategy)
    result = engine.run()
    assert result.filled_orders == 2
    assert result.final_capital <= result.initial_capital
    assert result.trade_analytics.total_trades >= 1
    assert not engine.portfolio.positions or all(p.quantity == 0 for p in engine.portfolio.positions.values())


def test_insufficient_cash_is_rejected_not_counted_as_fill():
    from decimal import Decimal
    from src.backtesting.engine.event_queue import FillEvent
    engine = BacktestEngine(BacktestConfig())
    engine.portfolio.cash = Decimal('10')
    engine.order_history.append({'order_id': 'too-large', 'status': 'PENDING'})
    event = FillEvent(timestamp=datetime(2024, 1, 1), order_id='too-large',
                      symbol='A', side='BUY', quantity=Decimal('2'),
                      fill_price=Decimal('100'), commission=Decimal('1'))
    engine._handle_fill(event)
    assert engine.portfolio.cash == Decimal('10')
    assert not engine.portfolio.positions
    assert engine.order_history[0]['status'] == 'REJECTED'
