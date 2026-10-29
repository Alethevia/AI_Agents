"""
Market data routes.

This module provides endpoints for market data including ticker information,
candlestick data, order books, and trade history.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...middleware.auth import get_optional_user
from ...models.api_models import (
    MarketDataRequest,
    MarketDataResponse,
    Candlestick,
    TickerResponse,
    OrderBookResponse,
    OrderBookLevel,
    TimeInterval,
)


router = APIRouter(prefix="/market-data", tags=["Market Data"])


# ============================================================================
# Ticker Endpoints
# ============================================================================

@router.get("/ticker/{symbol}", response_model=TickerResponse)
async def get_ticker(
    symbol: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get ticker information for a symbol.

    Args:
        symbol: Trading symbol
        current_user: Optional authenticated user

    Returns:
        Ticker information
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbol = symbol.upper()

    # Mock ticker data
    if symbol == "AAPL":
        return TickerResponse(
            symbol=symbol,
            last_price=Decimal("155.50"),
            bid_price=Decimal("155.48"),
            ask_price=Decimal("155.52"),
            bid_size=Decimal("1000"),
            ask_size=Decimal("800"),
            volume_24h=Decimal("75000000"),
            price_change_24h=Decimal("2.50"),
            price_change_percent_24h=Decimal("1.63"),
            high_24h=Decimal("157.00"),
            low_24h=Decimal("153.00"),
            timestamp=datetime.utcnow(),
        )
    elif symbol == "GOOGL":
        return TickerResponse(
            symbol=symbol,
            last_price=Decimal("2850.00"),
            bid_price=Decimal("2849.50"),
            ask_price=Decimal("2850.50"),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            volume_24h=Decimal("2500000"),
            price_change_24h=Decimal("50.00"),
            price_change_percent_24h=Decimal("1.79"),
            high_24h=Decimal("2875.00"),
            low_24h=Decimal("2800.00"),
            timestamp=datetime.utcnow(),
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Ticker data not available for {symbol}"
    )


@router.get("/ticker", response_model=List[TickerResponse])
async def get_all_tickers(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get ticker information for multiple symbols.

    Args:
        symbols: Optional comma-separated list of symbols
        current_user: Optional authenticated user

    Returns:
        List of ticker information
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    tickers = []

    for symbol in symbol_list:
        try:
            ticker = await get_ticker(symbol, current_user)
            tickers.append(ticker)
        except HTTPException:
            # Skip symbols that don't have data
            continue

    return tickers


# ============================================================================
# Candlestick Data Endpoints
# ============================================================================

@router.get("/candles/{symbol}", response_model=MarketDataResponse)
async def get_candlestick_data(
    symbol: str,
    interval: TimeInterval = Query(..., description="Time interval"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of candles"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get candlestick (OHLCV) data for a symbol.

    Args:
        symbol: Trading symbol
        interval: Time interval for candles
        start_time: Optional start time
        end_time: Optional end time
        limit: Maximum number of candles
        current_user: Optional authenticated user

    Returns:
        Candlestick data
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbol = symbol.upper()

    # Default time range
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        # Calculate start time based on interval
        interval_seconds = {
            TimeInterval.ONE_MIN: 60,
            TimeInterval.FIVE_MIN: 300,
            TimeInterval.FIFTEEN_MIN: 900,
            TimeInterval.THIRTY_MIN: 1800,
            TimeInterval.ONE_HOUR: 3600,
            TimeInterval.FOUR_HOUR: 14400,
            TimeInterval.ONE_DAY: 86400,
        }
        seconds = interval_seconds.get(interval, 3600) * limit
        start_time = end_time - timedelta(seconds=seconds)

    # Generate mock candlestick data
    candles = []
    current_time = start_time
    base_price = Decimal("155.00") if symbol == "AAPL" else Decimal("2800.00")

    interval_delta = {
        TimeInterval.ONE_MIN: timedelta(minutes=1),
        TimeInterval.FIVE_MIN: timedelta(minutes=5),
        TimeInterval.FIFTEEN_MIN: timedelta(minutes=15),
        TimeInterval.THIRTY_MIN: timedelta(minutes=30),
        TimeInterval.ONE_HOUR: timedelta(hours=1),
        TimeInterval.FOUR_HOUR: timedelta(hours=4),
        TimeInterval.ONE_DAY: timedelta(days=1),
    }

    delta = interval_delta.get(interval, timedelta(hours=1))

    while current_time <= end_time and len(candles) < limit:
        # Generate mock OHLCV data
        open_price = base_price
        high_price = base_price * Decimal("1.01")
        low_price = base_price * Decimal("0.99")
        close_price = base_price * Decimal("1.005")
        volume = Decimal("1000000")

        candles.append(
            Candlestick(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )
        )

        current_time += delta
        base_price = close_price  # Next candle starts at previous close

    return MarketDataResponse(
        symbol=symbol,
        interval=interval,
        data=candles,
    )


# ============================================================================
# Order Book Endpoints
# ============================================================================

@router.get("/orderbook/{symbol}", response_model=OrderBookResponse)
async def get_order_book(
    symbol: str,
    depth: int = Query(20, ge=1, le=100, description="Order book depth"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get order book (depth) for a symbol.

    Args:
        symbol: Trading symbol
        depth: Number of levels to return
        current_user: Optional authenticated user

    Returns:
        Order book data
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbol = symbol.upper()

    # Generate mock order book
    base_price = Decimal("155.00") if symbol == "AAPL" else Decimal("2800.00")

    bids = []
    asks = []

    for i in range(depth):
        # Bids (decreasing price)
        bid_price = base_price - Decimal(str(i * 0.01))
        bid_quantity = Decimal(str(1000 - i * 10))
        bids.append(OrderBookLevel(price=bid_price, quantity=bid_quantity))

        # Asks (increasing price)
        ask_price = base_price + Decimal(str(i * 0.01))
        ask_quantity = Decimal(str(800 - i * 8))
        asks.append(OrderBookLevel(price=ask_price, quantity=ask_quantity))

    return OrderBookResponse(
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=datetime.utcnow(),
    )


# ============================================================================
# Trade History Endpoints
# ============================================================================

@router.get("/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get recent trades for a symbol.

    Args:
        symbol: Trading symbol
        limit: Maximum number of trades
        current_user: Optional authenticated user

    Returns:
        List of recent trades
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbol = symbol.upper()
    base_price = Decimal("155.00") if symbol == "AAPL" else Decimal("2800.00")

    trades = []
    current_time = datetime.utcnow()

    for i in range(min(limit, 50)):
        trade_time = current_time - timedelta(seconds=i * 5)
        price = base_price + Decimal(str((i % 10 - 5) * 0.01))
        quantity = Decimal(str(100 + i * 10))
        side = "BUY" if i % 2 == 0 else "SELL"

        trades.append({
            "trade_id": f"trade_{i}",
            "symbol": symbol,
            "price": str(price),
            "quantity": str(quantity),
            "side": side,
            "timestamp": trade_time.isoformat(),
        })

    return trades


# ============================================================================
# Market Statistics Endpoints
# ============================================================================

@router.get("/stats/{symbol}")
async def get_market_statistics(
    symbol: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get market statistics for a symbol.

    Args:
        symbol: Trading symbol
        current_user: Optional authenticated user

    Returns:
        Market statistics
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbol = symbol.upper()

    return {
        "symbol": symbol,
        "volume_24h": "75000000",
        "volume_7d": "450000000",
        "volume_30d": "1800000000",
        "price_change_24h": "2.50",
        "price_change_7d": "8.75",
        "price_change_30d": "15.25",
        "price_change_percent_24h": "1.63",
        "price_change_percent_7d": "5.96",
        "price_change_percent_30d": "10.89",
        "high_24h": "157.00",
        "low_24h": "153.00",
        "high_7d": "159.00",
        "low_7d": "148.00",
        "high_30d": "165.00",
        "low_30d": "140.00",
        "vwap_24h": "155.25",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Symbol Information Endpoints
# ============================================================================

@router.get("/symbols")
async def list_symbols(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    current_user: dict = Depends(get_optional_user)
):
    """
    List available trading symbols.

    Args:
        exchange: Optional exchange filter
        current_user: Optional authenticated user

    Returns:
        List of available symbols
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbols = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "type": "stock",
            "currency": "USD",
            "tradable": True,
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "exchange": "NASDAQ",
            "type": "stock",
            "currency": "USD",
            "tradable": True,
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "exchange": "NASDAQ",
            "type": "stock",
            "currency": "USD",
            "tradable": True,
        },
        {
            "symbol": "TSLA",
            "name": "Tesla, Inc.",
            "exchange": "NASDAQ",
            "type": "stock",
            "currency": "USD",
            "tradable": True,
        },
    ]

    # Filter by exchange if provided
    if exchange:
        symbols = [s for s in symbols if s["exchange"].upper() == exchange.upper()]

    return symbols


@router.get("/symbols/{symbol}/info")
async def get_symbol_info(
    symbol: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get detailed information about a symbol.

    Args:
        symbol: Trading symbol
        current_user: Optional authenticated user

    Returns:
        Symbol information
    """
    # TODO: Integration with actual market data provider
    # This is a mock implementation

    symbol = symbol.upper()

    symbol_info = {
        "AAPL": {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "type": "stock",
            "currency": "USD",
            "tradable": True,
            "min_order_size": "1",
            "max_order_size": "1000000",
            "price_precision": 2,
            "quantity_precision": 0,
            "trading_hours": {
                "timezone": "America/New_York",
                "open": "09:30:00",
                "close": "16:00:00",
            },
        },
        "GOOGL": {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "exchange": "NASDAQ",
            "type": "stock",
            "currency": "USD",
            "tradable": True,
            "min_order_size": "1",
            "max_order_size": "1000000",
            "price_precision": 2,
            "quantity_precision": 0,
            "trading_hours": {
                "timezone": "America/New_York",
                "open": "09:30:00",
                "close": "16:00:00",
            },
        },
    }

    if symbol in symbol_info:
        return symbol_info[symbol]

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Symbol {symbol} not found"
    )
