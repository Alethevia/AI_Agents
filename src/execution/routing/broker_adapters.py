"""
Broker API Adapters

This module provides adapters for various broker APIs including:
- Generic broker interface
- Alpaca adapter
- Interactive Brokers adapter
- Connection management
- Error handling and retry logic
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal
from abc import ABC, abstractmethod
from enum import Enum
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


logger = logging.getLogger(__name__)


class BrokerStatus(Enum):
    """Broker connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OrderStatus(Enum):
    """Broker order status"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_id: str
    api_key: str
    api_secret: str
    base_url: str
    paper_trading: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BrokerOrder:
    """Standardized broker order representation"""
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.quantity = Decimal(str(self.quantity))
        self.filled_quantity = Decimal(str(self.filled_quantity))
        if self.limit_price:
            self.limit_price = Decimal(str(self.limit_price))
        if self.stop_price:
            self.stop_price = Decimal(str(self.stop_price))
        if self.avg_fill_price:
            self.avg_fill_price = Decimal(str(self.avg_fill_price))


class BrokerError(Exception):
    """Base exception for broker errors"""
    pass


class BrokerConnectionError(BrokerError):
    """Broker connection error"""
    pass


class BrokerAuthenticationError(BrokerError):
    """Broker authentication error"""
    pass


class BrokerOrderError(BrokerError):
    """Broker order error"""
    pass


class BrokerRateLimitError(BrokerError):
    """Broker rate limit exceeded"""
    pass


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters

    Defines the interface that all broker adapters must implement
    """

    def __init__(self, config: BrokerConfig):
        """
        Initialize broker adapter

        Args:
            config: Broker configuration
        """
        self.config = config
        self.status = BrokerStatus.DISCONNECTED
        self._session: Optional[aiohttp.ClientSession] = None
        self._callbacks: List[Callable] = []

        logger.info(f"Initialized {config.broker_id} adapter")

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to broker

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit an order to the broker

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            order_type: Order type
            quantity: Order quantity
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            client_order_id: Client-side order ID
            **kwargs: Additional broker-specific parameters

        Returns:
            Broker order ID
        """
        pass

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an order

        Args:
            broker_order_id: Broker order ID

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        """
        Get order details

        Args:
            broker_order_id: Broker order ID

        Returns:
            BrokerOrder object
        """
        pass

    @abstractmethod
    async def get_quote(
        self,
        symbol: str,
        side: str,
        quantity: Decimal
    ) -> Dict[str, Any]:
        """
        Get quote for a symbol

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            Quote data with bid, ask, sizes, etc.
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information

        Returns:
            Account data (balance, buying power, etc.)
        """
        pass

    def register_callback(self, callback: Callable) -> None:
        """Register a callback for broker events"""
        self._callbacks.append(callback)

    async def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger registered callbacks"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in broker callback: {e}")

    def is_connected(self) -> bool:
        """Check if broker is connected"""
        return self.status == BrokerStatus.CONNECTED


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Alpaca broker adapter

    Supports both paper and live trading through Alpaca API
    """

    def __init__(self, config: BrokerConfig):
        """Initialize Alpaca adapter"""
        super().__init__(config)

        # Alpaca endpoints
        if config.paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = config.base_url or "https://api.alpaca.markets"

        self.data_url = "https://data.alpaca.markets"

    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self.status = BrokerStatus.CONNECTING

            # Create session
            headers = {
                "APCA-API-KEY-ID": self.config.api_key,
                "APCA-API-SECRET-KEY": self.config.api_secret
            }

            self._session = aiohttp.ClientSession(headers=headers)

            # Test connection by getting account
            await self.get_account_info()

            self.status = BrokerStatus.CONNECTED
            logger.info(f"Connected to Alpaca ({self.config.broker_id})")

            return True

        except Exception as e:
            self.status = BrokerStatus.ERROR
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise BrokerConnectionError(f"Alpaca connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca"""
        if self._session:
            await self._session.close()
            self._session = None

        self.status = BrokerStatus.DISCONNECTED
        logger.info(f"Disconnected from Alpaca ({self.config.broker_id})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((BrokerConnectionError, BrokerRateLimitError))
    )
    async def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit order to Alpaca"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Alpaca")

        # Build order request
        order_data = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower()
        }

        if limit_price:
            order_data["limit_price"] = str(limit_price)

        if stop_price:
            order_data["stop_price"] = str(stop_price)

        if client_order_id:
            order_data["client_order_id"] = client_order_id

        try:
            async with self._session.post(
                f"{self.base_url}/v2/orders",
                json=order_data,
                timeout=self.config.timeout_seconds
            ) as response:
                if response.status == 429:
                    raise BrokerRateLimitError("Rate limit exceeded")
                elif response.status == 403:
                    raise BrokerAuthenticationError("Authentication failed")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise BrokerOrderError(f"Order submission failed: {error_text}")

                result = await response.json()
                broker_order_id = result["id"]

                logger.info(
                    f"Submitted order to Alpaca: {broker_order_id} "
                    f"({side} {quantity} {symbol})"
                )

                # Trigger callbacks
                await self._trigger_callbacks("order_submitted", result)

                return broker_order_id

        except aiohttp.ClientError as e:
            raise BrokerConnectionError(f"Connection error: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order on Alpaca"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Alpaca")

        try:
            async with self._session.delete(
                f"{self.base_url}/v2/orders/{broker_order_id}",
                timeout=self.config.timeout_seconds
            ) as response:
                if response.status == 204:
                    logger.info(f"Cancelled order on Alpaca: {broker_order_id}")
                    await self._trigger_callbacks("order_cancelled", {"order_id": broker_order_id})
                    return True
                elif response.status == 404:
                    raise BrokerOrderError(f"Order not found: {broker_order_id}")
                else:
                    error_text = await response.text()
                    raise BrokerOrderError(f"Cancellation failed: {error_text}")

        except aiohttp.ClientError as e:
            raise BrokerConnectionError(f"Connection error: {e}")

    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        """Get order from Alpaca"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Alpaca")

        try:
            async with self._session.get(
                f"{self.base_url}/v2/orders/{broker_order_id}",
                timeout=self.config.timeout_seconds
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise BrokerOrderError(f"Failed to get order: {error_text}")

                data = await response.json()
                return self._parse_alpaca_order(data)

        except aiohttp.ClientError as e:
            raise BrokerConnectionError(f"Connection error: {e}")

    async def get_quote(
        self,
        symbol: str,
        side: str,
        quantity: Decimal
    ) -> Dict[str, Any]:
        """Get quote from Alpaca"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Alpaca")

        try:
            # Get latest quote
            async with self._session.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
                timeout=self.config.timeout_seconds
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise BrokerError(f"Failed to get quote: {error_text}")

                data = await response.json()
                quote = data["quote"]

                return {
                    "symbol": symbol,
                    "bid": Decimal(str(quote["bp"])),
                    "ask": Decimal(str(quote["ap"])),
                    "bid_size": Decimal(str(quote["bs"])),
                    "ask_size": Decimal(str(quote["as"])),
                    "timestamp": quote["t"],
                    "fee_rate": Decimal("0.0")  # Alpaca has commission-free trading
                }

        except aiohttp.ClientError as e:
            raise BrokerConnectionError(f"Connection error: {e}")

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Alpaca"""
        if not self._session:
            raise BrokerConnectionError("Session not initialized")

        try:
            async with self._session.get(
                f"{self.base_url}/v2/account",
                timeout=self.config.timeout_seconds
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise BrokerAuthenticationError(f"Failed to get account: {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise BrokerConnectionError(f"Connection error: {e}")

    def _parse_alpaca_order(self, data: Dict[str, Any]) -> BrokerOrder:
        """Parse Alpaca order response to BrokerOrder"""
        # Map Alpaca status to our OrderStatus
        status_map = {
            "new": OrderStatus.ACCEPTED,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL_FILL,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED
        }

        return BrokerOrder(
            broker_order_id=data["id"],
            client_order_id=data.get("client_order_id", ""),
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["type"],
            quantity=Decimal(str(data["qty"])),
            status=status_map.get(data["status"], OrderStatus.PENDING),
            filled_quantity=Decimal(str(data.get("filled_qty", 0))),
            avg_fill_price=Decimal(str(data["filled_avg_price"])) if data.get("filled_avg_price") else None,
            limit_price=Decimal(str(data["limit_price"])) if data.get("limit_price") else None,
            stop_price=Decimal(str(data["stop_price"])) if data.get("stop_price") else None,
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            metadata=data
        )


class InteractiveBrokerAdapter(BrokerAdapter):
    """
    Interactive Brokers adapter

    Connects to IB Gateway or TWS via ib_insync library
    """

    def __init__(self, config: BrokerConfig):
        """Initialize Interactive Brokers adapter"""
        super().__init__(config)

        self.host = config.metadata.get("host", "127.0.0.1")
        self.port = config.metadata.get("port", 7497 if config.paper_trading else 7496)
        self.client_id = config.metadata.get("client_id", 1)

        self._ib = None
        self._orders: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to Interactive Brokers"""
        try:
            self.status = BrokerStatus.CONNECTING

            # Import ib_insync (lazy import)
            try:
                from ib_insync import IB
            except ImportError:
                raise BrokerError(
                    "ib_insync library not installed. "
                    "Install with: pip install ib_insync"
                )

            self._ib = IB()
            await self._ib.connectAsync(
                self.host,
                self.port,
                clientId=self.client_id,
                timeout=self.config.timeout_seconds
            )

            self.status = BrokerStatus.CONNECTED
            logger.info(f"Connected to Interactive Brokers ({self.config.broker_id})")

            # Register event handlers
            self._ib.orderStatusEvent += self._on_order_status

            return True

        except Exception as e:
            self.status = BrokerStatus.ERROR
            logger.error(f"Failed to connect to Interactive Brokers: {e}")
            raise BrokerConnectionError(f"IB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers"""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

        self.status = BrokerStatus.DISCONNECTED
        logger.info(f"Disconnected from Interactive Brokers ({self.config.broker_id})")

    async def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit order to Interactive Brokers"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Interactive Brokers")

        try:
            from ib_insync import Stock, Order

            # Create contract
            contract = Stock(symbol, "SMART", "USD")

            # Create order
            action = "BUY" if side.lower() == "buy" else "SELL"

            if order_type.lower() == "market":
                ib_order = Order(
                    action=action,
                    totalQuantity=float(quantity),
                    orderType="MKT"
                )
            elif order_type.lower() == "limit":
                ib_order = Order(
                    action=action,
                    totalQuantity=float(quantity),
                    orderType="LMT",
                    lmtPrice=float(limit_price)
                )
            elif order_type.lower() == "stop":
                ib_order = Order(
                    action=action,
                    totalQuantity=float(quantity),
                    orderType="STP",
                    auxPrice=float(stop_price)
                )
            else:
                raise BrokerOrderError(f"Unsupported order type: {order_type}")

            # Set time in force
            ib_order.tif = time_in_force.upper()

            # Place order
            trade = self._ib.placeOrder(contract, ib_order)
            await asyncio.sleep(0.1)  # Give IB time to process

            broker_order_id = str(trade.order.orderId)
            self._orders[broker_order_id] = trade

            logger.info(
                f"Submitted order to IB: {broker_order_id} "
                f"({side} {quantity} {symbol})"
            )

            await self._trigger_callbacks("order_submitted", trade)

            return broker_order_id

        except Exception as e:
            raise BrokerOrderError(f"Failed to submit order: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order on Interactive Brokers"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Interactive Brokers")

        try:
            trade = self._orders.get(broker_order_id)
            if not trade:
                raise BrokerOrderError(f"Order not found: {broker_order_id}")

            self._ib.cancelOrder(trade.order)
            await asyncio.sleep(0.1)

            logger.info(f"Cancelled order on IB: {broker_order_id}")
            await self._trigger_callbacks("order_cancelled", {"order_id": broker_order_id})

            return True

        except Exception as e:
            raise BrokerOrderError(f"Failed to cancel order: {e}")

    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        """Get order from Interactive Brokers"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Interactive Brokers")

        trade = self._orders.get(broker_order_id)
        if not trade:
            raise BrokerOrderError(f"Order not found: {broker_order_id}")

        return self._parse_ib_order(trade)

    async def get_quote(
        self,
        symbol: str,
        side: str,
        quantity: Decimal
    ) -> Dict[str, Any]:
        """Get quote from Interactive Brokers"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Interactive Brokers")

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            ticker = self._ib.reqMktData(contract)
            await asyncio.sleep(0.5)  # Wait for market data

            return {
                "symbol": symbol,
                "bid": Decimal(str(ticker.bid)),
                "ask": Decimal(str(ticker.ask)),
                "bid_size": Decimal(str(ticker.bidSize)),
                "ask_size": Decimal(str(ticker.askSize)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fee_rate": Decimal("0.005")  # IB typically charges ~$0.005/share
            }

        except Exception as e:
            raise BrokerError(f"Failed to get quote: {e}")

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Interactive Brokers"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Interactive Brokers")

        try:
            account_values = self._ib.accountValues()

            info = {}
            for av in account_values:
                info[av.tag] = {
                    "value": av.value,
                    "currency": av.currency,
                    "account": av.account
                }

            return info

        except Exception as e:
            raise BrokerError(f"Failed to get account info: {e}")

    def _on_order_status(self, trade):
        """Handle order status updates"""
        asyncio.create_task(
            self._trigger_callbacks("order_status", trade)
        )

    def _parse_ib_order(self, trade) -> BrokerOrder:
        """Parse IB trade to BrokerOrder"""
        order = trade.order
        order_status = trade.orderStatus

        # Map IB status to our OrderStatus
        status_map = {
            "PendingSubmit": OrderStatus.PENDING,
            "Submitted": OrderStatus.ACCEPTED,
            "PreSubmitted": OrderStatus.ACCEPTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED
        }

        return BrokerOrder(
            broker_order_id=str(order.orderId),
            client_order_id=str(order.orderId),
            symbol=trade.contract.symbol,
            side=order.action.lower(),
            order_type=order.orderType.lower(),
            quantity=Decimal(str(order.totalQuantity)),
            status=status_map.get(order_status.status, OrderStatus.PENDING),
            filled_quantity=Decimal(str(order_status.filled)),
            avg_fill_price=Decimal(str(order_status.avgFillPrice)) if order_status.avgFillPrice else None,
            limit_price=Decimal(str(order.lmtPrice)) if order.lmtPrice else None,
            stop_price=Decimal(str(order.auxPrice)) if order.auxPrice else None,
            metadata={"trade": trade}
        )


class BrokerAdapterManager:
    """
    Manager for multiple broker adapters

    Handles connection management, health checks, and failover
    """

    def __init__(self):
        """Initialize broker adapter manager"""
        self.adapters: Dict[str, BrokerAdapter] = {}
        self._health_check_task: Optional[asyncio.Task] = None

    def register_adapter(self, broker_id: str, adapter: BrokerAdapter):
        """Register a broker adapter"""
        self.adapters[broker_id] = adapter
        logger.info(f"Registered broker adapter: {broker_id}")

    async def connect_all(self) -> Dict[str, bool]:
        """Connect all adapters"""
        results = {}
        for broker_id, adapter in self.adapters.items():
            try:
                success = await adapter.connect()
                results[broker_id] = success
            except Exception as e:
                logger.error(f"Failed to connect {broker_id}: {e}")
                results[broker_id] = False

        return results

    async def disconnect_all(self):
        """Disconnect all adapters"""
        for adapter in self.adapters.values():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting adapter: {e}")

    def get_adapter(self, broker_id: str) -> Optional[BrokerAdapter]:
        """Get adapter by broker ID"""
        return self.adapters.get(broker_id)

    def get_connected_adapters(self) -> Dict[str, BrokerAdapter]:
        """Get all connected adapters"""
        return {
            broker_id: adapter
            for broker_id, adapter in self.adapters.items()
            if adapter.is_connected()
        }

    async def start_health_checks(self, interval_seconds: float = 60.0):
        """Start periodic health checks"""
        self._health_check_task = asyncio.create_task(
            self._health_check_loop(interval_seconds)
        )

    async def stop_health_checks(self):
        """Stop health checks"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self, interval_seconds: float):
        """Periodic health check loop"""
        while True:
            try:
                await asyncio.sleep(interval_seconds)

                for broker_id, adapter in self.adapters.items():
                    if adapter.status == BrokerStatus.CONNECTED:
                        try:
                            # Try to get account info as health check
                            await adapter.get_account_info()
                        except Exception as e:
                            logger.warning(f"Health check failed for {broker_id}: {e}")
                            adapter.status = BrokerStatus.ERROR

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
