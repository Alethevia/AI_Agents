"""
WebSocket server for real-time data streaming.

This module provides a WebSocket server for streaming real-time market data,
portfolio updates, order updates, and risk alerts to clients.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect, Query, Depends
from pydantic import ValidationError

from ..models.api_models import (
    WebSocketMessage,
    WebSocketMessageType,
    WebSocketSubscribeRequest,
    WebSocketUnsubscribeRequest,
)
from .subscription_manager import subscription_manager, WebSocketClient


logger = logging.getLogger(__name__)


# ============================================================================
# Message Handlers
# ============================================================================

class MessageHandler:
    """Handles incoming WebSocket messages."""

    def __init__(self, client: WebSocketClient):
        """
        Initialize message handler.

        Args:
            client: WebSocket client
        """
        self.client = client

    async def handle_message(self, raw_message: str) -> None:
        """
        Handle incoming message from client.

        Args:
            raw_message: Raw message string
        """
        try:
            # Parse message
            data = json.loads(raw_message)

            # Validate message structure
            try:
                message = WebSocketMessage(**data)
            except ValidationError as e:
                await self._send_error("Invalid message format", str(e))
                return

            # Route message based on type
            handlers = {
                WebSocketMessageType.SUBSCRIBE: self._handle_subscribe,
                WebSocketMessageType.UNSUBSCRIBE: self._handle_unsubscribe,
                WebSocketMessageType.PING: self._handle_ping,
            }

            handler = handlers.get(message.type)
            if handler:
                await handler(message)
            else:
                await self._send_error(
                    "Unknown message type",
                    f"Message type '{message.type}' is not supported"
                )

        except json.JSONDecodeError as e:
            await self._send_error("Invalid JSON", str(e))
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            await self._send_error("Internal error", "An error occurred processing your message")

    async def _handle_subscribe(self, message: WebSocketMessage) -> None:
        """
        Handle subscribe message.

        Args:
            message: WebSocket message
        """
        try:
            if not message.data or "topics" not in message.data:
                await self._send_error("Invalid subscribe request", "Topics are required")
                return

            # Validate subscription request
            try:
                request = WebSocketSubscribeRequest(**message.data)
            except ValidationError as e:
                await self._send_error("Invalid subscribe request", str(e))
                return

            # Subscribe to topics
            results = await subscription_manager.subscribe(
                self.client.client_id,
                request.topics
            )

            # Send response
            await self.client.send_json({
                "type": "subscribed",
                "data": {
                    "results": results,
                    "subscriptions": list(self.client.subscriptions),
                },
                "timestamp": datetime.utcnow().isoformat(),
            })

            logger.info(
                f"Client {self.client.client_id} subscribed to topics: {request.topics}"
            )

        except Exception as e:
            logger.exception(f"Error handling subscribe: {e}")
            await self._send_error("Subscribe failed", str(e))

    async def _handle_unsubscribe(self, message: WebSocketMessage) -> None:
        """
        Handle unsubscribe message.

        Args:
            message: WebSocket message
        """
        try:
            if not message.data or "topics" not in message.data:
                await self._send_error("Invalid unsubscribe request", "Topics are required")
                return

            # Validate unsubscribe request
            try:
                request = WebSocketUnsubscribeRequest(**message.data)
            except ValidationError as e:
                await self._send_error("Invalid unsubscribe request", str(e))
                return

            # Unsubscribe from topics
            results = await subscription_manager.unsubscribe(
                self.client.client_id,
                request.topics
            )

            # Send response
            await self.client.send_json({
                "type": "unsubscribed",
                "data": {
                    "results": results,
                    "subscriptions": list(self.client.subscriptions),
                },
                "timestamp": datetime.utcnow().isoformat(),
            })

            logger.info(
                f"Client {self.client.client_id} unsubscribed from topics: {request.topics}"
            )

        except Exception as e:
            logger.exception(f"Error handling unsubscribe: {e}")
            await self._send_error("Unsubscribe failed", str(e))

    async def _handle_ping(self, message: WebSocketMessage) -> None:
        """
        Handle ping message.

        Args:
            message: WebSocket message
        """
        await self.client.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def _send_error(self, error: str, details: str) -> None:
        """
        Send error message to client.

        Args:
            error: Error type
            details: Error details
        """
        await self.client.send_json({
            "type": "error",
            "error": error,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        })


# ============================================================================
# WebSocket Endpoint
# ============================================================================

async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time data streaming.

    Args:
        websocket: WebSocket connection
        token: Optional JWT token for authentication
    """
    client = None
    try:
        # Accept connection
        await websocket.accept()

        # Authenticate user if token provided
        user = None
        if token:
            user = await _authenticate_websocket(token)

        # Register client
        client = await subscription_manager.connect(
            websocket,
            user=user
        )

        # Send welcome message
        await client.send_json({
            "type": "connected",
            "data": {
                "client_id": client.client_id,
                "user": user.get("username") if user else None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        # Create message handler
        handler = MessageHandler(client)

        # Message loop
        while True:
            # Receive message
            raw_message = await websocket.receive_text()

            # Handle message
            await handler.handle_message(raw_message)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client.client_id if client else 'unknown'}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        # Disconnect client
        if client:
            await subscription_manager.disconnect(client.client_id)


async def _authenticate_websocket(token: str) -> Optional[Dict]:
    """
    Authenticate WebSocket connection with JWT token.

    Args:
        token: JWT token

    Returns:
        User data if authenticated, None otherwise
    """
    try:
        from ..middleware.auth import decode_token, get_user

        payload = decode_token(token)
        username = payload.get("sub")

        if username:
            user = get_user(username)
            if user and not user.get("disabled", False):
                return user

    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {e}")

    return None


# ============================================================================
# Broadcasting Functions
# ============================================================================

async def broadcast_market_data(
    symbol: str,
    data: Dict[str, Any],
    data_type: str = "ticker"
) -> int:
    """
    Broadcast market data update.

    Args:
        symbol: Trading symbol
        data: Market data
        data_type: Type of market data (ticker, orderbook, trades)

    Returns:
        Number of clients message was sent to
    """
    topic = f"market_data.{data_type}.{symbol}"

    message = {
        "type": "market_data",
        "data": {
            "symbol": symbol,
            "data_type": data_type,
            **data
        }
    }

    return await subscription_manager.broadcast(topic, message)


async def broadcast_portfolio_update(
    user_id: str,
    update_type: str,
    data: Dict[str, Any]
) -> int:
    """
    Broadcast portfolio update to user.

    Args:
        user_id: User identifier
        update_type: Type of update (positions, balance, performance)
        data: Update data

    Returns:
        Number of clients message was sent to
    """
    topic = f"portfolio.{update_type}"

    message = {
        "type": "portfolio_update",
        "data": {
            "update_type": update_type,
            **data
        }
    }

    # Find clients for this user
    user_clients = [
        client_id for client_id, client in subscription_manager.clients.items()
        if client.user and client.user.get("username") == user_id
    ]

    # Send to each user's client
    count = 0
    for client_id in user_clients:
        if await subscription_manager.send_to_client(client_id, message):
            count += 1

    return count


async def broadcast_order_update(
    user_id: str,
    order_data: Dict[str, Any]
) -> int:
    """
    Broadcast order update to user.

    Args:
        user_id: User identifier
        order_data: Order data

    Returns:
        Number of clients message was sent to
    """
    topic = "orders.updates"

    message = {
        "type": "order_update",
        "data": order_data
    }

    # Find clients for this user
    user_clients = [
        client_id for client_id, client in subscription_manager.clients.items()
        if client.user and client.user.get("username") == user_id
    ]

    # Send to each user's client
    count = 0
    for client_id in user_clients:
        if await subscription_manager.send_to_client(client_id, message):
            count += 1

    return count


async def broadcast_risk_alert(
    alert_data: Dict[str, Any],
    severity: str = "warning"
) -> int:
    """
    Broadcast risk alert.

    Args:
        alert_data: Alert data
        severity: Alert severity (info, warning, critical)

    Returns:
        Number of clients message was sent to
    """
    topic = "risk.alerts"

    message = {
        "type": "risk_alert",
        "data": {
            "severity": severity,
            **alert_data
        }
    }

    return await subscription_manager.broadcast(topic, message)


async def broadcast_strategy_update(
    strategy_id: str,
    update_type: str,
    data: Dict[str, Any]
) -> int:
    """
    Broadcast strategy update.

    Args:
        strategy_id: Strategy identifier
        update_type: Type of update (status, signal, performance)
        data: Update data

    Returns:
        Number of clients message was sent to
    """
    # Broadcast to strategy-specific topic
    strategy_topic = f"strategies.{strategy_id}"
    count = await subscription_manager.broadcast(strategy_topic, {
        "type": "strategy_update",
        "data": {
            "strategy_id": strategy_id,
            "update_type": update_type,
            **data
        }
    })

    # Also broadcast to general topics
    if update_type == "status":
        count += await subscription_manager.broadcast("strategies.status", {
            "type": "strategy_update",
            "data": {
                "strategy_id": strategy_id,
                "update_type": update_type,
                **data
            }
        })
    elif update_type == "signal":
        count += await subscription_manager.broadcast("strategies.signals", {
            "type": "strategy_update",
            "data": {
                "strategy_id": strategy_id,
                "update_type": update_type,
                **data
            }
        })

    return count


# ============================================================================
# Background Tasks
# ============================================================================

async def start_websocket_tasks():
    """Start background tasks for WebSocket server."""
    asyncio.create_task(cleanup_inactive_clients_task())
    asyncio.create_task(heartbeat_task())
    logger.info("WebSocket background tasks started")


async def cleanup_inactive_clients_task():
    """Background task to clean up inactive clients."""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute
            await subscription_manager.cleanup_inactive_clients(timeout_seconds=300)
        except Exception as e:
            logger.exception(f"Error in cleanup task: {e}")


async def heartbeat_task():
    """Background task to send heartbeat to all connected clients."""
    while True:
        try:
            await asyncio.sleep(30)  # Send every 30 seconds

            # Send heartbeat to all clients
            for client in subscription_manager.clients.values():
                await client.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })

        except Exception as e:
            logger.exception(f"Error in heartbeat task: {e}")


# ============================================================================
# Statistics and Monitoring
# ============================================================================

def get_websocket_stats() -> Dict[str, Any]:
    """Get WebSocket server statistics."""
    return subscription_manager.get_stats()
