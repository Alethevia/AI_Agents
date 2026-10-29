"""
WebSocket subscription manager.

This module manages WebSocket subscriptions, topics, and client connections
for real-time data streaming.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from fastapi import WebSocket


logger = logging.getLogger(__name__)


# ============================================================================
# Topic Definitions
# ============================================================================

# Available subscription topics
AVAILABLE_TOPICS = {
    # Market data topics
    "market_data.ticker.*",  # Ticker updates for all symbols
    "market_data.ticker.{symbol}",  # Ticker for specific symbol
    "market_data.orderbook.*",  # Order book updates for all symbols
    "market_data.orderbook.{symbol}",  # Order book for specific symbol
    "market_data.trades.*",  # Trade updates for all symbols
    "market_data.trades.{symbol}",  # Trades for specific symbol

    # Portfolio topics
    "portfolio.positions",  # Position updates
    "portfolio.balance",  # Balance updates
    "portfolio.performance",  # Performance metrics

    # Order topics
    "orders.updates",  # All order updates
    "orders.fills",  # Order fills
    "orders.cancellations",  # Order cancellations
    "orders.{order_id}",  # Specific order updates

    # Strategy topics
    "strategies.status",  # Strategy status changes
    "strategies.signals",  # Trading signals
    "strategies.{strategy_id}",  # Specific strategy updates

    # Risk topics
    "risk.alerts",  # Risk alerts
    "risk.metrics",  # Risk metrics updates
    "risk.limits",  # Risk limit violations

    # System topics
    "system.status",  # System status updates
    "system.health",  # Health check updates
}


# ============================================================================
# Client Connection
# ============================================================================

class WebSocketClient:
    """Represents a WebSocket client connection."""

    def __init__(self, websocket: WebSocket, client_id: Optional[str] = None):
        """
        Initialize WebSocket client.

        Args:
            websocket: The WebSocket connection
            client_id: Optional client identifier
        """
        self.client_id = client_id or str(uuid4())
        self.websocket = websocket
        self.subscriptions: Set[str] = set()
        self.user: Optional[Dict] = None
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        self.message_count = 0

    async def send_json(self, data: Dict[str, Any]) -> bool:
        """
        Send JSON data to client.

        Args:
            data: Data to send

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.websocket.send_json(data)
            self.last_activity = datetime.utcnow()
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Error sending to client {self.client_id}: {e}")
            return False

    async def send_text(self, message: str) -> bool:
        """
        Send text message to client.

        Args:
            message: Message to send

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.websocket.send_text(message)
            self.last_activity = datetime.utcnow()
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Error sending to client {self.client_id}: {e}")
            return False

    def subscribe(self, topic: str) -> bool:
        """
        Subscribe to a topic.

        Args:
            topic: Topic to subscribe to

        Returns:
            True if subscribed, False if already subscribed
        """
        if topic not in self.subscriptions:
            self.subscriptions.add(topic)
            return True
        return False

    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a topic.

        Args:
            topic: Topic to unsubscribe from

        Returns:
            True if unsubscribed, False if not subscribed
        """
        if topic in self.subscriptions:
            self.subscriptions.remove(topic)
            return True
        return False

    def is_subscribed(self, topic: str) -> bool:
        """Check if client is subscribed to a topic."""
        return topic in self.subscriptions

    def to_dict(self) -> Dict[str, Any]:
        """Convert client to dictionary."""
        return {
            "client_id": self.client_id,
            "subscriptions": list(self.subscriptions),
            "user": self.user.get("username") if self.user else None,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "metadata": self.metadata,
        }


# ============================================================================
# Subscription Manager
# ============================================================================

class SubscriptionManager:
    """Manages WebSocket subscriptions and message routing."""

    def __init__(self):
        """Initialize subscription manager."""
        self.clients: Dict[str, WebSocketClient] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        user: Optional[Dict] = None
    ) -> WebSocketClient:
        """
        Register a new client connection.

        Args:
            websocket: The WebSocket connection
            client_id: Optional client identifier
            user: Optional authenticated user

        Returns:
            WebSocketClient instance
        """
        async with self.lock:
            client = WebSocketClient(websocket, client_id)
            client.user = user

            self.clients[client.client_id] = client

            logger.info(
                f"Client connected: {client.client_id} "
                f"(user: {user.get('username') if user else 'anonymous'})"
            )

            return client

    async def disconnect(self, client_id: str) -> bool:
        """
        Unregister a client connection.

        Args:
            client_id: Client identifier

        Returns:
            True if disconnected, False if client not found
        """
        async with self.lock:
            client = self.clients.get(client_id)
            if not client:
                return False

            # Unsubscribe from all topics
            for topic in list(client.subscriptions):
                await self._unsubscribe_internal(client, topic)

            # Remove client
            del self.clients[client_id]

            logger.info(f"Client disconnected: {client_id}")

            return True

    async def subscribe(self, client_id: str, topics: List[str]) -> Dict[str, bool]:
        """
        Subscribe client to topics.

        Args:
            client_id: Client identifier
            topics: List of topics to subscribe to

        Returns:
            Dictionary of topic -> success status
        """
        async with self.lock:
            client = self.clients.get(client_id)
            if not client:
                return {topic: False for topic in topics}

            results = {}
            for topic in topics:
                # Validate topic
                if not self._validate_topic(topic):
                    logger.warning(f"Invalid topic: {topic}")
                    results[topic] = False
                    continue

                # Check permissions
                if not self._check_topic_permission(client, topic):
                    logger.warning(
                        f"Permission denied for client {client_id} on topic {topic}"
                    )
                    results[topic] = False
                    continue

                # Subscribe
                success = await self._subscribe_internal(client, topic)
                results[topic] = success

            return results

    async def unsubscribe(self, client_id: str, topics: List[str]) -> Dict[str, bool]:
        """
        Unsubscribe client from topics.

        Args:
            client_id: Client identifier
            topics: List of topics to unsubscribe from

        Returns:
            Dictionary of topic -> success status
        """
        async with self.lock:
            client = self.clients.get(client_id)
            if not client:
                return {topic: False for topic in topics}

            results = {}
            for topic in topics:
                success = await self._unsubscribe_internal(client, topic)
                results[topic] = success

            return results

    async def _subscribe_internal(self, client: WebSocketClient, topic: str) -> bool:
        """Internal subscribe method (not thread-safe, use within lock)."""
        if client.subscribe(topic):
            self.topic_subscribers[topic].add(client.client_id)
            logger.debug(f"Client {client.client_id} subscribed to {topic}")
            return True
        return False

    async def _unsubscribe_internal(self, client: WebSocketClient, topic: str) -> bool:
        """Internal unsubscribe method (not thread-safe, use within lock)."""
        if client.unsubscribe(topic):
            self.topic_subscribers[topic].discard(client.client_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
            logger.debug(f"Client {client.client_id} unsubscribed from {topic}")
            return True
        return False

    async def broadcast(
        self,
        topic: str,
        message: Dict[str, Any],
        exclude_clients: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast message to all subscribers of a topic.

        Args:
            topic: Topic to broadcast to
            message: Message to send
            exclude_clients: Optional set of client IDs to exclude

        Returns:
            Number of clients message was sent to
        """
        exclude_clients = exclude_clients or set()

        # Get matching subscribers
        subscribers = await self._get_subscribers(topic)

        # Filter out excluded clients
        subscribers = [
            client_id for client_id in subscribers
            if client_id not in exclude_clients
        ]

        if not subscribers:
            return 0

        # Add topic to message
        message_with_topic = {
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            **message
        }

        # Send to all subscribers
        success_count = 0
        failed_clients = []

        for client_id in subscribers:
            client = self.clients.get(client_id)
            if not client:
                continue

            success = await client.send_json(message_with_topic)
            if success:
                success_count += 1
            else:
                failed_clients.append(client_id)

        # Disconnect failed clients
        for client_id in failed_clients:
            await self.disconnect(client_id)

        return success_count

    async def send_to_client(
        self,
        client_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send message to a specific client.

        Args:
            client_id: Client identifier
            message: Message to send

        Returns:
            True if successful, False otherwise
        """
        client = self.clients.get(client_id)
        if not client:
            return False

        return await client.send_json(message)

    async def _get_subscribers(self, topic: str) -> List[str]:
        """
        Get list of client IDs subscribed to a topic.

        Supports wildcard matching:
        - "market_data.ticker.*" matches all ticker topics
        - "market_data.ticker.BTC" matches specific symbol

        Args:
            topic: Topic to get subscribers for

        Returns:
            List of client IDs
        """
        subscribers = set()

        # Exact match
        if topic in self.topic_subscribers:
            subscribers.update(self.topic_subscribers[topic])

        # Wildcard match - check if any subscription patterns match this topic
        for subscription, clients in self.topic_subscribers.items():
            if self._topic_matches(topic, subscription):
                subscribers.update(clients)

        return list(subscribers)

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """
        Check if topic matches a subscription pattern.

        Args:
            topic: The actual topic
            pattern: The subscription pattern (may contain wildcards)

        Returns:
            True if matches, False otherwise
        """
        # Exact match
        if topic == pattern:
            return True

        # Wildcard matching
        topic_parts = topic.split(".")
        pattern_parts = pattern.split(".")

        if len(topic_parts) != len(pattern_parts):
            return False

        for topic_part, pattern_part in zip(topic_parts, pattern_parts):
            if pattern_part == "*":
                continue
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                continue
            if topic_part != pattern_part:
                return False

        return True

    def _validate_topic(self, topic: str) -> bool:
        """
        Validate that a topic exists.

        Args:
            topic: Topic to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if topic matches any available topic pattern
        for available_topic in AVAILABLE_TOPICS:
            if self._topic_matches(topic, available_topic):
                return True
            if self._topic_matches(available_topic, topic):
                return True

        return False

    def _check_topic_permission(self, client: WebSocketClient, topic: str) -> bool:
        """
        Check if client has permission to access topic.

        Args:
            client: WebSocket client
            topic: Topic to check

        Returns:
            True if allowed, False otherwise
        """
        # If no user (anonymous), only allow public topics
        if not client.user:
            public_topics = ["market_data"]
            return any(topic.startswith(prefix) for prefix in public_topics)

        # For authenticated users, check permissions based on topic category
        user = client.user
        topic_category = topic.split(".")[0]

        permission_map = {
            "market_data": "read:market_data",
            "portfolio": "read:portfolio",
            "orders": "read:orders",
            "strategies": "read:strategies",
            "risk": "read:risk",
            "system": "admin:users",
        }

        required_permission = permission_map.get(topic_category)
        if not required_permission:
            return False

        # Check if user has permission (simplified - should use proper auth system)
        user_permissions = set()
        for role in user.get("roles", []):
            from ..middleware.auth import ROLE_PERMISSIONS
            user_permissions.update(ROLE_PERMISSIONS.get(role, set()))

        return required_permission in user_permissions

    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        """Get client by ID."""
        return self.clients.get(client_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get subscription manager statistics."""
        total_subscriptions = sum(
            len(client.subscriptions) for client in self.clients.values()
        )

        topic_stats = {}
        for topic, subscribers in self.topic_subscribers.items():
            topic_stats[topic] = len(subscribers)

        return {
            "total_clients": len(self.clients),
            "total_subscriptions": total_subscriptions,
            "total_topics": len(self.topic_subscribers),
            "topics": topic_stats,
            "clients": [client.to_dict() for client in self.clients.values()],
        }

    async def cleanup_inactive_clients(self, timeout_seconds: int = 300):
        """
        Clean up clients that haven't been active.

        Args:
            timeout_seconds: Inactivity timeout in seconds
        """
        now = datetime.utcnow()
        inactive_clients = []

        for client_id, client in self.clients.items():
            inactivity = (now - client.last_activity).total_seconds()
            if inactivity > timeout_seconds:
                inactive_clients.append(client_id)

        for client_id in inactive_clients:
            logger.info(f"Cleaning up inactive client: {client_id}")
            await self.disconnect(client_id)


# Global subscription manager instance
subscription_manager = SubscriptionManager()
