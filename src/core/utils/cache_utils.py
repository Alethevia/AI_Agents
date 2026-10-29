"""
Caching utilities for trading system.
Provides in-memory and distributed caching with Redis support.
"""

import logging
import asyncio
from typing import Any, Optional, Callable, Union, Dict
from datetime import timedelta
from functools import wraps
import hashlib
import pickle

import redis.asyncio as redis
from redis.asyncio import Redis

from src.core.utils.serializers import (
    serialize_pickle,
    deserialize_pickle,
    serialize_json,
    deserialize_json
)

logger = logging.getLogger(__name__)


class MemoryCache:
    """Simple in-memory LRU cache."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: list = []

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if exists
        if key in self.cache:
            self.access_order.remove(key)

        # Add to cache
        self.cache[key] = value
        self.access_order.append(key)

        # Evict if over size limit
        while len(self.cache) > self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

    def delete(self, key: str) -> None:
        """
        Delete item from cache.

        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)

    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()
        self.access_order.clear()

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        return key in self.cache

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of items in cache
        """
        return len(self.cache)


class RedisCache:
    """Redis-backed distributed cache."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "trading:",
        default_ttl: int = 3600
    ):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.client: Optional[Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
            await self.client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis cache")

    def _make_key(self, key: str) -> str:
        """
        Make full key with prefix.

        Args:
            key: Base key

        Returns:
            Full key
        """
        return f"{self.prefix}{key}"

    async def get(
        self,
        key: str,
        deserialize: bool = True
    ) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key
            deserialize: Whether to deserialize value

        Returns:
            Cached value or None
        """
        if not self.client:
            logger.warning("Redis client not connected")
            return None

        try:
            full_key = self._make_key(key)
            value = await self.client.get(full_key)

            if value is None:
                return None

            if deserialize:
                return deserialize_pickle(value)
            else:
                return value

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize value

        Returns:
            True if successful
        """
        if not self.client:
            logger.warning("Redis client not connected")
            return False

        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl

            if serialize:
                value = serialize_pickle(value)

            await self.client.setex(full_key, ttl, value)
            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete item from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if not self.client:
            logger.warning("Redis client not connected")
            return False

        try:
            full_key = self._make_key(key)
            result = await self.client.delete(full_key)
            return result > 0

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        if not self.client:
            return False

        try:
            full_key = self._make_key(key)
            result = await self.client.exists(full_key)
            return result > 0

        except Exception as e:
            logger.error(f"Error checking existence: {e}")
            return False

    async def clear(self, pattern: str = "*") -> int:
        """
        Clear keys matching pattern.

        Args:
            pattern: Key pattern

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        try:
            full_pattern = self._make_key(pattern)
            keys = await self.client.keys(full_pattern)

            if not keys:
                return 0

            result = await self.client.delete(*keys)
            return result

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment counter.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            New value
        """
        if not self.client:
            return 0

        try:
            full_key = self._make_key(key)
            return await self.client.incrby(full_key, amount)

        except Exception as e:
            logger.error(f"Error incrementing: {e}")
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            full_key = self._make_key(key)
            return await self.client.expire(full_key, ttl)

        except Exception as e:
            logger.error(f"Error setting expiration: {e}")
            return False


class CacheManager:
    """Unified cache manager with fallback."""

    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        redis_cache: Optional[RedisCache] = None
    ):
        """
        Initialize cache manager.

        Args:
            memory_cache: Memory cache instance
            redis_cache: Redis cache instance
        """
        self.memory_cache = memory_cache or MemoryCache()
        self.redis_cache = redis_cache

    async def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache (checks memory first, then Redis).

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        # Check memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Check Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Populate memory cache
                self.memory_cache.set(key, value)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set item in cache (both memory and Redis).

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Set in memory cache
        self.memory_cache.set(key, value)

        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl=ttl)

    async def delete(self, key: str) -> None:
        """
        Delete item from cache.

        Args:
            key: Cache key
        """
        self.memory_cache.delete(key)

        if self.redis_cache:
            await self.redis_cache.delete(key)

    async def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()

        if self.redis_cache:
            await self.redis_cache.clear()


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key
    """
    # Create string representation
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = ":".join(key_parts)

    # Hash if too long
    if len(key_str) > 200:
        return hashlib.md5(key_str.encode()).hexdigest()

    return key_str


def cached(
    ttl: int = 3600,
    key_prefix: str = ""
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Key prefix

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = MemoryCache()

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(key, result)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result)

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class TimedCache:
    """Cache with automatic expiration."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize timed cache.

        Args:
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
        self.cache: Dict[str, tuple] = {}  # {key: (value, expiry_time)}

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key not in self.cache:
            return None

        value, expiry = self.cache[key]

        # Check if expired
        import time
        if time.time() > expiry:
            del self.cache[key]
            return None

        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        import time
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """
        Delete item from cache.

        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]

    def clear_expired(self) -> int:
        """
        Clear all expired items.

        Returns:
            Number of items cleared
        """
        import time
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if expiry < current_time
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)


# Global cache instance
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """
    Get global cache instance.

    Returns:
        CacheManager instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


async def init_cache(redis_url: Optional[str] = None) -> CacheManager:
    """
    Initialize global cache.

    Args:
        redis_url: Redis connection URL

    Returns:
        CacheManager instance
    """
    global _global_cache

    memory_cache = MemoryCache(max_size=1000)
    redis_cache = None

    if redis_url:
        redis_cache = RedisCache(redis_url=redis_url)
        await redis_cache.connect()

    _global_cache = CacheManager(
        memory_cache=memory_cache,
        redis_cache=redis_cache
    )

    return _global_cache
