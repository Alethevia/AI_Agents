"""
Rate limiting middleware.

This module provides flexible rate limiting for API endpoints with support for
per-user, per-IP, and per-endpoint rate limits.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware


# ============================================================================
# Rate Limit Configuration
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    key_func: Optional[Callable] = None  # Custom key function


# Default rate limits by endpoint pattern
DEFAULT_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    # Authentication endpoints
    "/api/v1/auth/login": RateLimitConfig(requests=5, window=60),  # 5 per minute
    "/api/v1/auth/refresh": RateLimitConfig(requests=10, window=60),

    # Order endpoints - more restrictive
    "/api/v1/orders/create": RateLimitConfig(requests=10, window=1),  # 10 per second
    "/api/v1/orders/cancel": RateLimitConfig(requests=20, window=1),

    # Portfolio endpoints
    "/api/v1/portfolio": RateLimitConfig(requests=30, window=1),

    # Market data - high throughput
    "/api/v1/market-data/ticker": RateLimitConfig(requests=100, window=1),
    "/api/v1/market-data/orderbook": RateLimitConfig(requests=50, window=1),
    "/api/v1/market-data/candles": RateLimitConfig(requests=60, window=60),

    # Strategy endpoints
    "/api/v1/strategies": RateLimitConfig(requests=30, window=60),
    "/api/v1/strategies/start": RateLimitConfig(requests=5, window=60),
    "/api/v1/strategies/stop": RateLimitConfig(requests=10, window=60),

    # Risk endpoints
    "/api/v1/risk/metrics": RateLimitConfig(requests=30, window=1),
    "/api/v1/risk/alerts": RateLimitConfig(requests=20, window=1),
}

# Global rate limit (applies to all endpoints)
GLOBAL_RATE_LIMIT = RateLimitConfig(requests=1000, window=60)  # 1000 per minute

# Per-user rate limit
USER_RATE_LIMIT = RateLimitConfig(requests=500, window=60)  # 500 per minute

# Per-IP rate limit (for unauthenticated requests)
IP_RATE_LIMIT = RateLimitConfig(requests=100, window=60)  # 100 per minute


# ============================================================================
# Rate Limiter Implementation
# ============================================================================

class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, rate: int, window: int):
        """
        Initialize token bucket.

        Args:
            rate: Number of tokens (requests) allowed
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limit exceeded
        """
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            refill = (elapsed / self.window) * self.rate
            self.tokens = min(self.rate, self.tokens + refill)
            self.last_update = now

            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.tokens >= 1:
            return 0.0

        # Calculate time needed to refill one token
        tokens_needed = 1 - self.tokens
        return (tokens_needed / self.rate) * self.window


class SlidingWindowCounter:
    """Sliding window counter rate limiter implementation."""

    def __init__(self, rate: int, window: int):
        """
        Initialize sliding window counter.

        Args:
            rate: Number of requests allowed
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.requests: deque = deque()
        self.lock = asyncio.Lock()

    async def consume(self) -> bool:
        """
        Try to consume a request slot.

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.window

            # Remove old requests outside the window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # Check if we're within the limit
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True

            return False

    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if len(self.requests) < self.rate:
            return 0.0

        # Oldest request will expire at this time
        oldest = self.requests[0]
        return max(0.0, (oldest + self.window) - time.time())

    def get_remaining(self) -> int:
        """Get number of remaining requests in the current window."""
        now = time.time()
        cutoff = now - self.window

        # Count requests within window
        count = sum(1 for req_time in self.requests if req_time >= cutoff)
        return max(0, self.rate - count)


# ============================================================================
# Rate Limiter Storage
# ============================================================================

class RateLimiterStorage:
    """Storage for rate limiters."""

    def __init__(self):
        """Initialize storage."""
        self.limiters: Dict[str, SlidingWindowCounter] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def get_limiter(self, key: str, config: RateLimitConfig) -> SlidingWindowCounter:
        """Get or create a rate limiter for a key."""
        if key not in self.limiters:
            self.limiters[key] = SlidingWindowCounter(config.requests, config.window)

        # Periodic cleanup of old limiters
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self.cleanup()

        return self.limiters[key]

    def cleanup(self):
        """Remove inactive limiters to prevent memory leaks."""
        now = time.time()
        inactive_keys = []

        for key, limiter in self.limiters.items():
            # Remove if no requests in the last 2 windows
            if not limiter.requests or (now - limiter.requests[-1]) > (limiter.window * 2):
                inactive_keys.append(key)

        for key in inactive_keys:
            del self.limiters[key]

        self.last_cleanup = now


# Global storage instance
rate_limiter_storage = RateLimiterStorage()


# ============================================================================
# Rate Limiting Helpers
# ============================================================================

def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get authenticated user
    user = getattr(request.state, "user", None)
    if user:
        return f"user:{user.get('username', 'unknown')}"

    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"

    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def get_rate_limit_config(path: str) -> Optional[RateLimitConfig]:
    """Get rate limit configuration for a path."""
    # Exact match
    if path in DEFAULT_RATE_LIMITS:
        return DEFAULT_RATE_LIMITS[path]

    # Prefix match
    for pattern, config in DEFAULT_RATE_LIMITS.items():
        if path.startswith(pattern):
            return config

    return None


async def check_rate_limit(
    request: Request,
    config: RateLimitConfig,
    identifier: str
) -> Tuple[bool, Optional[float]]:
    """
    Check if request is within rate limit.

    Args:
        request: The request to check
        config: Rate limit configuration
        identifier: Client identifier

    Returns:
        Tuple of (is_allowed, wait_time)
    """
    # Create composite key
    key = f"{identifier}:{request.url.path}"

    # Get limiter
    limiter = rate_limiter_storage.get_limiter(key, config)

    # Check if request is allowed
    allowed = await limiter.consume()

    if allowed:
        return True, None
    else:
        wait_time = limiter.get_wait_time()
        return False, wait_time


def add_rate_limit_headers(
    response: Response,
    config: RateLimitConfig,
    identifier: str,
    path: str
) -> None:
    """Add rate limit headers to response."""
    key = f"{identifier}:{path}"
    limiter = rate_limiter_storage.limiters.get(key)

    if limiter:
        response.headers["X-RateLimit-Limit"] = str(config.requests)
        response.headers["X-RateLimit-Remaining"] = str(limiter.get_remaining())
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time() + config.window)
        )


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for certain paths
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health"]:
            return await call_next(request)

        identifier = get_client_identifier(request)

        # Check global rate limit
        allowed, wait_time = await check_rate_limit(
            request,
            GLOBAL_RATE_LIMIT,
            "global"
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Global rate limit exceeded. Retry after {wait_time:.2f} seconds",
                headers={"Retry-After": str(int(wait_time) + 1)},
            )

        # Check user/IP rate limit
        user_limit = USER_RATE_LIMIT if identifier.startswith("user:") else IP_RATE_LIMIT
        allowed, wait_time = await check_rate_limit(
            request,
            user_limit,
            identifier
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {wait_time:.2f} seconds",
                headers={"Retry-After": str(int(wait_time) + 1)},
            )

        # Check endpoint-specific rate limit
        endpoint_config = get_rate_limit_config(request.url.path)
        if endpoint_config:
            allowed, wait_time = await check_rate_limit(
                request,
                endpoint_config,
                identifier
            )

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Endpoint rate limit exceeded. Retry after {wait_time:.2f} seconds",
                    headers={"Retry-After": str(int(wait_time) + 1)},
                )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if endpoint_config:
            add_rate_limit_headers(
                response,
                endpoint_config,
                identifier,
                request.url.path
            )

        return response


# ============================================================================
# Decorator for Custom Rate Limits
# ============================================================================

def rate_limit(requests: int, window: int, key_func: Optional[Callable] = None):
    """
    Decorator for applying custom rate limits to endpoints.

    Args:
        requests: Number of requests allowed
        window: Time window in seconds
        key_func: Optional custom key function

    Example:
        @app.get("/api/expensive")
        @rate_limit(requests=5, window=60)
        async def expensive_endpoint():
            return {"status": "ok"}
    """
    config = RateLimitConfig(requests=requests, window=window, key_func=key_func)

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            identifier = get_client_identifier(request)

            # Use custom key function if provided
            if config.key_func:
                identifier = config.key_func(request)

            allowed, wait_time = await check_rate_limit(request, config, identifier)

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry after {wait_time:.2f} seconds",
                    headers={"Retry-After": str(int(wait_time) + 1)},
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Rate Limit Monitoring
# ============================================================================

def get_rate_limit_stats() -> Dict:
    """Get statistics about rate limiting."""
    stats = {
        "total_limiters": len(rate_limiter_storage.limiters),
        "active_clients": defaultdict(int),
        "top_consumers": [],
    }

    # Count by client type
    for key in rate_limiter_storage.limiters.keys():
        client_type = key.split(":")[0]
        stats["active_clients"][client_type] += 1

    # Find top consumers
    consumers = []
    for key, limiter in rate_limiter_storage.limiters.items():
        if limiter.requests:
            consumers.append({
                "key": key,
                "requests_in_window": len(limiter.requests),
                "rate_limit": limiter.rate,
            })

    consumers.sort(key=lambda x: x["requests_in_window"], reverse=True)
    stats["top_consumers"] = consumers[:10]

    return stats
