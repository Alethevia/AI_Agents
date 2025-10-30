"""
Error handlers and recovery strategies.
"""

import asyncio
from functools import wraps
from typing import Callable, Optional, Type, Tuple, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.logging.logger import get_logger
from src.core.monitoring.metrics import get_metrics
from .exceptions import TradingSystemError, TimeoutError

logger = get_logger(__name__)
metrics = get_metrics()


def error_handler(
    reraise: bool = True,
    default_return: Any = None,
    component: str = "unknown",
):
    """
    Decorator for error handling.

    Args:
        reraise: Whether to reraise the exception
        default_return: Default return value on error
        component: Component name for metrics
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}",
                    function=func.__name__,
                    component=component,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                metrics.record_error(component, type(e).__name__)

                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def async_error_handler(
    reraise: bool = True,
    default_return: Any = None,
    component: str = "unknown",
):
    """
    Decorator for async error handling.

    Args:
        reraise: Whether to reraise the exception
        default_return: Default return value on error
        component: Component name for metrics
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}",
                    function=func.__name__,
                    component=component,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                metrics.record_error(component, type(e).__name__)

                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def with_retry(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    wait_min: float = 1.0,
    wait_max: float = 10.0,
    exponential_base: float = 2.0,
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        exceptions: Exception types to retry on
        wait_min: Minimum wait time between retries
        wait_max: Maximum wait time between retries
        exponential_base: Base for exponential backoff
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            min=wait_min,
            max=wait_max,
            multiplier=1,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )


def with_timeout(seconds: float):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds}s"
                )

        return wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting to close circuit
            expected_exceptions: Exceptions that count as failures
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout_seconds
        self.expected_exceptions = expected_exceptions

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        import time

        if self.state == "open":
            if self.last_failure_time and (
                time.time() - self.last_failure_time > self.timeout
            ):
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                from .exceptions import CircuitBreakerError

                raise CircuitBreakerError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
            return result

        except self.expected_exceptions as e:
            import time

            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

            raise

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Async version of call.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        import time

        if self.state == "open":
            if self.last_failure_time and (
                time.time() - self.last_failure_time > self.timeout
            ):
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                from .exceptions import CircuitBreakerError

                raise CircuitBreakerError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
            return result

        except self.expected_exceptions as e:
            import time

            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

            raise
