"""
Production-grade structured logging using structlog.
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from structlog.stdlib import LoggerFactory
from pythonjsonlogger import jsonlogger

from src.config.settings import get_settings


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add timestamp to log events."""
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def add_log_level(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add log level to event dict."""
    if method_name == "warn":
        method_name = "warning"
    event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Setup structured logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'text')
        log_file: Optional log file path
    """
    settings = get_settings()

    # Use settings if not provided
    log_level = log_level or settings.log_level
    log_format = log_format or settings.log_format
    log_file = log_file or (settings.logs_dir / "trading_system.log")

    # Create logs directory if needed
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_log_level,
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        if log_format == "json":
            formatter = jsonlogger.JsonFormatter(
                "%(timestamp)s %(name)s %(level)s %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


# Context management for request tracing
class LogContext:
    """Context manager for adding context to logs."""

    def __init__(self, **kwargs: Any):
        """
        Initialize log context.

        Args:
            **kwargs: Context key-value pairs to add to logs
        """
        self.context = kwargs

    def __enter__(self) -> "LogContext":
        """Enter context."""
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


# Decorators for logging
def log_execution_time(logger: Optional[structlog.stdlib.BoundLogger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Optional logger instance
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                _logger.info(
                    f"{func.__name__} completed",
                    execution_time=execution_time,
                    function=func.__name__,
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(
                    f"{func.__name__} failed",
                    execution_time=execution_time,
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_async_execution_time(logger: Optional[structlog.stdlib.BoundLogger] = None):
    """
    Decorator to log async function execution time.

    Args:
        logger: Optional logger instance
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                _logger.info(
                    f"{func.__name__} completed",
                    execution_time=execution_time,
                    function=func.__name__,
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(
                    f"{func.__name__} failed",
                    execution_time=execution_time,
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator
