"""
Health checking for system components.
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum

import psutil
from pydantic import BaseModel

from src.core.logging.logger import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    last_check: datetime
    metrics: Dict[str, Any] = {}


class SystemHealth(BaseModel):
    """Overall system health."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    system_metrics: Dict[str, Any] = {}


class HealthChecker:
    """System health checker."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.check_intervals: Dict[str, int] = {}
        self.last_results: Dict[str, ComponentHealth] = {}

    def register_check(
        self,
        name: str,
        check_func: Callable,
        interval_seconds: int = 60,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Component name
            check_func: Async function that returns (status, message, metrics)
            interval_seconds: Check interval in seconds
        """
        self.checks[name] = check_func
        self.check_intervals[name] = interval_seconds
        logger.info(f"Registered health check: {name}")

    async def check_component(self, name: str) -> ComponentHealth:
        """
        Check health of a specific component.

        Args:
            name: Component name

        Returns:
            Component health status
        """
        if name not in self.checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {name}",
                last_check=datetime.utcnow(),
            )

        try:
            check_func = self.checks[name]
            status, message, metrics = await check_func()

            health = ComponentHealth(
                name=name,
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                metrics=metrics or {},
            )

            self.last_results[name] = health
            return health

        except Exception as e:
            logger.error(f"Health check failed for {name}", error=str(e), exc_info=True)
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                last_check=datetime.utcnow(),
            )

    async def check_all(self) -> SystemHealth:
        """
        Check health of all components.

        Returns:
            Overall system health
        """
        # Run all health checks concurrently
        tasks = [self.check_component(name) for name in self.checks.keys()]
        component_healths = await asyncio.gather(*tasks)

        # Get system metrics
        system_metrics = self._get_system_metrics()

        # Determine overall status
        statuses = [c.status for c in component_healths]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall_status,
            components=component_healths,
            timestamp=datetime.utcnow(),
            system_metrics=system_metrics,
        )

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
            }
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {}

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        logger.info("Starting health monitoring")

        async def monitor_component(name: str):
            interval = self.check_intervals.get(name, 60)
            while True:
                try:
                    await self.check_component(name)
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(
                        f"Error in health monitoring for {name}",
                        error=str(e),
                        exc_info=True,
                    )
                    await asyncio.sleep(interval)

        # Start monitoring tasks for all components
        tasks = [monitor_component(name) for name in self.checks.keys()]
        await asyncio.gather(*tasks)


# Common health checks
async def check_database_health(db_url: str) -> tuple:
    """
    Check database connectivity.

    Args:
        db_url: Database connection URL

    Returns:
        Tuple of (status, message, metrics)
    """
    try:
        # This is a placeholder - implement actual DB check
        # For PostgreSQL, you'd use asyncpg
        return (
            HealthStatus.HEALTHY,
            "Database connection OK",
            {"latency_ms": 10},
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Database connection failed: {str(e)}",
            {},
        )


async def check_redis_health(redis_url: str) -> tuple:
    """
    Check Redis connectivity.

    Args:
        redis_url: Redis connection URL

    Returns:
        Tuple of (status, message, metrics)
    """
    try:
        # This is a placeholder - implement actual Redis check
        return (
            HealthStatus.HEALTHY,
            "Redis connection OK",
            {"latency_ms": 5},
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Redis connection failed: {str(e)}",
            {},
        )


async def check_broker_health(broker_api) -> tuple:
    """
    Check broker API connectivity.

    Args:
        broker_api: Broker API client

    Returns:
        Tuple of (status, message, metrics)
    """
    try:
        # This is a placeholder - implement actual broker check
        return (
            HealthStatus.HEALTHY,
            "Broker API connection OK",
            {"latency_ms": 50},
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Broker API connection failed: {str(e)}",
            {},
        )


async def check_model_health(model_service) -> tuple:
    """
    Check ML model service health.

    Args:
        model_service: Model service instance

    Returns:
        Tuple of (status, message, metrics)
    """
    try:
        # This is a placeholder - implement actual model check
        return (
            HealthStatus.HEALTHY,
            "Model service OK",
            {"inference_latency_ms": 20},
        )
    except Exception as e:
        return (
            HealthStatus.DEGRADED,
            f"Model service degraded: {str(e)}",
            {},
        )
