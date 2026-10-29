"""
System orchestrator for managing component lifecycle, health monitoring,
automatic restarts, and graceful degradation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import traceback

from src.config.settings import Settings, get_settings
from src.scheduler import SystemScheduler

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    DEGRADED = "degraded"


class ComponentHealth:
    """Component health information."""

    def __init__(self, name: str):
        """
        Initialize component health.

        Args:
            name: Component name
        """
        self.name = name
        self.status = ComponentStatus.STOPPED
        self.last_heartbeat: Optional[datetime] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.restart_count = 0
        self.start_time: Optional[datetime] = None

    def update_heartbeat(self) -> None:
        """Update last heartbeat time."""
        self.last_heartbeat = datetime.now()

    def record_error(self, error: str) -> None:
        """
        Record an error.

        Args:
            error: Error message
        """
        self.error_count += 1
        self.last_error = error

    def record_restart(self) -> None:
        """Record a restart."""
        self.restart_count += 1

    def is_healthy(self, timeout: int = 60) -> bool:
        """
        Check if component is healthy.

        Args:
            timeout: Heartbeat timeout in seconds

        Returns:
            True if healthy
        """
        if self.status != ComponentStatus.RUNNING:
            return False

        if self.last_heartbeat is None:
            return False

        age = (datetime.now() - self.last_heartbeat).total_seconds()
        return age < timeout


class ManagedComponent:
    """Managed system component."""

    def __init__(
        self,
        name: str,
        initialize: Callable,
        start: Callable,
        stop: Callable,
        health_check: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        critical: bool = True,
        auto_restart: bool = True,
        max_restarts: int = 3
    ):
        """
        Initialize managed component.

        Args:
            name: Component name
            initialize: Initialization function
            start: Start function
            stop: Stop function
            health_check: Optional health check function
            dependencies: List of component dependencies
            critical: Whether component is critical
            auto_restart: Whether to auto-restart on failure
            max_restarts: Maximum restart attempts
        """
        self.name = name
        self.initialize = initialize
        self.start = start
        self.stop = stop
        self.health_check = health_check
        self.dependencies = dependencies or []
        self.critical = critical
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.health = ComponentHealth(name)
        self.instance: Optional[Any] = None


class SystemOrchestrator:
    """System orchestrator for managing all components."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize system orchestrator.

        Args:
            settings: Settings instance
        """
        self.settings = settings or get_settings()
        self.components: Dict[str, ManagedComponent] = {}
        self.scheduler: Optional[SystemScheduler] = None
        self.running = False
        self.health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the orchestrator and register components."""
        logger.info("Initializing system orchestrator...")

        try:
            # Register all components
            await self._register_components()

            # Initialize scheduler
            self.scheduler = SystemScheduler(settings=self.settings)
            await self.scheduler.initialize()

            logger.info("System orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
            raise

    async def _register_components(self) -> None:
        """Register all system components."""
        logger.info("Registering system components...")

        # Database component
        self.register_component(
            ManagedComponent(
                name="database",
                initialize=self._init_database,
                start=self._start_database,
                stop=self._stop_database,
                health_check=self._check_database_health,
                critical=True,
                auto_restart=True
            )
        )

        # Data collectors component
        self.register_component(
            ManagedComponent(
                name="data_collectors",
                initialize=self._init_data_collectors,
                start=self._start_data_collectors,
                stop=self._stop_data_collectors,
                health_check=self._check_data_collectors_health,
                dependencies=["database"],
                critical=True,
                auto_restart=True
            )
        )

        # Feature engineering component
        self.register_component(
            ManagedComponent(
                name="feature_engine",
                initialize=self._init_feature_engine,
                start=self._start_feature_engine,
                stop=self._stop_feature_engine,
                dependencies=["data_collectors"],
                critical=False,
                auto_restart=True
            )
        )

        # Agents component
        self.register_component(
            ManagedComponent(
                name="agents",
                initialize=self._init_agents,
                start=self._start_agents,
                stop=self._stop_agents,
                health_check=self._check_agents_health,
                dependencies=["data_collectors", "feature_engine"],
                critical=True,
                auto_restart=True
            )
        )

        # Strategy engines component
        self.register_component(
            ManagedComponent(
                name="strategy_engines",
                initialize=self._init_strategy_engines,
                start=self._start_strategy_engines,
                stop=self._stop_strategy_engines,
                dependencies=["agents"],
                critical=True,
                auto_restart=True
            )
        )

        # Risk monitoring component
        self.register_component(
            ManagedComponent(
                name="risk_monitor",
                initialize=self._init_risk_monitor,
                start=self._start_risk_monitor,
                stop=self._stop_risk_monitor,
                health_check=self._check_risk_monitor_health,
                dependencies=["database"],
                critical=True,
                auto_restart=True
            )
        )

        # Execution engine component
        self.register_component(
            ManagedComponent(
                name="execution_engine",
                initialize=self._init_execution_engine,
                start=self._start_execution_engine,
                stop=self._stop_execution_engine,
                dependencies=["database", "risk_monitor"],
                critical=True,
                auto_restart=True
            )
        )

        # API servers component
        self.register_component(
            ManagedComponent(
                name="api_servers",
                initialize=self._init_api_servers,
                start=self._start_api_servers,
                stop=self._stop_api_servers,
                dependencies=["database"],
                critical=False,
                auto_restart=True
            )
        )

        logger.info(f"Registered {len(self.components)} components")

    def register_component(self, component: ManagedComponent) -> None:
        """
        Register a component.

        Args:
            component: Component to register
        """
        self.components[component.name] = component
        logger.debug(f"Registered component: {component.name}")

    async def start(self) -> None:
        """Start all components in dependency order."""
        if self.running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting system orchestrator...")

        try:
            # Get start order based on dependencies
            start_order = self._get_start_order()

            # Initialize components
            for name in start_order:
                await self._initialize_component(name)

            # Start components
            for name in start_order:
                await self._start_component(name)

            # Start health monitoring
            self.health_check_task = asyncio.create_task(
                self._health_monitor_loop()
            )

            # Start scheduler
            if self.scheduler:
                await self.scheduler.start()

            self.running = True
            logger.info("System orchestrator started successfully")

        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}", exc_info=True)
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        if not self.running:
            return

        logger.info("Shutting down system orchestrator...")

        try:
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass

            # Stop scheduler
            if self.scheduler:
                await self.scheduler.shutdown()

            # Get shutdown order (reverse of start order)
            shutdown_order = list(reversed(self._get_start_order()))

            # Stop components
            for name in shutdown_order:
                await self._stop_component(name)

            self.running = False
            logger.info("System orchestrator shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise

    def _get_start_order(self) -> List[str]:
        """
        Get component start order based on dependencies.

        Returns:
            List of component names in start order
        """
        order = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            component = self.components[name]
            for dep in component.dependencies:
                if dep in self.components:
                    visit(dep)

            order.append(name)

        for name in self.components:
            visit(name)

        return order

    async def _initialize_component(self, name: str) -> None:
        """
        Initialize a component.

        Args:
            name: Component name
        """
        component = self.components[name]
        logger.info(f"Initializing component: {name}")

        try:
            component.health.status = ComponentStatus.STARTING
            component.instance = await component.initialize()
            logger.info(f"Component initialized: {name}")

        except Exception as e:
            logger.error(f"Failed to initialize component {name}: {e}", exc_info=True)
            component.health.status = ComponentStatus.FAILED
            component.health.record_error(str(e))
            raise

    async def _start_component(self, name: str) -> None:
        """
        Start a component.

        Args:
            name: Component name
        """
        component = self.components[name]
        logger.info(f"Starting component: {name}")

        try:
            component.health.status = ComponentStatus.STARTING
            component.health.start_time = datetime.now()

            await component.start(component.instance)

            component.health.status = ComponentStatus.RUNNING
            component.health.update_heartbeat()

            logger.info(f"Component started: {name}")

        except Exception as e:
            logger.error(f"Failed to start component {name}: {e}", exc_info=True)
            component.health.status = ComponentStatus.FAILED
            component.health.record_error(str(e))

            if component.critical:
                raise

    async def _stop_component(self, name: str) -> None:
        """
        Stop a component.

        Args:
            name: Component name
        """
        component = self.components[name]

        if component.health.status == ComponentStatus.STOPPED:
            return

        logger.info(f"Stopping component: {name}")

        try:
            component.health.status = ComponentStatus.STOPPING

            await component.stop(component.instance)

            component.health.status = ComponentStatus.STOPPED
            logger.info(f"Component stopped: {name}")

        except Exception as e:
            logger.error(f"Error stopping component {name}: {e}", exc_info=True)
            component.health.status = ComponentStatus.FAILED
            component.health.record_error(str(e))

    async def _restart_component(self, name: str) -> None:
        """
        Restart a component.

        Args:
            name: Component name
        """
        component = self.components[name]

        if component.health.restart_count >= component.max_restarts:
            logger.error(
                f"Component {name} exceeded max restarts "
                f"({component.max_restarts}), giving up"
            )
            component.health.status = ComponentStatus.FAILED
            return

        logger.warning(f"Restarting component: {name}")

        try:
            # Stop component
            await self._stop_component(name)

            # Wait before restart
            await asyncio.sleep(5)

            # Start component
            await self._start_component(name)

            component.health.record_restart()

        except Exception as e:
            logger.error(f"Failed to restart component {name}: {e}", exc_info=True)
            component.health.status = ComponentStatus.FAILED
            component.health.record_error(str(e))

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        logger.info("Starting health monitor loop")

        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for name, component in self.components.items():
                    if component.health.status != ComponentStatus.RUNNING:
                        continue

                    # Run health check if available
                    if component.health_check:
                        try:
                            healthy = await component.health_check(component.instance)
                            if healthy:
                                component.health.update_heartbeat()
                            else:
                                logger.warning(f"Component {name} health check failed")
                                if component.auto_restart:
                                    await self._restart_component(name)
                        except Exception as e:
                            logger.error(
                                f"Health check error for {name}: {e}",
                                exc_info=True
                            )
                            component.health.record_error(str(e))

                    # Check heartbeat timeout
                    if not component.health.is_healthy():
                        logger.warning(f"Component {name} heartbeat timeout")
                        if component.auto_restart:
                            await self._restart_component(name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}", exc_info=True)

        logger.info("Health monitor loop stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.

        Returns:
            System status dictionary
        """
        return {
            "running": self.running,
            "components": {
                name: {
                    "status": component.health.status.value,
                    "last_heartbeat": component.health.last_heartbeat.isoformat()
                    if component.health.last_heartbeat else None,
                    "error_count": component.health.error_count,
                    "restart_count": component.health.restart_count,
                    "uptime": (
                        (datetime.now() - component.health.start_time).total_seconds()
                        if component.health.start_time else 0
                    )
                }
                for name, component in self.components.items()
            }
        }

    # Component initialization methods (placeholders)
    async def _init_database(self) -> Any:
        """Initialize database."""
        logger.info("Initializing database connections...")
        # TODO: Implement actual database initialization
        return None

    async def _start_database(self, instance: Any) -> None:
        """Start database."""
        pass

    async def _stop_database(self, instance: Any) -> None:
        """Stop database."""
        pass

    async def _check_database_health(self, instance: Any) -> bool:
        """Check database health."""
        return True

    async def _init_data_collectors(self) -> Any:
        """Initialize data collectors."""
        logger.info("Initializing data collectors...")
        # TODO: Implement actual data collector initialization
        return None

    async def _start_data_collectors(self, instance: Any) -> None:
        """Start data collectors."""
        pass

    async def _stop_data_collectors(self, instance: Any) -> None:
        """Stop data collectors."""
        pass

    async def _check_data_collectors_health(self, instance: Any) -> bool:
        """Check data collectors health."""
        return True

    async def _init_feature_engine(self) -> Any:
        """Initialize feature engine."""
        logger.info("Initializing feature engine...")
        return None

    async def _start_feature_engine(self, instance: Any) -> None:
        """Start feature engine."""
        pass

    async def _stop_feature_engine(self, instance: Any) -> None:
        """Stop feature engine."""
        pass

    async def _init_agents(self) -> Any:
        """Initialize agents."""
        logger.info("Initializing trading agents...")
        return None

    async def _start_agents(self, instance: Any) -> None:
        """Start agents."""
        pass

    async def _stop_agents(self, instance: Any) -> None:
        """Stop agents."""
        pass

    async def _check_agents_health(self, instance: Any) -> bool:
        """Check agents health."""
        return True

    async def _init_strategy_engines(self) -> Any:
        """Initialize strategy engines."""
        logger.info("Initializing strategy engines...")
        return None

    async def _start_strategy_engines(self, instance: Any) -> None:
        """Start strategy engines."""
        pass

    async def _stop_strategy_engines(self, instance: Any) -> None:
        """Stop strategy engines."""
        pass

    async def _init_risk_monitor(self) -> Any:
        """Initialize risk monitor."""
        logger.info("Initializing risk monitor...")
        return None

    async def _start_risk_monitor(self, instance: Any) -> None:
        """Start risk monitor."""
        pass

    async def _stop_risk_monitor(self, instance: Any) -> None:
        """Stop risk monitor."""
        pass

    async def _check_risk_monitor_health(self, instance: Any) -> bool:
        """Check risk monitor health."""
        return True

    async def _init_execution_engine(self) -> Any:
        """Initialize execution engine."""
        logger.info("Initializing execution engine...")
        return None

    async def _start_execution_engine(self, instance: Any) -> None:
        """Start execution engine."""
        pass

    async def _stop_execution_engine(self, instance: Any) -> None:
        """Stop execution engine."""
        pass

    async def _init_api_servers(self) -> Any:
        """Initialize API servers."""
        logger.info("Initializing API servers...")
        return None

    async def _start_api_servers(self, instance: Any) -> None:
        """Start API servers."""
        pass

    async def _stop_api_servers(self, instance: Any) -> None:
        """Stop API servers."""
        pass
