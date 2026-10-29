"""
Main application entry point for the trading system.
Initializes and orchestrates all components with graceful shutdown handling.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, List
from pathlib import Path
import argparse

from src.config.settings import get_settings
from src.core.logging.setup import setup_logging
from src.core.monitoring.metrics import MetricsCollector
from src.core.utils.cache_utils import init_cache
from src.orchestrator import SystemOrchestrator

logger = logging.getLogger(__name__)


class TradingSystemApplication:
    """Main trading system application."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize application.

        Args:
            config_path: Optional path to config file
        """
        self.settings = get_settings()
        self.config_path = config_path
        self.orchestrator: Optional[SystemOrchestrator] = None
        self.shutdown_event = asyncio.Event()
        self._running = False

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing trading system...")

        try:
            # Initialize cache
            logger.info("Initializing cache...")
            await init_cache(redis_url=self.settings.redis_url)

            # Initialize metrics collector
            logger.info("Initializing metrics collector...")
            metrics_collector = MetricsCollector(
                port=self.settings.prometheus_port
            )
            await metrics_collector.start()

            # Initialize orchestrator
            logger.info("Initializing system orchestrator...")
            self.orchestrator = SystemOrchestrator(settings=self.settings)
            await self.orchestrator.initialize()

            logger.info("Trading system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}", exc_info=True)
            raise

    async def start(self) -> None:
        """Start the trading system."""
        if self._running:
            logger.warning("Trading system is already running")
            return

        logger.info("Starting trading system...")

        try:
            # Initialize components
            await self.initialize()

            # Start orchestrator
            if self.orchestrator:
                await self.orchestrator.start()

            self._running = True
            logger.info("Trading system started successfully")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Error starting trading system: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown the trading system."""
        if not self._running:
            return

        logger.info("Shutting down trading system...")

        try:
            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()

            self._running = False
            logger.info("Trading system shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise

    def signal_handler(self, sig, frame):
        """
        Handle shutdown signals.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {sig}, initiating shutdown...")
        self.shutdown_event.set()

    async def run(self) -> None:
        """Run the trading system."""
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Start the system
        await self.start()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AI-Powered Trading Robot System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="development",
        help="Runtime environment"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--log-format",
        choices=["json", "text"],
        default="json",
        help="Log format"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis cache"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no real trades)"
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtesting mode"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Trading symbols to focus on"
    )

    parser.add_argument(
        "--paper-trading",
        action="store_true",
        default=True,
        help="Use paper trading mode"
    )

    parser.add_argument(
        "--disable-agents",
        nargs="+",
        help="Disable specific agents"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_format=args.log_format,
        service_name="trading-system"
    )

    logger.info("=" * 80)
    logger.info("AI-Powered Trading Robot System")
    logger.info("=" * 80)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Paper Trading: {args.paper_trading}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"Backtesting: {args.backtest}")

    if args.symbols:
        logger.info(f"Symbols: {', '.join(args.symbols)}")

    if args.disable_agents:
        logger.info(f"Disabled Agents: {', '.join(args.disable_agents)}")

    logger.info("=" * 80)

    try:
        # Create application
        app = TradingSystemApplication(config_path=args.config)

        # Run application
        asyncio.run(app.run())

        logger.info("Application exited normally")
        return 0

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
