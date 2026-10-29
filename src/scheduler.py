"""
Job scheduling system for periodic tasks.
Handles model retraining, portfolio rebalancing, data refresh, risk monitoring, and reporting.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.job import Job
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from src.config.settings import Settings, get_settings
from src.core.utils.date_utils import get_market_calendar

logger = logging.getLogger(__name__)


class JobMetrics:
    """Job execution metrics."""

    def __init__(self, name: str):
        """
        Initialize job metrics.

        Args:
            name: Job name
        """
        self.name = name
        self.execution_count = 0
        self.error_count = 0
        self.last_execution: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.total_execution_time = 0.0
        self.avg_execution_time = 0.0

    def record_execution(self, duration: float, error: Optional[str] = None) -> None:
        """
        Record job execution.

        Args:
            duration: Execution duration in seconds
            error: Optional error message
        """
        self.execution_count += 1
        self.last_execution = datetime.now()
        self.total_execution_time += duration
        self.avg_execution_time = self.total_execution_time / self.execution_count

        if error:
            self.error_count += 1
            self.last_error = error


class SystemScheduler:
    """System job scheduler."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize scheduler.

        Args:
            settings: Settings instance
        """
        self.settings = settings or get_settings()
        self.scheduler = AsyncIOScheduler()
        self.jobs: Dict[str, Job] = {}
        self.metrics: Dict[str, JobMetrics] = {}
        self.market_calendar = get_market_calendar()
        self._running = False

    async def initialize(self) -> None:
        """Initialize the scheduler and register jobs."""
        logger.info("Initializing system scheduler...")

        try:
            # Register event listeners
            self.scheduler.add_listener(
                self._job_executed_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )

            # Register all jobs
            self._register_jobs()

            logger.info(f"Scheduler initialized with {len(self.jobs)} jobs")

        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}", exc_info=True)
            raise

    def _register_jobs(self) -> None:
        """Register all scheduled jobs."""
        logger.info("Registering scheduled jobs...")

        # Model retraining (daily at 1 AM)
        self.add_job(
            "model_retraining",
            self._run_model_retraining,
            CronTrigger(hour=1, minute=0),
            description="Retrain ML models"
        )

        # Portfolio rebalancing (daily at market close)
        self.add_job(
            "portfolio_rebalancing",
            self._run_portfolio_rebalancing,
            CronTrigger(hour=16, minute=5),  # 5 minutes after close
            description="Rebalance portfolio"
        )

        # Data refresh (every hour during market hours)
        self.add_job(
            "data_refresh",
            self._run_data_refresh,
            IntervalTrigger(hours=1),
            description="Refresh market data"
        )

        # Risk monitoring (every 15 minutes during market hours)
        self.add_job(
            "risk_monitoring",
            self._run_risk_monitoring,
            IntervalTrigger(minutes=15),
            description="Monitor risk metrics"
        )

        # Performance report (daily at 5 PM)
        self.add_job(
            "daily_report",
            self._generate_daily_report,
            CronTrigger(hour=17, minute=0),
            description="Generate daily performance report"
        )

        # Weekly report (Sundays at 6 PM)
        self.add_job(
            "weekly_report",
            self._generate_weekly_report,
            CronTrigger(day_of_week="sun", hour=18, minute=0),
            description="Generate weekly performance report"
        )

        # Database cleanup (daily at 2 AM)
        self.add_job(
            "database_cleanup",
            self._run_database_cleanup,
            CronTrigger(hour=2, minute=0),
            description="Clean up old database records"
        )

        # Cache cleanup (every 6 hours)
        self.add_job(
            "cache_cleanup",
            self._run_cache_cleanup,
            IntervalTrigger(hours=6),
            description="Clean up expired cache entries"
        )

        # Health check (every 5 minutes)
        self.add_job(
            "health_check",
            self._run_health_check,
            IntervalTrigger(minutes=5),
            description="System health check"
        )

        # Market data backup (daily at 6 AM)
        self.add_job(
            "data_backup",
            self._run_data_backup,
            CronTrigger(hour=6, minute=0),
            description="Backup market data"
        )

        logger.info(f"Registered {len(self.jobs)} scheduled jobs")

    def add_job(
        self,
        job_id: str,
        func: Callable,
        trigger: Any,
        description: str = "",
        **kwargs
    ) -> Job:
        """
        Add a scheduled job.

        Args:
            job_id: Unique job ID
            func: Job function to execute
            trigger: APScheduler trigger
            description: Job description
            **kwargs: Additional job arguments

        Returns:
            Job instance
        """
        job = self.scheduler.add_job(
            func,
            trigger,
            id=job_id,
            name=description or job_id,
            **kwargs
        )

        self.jobs[job_id] = job
        self.metrics[job_id] = JobMetrics(job_id)

        logger.debug(f"Added job: {job_id} - {description}")
        return job

    def remove_job(self, job_id: str) -> None:
        """
        Remove a scheduled job.

        Args:
            job_id: Job ID to remove
        """
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.debug(f"Removed job: {job_id}")

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        logger.info("Starting scheduler...")

        try:
            self.scheduler.start()
            self._running = True
            logger.info("Scheduler started successfully")

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Shutdown the scheduler."""
        if not self._running:
            return

        logger.info("Shutting down scheduler...")

        try:
            self.scheduler.shutdown(wait=True)
            self._running = False
            logger.info("Scheduler shut down successfully")

        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}", exc_info=True)
            raise

    def pause_job(self, job_id: str) -> None:
        """
        Pause a job.

        Args:
            job_id: Job ID to pause
        """
        if job_id in self.jobs:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")

    def resume_job(self, job_id: str) -> None:
        """
        Resume a paused job.

        Args:
            job_id: Job ID to resume
        """
        if job_id in self.jobs:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")

    def get_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of all jobs.

        Returns:
            List of job information
        """
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "metrics": {
                    "executions": self.metrics[job.id].execution_count,
                    "errors": self.metrics[job.id].error_count,
                    "last_execution": (
                        self.metrics[job.id].last_execution.isoformat()
                        if self.metrics[job.id].last_execution else None
                    ),
                    "avg_execution_time": self.metrics[job.id].avg_execution_time
                }
            }
            for job in self.jobs.values()
        ]

    def _job_executed_listener(self, event) -> None:
        """
        Listen to job execution events.

        Args:
            event: Job event
        """
        job_id = event.job_id

        if job_id not in self.metrics:
            return

        # Record execution
        if event.exception:
            error_msg = str(event.exception)
            self.metrics[job_id].record_execution(0.0, error=error_msg)
            logger.error(f"Job {job_id} failed: {error_msg}")
        else:
            duration = (event.scheduled_run_time - datetime.now()).total_seconds()
            self.metrics[job_id].record_execution(abs(duration))

    # Job execution methods
    async def _run_model_retraining(self) -> None:
        """Run model retraining job."""
        logger.info("Running model retraining job...")

        try:
            # TODO: Implement model retraining logic
            # 1. Load training data
            # 2. Train models
            # 3. Validate models
            # 4. Deploy new models

            logger.info("Model retraining completed successfully")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}", exc_info=True)
            raise

    async def _run_portfolio_rebalancing(self) -> None:
        """Run portfolio rebalancing job."""
        logger.info("Running portfolio rebalancing job...")

        try:
            # TODO: Implement portfolio rebalancing logic
            # 1. Get current positions
            # 2. Calculate target allocations
            # 3. Generate rebalancing orders
            # 4. Execute orders

            logger.info("Portfolio rebalancing completed successfully")

        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}", exc_info=True)
            raise

    async def _run_data_refresh(self) -> None:
        """Run data refresh job."""
        logger.info("Running data refresh job...")

        try:
            # Only run during market hours
            if not self.market_calendar.is_market_open():
                logger.debug("Market closed, skipping data refresh")
                return

            # TODO: Implement data refresh logic
            # 1. Fetch latest market data
            # 2. Update database
            # 3. Invalidate caches

            logger.info("Data refresh completed successfully")

        except Exception as e:
            logger.error(f"Data refresh failed: {e}", exc_info=True)
            raise

    async def _run_risk_monitoring(self) -> None:
        """Run risk monitoring job."""
        logger.info("Running risk monitoring job...")

        try:
            # TODO: Implement risk monitoring logic
            # 1. Calculate portfolio risk metrics
            # 2. Check risk limits
            # 3. Send alerts if needed

            logger.info("Risk monitoring completed successfully")

        except Exception as e:
            logger.error(f"Risk monitoring failed: {e}", exc_info=True)
            raise

    async def _generate_daily_report(self) -> None:
        """Generate daily performance report."""
        logger.info("Generating daily report...")

        try:
            # TODO: Implement daily report generation
            # 1. Collect performance metrics
            # 2. Generate report
            # 3. Send notifications

            logger.info("Daily report generated successfully")

        except Exception as e:
            logger.error(f"Daily report generation failed: {e}", exc_info=True)
            raise

    async def _generate_weekly_report(self) -> None:
        """Generate weekly performance report."""
        logger.info("Generating weekly report...")

        try:
            # TODO: Implement weekly report generation
            # 1. Collect weekly metrics
            # 2. Generate comprehensive report
            # 3. Send notifications

            logger.info("Weekly report generated successfully")

        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}", exc_info=True)
            raise

    async def _run_database_cleanup(self) -> None:
        """Run database cleanup job."""
        logger.info("Running database cleanup...")

        try:
            # TODO: Implement database cleanup logic
            # 1. Delete old data
            # 2. Vacuum database
            # 3. Update statistics

            logger.info("Database cleanup completed successfully")

        except Exception as e:
            logger.error(f"Database cleanup failed: {e}", exc_info=True)
            raise

    async def _run_cache_cleanup(self) -> None:
        """Run cache cleanup job."""
        logger.info("Running cache cleanup...")

        try:
            # TODO: Implement cache cleanup logic
            # 1. Remove expired entries
            # 2. Compact cache storage

            logger.info("Cache cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}", exc_info=True)
            raise

    async def _run_health_check(self) -> None:
        """Run system health check."""
        logger.debug("Running health check...")

        try:
            # TODO: Implement health check logic
            # 1. Check database connectivity
            # 2. Check data feeds
            # 3. Check component health
            # 4. Send alerts if unhealthy

            logger.debug("Health check completed")

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            raise

    async def _run_data_backup(self) -> None:
        """Run data backup job."""
        logger.info("Running data backup...")

        try:
            # TODO: Implement data backup logic
            # 1. Backup databases
            # 2. Backup configuration
            # 3. Upload to cloud storage

            logger.info("Data backup completed successfully")

        except Exception as e:
            logger.error(f"Data backup failed: {e}", exc_info=True)
            raise
