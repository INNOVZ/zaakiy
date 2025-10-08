"# services/worker_scheduler.py\n\n"
# This module manages the background worker for processing uploads using APScheduler.

import threading
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from ..scraping.ingestion_worker import process_pending_uploads

logger = logging.getLogger(__name__)


class IngestionWorkerScheduler:
    """
    Thread-safe scheduler to manage background worker for processing uploads

    This class uses a threading lock to prevent race conditions when starting/stopping
    the scheduler from multiple threads or coroutines.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        self._lock = threading.Lock()
        self._start_count = 0
        self._stop_count = 0
        logger.info("IngestionWorkerScheduler initialized")

    def start(self):
        """
        Start the background worker scheduler (thread-safe)

        This method is idempotent - calling it multiple times will only start
        the scheduler once. Uses a lock to prevent race conditions.
        """
        with self._lock:
            if self.is_running:
                logger.warning(
                    "Scheduler start() called but already running "
                    f"(start_count: {self._start_count})"
                )
                return

            try:
                # Process pending uploads every 30 seconds
                self.scheduler.add_job(
                    process_pending_uploads,
                    IntervalTrigger(seconds=30),
                    id='process_uploads',
                    replace_existing=True,
                    max_instances=1,  # Prevent concurrent job execution
                    coalesce=True  # Combine missed runs into one
                )

                self.scheduler.start()
                self.is_running = True
                self._start_count += 1

                logger.info(
                    f"Ingestion worker scheduler started successfully "
                    f"(start_count: {self._start_count})"
                )

            except Exception as e:
                logger.error(f"Failed to start scheduler: {e}", exc_info=True)
                self.is_running = False
                raise

    def stop(self):
        """
        Stop the background worker scheduler (thread-safe)

        This method is idempotent - calling it multiple times will only stop
        the scheduler once. Uses a lock to prevent race conditions.
        """
        with self._lock:
            if not self.is_running:
                logger.warning(
                    "Scheduler stop() called but not running "
                    f"(stop_count: {self._stop_count})"
                )
                return

            try:
                # Shutdown with wait to ensure clean shutdown
                self.scheduler.shutdown(wait=True)
                self.is_running = False
                self._stop_count += 1

                logger.info(
                    f"Ingestion worker scheduler stopped successfully "
                    f"(stop_count: {self._stop_count})"
                )

            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}", exc_info=True)
                # Force state to False even if shutdown failed
                self.is_running = False
                raise

    def get_status(self):
        """
        Get current scheduler status (thread-safe)

        Returns:
            dict: Status information including running state and statistics
        """
        with self._lock:
            return {
                "is_running": self.is_running,
                "start_count": self._start_count,
                "stop_count": self._stop_count,
                "scheduler_state": self.scheduler.state if self.scheduler else None,
                "jobs": [
                    {
                        "id": job.id,
                        "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None
                    }
                    for job in self.scheduler.get_jobs()
                ] if self.is_running else []
            }

    def restart(self):
        """
        Restart the scheduler (thread-safe)

        Useful for applying configuration changes or recovering from errors.
        """
        logger.info("Restarting scheduler...")
        self.stop()
        self.start()
        logger.info("Scheduler restarted successfully")


# Global scheduler instance
worker_scheduler = IngestionWorkerScheduler()


def start_background_worker():
    """Function to start the background worker"""
    worker_scheduler.start()


def stop_background_worker():
    """Function to stop the background worker"""
    worker_scheduler.stop()
