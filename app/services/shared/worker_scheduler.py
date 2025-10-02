"# services/worker_scheduler.py\n\n"
# This module manages the background worker for processing uploads using APScheduler.   

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from ..scraping.ingestion_worker import process_pending_uploads


class IngestionWorkerScheduler:
    """Scheduler to manage background worker for processing uploads"""
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False

    def start(self):
        """Start the background worker scheduler"""
        if not self.is_running:
            # Process pending uploads every 30 seconds
            self.scheduler.add_job(
                process_pending_uploads,
                IntervalTrigger(seconds=30),
                id='process_uploads',
                replace_existing=True
            )

            self.scheduler.start()
            self.is_running = True
            print("[Info] Ingestion worker scheduler started")

    def stop(self):
        """Stop the background worker scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            print("[Info] Ingestion worker scheduler stopped")


# Global scheduler instance
worker_scheduler = IngestionWorkerScheduler()


def start_background_worker():
    """Function to start the background worker"""
    worker_scheduler.start()


def stop_background_worker():
    """Function to stop the background worker"""
    worker_scheduler.stop()
