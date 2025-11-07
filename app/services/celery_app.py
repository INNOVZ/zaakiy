"""
Celery application configuration for distributed task processing.

This module provides:
- Celery app instance with Redis broker and result backend
- Task configuration with retry logic
- Support for async tasks using celery-async
"""
import logging
import os
from typing import Any, Dict
from urllib.parse import quote, urlparse, urlunparse

from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# Get Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Use separate Redis DBs for broker and results to avoid conflicts
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)


def add_password_to_redis_url(url: str, password: str) -> str:
    """
    Add password to Redis URL with proper URL encoding.

    Handles passwords containing special characters (@, /, %, :, etc.)
    by using urllib.parse for proper URL construction.

    Args:
        url: Redis URL (e.g., "redis://host:6379/0")
        password: Password to add (may contain special characters)

    Returns:
        URL with properly encoded password
    """
    if not password:
        return url

    # Parse the URL
    parsed = urlparse(url)

    # If password already in URL, return as-is
    if parsed.password:
        return url

    # URL-encode the password to handle special characters
    encoded_password = quote(password, safe="")

    # Reconstruct URL with password, preserving username if present
    # Format: redis://username:password@host:port/path or redis://:password@host:port/path
    if parsed.username:
        # Preserve existing username
        encoded_username = quote(parsed.username, safe="")
        netloc = f"{encoded_username}:{encoded_password}@{parsed.hostname}"
    else:
        # No username, just add password
        netloc = f":{encoded_password}@{parsed.hostname}"

    if parsed.port:
        netloc += f":{parsed.port}"

    new_url = urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return new_url


# Add password to URLs if provided (with proper URL encoding)
if REDIS_PASSWORD:
    if "://" in CELERY_BROKER_URL and "@" not in CELERY_BROKER_URL:
        CELERY_BROKER_URL = add_password_to_redis_url(CELERY_BROKER_URL, REDIS_PASSWORD)
    if "://" in CELERY_RESULT_BACKEND and "@" not in CELERY_RESULT_BACKEND:
        CELERY_RESULT_BACKEND = add_password_to_redis_url(
            CELERY_RESULT_BACKEND, REDIS_PASSWORD
        )

# Create Celery app
celery_app = Celery(
    "zaakiy",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.services.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,  # Re-queue tasks if worker dies
    worker_prefetch_multiplier=1,  # Process one task at a time for fairness
    # Prevent multiple instances of singleton tasks from running concurrently
    # Note: We use distributed locks in task code for process_pending_uploads
    # This setting helps but distributed lock is the primary protection
    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    # Broker transport options (for regular Redis)
    broker_transport_options={
        "visibility_timeout": 3600,  # Task visibility timeout in seconds
    },
    # Retry configuration
    task_default_retry_delay=60,  # Retry after 60 seconds
    task_max_retries=3,
    # Task routes (if needed)
    task_routes={
        "app.services.tasks.process_upload": {"queue": "uploads"},
        "app.services.tasks.process_pending_uploads": {"queue": "uploads"},
        "app.services.tasks.reindex_upload": {"queue": "uploads"},
    },
    # Worker settings
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks (memory leak prevention)
    worker_disable_rate_limits=False,
    # Beat schedule (periodic tasks)
    beat_schedule={
        "process-pending-uploads": {
            "task": "app.services.tasks.process_pending_uploads",
            "schedule": 30.0,  # Every 30 seconds
            "options": {"queue": "uploads"},
        },
        # Add more scheduled tasks here as needed
        # "cleanup-old-results": {
        #     "task": "app.services.tasks.cleanup_old_results",
        #     "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        # },
    },
    # Task result compression (for large results)
    task_compression="gzip",
    result_compression="gzip",
)

logger.info(
    "Celery app configured",
    extra={
        "broker": CELERY_BROKER_URL.split("@")[-1]
        if "@" in CELERY_BROKER_URL
        else CELERY_BROKER_URL,
        "backend": CELERY_RESULT_BACKEND.split("@")[-1]
        if "@" in CELERY_RESULT_BACKEND
        else CELERY_RESULT_BACKEND,
    },
)
