"""
Celery tasks for background processing.

This module contains all Celery tasks for:
- Upload processing (PDF, JSON, URL)
- Document ingestion and vectorization
- Scheduled maintenance tasks
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from celery import shared_task
from celery.exceptions import Retry

from .scraping.ingestion_worker import (
    process_pending_uploads as async_process_pending_uploads,
)
from .shared.distributed_lock import DistributedLock, get_redis_client_for_lock

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run an async coroutine in a new event loop"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # If loop is already running, create a new task
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)


@shared_task(
    bind=True,
    name="app.services.tasks.process_pending_uploads",
    max_retries=3,
    default_retry_delay=60,
    queue="uploads",
)
def process_pending_uploads(self):
    """
    Process all pending uploads from the database.

    This task is scheduled to run every 30 seconds via Celery Beat.
    It processes PDFs, JSON files, and URLs, extracts text, generates embeddings,
    and stores them in Pinecone.

    CRITICAL: Uses distributed lock to prevent concurrent execution across workers.
    This prevents duplicate ingestion when multiple workers process the same backlog.

    Returns:
        dict: Processing statistics
    """
    # Get Redis client for distributed lock
    redis_client = get_redis_client_for_lock()

    if not redis_client:
        logger.warning(
            "Redis unavailable for distributed lock - proceeding without lock protection. "
            "This may cause duplicate ingestion if multiple workers run concurrently."
        )
        # Fall back to execution without lock (not ideal but better than failing)
        try:
            logger.info("Starting pending uploads processing task (no lock)")
            run_async(async_process_pending_uploads())
            logger.info("Completed pending uploads processing task")
            return {"status": "success", "message": "Uploads processed successfully"}
        except Exception as exc:
            logger.error(
                f"Error processing pending uploads: {exc}",
                exc_info=True,
                extra={"retry_count": self.request.retries},
            )
            if self.request.retries < self.max_retries:
                raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
            else:
                logger.error(
                    f"Max retries ({self.max_retries}) exceeded for process_pending_uploads"
                )
                raise

    # Use distributed lock to prevent concurrent execution
    lock = DistributedLock(
        redis_client=redis_client,
        lock_key="process_pending_uploads",
        timeout=300,  # 5 minute lock timeout (safety net for stuck tasks)
    )

    # Track whether we successfully acquired the lock
    lock_acquired = False

    # Try to acquire lock (non-blocking, fail fast if already running)
    if not lock.acquire(blocking=False, timeout=0):
        logger.info(
            "process_pending_uploads already running in another worker - skipping this execution",
            extra={
                "task_id": self.request.id,
                "lock_key": lock.lock_key,
            },
        )
        return {
            "status": "skipped",
            "message": "Task already running in another worker",
            "reason": "distributed_lock_held",
        }

    # Lock was successfully acquired
    lock_acquired = True

    try:
        logger.info(
            "Starting pending uploads processing task (lock acquired)",
            extra={
                "task_id": self.request.id,
                "lock_key": lock.lock_key,
            },
        )

        # Execute the actual processing
        run_async(async_process_pending_uploads())

        logger.info(
            "Completed pending uploads processing task",
            extra={
                "task_id": self.request.id,
                "lock_key": lock.lock_key,
            },
        )

        return {"status": "success", "message": "Uploads processed successfully"}

    except Exception as exc:
        logger.error(
            f"Error processing pending uploads: {exc}",
            exc_info=True,
            extra={
                "retry_count": self.request.retries,
                "task_id": self.request.id,
                "lock_key": lock.lock_key,
            },
        )

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
        else:
            logger.error(
                f"Max retries ({self.max_retries}) exceeded for process_pending_uploads"
            )
            raise

    finally:
        # Only release lock if we successfully acquired it
        if lock_acquired:
            lock.release()
            logger.debug(
                "Released distributed lock",
                extra={
                    "task_id": self.request.id,
                    "lock_key": lock.lock_key,
                },
            )


@shared_task(
    bind=True,
    name="app.services.tasks.process_upload",
    max_retries=3,
    default_retry_delay=60,
    queue="uploads",
)
def process_upload(self, upload_id: str, org_id: Optional[str] = None):
    """
    Process a single upload by ID.

    This task triggers processing of a specific upload by marking it as pending,
    which will be picked up by the scheduled process_pending_uploads task.

    Args:
        upload_id: The upload ID to process
        org_id: Optional organization ID for validation

    Returns:
        dict: Processing result
    """
    try:
        from ..services.storage.supabase_client import get_supabase_client, run_supabase

        logger.info(
            "Queuing upload for processing",
            extra={"upload_id": upload_id, "org_id": org_id},
        )

        supabase = get_supabase_client()

        # Verify upload exists and get details
        async def check_upload():
            return await run_supabase(
                lambda: (
                    supabase.table("uploads").select("*").eq("id", upload_id).execute()
                )
            )

        result = run_async(check_upload())

        if not result.data:
            raise ValueError(f"Upload {upload_id} not found")

        upload = result.data[0]

        # Validate org_id if provided
        if org_id and upload.get("org_id") != org_id:
            raise ValueError(f"Upload {upload_id} does not belong to org {org_id}")

        # Mark as pending - will be processed by scheduled task
        async def mark_pending():
            return await run_supabase(
                lambda: (
                    supabase.table("uploads")
                    .update({"status": "pending", "error_message": None})
                    .eq("id", upload_id)
                    .execute()
                )
            )

        run_async(mark_pending())

        logger.info(
            "Upload queued for processing",
            extra={"upload_id": upload_id},
        )

        return {
            "status": "queued",
            "upload_id": upload_id,
            "message": "Upload queued for processing",
        }

    except Exception as exc:
        logger.error(
            f"Error queueing upload {upload_id}: {exc}",
            exc_info=True,
            extra={"upload_id": upload_id, "retry_count": self.request.retries},
        )

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
        else:
            logger.error(
                f"Max retries ({self.max_retries}) exceeded for upload {upload_id}"
            )
            raise


@shared_task(
    bind=True,
    name="app.services.tasks.reindex_upload",
    max_retries=3,
    default_retry_delay=60,
    queue="uploads",
)
def reindex_upload(self, upload_id: str, org_id: Optional[str] = None):
    """
    Reindex an existing upload (delete old vectors and reprocess).

    Args:
        upload_id: The upload ID to reindex
        org_id: Optional organization ID for validation

    Returns:
        dict: Reindexing result
    """
    try:
        from ..routers.uploads import delete_vectors_from_pinecone
        from ..services.storage.pinecone_client import get_pinecone_index
        from ..services.storage.supabase_client import get_supabase_client, run_supabase

        logger.info(
            "Reindexing upload",
            extra={"upload_id": upload_id, "org_id": org_id},
        )

        supabase = get_supabase_client()

        # Get upload details
        async def get_upload():
            return await run_supabase(
                lambda: (
                    supabase.table("uploads").select("*").eq("id", upload_id).execute()
                )
            )

        upload_result = run_async(get_upload())

        if not upload_result.data:
            raise ValueError(f"Upload {upload_id} not found")

        upload = upload_result.data[0]

        # Validate org_id if provided
        if org_id and upload.get("org_id") != org_id:
            raise ValueError(f"Upload {upload_id} does not belong to org {org_id}")

        namespace = upload["pinecone_namespace"]

        # Delete existing vectors
        delete_vectors_from_pinecone(upload_id, namespace)

        # Reset status to pending for reprocessing
        async def reset_status():
            return await run_supabase(
                lambda: (
                    supabase.table("uploads")
                    .update(
                        {
                            "status": "pending",
                            "error_message": None,
                            "updated_at": "now()",
                        }
                    )
                    .eq("id", upload_id)
                    .execute()
                )
            )

        run_async(reset_status())

        # Trigger processing
        process_upload.delay(upload_id, org_id)

        logger.info(
            "Upload reindexed successfully",
            extra={"upload_id": upload_id},
        )

        return {
            "status": "success",
            "upload_id": upload_id,
            "message": "Upload queued for reindexing",
        }

    except Exception as exc:
        logger.error(
            f"Error reindexing upload {upload_id}: {exc}",
            exc_info=True,
            extra={"upload_id": upload_id, "retry_count": self.request.retries},
        )

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
        else:
            logger.error(
                f"Max retries ({self.max_retries}) exceeded for reindexing upload {upload_id}"
            )
            raise


@shared_task(
    name="app.services.tasks.cleanup_old_results",
    queue="maintenance",
)
def cleanup_old_results():
    """
    Clean up old Celery task results from Redis.

    This task should be scheduled to run periodically (e.g., daily).
    """
    try:
        from .celery_app import celery_app

        # Get result backend
        backend = celery_app.backend

        # Clean up old results (older than 24 hours)
        # This is a simplified version - actual implementation depends on backend type
        logger.info("Cleaning up old task results")
        # Implementation would go here based on backend type
        return {"status": "success", "message": "Cleanup completed"}

    except Exception as exc:
        logger.error(f"Error cleaning up old results: {exc}", exc_info=True)
        raise


# Health check task
@shared_task(name="app.services.tasks.health_check")
def health_check():
    """Simple health check task to verify Celery is working"""
    return {"status": "healthy", "message": "Celery is operational"}
