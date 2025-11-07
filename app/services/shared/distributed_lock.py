"""
Distributed lock implementation for Celery tasks using Redis.

Prevents concurrent execution of singleton tasks like process_pending_uploads.
"""
import logging
import time
import uuid
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class DistributedLock:
    """
    Distributed lock using Redis SET NX (set if not exists).

    Prevents multiple workers from executing the same task concurrently.
    """

    def __init__(self, redis_client: redis.Redis, lock_key: str, timeout: int = 300):
        """
        Initialize distributed lock.

        Args:
            redis_client: Redis client instance
            lock_key: Unique key for the lock
            timeout: Lock timeout in seconds (default: 5 minutes)
        """
        self.redis = redis_client
        self.lock_key = f"celery:lock:{lock_key}"
        self.timeout = timeout
        self.lock_value = None

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: If True, wait for lock to be available
            timeout: Maximum time to wait for lock (seconds)

        Returns:
            True if lock acquired, False otherwise
        """
        if timeout is None:
            timeout = 10.0  # Default 10 second wait

        self.lock_value = str(uuid.uuid4())
        start_time = time.time()

        while True:
            # Try to acquire lock using SET NX (set if not exists)
            # EX sets expiration time
            acquired = self.redis.set(
                self.lock_key,
                self.lock_value,
                nx=True,  # Only set if key doesn't exist
                ex=self.timeout,  # Expire after timeout
            )

            if acquired:
                logger.info(
                    f"Acquired distributed lock: {self.lock_key}",
                    extra={"lock_key": self.lock_key, "lock_value": self.lock_value},
                )
                return True

            if not blocking:
                logger.debug(
                    f"Lock not available (non-blocking): {self.lock_key}",
                    extra={"lock_key": self.lock_key},
                )
                return False

            # Check if we've exceeded timeout
            if time.time() - start_time >= timeout:
                logger.warning(
                    f"Failed to acquire lock within timeout: {self.lock_key}",
                    extra={"lock_key": self.lock_key, "timeout": timeout},
                )
                return False

            # Wait a bit before retrying
            time.sleep(0.1)

    def release(self) -> bool:
        """
        Release the lock.

        Uses Lua script to ensure we only delete our own lock value,
        preventing accidental deletion of another process's lock.

        Returns:
            True if lock released, False otherwise
        """
        if not self.lock_value:
            return False

        # Lua script to atomically check and delete lock
        # Only deletes if the value matches (prevents deleting another process's lock)
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        try:
            result = self.redis.eval(lua_script, 1, self.lock_key, self.lock_value)
            if result:
                logger.info(
                    f"Released distributed lock: {self.lock_key}",
                    extra={"lock_key": self.lock_key},
                )
                return True
            else:
                logger.warning(
                    f"Lock was not owned by this process: {self.lock_key}",
                    extra={"lock_key": self.lock_key},
                )
                return False
        except Exception as e:
            logger.error(
                f"Error releasing lock: {e}",
                extra={"lock_key": self.lock_key},
                exc_info=True,
            )
            return False

    def is_locked(self) -> bool:
        """Check if lock is currently held by any process."""
        return self.redis.exists(self.lock_key) > 0

    def __enter__(self):
        """
        Context manager support - acquire lock on entry.

        Usage:
            with DistributedLock(redis_client, "my_lock") as lock:
                # Lock is acquired here
                # Do work...
            # Lock is automatically released here
        """
        if not self.acquire():
            raise RuntimeError(f"Failed to acquire lock: {self.lock_key}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager support - release lock on exit.

        Always releases the lock, even if an exception occurred.
        """
        self.release()
        return False  # Don't suppress exceptions


def get_redis_client_for_lock() -> Optional[redis.Redis]:
    """
    Get Redis client for distributed locking.

    Uses redis.from_url() for robust URL parsing that handles:
    - Database suffixes (e.g., redis://host:6379/0)
    - Passwords with colons
    - Authentication in URL
    - All standard Redis URL formats

    Returns:
        Redis client or None if Redis unavailable
    """
    redis_url = "unknown"
    try:
        import os

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_password = os.getenv("REDIS_PASSWORD")

        # Handle REDIS_DB environment variable with validation
        redis_db_str = os.getenv("REDIS_DB", "0")
        try:
            redis_db = int(redis_db_str)
            if redis_db < 0:
                logger.warning(
                    f"Invalid REDIS_DB value '{redis_db_str}' (must be >= 0), using default 0"
                )
                redis_db = 0
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid REDIS_DB value '{redis_db_str}' (must be integer), using default 0"
            )
            redis_db = 0

        # Use redis.from_url() for proper URL parsing
        # This handles database suffixes, passwords with colons, and all URL formats
        client = redis.from_url(
            redis_url,
            password=redis_password,  # Override password from env if provided
            db=redis_db,  # Override DB from env if provided
            decode_responses=False,  # Keep binary for Lua scripts
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        # Test connection
        client.ping()

        # Mask URL to prevent credential leakage in logs
        # Extract host:port only, remove credentials
        safe_url = redis_url
        if "@" in redis_url:
            # Format: redis://user:pass@host:port/db -> redis://host:port/db
            safe_url = redis_url.split("@")[-1]
        elif "://" in redis_url:
            # Format: redis://host:port/db
            safe_url = redis_url.split("://")[-1]

        logger.debug(
            "Redis client initialized for distributed locking",
            extra={
                "redis_host": safe_url.split("/")[0] if "/" in safe_url else safe_url
            },
        )
        return client

    except Exception as e:
        # Mask URL to prevent credential leakage in error logs
        safe_url = redis_url
        if "@" in redis_url:
            safe_url = redis_url.split("@")[-1]
        elif "://" in redis_url:
            safe_url = redis_url.split("://")[-1]

        logger.error(
            "Redis client unavailable for distributed locking",
            extra={
                "redis_host": safe_url.split("/")[0] if "/" in safe_url else safe_url,
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return None
