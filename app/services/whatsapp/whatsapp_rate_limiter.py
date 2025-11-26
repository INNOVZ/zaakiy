"""
WhatsApp Rate Limiter
---------------------
Robust rate limiting for WhatsApp messages to prevent abuse and manage costs.

Features:
- Per-user rate limiting
- Per-organization rate limiting
- Sliding window algorithm
- Redis-based for distributed systems
- Configurable limits
- Graceful degradation
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""

    def __init__(self, message: str, retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class WhatsAppRateLimiter:
    """
    Rate limiter for WhatsApp messages using sliding window algorithm.

    Limits:
    - Per user: 10 messages per minute, 50 per hour, 200 per day
    - Per organization: 100 messages per minute, 1000 per hour, 5000 per day
    - Global: 1000 messages per minute
    """

    # Default rate limits (messages per time window)
    DEFAULT_LIMITS = {
        "user": {
            "per_minute": 10,
            "per_hour": 50,
            "per_day": 200,
        },
        "organization": {
            "per_minute": 100,
            "per_hour": 1000,
            "per_day": 5000,
        },
        "global": {
            "per_minute": 1000,
            "per_hour": 10000,
            "per_day": 50000,
        },
    }

    # Time windows in seconds
    WINDOWS = {
        "per_minute": 60,
        "per_hour": 3600,
        "per_day": 86400,
    }

    def __init__(self, redis_client=None, limits: Optional[Dict] = None):
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis client for distributed rate limiting
            limits: Custom rate limits (overrides defaults)
        """
        self.redis = redis_client
        self.limits = limits or self.DEFAULT_LIMITS
        self.in_memory_cache = {}  # Fallback if Redis unavailable

    def check_rate_limit(
        self, phone_number: str, org_id: str, action: str = "message"
    ) -> Tuple[bool, Optional[str], int]:
        """
        Check if request is within rate limits.

        Args:
            phone_number: User's phone number
            org_id: Organization ID
            action: Action type (default: "message")

        Returns:
            Tuple of (allowed, error_message, retry_after_seconds)
        """
        current_time = int(time.time())

        # Check user rate limit
        user_allowed, user_msg, user_retry = self._check_limit(
            key_type="user", identifier=phone_number, current_time=current_time
        )

        if not user_allowed:
            logger.warning(f"Rate limit exceeded for user {phone_number}: {user_msg}")
            return False, user_msg, user_retry

        # Check organization rate limit
        org_allowed, org_msg, org_retry = self._check_limit(
            key_type="organization", identifier=org_id, current_time=current_time
        )

        if not org_allowed:
            logger.warning(f"Rate limit exceeded for org {org_id}: {org_msg}")
            return False, org_msg, org_retry

        # Check global rate limit
        global_allowed, global_msg, global_retry = self._check_limit(
            key_type="global", identifier="whatsapp", current_time=current_time
        )

        if not global_allowed:
            logger.warning(f"Global rate limit exceeded: {global_msg}")
            return False, global_msg, global_retry

        # All checks passed - record the request
        self._record_request(phone_number, org_id, current_time)

        return True, None, 0

    def _check_limit(
        self, key_type: str, identifier: str, current_time: int
    ) -> Tuple[bool, Optional[str], int]:
        """
        Check rate limit for a specific key type.

        Args:
            key_type: Type of limit (user, organization, global)
            identifier: Unique identifier
            current_time: Current timestamp

        Returns:
            Tuple of (allowed, error_message, retry_after_seconds)
        """
        limits = self.limits.get(key_type, {})

        for window_name, limit in limits.items():
            window_seconds = self.WINDOWS[window_name]

            # Get request count for this window
            count = self._get_request_count(
                key_type=key_type,
                identifier=identifier,
                window_name=window_name,
                window_seconds=window_seconds,
                current_time=current_time,
            )

            if count >= limit:
                # Calculate retry after
                retry_after = self._calculate_retry_after(
                    key_type=key_type,
                    identifier=identifier,
                    window_name=window_name,
                    window_seconds=window_seconds,
                    current_time=current_time,
                )

                error_msg = (
                    f"Rate limit exceeded: {count}/{limit} messages "
                    f"{window_name.replace('_', ' ')}. "
                    f"Please try again in {retry_after} seconds."
                )

                return False, error_msg, retry_after

        return True, None, 0

    def _get_request_count(
        self,
        key_type: str,
        identifier: str,
        window_name: str,
        window_seconds: int,
        current_time: int,
    ) -> int:
        """Get number of requests in the time window"""

        key = self._make_key(key_type, identifier, window_name)

        if self.redis:
            try:
                # Use Redis sorted set for sliding window
                # Remove old entries
                cutoff_time = current_time - window_seconds
                self.redis.zremrangebyscore(key, 0, cutoff_time)

                # Count entries in window
                count = self.redis.zcard(key)

                return count
            except Exception as e:
                logger.error(f"Redis error in rate limiter: {e}")
                # Fall through to in-memory cache

        # Fallback to in-memory cache
        return self._get_in_memory_count(key, window_seconds, current_time)

    def _record_request(self, phone_number: str, org_id: str, current_time: int):
        """Record a request for rate limiting"""

        # Record for all window types
        for key_type in ["user", "organization", "global"]:
            identifier = {
                "user": phone_number,
                "organization": org_id,
                "global": "whatsapp",
            }[key_type]

            for window_name in self.WINDOWS.keys():
                key = self._make_key(key_type, identifier, window_name)

                if self.redis:
                    try:
                        # Add to sorted set with current timestamp as score
                        self.redis.zadd(key, {str(current_time): current_time})

                        # Set expiry to window duration + buffer
                        window_seconds = self.WINDOWS[window_name]
                        self.redis.expire(key, window_seconds + 60)
                    except Exception as e:
                        logger.error(f"Redis error recording request: {e}")
                        # Fall through to in-memory

                # Also record in-memory as fallback
                self._record_in_memory(key, current_time)

    def _calculate_retry_after(
        self,
        key_type: str,
        identifier: str,
        window_name: str,
        window_seconds: int,
        current_time: int,
    ) -> int:
        """Calculate seconds until rate limit resets"""

        key = self._make_key(key_type, identifier, window_name)

        if self.redis:
            try:
                # Get oldest entry in window
                oldest_entries = self.redis.zrange(key, 0, 0, withscores=True)
                if oldest_entries:
                    oldest_time = int(oldest_entries[0][1])
                    retry_after = window_seconds - (current_time - oldest_time)
                    return max(1, retry_after)
            except Exception as e:
                logger.error(f"Redis error calculating retry: {e}")

        # Default retry after
        return 60

    def _make_key(self, key_type: str, identifier: str, window_name: str) -> str:
        """Generate Redis/cache key"""
        # Hash identifier for privacy
        hashed_id = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"whatsapp:ratelimit:{key_type}:{hashed_id}:{window_name}"

    def _get_in_memory_count(
        self, key: str, window_seconds: int, current_time: int
    ) -> int:
        """Get count from in-memory cache (fallback)"""

        if key not in self.in_memory_cache:
            return 0

        # Clean old entries
        cutoff_time = current_time - window_seconds
        self.in_memory_cache[key] = [
            t for t in self.in_memory_cache[key] if t > cutoff_time
        ]

        return len(self.in_memory_cache[key])

    def _record_in_memory(self, key: str, current_time: int):
        """Record request in in-memory cache (fallback)"""

        if key not in self.in_memory_cache:
            self.in_memory_cache[key] = []

        self.in_memory_cache[key].append(current_time)

        # Limit in-memory cache size
        if len(self.in_memory_cache[key]) > 1000:
            self.in_memory_cache[key] = self.in_memory_cache[key][-500:]

    def get_rate_limit_info(self, phone_number: str, org_id: str) -> Dict[str, Any]:
        """
        Get current rate limit status for a user/org.

        Returns:
            Dictionary with current usage and limits
        """
        current_time = int(time.time())

        info = {"user": {}, "organization": {}, "timestamp": current_time}

        # Get user limits
        for window_name, window_seconds in self.WINDOWS.items():
            count = self._get_request_count(
                key_type="user",
                identifier=phone_number,
                window_name=window_name,
                window_seconds=window_seconds,
                current_time=current_time,
            )
            limit = self.limits["user"][window_name]

            info["user"][window_name] = {
                "count": count,
                "limit": limit,
                "remaining": max(0, limit - count),
                "percentage": round((count / limit) * 100, 1),
            }

        # Get org limits
        for window_name, window_seconds in self.WINDOWS.items():
            count = self._get_request_count(
                key_type="organization",
                identifier=org_id,
                window_name=window_name,
                window_seconds=window_seconds,
                current_time=current_time,
            )
            limit = self.limits["organization"][window_name]

            info["organization"][window_name] = {
                "count": count,
                "limit": limit,
                "remaining": max(0, limit - count),
                "percentage": round((count / limit) * 100, 1),
            }

        return info

    def reset_limits(
        self, phone_number: Optional[str] = None, org_id: Optional[str] = None
    ):
        """
        Reset rate limits for a user or organization.

        Args:
            phone_number: User phone number to reset
            org_id: Organization ID to reset
        """
        if phone_number:
            for window_name in self.WINDOWS.keys():
                key = self._make_key("user", phone_number, window_name)
                if self.redis:
                    try:
                        self.redis.delete(key)
                    except Exception as e:
                        logger.error(f"Redis error resetting limits: {e}")
                if key in self.in_memory_cache:
                    del self.in_memory_cache[key]

            logger.info(f"Reset rate limits for user {phone_number}")

        if org_id:
            for window_name in self.WINDOWS.keys():
                key = self._make_key("organization", org_id, window_name)
                if self.redis:
                    try:
                        self.redis.delete(key)
                    except Exception as e:
                        logger.error(f"Redis error resetting limits: {e}")
                if key in self.in_memory_cache:
                    del self.in_memory_cache[key]

            logger.info(f"Reset rate limits for org {org_id}")


# Convenience function
def check_whatsapp_rate_limit(
    phone_number: str, org_id: str, redis_client=None
) -> Tuple[bool, Optional[str], int]:
    """
    Quick function to check WhatsApp rate limit.

    Args:
        phone_number: User's phone number
        org_id: Organization ID
        redis_client: Optional Redis client

    Returns:
        Tuple of (allowed, error_message, retry_after_seconds)
    """
    limiter = WhatsAppRateLimiter(redis_client=redis_client)
    return limiter.check_rate_limit(phone_number, org_id)


# Export
__all__ = ["WhatsAppRateLimiter", "RateLimitExceeded", "check_whatsapp_rate_limit"]
