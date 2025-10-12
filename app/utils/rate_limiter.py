"""
Rate limiting utilities for API endpoints

This module provides rate limiting functionality to prevent abuse and
protect against cost explosion from excessive API calls.
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Callable, Optional

from fastapi import Depends

from ..services.auth import verify_jwt_token_from_header

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm

    This is a basic implementation suitable for single-server deployments.
    For multi-server deployments, consider using Redis-based rate limiting.
    """

    def __init__(self):
        self._requests = defaultdict(list)
        self._lock = threading.Lock()
        self._cleanup_interval = 60  # Cleanup old entries every 60 seconds
        self._last_cleanup = datetime.now(timezone.utc)

    def is_allowed(
        self, key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, dict]:
        """
        Check if a request is allowed based on rate limits

        Args:
            key: Unique identifier for the client (e.g., user_id, IP address)
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, info_dict)
            info_dict contains: remaining, reset_time, retry_after
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=window_seconds)

            # Get request history for this key
            request_times = self._requests[key]

            # Remove requests outside the current window
            request_times = [t for t in request_times if t > window_start]
            self._requests[key] = request_times

            # Check if limit exceeded
            current_count = len(request_times)
            is_allowed = current_count < max_requests

            if is_allowed:
                # Add current request
                request_times.append(now)

            # Calculate info
            remaining = max(0, max_requests - current_count - (1 if is_allowed else 0))
            reset_time = now + timedelta(seconds=window_seconds)
            retry_after = 0

            if not is_allowed and request_times:
                # Calculate when the oldest request will expire
                oldest_request = min(request_times)
                retry_after = int(
                    (
                        oldest_request + timedelta(seconds=window_seconds) - now
                    ).total_seconds()
                )

            # Periodic cleanup
            if (now - self._last_cleanup).total_seconds() > self._cleanup_interval:
                self._cleanup_old_entries(window_start)
                self._last_cleanup = now

            info = {
                "remaining": remaining,
                "reset_time": reset_time.isoformat(),
                "retry_after": max(0, retry_after),
                "limit": max_requests,
                "window": window_seconds,
            }

            return is_allowed, info

    def _cleanup_old_entries(self, cutoff_time: datetime):
        """Remove old entries to prevent memory growth"""
        keys_to_delete = []

        for key, request_times in self._requests.items():
            # Remove old requests
            self._requests[key] = [t for t in request_times if t > cutoff_time]

            # Mark empty keys for deletion
            if not self._requests[key]:
                keys_to_delete.append(key)

        # Delete empty keys
        for key in keys_to_delete:
            del self._requests[key]

        if keys_to_delete:
            logger.debug(f"Cleaned up {len(keys_to_delete)} empty rate limit keys")

    def reset(self, key: str):
        """Reset rate limit for a specific key"""
        with self._lock:
            if key in self._requests:
                del self._requests[key]
                logger.info(f"Rate limit reset for key: {key}")

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        with self._lock:
            return {
                "total_keys": len(self._requests),
                "total_requests": sum(len(times) for times in self._requests.values()),
                "last_cleanup": self._last_cleanup.isoformat(),
            }


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()


def get_rate_limiter() -> InMemoryRateLimiter:
    """Get the global rate limiter instance"""
    return _rate_limiter


def rate_limit(
    max_requests: int = 10,
    window_seconds: int = 60,
    key_func: Optional[Callable] = None,
    error_message: str = "Rate limit exceeded",
):
    """
    Decorator for rate limiting endpoints

    Args:
        max_requests: Maximum requests allowed in the window
        window_seconds: Time window in seconds
        key_func: Function to extract rate limit key from request
                  If None, uses user_id from JWT token
        error_message: Custom error message

    Example:
        @rate_limit(max_requests=10, window_seconds=60)
        async def my_endpoint(user=Depends(verify_jwt_token_from_header)):
            return {"message": "success"}
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import HTTPException

            # Extract rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Try to get user from kwargs (from JWT token)
                user = kwargs.get("user")
                if user and isinstance(user, dict):
                    key = f"user:{user.get('user_id', 'unknown')}"
                else:
                    # Fallback to a generic key (not ideal for production)
                    key = "anonymous"

            # Check rate limit
            limiter = get_rate_limiter()
            is_allowed, info = limiter.is_allowed(key, max_requests, window_seconds)

            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded for key: {key}",
                    extra={
                        "key": key,
                        "limit": max_requests,
                        "window": window_seconds,
                        "retry_after": info["retry_after"],
                    },
                )

                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": error_message,
                        "retry_after": info["retry_after"],
                        "limit": info["limit"],
                        "window": info["window"],
                    },
                    headers={
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": info["reset_time"],
                        "Retry-After": str(info["retry_after"]),
                    },
                )

            # Add rate limit headers to response
            try:
                result = await func(*args, **kwargs)

                # Note: Adding headers to response requires middleware or response model
                # For now, we just log the info
                logger.debug(
                    f"Rate limit check passed for key: {key}",
                    extra={"remaining": info["remaining"], "limit": info["limit"]},
                )

                return result
            except Exception as e:
                # Re-raise the exception
                raise e

        return wrapper

    return decorator


class RateLimitMiddleware:
    """
    Middleware to add rate limit headers to all responses

    This middleware adds X-RateLimit-* headers to responses
    """

    def __init__(self, app, default_limit: int = 100, default_window: int = 60):
        self.app = app
        self.default_limit = default_limit
        self.default_window = default_window

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract user info from scope if available
        # This is a simplified version - in production, you'd extract from JWT
        user_id = scope.get("user", {}).get("user_id", "anonymous")
        key = f"user:{user_id}"

        # Check rate limit
        limiter = get_rate_limiter()
        is_allowed, info = limiter.is_allowed(
            key, self.default_limit, self.default_window
        )

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                # Add rate limit headers
                headers.append((b"x-ratelimit-limit", str(self.default_limit).encode()))
                headers.append(
                    (b"x-ratelimit-remaining", str(info["remaining"]).encode())
                )
                headers.append((b"x-ratelimit-reset", info["reset_time"].encode()))

                message["headers"] = headers

            await send(message)

        if not is_allowed:
            # Send 429 response
            await send_with_headers(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"retry-after", str(info["retry_after"]).encode()),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "Rate limit exceeded"}',
                }
            )
        else:
            await self.app(scope, receive, send_with_headers)


# Predefined rate limit configurations
RATE_LIMITS = {
    "chat": {
        "max_requests": 100,
        "window_seconds": 60,
        "error_message": "Too many chat requests. Please wait before sending more messages.",
    },
    "upload": {
        "max_requests": 10,
        "window_seconds": 300,  # 5 minutes
        "error_message": "Too many upload requests. Please wait before uploading more files.",
    },
    "search": {
        "max_requests": 30,
        "window_seconds": 60,
        "error_message": "Too many search requests. Please slow down.",
    },
    "public_chat": {
        "max_requests": 100,
        "window_seconds": 60,
        "error_message": "Too many messages. Please wait before sending more.",
    },
    "api_general": {
        "max_requests": 100,
        "window_seconds": 60,
        "error_message": "API rate limit exceeded. Please slow down.",
    },
}


def get_rate_limit_config(endpoint_type: str) -> dict:
    """Get rate limit configuration for an endpoint type"""
    return RATE_LIMITS.get(endpoint_type, RATE_LIMITS["api_general"])
