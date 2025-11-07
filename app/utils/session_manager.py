"""
Session Management Utilities

Handles consistent session ID generation and caching to prevent rate limit bypass
and ensure proper conversation context continuity.
"""
import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from ..services.shared import cache_service

logger = logging.getLogger(__name__)

# Cache for active user sessions to prevent session ID manipulation
# Using OrderedDict for LRU eviction: stores (cache_key, session_data) pairs
# session_data is a dict: {"user_id": str, "chatbot_id": str, "org_id": str, "timestamp": float}
_active_sessions_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
# Async lock for thread-safe access to shared mutable state in async context
_cache_lock = asyncio.Lock()
# Maximum cache size before eviction (LRU)
_MAX_CACHE_SIZE = 10000
# Cache entry TTL in seconds (24 hours)
_CACHE_TTL = 86400


def _generate_consistent_session_id(user_id: str, chatbot_id: Optional[str]) -> str:
    """
    Generate a consistent session ID based on user_id and chatbot_id.
    This prevents users from bypassing rate limits by creating new sessions.

    Security: Uses hashed identifiers only, no plaintext user_id or chatbot_id.

    Args:
        user_id: User ID from JWT token
        chatbot_id: Chatbot ID (optional)

    Returns:
        Consistent, deterministic session ID (hashed, no plaintext identifiers)
    """
    chatbot_id = chatbot_id or "default"
    # Create a deterministic hash from user_id + chatbot_id
    # Use full hash for better security (no plaintext identifiers)
    session_key = f"{user_id}:{chatbot_id}"
    session_hash = hashlib.sha256(session_key.encode()).hexdigest()
    # Return only the hash (no plaintext identifiers)
    return f"session-{session_hash}"


def _evict_old_entries():
    """
    Evict old entries from cache based on TTL and size (LRU).

    WARNING: This function must be called while holding _cache_lock.
    It directly modifies shared mutable state and is not thread-safe on its own.
    """
    current_time = time.time()

    # Remove expired entries (based on TTL)
    expired_keys = [
        key
        for key, data in _active_sessions_cache.items()
        if current_time - data.get("timestamp", 0) > _CACHE_TTL
    ]
    for key in expired_keys:
        _active_sessions_cache.pop(key, None)

    # If still over limit, remove oldest entries (LRU)
    while len(_active_sessions_cache) > _MAX_CACHE_SIZE:
        _active_sessions_cache.popitem(last=False)  # Remove oldest (FIFO)


async def get_or_create_session_id(
    user_id: str,
    chatbot_id: Optional[str],
    org_id: str,
    provided_conversation_id: Optional[str] = None,
) -> str:
    """
    Get or create a consistent session ID for a user/chatbot combination.
    This prevents rate limit bypass by ensuring session IDs are consistent.

    Args:
        user_id: User ID from JWT token
        chatbot_id: Chatbot ID (optional)
        org_id: Organization ID
        provided_conversation_id: Optional conversation ID from request

    Returns:
        Consistent session ID
    """
    chatbot_id = chatbot_id or "default"

    # If conversation_id is provided, validate and use it
    # (client may be continuing an existing conversation)
    if provided_conversation_id:
        # Validate that the conversation belongs to this user/org/chatbot
        cache_key = f"session:{org_id}:{provided_conversation_id}"
        cached_data = None

        # Check in-memory cache first (with lock for thread-safe access)
        async with _cache_lock:
            if cache_key in _active_sessions_cache:
                cached_data = _active_sessions_cache[
                    cache_key
                ].copy()  # Copy to avoid holding lock during await
                # Refresh timestamp and move to end (LRU - mark as recently used)
                _active_sessions_cache[cache_key]["timestamp"] = time.time()
                _active_sessions_cache.move_to_end(cache_key)

        # If not in memory, check Redis (no lock held during await)
        if not cached_data:
            try:
                if cache_service:
                    cached_data = await cache_service.get(cache_key)
            except Exception as e:
                logger.debug("Failed to retrieve cached session: %s", e)

        # Validate: must match user_id, chatbot_id, and org_id
        if cached_data and isinstance(cached_data, dict):
            if (
                cached_data.get("user_id") == user_id
                and cached_data.get("chatbot_id") == chatbot_id
                and cached_data.get("org_id") == org_id
            ):
                # Refresh timestamp in Redis to keep it in sync (no lock held during await)
                try:
                    if cache_service:
                        # Update Redis with refreshed timestamp
                        refreshed_data = cached_data.copy()
                        refreshed_data["timestamp"] = time.time()
                        await cache_service.set(
                            cache_key,
                            refreshed_data,
                            ttl=_CACHE_TTL,
                        )
                except Exception as cache_error:
                    logger.debug(
                        "Failed to refresh session timestamp in Redis: %s", cache_error
                    )

                logger.debug(
                    "Using provided conversation_id (validated, timestamp refreshed)",
                    extra={
                        "conversation_id": provided_conversation_id,
                        "user_id": user_id,
                    },
                )
                return provided_conversation_id
            else:
                logger.warning(
                    "Provided conversation_id validation failed - mismatch detected",
                    extra={
                        "conversation_id": provided_conversation_id,
                        "expected_user": user_id,
                        "expected_chatbot": chatbot_id,
                        "expected_org": org_id,
                        "cached_user": cached_data.get("user_id"),
                        "cached_chatbot": cached_data.get("chatbot_id"),
                        "cached_org": cached_data.get("org_id"),
                    },
                )
        # Otherwise, we'll generate a consistent one below

    # Generate consistent session ID
    session_id = _generate_consistent_session_id(user_id, chatbot_id)

    # Cache the session mapping to prevent manipulation
    cache_key = f"session:{org_id}:{session_id}"

    # Prepare session data (consistent structure for both caches)
    session_data = {
        "user_id": user_id,
        "chatbot_id": chatbot_id,
        "org_id": org_id,
        "timestamp": time.time(),
    }

    # Update in-memory cache atomically (check-and-set/update within lock to prevent TOCTOU)
    session_was_new = False
    async with _cache_lock:
        # Atomic check-and-set/update: prevents race condition
        if cache_key not in _active_sessions_cache:
            # New session: create it
            _active_sessions_cache[cache_key] = session_data
            _active_sessions_cache.move_to_end(cache_key)  # Mark as recently used
            session_was_new = True
            # Evict old entries if needed (called within lock context)
            _evict_old_entries()
        else:
            # Existing session: refresh timestamp to indicate it's still active
            _active_sessions_cache[cache_key]["timestamp"] = time.time()
            _active_sessions_cache.move_to_end(cache_key)  # Mark as recently used
            # Update session_data with refreshed timestamp for Redis sync
            session_data = _active_sessions_cache[cache_key].copy()

    # Cache/refresh in Redis for persistence across restarts (async, no lock held)
    try:
        if cache_service:
            await cache_service.set(
                cache_key,
                session_data,  # Same structure as in-memory cache (with refreshed timestamp if existing)
                ttl=_CACHE_TTL,
            )
            action = "Cached" if session_was_new else "Refreshed"
            logger.debug(
                f"{action} session mapping",
                extra={
                    "session_id": session_id,
                    "user_id": user_id,
                    "org_id": org_id,
                    "was_new": session_was_new,
                },
            )
    except Exception as cache_error:
        logger.warning("Failed to cache/refresh session: %s", cache_error)

    return session_id


async def clear_session_cache(session_id: str, org_id: str) -> None:
    """
    Clear a session from both in-memory and Redis cache (e.g., when conversation ends).

    Args:
        session_id: Session ID to clear
        org_id: Organization ID
    """
    cache_key = f"session:{org_id}:{session_id}"

    # Clear from in-memory cache atomically (check-and-delete within lock to prevent race condition)
    async with _cache_lock:
        if cache_key in _active_sessions_cache:
            del _active_sessions_cache[cache_key]
            logger.debug(
                "Cleared session from in-memory cache", extra={"session_id": session_id}
            )

    # Clear from Redis cache for consistency (no lock held during await)
    try:
        if cache_service:
            await cache_service.delete(cache_key)
            logger.debug(
                "Cleared session from Redis cache", extra={"session_id": session_id}
            )
    except Exception as cache_error:
        logger.warning("Failed to clear session from Redis cache: %s", cache_error)
