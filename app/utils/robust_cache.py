"""
Robust Caching Utility
Provides consistent, high-performance caching with automatic optimization
"""
import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """Generates consistent, deterministic cache keys"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent caching

        Handles:
        - Case sensitivity
        - Extra whitespace
        - Punctuation variations
        - Common variations
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove trailing punctuation
        text = text.rstrip("?!.,;:")

        # Remove common filler phrases (must be done before word splitting)
        filler_phrases = [
            "please ",
            "can you ",
            "could you ",
            "would you ",
            "i want to ",
            "i need to ",
        ]
        for phrase in filler_phrases:
            if text.startswith(phrase):
                text = text[len(phrase) :]

        # Clean up any extra whitespace again
        text = " ".join(text.split())

        return text

    @staticmethod
    def generate_key(prefix: str, data: Any, normalize: bool = True) -> str:
        """
        Generate deterministic cache key

        Args:
            prefix: Key prefix (e.g., 'embedding', 'conversation')
            data: Data to hash (string, dict, list, etc.)
            normalize: Whether to normalize text data

        Returns:
            Consistent cache key
        """
        # Handle different data types
        if isinstance(data, str):
            if normalize:
                data = CacheKeyGenerator.normalize_text(data)
            hash_input = data
        elif isinstance(data, dict):
            # Sort keys for consistency
            hash_input = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            hash_input = json.dumps(
                sorted(data) if all(isinstance(x, str) for x in data) else list(data)
            )
        else:
            hash_input = str(data)

        # Generate hash
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()

        return f"{prefix}:{hash_value}"

    @staticmethod
    def generate_embedding_key(
        query: str, model: str = "text-embedding-3-small"
    ) -> str:
        """Generate cache key for embeddings"""
        normalized_query = CacheKeyGenerator.normalize_text(query)
        data = {"query": normalized_query, "model": model}
        return CacheKeyGenerator.generate_key("embedding", data, normalize=False)

    @staticmethod
    def generate_conversation_key(session_id: str, org_id: str) -> str:
        """Generate cache key for conversations"""
        return f"conversation:session:{org_id}:{session_id}"


class SmartTTLManager:
    """Manages TTL based on data type and access patterns"""

    # Default TTLs for different data types (in seconds)
    DEFAULT_TTLS = {
        "embedding": 3600,  # 1 hour - stable
        "conversation": 1800,  # 30 minutes - moderate
        "conversation_history": 900,  # 15 minutes - dynamic
        "chatbot_config": 7200,  # 2 hours - very stable
        "document_retrieval": 3600,  # 1 hour - stable
        "user_session": 1800,  # 30 minutes - moderate
        "cache_warming": 14400,  # 4 hours - pre-warmed data
    }

    def __init__(self):
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}

    def get_ttl(
        self,
        data_type: str,
        access_count: Optional[int] = None,
        base_ttl: Optional[int] = None,
    ) -> int:
        """
        Get appropriate TTL for data type

        Args:
            data_type: Type of data being cached
            access_count: Number of times this key has been accessed
            base_ttl: Override default TTL

        Returns:
            TTL in seconds
        """
        # Get base TTL
        ttl = base_ttl or self.DEFAULT_TTLS.get(data_type, 3600)

        # Adjust based on access frequency
        if access_count is not None:
            if access_count > 100:
                ttl *= 4  # Very frequently accessed
            elif access_count > 50:
                ttl *= 2  # Frequently accessed
            elif access_count > 20:
                ttl *= 1.5  # Moderately accessed

        # Cap at 24 hours
        return min(int(ttl), 86400)

    def record_access(self, key: str):
        """Record key access for TTL optimization"""
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.last_access[key] = datetime.now()

    def get_access_count(self, key: str) -> int:
        """Get access count for a key"""
        return self.access_counts.get(key, 0)


class RobustCacheService:
    """
    Robust caching service with automatic optimization

    Features:
    - Consistent key generation
    - Smart TTL management
    - Access pattern tracking
    - Error handling with fallback
    - Cache warming support
    - Monitoring and metrics
    """

    def __init__(self, cache_service):
        self.cache_service = cache_service
        self.key_generator = CacheKeyGenerator()
        self.ttl_manager = SmartTTLManager()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.sets = 0

    async def get(self, key: str, track_access: bool = True) -> Optional[Any]:
        """
        Get value from cache with error handling

        Args:
            key: Cache key
            track_access: Whether to track access for TTL optimization

        Returns:
            Cached value or None
        """
        try:
            value = await self.cache_service.get(key)

            if value is not None:
                self.hits += 1
                if track_access:
                    self.ttl_manager.record_access(key)
                logger.debug(f"Cache HIT: {key}")
                return value
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {key}")
                return None

        except Exception as e:
            self.errors += 1
            logger.warning(f"Cache GET error for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        data_type: str = "default",
    ) -> bool:
        """
        Set value in cache with smart TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom TTL (if None, uses smart TTL)
            data_type: Type of data for TTL calculation

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get smart TTL if not provided
            if ttl is None:
                access_count = self.ttl_manager.get_access_count(key)
                ttl = self.ttl_manager.get_ttl(data_type, access_count)

            await self.cache_service.set(key, value, ttl=ttl)
            self.sets += 1
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            self.errors += 1
            logger.warning(f"Cache SET error for {key}: {e}")
            return False

    async def get_or_set(
        self,
        key: str,
        compute_fn: Callable,
        ttl: Optional[int] = None,
        data_type: str = "default",
    ) -> Any:
        """
        Get from cache or compute and cache (Cache-Aside pattern)

        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached
            ttl: Custom TTL
            data_type: Type of data

        Returns:
            Cached or computed value
        """
        # Try cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        value = await compute_fn()

        # Cache it
        await self.set(key, value, ttl=ttl, data_type=data_type)

        return value

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.cache_service.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache DELETE error for {key}: {e}")
            return False

    async def warm_cache(self, warm_data: List[Dict[str, Any]]):
        """
        Warm cache with common data

        Args:
            warm_data: List of dicts with 'key', 'value', 'ttl' (optional)
        """
        logger.info(f"🔥 Warming cache with {len(warm_data)} items...")

        for item in warm_data:
            key = item["key"]
            value = item["value"]
            ttl = item.get("ttl", self.ttl_manager.get_ttl("cache_warming"))

            await self.set(key, value, ttl=ttl, data_type="cache_warming")

        logger.info(f"✅ Cache warmed with {len(warm_data)} items")

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "sets": self.sets,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

    def reset_metrics(self):
        """Reset metrics counters"""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.sets = 0


def cached(
    data_type: str = "default",
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    normalize_args: bool = True,
):
    """
    Decorator for automatic caching of async functions

    Usage:
        @cached(data_type="embedding", ttl=3600)
        async def generate_embedding(query: str):
            # ... expensive operation
            return embedding

    Args:
        data_type: Type of data for TTL calculation
        ttl: Custom TTL
        key_prefix: Custom key prefix (defaults to function name)
        normalize_args: Whether to normalize string arguments
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Import here to avoid circular dependency
            from app.services.shared import cache_service

            if not cache_service:
                # No cache available, just execute function
                return await func(*args, **kwargs)

            # Create robust cache service
            robust_cache = RobustCacheService(cache_service)

            # Generate cache key from function name and arguments
            prefix = key_prefix or f"func:{func.__name__}"

            # Combine args and kwargs for key generation
            key_data = {"args": args, "kwargs": kwargs}

            cache_key = CacheKeyGenerator.generate_key(
                prefix, key_data, normalize=normalize_args
            )

            # Use get_or_set pattern
            return await robust_cache.get_or_set(
                cache_key, lambda: func(*args, **kwargs), ttl=ttl, data_type=data_type
            )

        return wrapper

    return decorator


# Global instance (initialized on first use)
_robust_cache_instance: Optional[RobustCacheService] = None


def get_robust_cache() -> Optional[RobustCacheService]:
    """Get or create robust cache instance"""
    global _robust_cache_instance

    if _robust_cache_instance is None:
        from app.services.shared import cache_service

        if cache_service:
            _robust_cache_instance = RobustCacheService(cache_service)

    return _robust_cache_instance


async def warm_common_queries():
    """Warm cache with common queries"""
    robust_cache = get_robust_cache()
    if not robust_cache:
        logger.warning("Cache service not available for warming")
        return

    # Common queries to pre-cache
    common_queries = [
        "What are your products?",
        "How do I contact you?",
        "What are your prices?",
        "Tell me about your services",
        "Where are you located?",
        "What is your email?",
        "What is your phone number?",
        "How can I reach you?",
        "What do you offer?",
        "What are your hours?",
    ]

    warm_data = []

    # Generate embeddings for common queries (if embedding service available)
    try:
        from app.services.chat.document_retrieval_service import (
            DocumentRetrievalService,
        )

        # This would need actual implementation
        # For now, we'll just warm the keys
        for query in common_queries:
            key = CacheKeyGenerator.generate_embedding_key(query)
            # In production, you'd generate actual embedding here
            warm_data.append(
                {
                    "key": key,
                    "value": {"query": query, "warmed": True},
                    "ttl": 14400,  # 4 hours
                }
            )
    except Exception as e:
        logger.warning(f"Could not warm embeddings: {e}")

    if warm_data:
        await robust_cache.warm_cache(warm_data)


# Export main components
__all__ = [
    "RobustCacheService",
    "CacheKeyGenerator",
    "SmartTTLManager",
    "cached",
    "get_robust_cache",
    "warm_common_queries",
]
