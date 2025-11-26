"""
WhatsApp Configuration Cache Service

Integrates with the existing centralized caching infrastructure to provide
high-performance caching for WhatsApp configurations.

This service uses the shared CacheService for Redis operations and adds
WhatsApp-specific caching logic on top.
"""
import json
import logging
import time
from typing import Any, Dict, Optional

from ...utils.logging_config import get_logger
from ..shared.cache_service import cache_service

logger = get_logger(__name__)


class WhatsAppConfigCache:
    """
    WhatsApp configuration cache using centralized cache service

    """

    def __init__(
        self,
        ttl_seconds: int = 300,  # 5 minutes default
        max_memory_cache_size: int = 100,  # Max orgs in memory
    ):
        """
        Initialize configuration cache

        """
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_memory_cache_size

        # In-memory cache: {org_id: (config_dict, timestamp)}
        # This provides ultra-fast access for frequently accessed configs
        self._memory_cache: Dict[str, tuple[Dict[str, Any], float]] = {}

        # Track cache statistics
        self._stats = {
            "memory_hits": 0,
            "redis_hits": 0,
            "db_fetches": 0,
            "cache_misses": 0,
            "invalidations": 0,
        }

        logger.info(
            "WhatsApp config cache initialized (integrated with shared cache service)",
            extra={
                "ttl_seconds": ttl_seconds,
                "max_memory_size": max_memory_cache_size,
            },
        )

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cached entry has expired"""
        return (time.time() - timestamp) > self.ttl_seconds

    def _get_redis_key(self, org_id: str) -> str:
        """Generate Redis key for organization config (follows existing naming convention)"""
        # Use the same naming convention as other cache keys in the system
        return f"whatsapp:config:v1:{org_id}"

    async def get(self, org_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration from cache (memory -> Redis -> None)

        Args:
            org_id: Organization ID

        Returns:
            Configuration dict if found in cache, None otherwise
        """
        # Layer 1: Check in-memory cache (fastest - ~1-2ms)
        if org_id in self._memory_cache:
            config, timestamp = self._memory_cache[org_id]
            if not self._is_expired(timestamp):
                self._stats["memory_hits"] += 1
                logger.debug(
                    f"✅ Config cache HIT (memory) for org {org_id}",
                    extra={"org_id": org_id, "cache_layer": "memory"},
                )
                return config
            else:
                # Expired, remove from memory
                del self._memory_cache[org_id]
                logger.debug(f"⏰ Memory cache expired for org {org_id}")

        # Layer 2: Check Redis cache using shared cache service (~5-10ms)
        try:
            redis_key = self._get_redis_key(org_id)
            cached_data = await cache_service.get(redis_key)

            if cached_data:
                # Parse the config (it's already deserialized by cache_service)
                config = (
                    cached_data
                    if isinstance(cached_data, dict)
                    else json.loads(cached_data)
                )
                self._stats["redis_hits"] += 1

                # Populate memory cache for next access
                self._set_memory_cache(org_id, config)

                logger.debug(
                    f"✅ Config cache HIT (Redis) for org {org_id}",
                    extra={"org_id": org_id, "cache_layer": "redis"},
                )
                return config
        except Exception as e:
            logger.warning(
                f"Redis cache read failed for org {org_id}: {e}",
                extra={"org_id": org_id, "error": str(e)},
            )

        # Cache miss
        self._stats["cache_misses"] += 1
        logger.debug(f"❌ Config cache MISS for org {org_id}", extra={"org_id": org_id})
        return None

    async def set(self, org_id: str, config: Dict[str, Any]) -> None:
        """
        Store configuration in all cache layers

        Args:
            org_id: Organization ID
            config: Configuration dictionary to cache
        """
        # Store in memory cache (fastest layer)
        self._set_memory_cache(org_id, config)

        # Store in Redis cache using shared cache service
        try:
            redis_key = self._get_redis_key(org_id)
            await cache_service.set(redis_key, config, ttl_seconds=self.ttl_seconds)

            logger.debug(
                f"💾 Stored config in cache for org {org_id}",
                extra={"org_id": org_id, "ttl": self.ttl_seconds},
            )
        except Exception as e:
            logger.warning(
                f"Failed to store config in Redis for org {org_id}: {e}",
                extra={"org_id": org_id, "error": str(e)},
            )

    def _set_memory_cache(self, org_id: str, config: Dict[str, Any]) -> None:
        """Store configuration in memory cache with LRU eviction"""
        # Implement simple LRU: if cache is full, remove oldest entry
        if len(self._memory_cache) >= self.max_cache_size:
            # Find and remove the oldest entry
            oldest_org = min(
                self._memory_cache.keys(), key=lambda k: self._memory_cache[k][1]
            )
            del self._memory_cache[oldest_org]
            logger.debug(f"🗑️ Evicted oldest config from memory: {oldest_org}")

        self._memory_cache[org_id] = (config, time.time())
        logger.debug(
            f"💾 Stored config in memory cache for org {org_id}",
            extra={"org_id": org_id, "cache_size": len(self._memory_cache)},
        )

    async def invalidate(self, org_id: str) -> None:
        """
        Invalidate cached configuration for an organization

        Args:
            org_id: Organization ID to invalidate
        """
        self._stats["invalidations"] += 1

        # Remove from memory cache
        if org_id in self._memory_cache:
            del self._memory_cache[org_id]
            logger.debug(f"🗑️ Invalidated memory cache for org {org_id}")

        # Remove from Redis cache using shared cache service
        try:
            redis_key = self._get_redis_key(org_id)
            await cache_service.delete(redis_key)
            logger.debug(f"🗑️ Invalidated Redis cache for org {org_id}")
        except Exception as e:
            logger.warning(
                f"Failed to invalidate Redis cache for org {org_id}: {e}",
                extra={"org_id": org_id, "error": str(e)},
            )

        logger.info(f"♻️ Cache invalidated for org {org_id}", extra={"org_id": org_id})

    def clear_all(self) -> None:
        """Clear all cached configurations from memory (Redis handled by cache service)"""
        cache_size = len(self._memory_cache)
        self._memory_cache.clear()

        logger.info(
            f"🧹 Cleared all memory cache ({cache_size} entries)",
            extra={"cleared_entries": cache_size},
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics

        Returns:
            Dictionary with cache hit/miss statistics
        """
        total_requests = (
            self._stats["memory_hits"]
            + self._stats["redis_hits"]
            + self._stats["cache_misses"]
        )

        hit_rate = 0.0
        if total_requests > 0:
            total_hits = self._stats["memory_hits"] + self._stats["redis_hits"]
            hit_rate = (total_hits / total_requests) * 100

        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self._memory_cache),
            "max_cache_size": self.max_cache_size,
            "ttl_seconds": self.ttl_seconds,
            "integration": "shared_cache_service",  # Indicates integration with shared cache
        }

    def reset_stats(self) -> None:
        """Reset cache statistics"""
        self._stats = {
            "memory_hits": 0,
            "redis_hits": 0,
            "db_fetches": 0,
            "cache_misses": 0,
            "invalidations": 0,
        }
        logger.info("📊 Cache statistics reset")


# Global cache instance (singleton pattern)
_config_cache: Optional[WhatsAppConfigCache] = None


def get_config_cache(redis_client=None) -> WhatsAppConfigCache:
    """
    Get the global WhatsApp configuration cache instance

    Note: redis_client parameter is kept for backward compatibility but not used
    since we now use the shared cache service

    """
    global _config_cache

    if _config_cache is None:
        _config_cache = WhatsAppConfigCache(
            ttl_seconds=300,  # 5 minutes
            max_memory_cache_size=100,  # Cache up to 100 org configs
        )
        logger.info(
            "🚀 Global WhatsApp config cache initialized (using shared cache service)"
        )

    return _config_cache


async def invalidate_org_config(org_id: str) -> None:
    """
    Convenience function to invalidate a specific org's config

    Args:
        org_id: Organization ID to invalidate
    """
    cache = get_config_cache()
    await cache.invalidate(org_id)
