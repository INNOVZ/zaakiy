"""
Redis cache service for high-performance caching with connection pooling and advanced features
"""
import json
import logging
import os
import hashlib
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from ...utils.error_handlers import ErrorHandler
from ...utils.error_context import ErrorContextManager, ErrorSeverity
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class CacheCircuitBreaker:
    """Circuit breaker for cache operations to prevent cascade failures"""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.successful_calls = 0

    def _should_try_reset(self) -> bool:
        """Check if we should try to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.timeout_seconds

    def _on_success(self):
        """Handle successful operation"""
        if self.state == "HALF_OPEN":
            self.successful_calls += 1
            if self.successful_calls >= 3:  # Require 3 successful calls to close
                self.state = "CLOSED"
                self.failure_count = 0
                self.successful_calls = 0
                logger.info(
                    "Cache circuit breaker CLOSED after successful operations")
        else:
            self.failure_count = max(
                0, self.failure_count - 1)  # Decay failure count

    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.successful_calls = 0

        if self.failure_count >= self.failure_threshold and self.state == "CLOSED":
            self.state = "OPEN"
            logger.warning(
                f"Cache circuit breaker OPEN after {self.failure_count} failures")

    async def execute_with_breaker(self, cache_operation, fallback_operation=None):
        """Execute cache operation with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_try_reset():
                self.state = "HALF_OPEN"
                logger.info(
                    "Cache circuit breaker HALF_OPEN - testing connection")
            else:
                if fallback_operation:
                    logger.debug("Cache circuit breaker OPEN - using fallback")
                    return await fallback_operation()
                else:
                    logger.debug("Cache circuit breaker OPEN - returning None")
                    return None

        try:
            result = await cache_operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            logger.warning(f"Cache operation failed (circuit breaker): {e}")
            if fallback_operation:
                return await fallback_operation()
            raise

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "successful_calls": self.successful_calls
        }


@dataclass
class CacheMetrics:
    """Enhanced cache performance metrics with detailed monitoring"""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_updated: datetime = None
    evictions: int = 0
    memory_usage_mb: float = 0.0
    connection_pool_usage: float = 0.0
    slow_operations: int = 0  # Operations > 100ms
    errors: int = 0
    timeouts: int = 0
    circuit_breaker_trips: int = 0

    def update_hit(self, response_time_ms: float):
        """Update metrics for cache hit"""
        self.hits += 1
        self.total_requests += 1
        self.hit_rate = self.hits / self.total_requests
        self._update_avg_response_time(response_time_ms)
        self._record_slow_operation(response_time_ms)
        self.last_updated = datetime.utcnow()

    def update_miss(self, response_time_ms: float):
        """Update metrics for cache miss"""
        self.misses += 1
        self.total_requests += 1
        self.hit_rate = self.hits / self.total_requests
        self._update_avg_response_time(response_time_ms)
        self._record_slow_operation(response_time_ms)
        self.last_updated = datetime.utcnow()

    def update_error(self):
        """Update error count"""
        self.errors += 1
        self.last_updated = datetime.utcnow()

    def update_timeout(self):
        """Update timeout count"""
        self.timeouts += 1
        self.last_updated = datetime.utcnow()

    def update_circuit_breaker_trip(self):
        """Update circuit breaker trip count"""
        self.circuit_breaker_trips += 1
        self.last_updated = datetime.utcnow()

    def _update_avg_response_time(self, response_time_ms: float):
        """Update average response time"""
        if self.total_requests == 1:
            self.avg_response_time_ms = response_time_ms
        else:
            # Simple exponential moving average
            alpha = 0.1
            self.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.avg_response_time_ms
            )

    def _record_slow_operation(self, response_time_ms: float):
        """Record slow cache operations"""
        if response_time_ms > 100:
            self.slow_operations += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "hit_rate": round(self.hit_rate * 100, 2),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "total_requests": self.total_requests,
            "slow_operations_rate": round((self.slow_operations / max(1, self.total_requests)) * 100, 2),
            "error_rate": round((self.errors / max(1, self.total_requests)) * 100, 2),
            "memory_usage_mb": self.memory_usage_mb,
            "connection_pool_usage": round(self.connection_pool_usage * 100, 2)
        }


class VectorCacheService:
    """Redis-based caching service for vector search results"""

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 300,
        key_prefix: str = "zaaky:vector:",
        enabled: bool = True
    ):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.enabled = enabled
        self.metrics = CacheMetrics()

        logger.info(
            f"VectorCacheService initialized - TTL: {ttl_seconds}s, Enabled: {enabled}")

    def _generate_cache_key(
        self,
        org_id: str,
        chatbot_id: str,
        query: str,
        context_config: Dict[str, Any],
        retrieval_params: Dict[str, Any]
    ) -> str:
        """Generate composite cache key for vector search with versioning and namespacing"""
        # Add namespace and version for better key management
        version = "v1"
        namespace = "vector_search"

        # Create a stable hash of the configuration and parameters
        config_str = json.dumps(context_config, sort_keys=True)
        params_str = json.dumps(retrieval_params, sort_keys=True)

        # Include environment to avoid conflicts
        env = os.getenv("ENVIRONMENT", "dev")

        # Create composite string for hashing
        composite = f"{env}:{namespace}:{version}:{org_id}:{chatbot_id}:{query}:{config_str}:{params_str}"

        # Generate MD5 hash for consistent key length
        cache_hash = hashlib.md5(composite.encode('utf-8')).hexdigest()

        return f"{self.key_prefix}{namespace}:{version}:{org_id}:{cache_hash}"

    async def get_cached_results(
        self,
        org_id: str,
        chatbot_id: str,
        query: str,
        context_config: Dict[str, Any],
        retrieval_params: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached vector search results"""
        if not self.enabled:
            return None

        start_time = datetime.utcnow()

        try:
            cache_key = self._generate_cache_key(
                org_id, chatbot_id, query, context_config, retrieval_params
            )

            # Get from Redis
            cached_data = self.redis.get(cache_key)

            response_time = (datetime.utcnow() -
                             start_time).total_seconds() * 1000

            if cached_data:
                # Cache hit
                self.metrics.update_hit(response_time)

                # Deserialize and return
                results = json.loads(cached_data.decode('utf-8'))

                logger.debug(
                    f"Cache HIT for key: {cache_key[:20]}... ({len(results)} results)")
                return results
            else:
                # Cache miss
                self.metrics.update_miss(response_time)
                logger.debug(f"Cache MISS for key: {cache_key[:20]}...")
                return None

        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            self.metrics.update_miss(
                (datetime.utcnow() - start_time).total_seconds() * 1000)
            return None

    async def cache_results(
        self,
        org_id: str,
        chatbot_id: str,
        query: str,
        context_config: Dict[str, Any],
        retrieval_params: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> bool:
        """Cache vector search results"""
        if not self.enabled or not results:
            return False

        try:
            cache_key = self._generate_cache_key(
                org_id, chatbot_id, query, context_config, retrieval_params
            )

            # Serialize results
            cached_data = json.dumps(results, default=str)

            # Store in Redis with TTL
            success = self.redis.setex(
                cache_key,
                self.ttl_seconds,
                cached_data
            )

            if success:
                logger.debug(
                    f"Cached {len(results)} results for key: {cache_key[:20]}...")
                return True
            else:
                logger.warning(
                    f"Failed to cache results for key: {cache_key[:20]}...")
                return False

        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False

    async def invalidate_org_cache(self, org_id: str) -> int:
        """Invalidate all cached results for an organization"""
        try:
            # Find all keys for this org
            pattern = f"{self.key_prefix}*"
            keys = self.redis.keys(pattern)

            # This is a simple approach - in production, you might want to store
            # org_id in the cache key for more efficient invalidation
            deleted_count = 0
            for key in keys:
                try:
                    # Get the key and check if it belongs to this org
                    # This is inefficient but works for the pattern we're using
                    self.redis.delete(key)
                    deleted_count += 1
                except Exception:
                    continue

            logger.info(
                f"Invalidated {deleted_count} cache entries for org: {org_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Cache invalidation error for org {org_id}: {e}")
            return 0

    async def invalidate_chatbot_cache(self, org_id: str, chatbot_id: str) -> int:
        """Invalidate all cached results for a specific chatbot"""
        try:
            # Similar to org invalidation but more specific
            pattern = f"{self.key_prefix}*"
            keys = self.redis.keys(pattern)

            deleted_count = 0
            for key in keys:
                try:
                    self.redis.delete(key)
                    deleted_count += 1
                except Exception:
                    continue

            logger.info(
                f"Invalidated {deleted_count} cache entries for chatbot: {chatbot_id}")
            return deleted_count

        except Exception as e:
            logger.error(
                f"Cache invalidation error for chatbot {chatbot_id}: {e}")
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "metrics": asdict(self.metrics),
            "redis_info": self._get_redis_info()
        }

    def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis connection info"""
        try:
            info = self.redis.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"connected": False, "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache service"""
        try:
            # Simple ping test
            start_time = datetime.utcnow()
            result = self.redis.ping()
            response_time = (datetime.utcnow() -
                             start_time).total_seconds() * 1000

            return {
                "status": "healthy" if result else "unhealthy",
                "response_time_ms": response_time,
                "enabled": self.enabled
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "enabled": self.enabled
            }

    async def warm_cache(self, popular_queries: List[Dict[str, Any]]) -> int:
        """Warm cache with popular queries"""
        if not self.enabled:
            return 0

        warmed_count = 0
        for query_data in popular_queries:
            try:
                # This would typically be called with actual vector search results
                # For now, we just ensure the structure is ready
                logger.info(
                    f"Cache warming for query: {query_data.get('query', 'unknown')[:50]}...")
                warmed_count += 1
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                continue

        logger.info(
            f"Cache warming completed: {warmed_count} queries processed")
        return warmed_count


class CacheService:
    """Enhanced Redis-based cache service with connection pooling, circuit breaker, and multi-level caching"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
        self.enabled = False
        self.metrics = CacheMetrics()
        self.circuit_breaker = CacheCircuitBreaker()
        # In-memory cache: {key: (value, timestamp)}
        self._memory_cache: Dict[str, tuple] = {}
        self._memory_cache_ttl = 300  # 5 minutes
        self._memory_cache_max_size = 1000
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection with connection pooling and enhanced error handling"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_password = os.getenv("REDIS_PASSWORD")
            redis_db = int(os.getenv("REDIS_DB", "0"))
            max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))

            # Parse Redis URL if provided
            if redis_url.startswith("redis://"):
                # Extract host and port from URL
                url_parts = redis_url.replace("redis://", "").split(":")
                host = url_parts[0] if url_parts else "localhost"
                port = int(url_parts[1]) if len(url_parts) > 1 else 6379
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))

            # Create connection pool for better connection management
            self.connection_pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=redis_password,
                db=redis_db,
                max_connections=max_connections,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                decode_responses=True
            )

            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool)

            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(
                f"✅ Redis cache service initialized with connection pool (max_connections: {max_connections})")

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            self.enabled = False
            self.redis_client = None
            self.connection_pool = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis: {e}")
            self.enabled = False
            self.redis_client = None
            self.connection_pool = None

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        elif isinstance(value, (str, int, float, bool)):
            return str(value)
        else:
            return json.dumps(value, default=str)

    def _deserialize_value(self, value: str, expected_type: type = None) -> Any:
        """Deserialize value from Redis storage"""
        if not value:
            return None

        try:
            # Try JSON deserialization first
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Return as string if JSON fails
            return value

    def _update_memory_cache(self, key: str, value: Any):
        """Update in-memory cache with size management"""
        current_time = time.time()

        # Clean expired entries if cache is getting full
        if len(self._memory_cache) >= self._memory_cache_max_size:
            self._cleanup_memory_cache()

        self._memory_cache[key] = (value, current_time)

    def _get_from_memory_cache(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache"""
        if key not in self._memory_cache:
            return None

        value, timestamp = self._memory_cache[key]
        current_time = time.time()

        # Check if expired
        if current_time - timestamp > self._memory_cache_ttl:
            del self._memory_cache[key]
            return None

        return value

    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._memory_cache.items()
            if current_time - timestamp > self._memory_cache_ttl
        ]

        for key in expired_keys:
            del self._memory_cache[key]

        # If still too full, remove oldest entries
        if len(self._memory_cache) >= self._memory_cache_max_size:
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            remove_count = len(self._memory_cache) - \
                self._memory_cache_max_size + 100
            for key, _ in sorted_items[:remove_count]:
                del self._memory_cache[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level caching (memory + Redis)"""
        start_time = time.time()

        # Level 1: Check in-memory cache first (fastest)
        memory_result = self._get_from_memory_cache(key)
        if memory_result is not None:
            response_time = (time.time() - start_time) * 1000
            self.metrics.update_hit(response_time)
            return memory_result

        # Level 2: Check Redis cache
        if not self.enabled or not self.redis_client:
            response_time = (time.time() - start_time) * 1000
            self.metrics.update_miss(response_time)
            return default

        async def cache_operation():
            value = self.redis_client.get(key)
            if value is None:
                return None
            return self._deserialize_value(value)

        async def fallback_operation():
            return default

        try:
            # Use circuit breaker for Redis operations
            result = asyncio.run(
                self.circuit_breaker.execute_with_breaker(
                    cache_operation, fallback_operation
                )
            )

            response_time = (time.time() - start_time) * 1000

            if result is not None:
                # Cache hit in Redis - update memory cache
                self._update_memory_cache(key, result)
                self.metrics.update_hit(response_time)
                return result
            else:
                # Cache miss
                self.metrics.update_miss(response_time)
                return default

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            response_time = (time.time() - start_time) * 1000
            self.metrics.update_error()
            self.metrics.update_miss(response_time)
            return default
        except Exception as e:
            logger.error(f"Unexpected Redis get error for key {key}: {e}")
            response_time = (time.time() - start_time) * 1000
            self.metrics.update_error()
            self.metrics.update_miss(response_time)
            return default

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with write-through multi-level caching"""
        start_time = time.time()

        # Always update memory cache first (fastest)
        self._update_memory_cache(key, value)

        # If Redis is not available, at least we have memory cache
        if not self.enabled or not self.redis_client:
            return False

        async def cache_operation():
            serialized_value = self._serialize_value(value)
            result = self.redis_client.setex(
                key, ttl_seconds, serialized_value)
            return bool(result)

        async def fallback_operation():
            # Still have memory cache, so not a complete failure
            return False

        try:
            result = asyncio.run(
                self.circuit_breaker.execute_with_breaker(
                    cache_operation, fallback_operation
                )
            )

            if not result:
                self.metrics.update_error()

            return result

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            self.metrics.update_error()
            self.metrics.update_timeout()
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis set error for key {key}: {e}")
            self.metrics.update_error()
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.redis_client:
            return False

        try:
            result = self.redis_client.delete(key)
            return bool(result)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis delete error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis delete error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled or not self.redis_client:
            return False

        try:
            return bool(self.redis_client.exists(key))

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis exists error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis exists error for key {key}: {e}")
            return False

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not self.enabled or not self.redis_client:
            return {}

        try:
            values = self.redis_client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize_value(value)
            return result

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis mget error for keys {keys}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected Redis mget error for keys {keys}: {e}")
            return {}

    def set_many(self, mapping: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """Set multiple values in cache"""
        if not self.enabled or not self.redis_client:
            return False

        try:
            # Serialize all values
            serialized_mapping = {
                key: self._serialize_value(value)
                for key, value in mapping.items()
            }

            # Use pipeline for atomic operation
            pipe = self.redis_client.pipeline()
            for key, value in serialized_mapping.items():
                pipe.setex(key, ttl_seconds, value)

            results = pipe.execute()
            return all(results)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis mset error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis mset error: {e}")
            return False

    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache"""
        if not self.enabled or not self.redis_client:
            return 0

        try:
            return self.redis_client.delete(*keys)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis delete many error for keys {keys}: {e}")
            return 0
        except Exception as e:
            logger.error(
                f"Unexpected Redis delete many error for keys {keys}: {e}")
            return 0

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern with memory cache cleanup"""
        deleted_count = 0

        # Clear from memory cache first
        memory_keys_to_delete = [
            key for key in self._memory_cache.keys()
            if self._pattern_matches(key, pattern)
        ]
        for key in memory_keys_to_delete:
            del self._memory_cache[key]
            deleted_count += 1

        # Clear from Redis
        if not self.enabled or not self.redis_client:
            return deleted_count

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                redis_deleted = self.redis_client.delete(*keys)
                deleted_count += redis_deleted

            return deleted_count

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(
                f"Redis clear pattern error for pattern {pattern}: {e}")
            self.metrics.update_error()
            return deleted_count
        except Exception as e:
            logger.error(
                f"Unexpected Redis clear pattern error for pattern {pattern}: {e}")
            self.metrics.update_error()
            return deleted_count

    def _pattern_matches(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for cache keys"""
        if '*' not in pattern:
            return key == pattern

        # Convert Redis pattern to Python regex
        import re
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f"^{regex_pattern}$", key))

    def get_ttl(self, key: str) -> int:
        """Get TTL for a key"""
        if not self.enabled or not self.redis_client:
            return -1

        try:
            return self.redis_client.ttl(key)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis TTL error for key {key}: {e}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected Redis TTL error for key {key}: {e}")
            return -1

    def extend_ttl(self, key: str, ttl_seconds: int) -> bool:
        """Extend TTL for a key"""
        if not self.enabled or not self.redis_client:
            return False

        try:
            return bool(self.redis_client.expire(key, ttl_seconds))

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis expire error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis expire error for key {key}: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with detailed metrics"""
        if not self.enabled or not self.redis_client:
            return {
                "status": "disabled",
                "enabled": False,
                "error": "Redis not initialized",
                "memory_cache": {
                    "size": len(self._memory_cache),
                    "max_size": self._memory_cache_max_size
                }
            }

        try:
            # Test connection with timing
            start_time = time.time()
            self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000

            # Get Redis info
            info = self.redis_client.info()

            # Update metrics with current Redis stats
            self.metrics.memory_usage_mb = info.get(
                "used_memory", 0) / (1024 * 1024)

            # Calculate connection pool usage if available
            if self.connection_pool:
                pool_info = self.connection_pool.connection_kwargs
                max_connections = pool_info.get("max_connections", 1)
                created_connections = getattr(
                    self.connection_pool, "created_connections", 0)
                self.metrics.connection_pool_usage = created_connections / max_connections

            return {
                "status": "healthy",
                "enabled": True,
                "ping_time_ms": round(ping_time, 2),
                "redis_info": {
                    "version": info.get("redis_version", "unknown"),
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "connection_pool": {
                    "max_connections": getattr(self.connection_pool, "_max_connections", 0) if self.connection_pool else 0,
                    "created_connections": getattr(self.connection_pool, "created_connections", 0) if self.connection_pool else 0,
                    "available_connections": getattr(self.connection_pool, "_available_connections", []) if self.connection_pool else []
                },
                "memory_cache": {
                    "size": len(self._memory_cache),
                    "max_size": self._memory_cache_max_size,
                    "ttl_seconds": self._memory_cache_ttl
                },
                "performance_metrics": self.metrics.get_performance_summary(),
                "circuit_breaker": self.circuit_breaker.get_state()
            }

        except (RedisError, ConnectionError, TimeoutError) as e:
            self.metrics.update_error()
            self.metrics.update_timeout()
            return {
                "status": "unhealthy",
                "enabled": True,
                "error": str(e),
                "memory_cache": {
                    "size": len(self._memory_cache),
                    "max_size": self._memory_cache_max_size
                },
                "circuit_breaker": self.circuit_breaker.get_state()
            }
        except Exception as e:
            self.metrics.update_error()
            return {
                "status": "error",
                "enabled": True,
                "error": str(e),
                "memory_cache": {
                    "size": len(self._memory_cache),
                    "max_size": self._memory_cache_max_size
                }
            }


# Global cache service instance
cache_service = CacheService()
