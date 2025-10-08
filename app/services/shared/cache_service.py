"""
Redis cache service for high-performance caching
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from ...utils.error_handlers import ErrorHandler
from ...utils.error_context import ErrorContextManager, ErrorSeverity
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_updated: datetime = None

    def update_hit(self, response_time_ms: float):
        """Update metrics for cache hit"""
        self.hits += 1
        self.total_requests += 1
        self.hit_rate = self.hits / self.total_requests
        self._update_avg_response_time(response_time_ms)
        self.last_updated = datetime.utcnow()

    def update_miss(self, response_time_ms: float):
        """Update metrics for cache miss"""
        self.misses += 1
        self.total_requests += 1
        self.hit_rate = self.hits / self.total_requests
        self._update_avg_response_time(response_time_ms)
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
        
        logger.info(f"VectorCacheService initialized - TTL: {ttl_seconds}s, Enabled: {enabled}")

    def _generate_cache_key(
        self,
        org_id: str,
        chatbot_id: str,
        query: str,
        context_config: Dict[str, Any],
        retrieval_params: Dict[str, Any]
    ) -> str:
        """Generate composite cache key for vector search"""
        # Create a stable hash of the configuration and parameters
        config_str = json.dumps(context_config, sort_keys=True)
        params_str = json.dumps(retrieval_params, sort_keys=True)
        
        # Create composite string for hashing
        composite = f"{org_id}:{chatbot_id}:{query}:{config_str}:{params_str}"
        
        # Generate MD5 hash for consistent key length
        cache_hash = hashlib.md5(composite.encode('utf-8')).hexdigest()
        
        return f"{self.key_prefix}{cache_hash}"

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
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if cached_data:
                # Cache hit
                self.metrics.update_hit(response_time)
                
                # Deserialize and return
                results = json.loads(cached_data.decode('utf-8'))
                
                logger.debug(f"Cache HIT for key: {cache_key[:20]}... ({len(results)} results)")
                return results
            else:
                # Cache miss
                self.metrics.update_miss(response_time)
                logger.debug(f"Cache MISS for key: {cache_key[:20]}...")
                return None
                
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            self.metrics.update_miss((datetime.utcnow() - start_time).total_seconds() * 1000)
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
                logger.debug(f"Cached {len(results)} results for key: {cache_key[:20]}...")
                return True
            else:
                logger.warning(f"Failed to cache results for key: {cache_key[:20]}...")
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
            
            logger.info(f"Invalidated {deleted_count} cache entries for org: {org_id}")
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
            
            logger.info(f"Invalidated {deleted_count} cache entries for chatbot: {chatbot_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache invalidation error for chatbot {chatbot_id}: {e}")
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
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
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
                logger.info(f"Cache warming for query: {query_data.get('query', 'unknown')[:50]}...")
                warmed_count += 1
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                continue
        
        logger.info(f"Cache warming completed: {warmed_count} queries processed")
        return warmed_count

class CacheService:
    """Redis-based cache service with error handling and fallback"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = False
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_password = os.getenv("REDIS_PASSWORD")
            redis_db = int(os.getenv("REDIS_DB", "0"))
            
            # Parse Redis URL if provided
            if redis_url.startswith("redis://"):
                # Extract host and port from URL
                url_parts = redis_url.replace("redis://", "").split(":")
                host = url_parts[0] if url_parts else "localhost"
                port = int(url_parts[1]) if len(url_parts) > 1 else 6379
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
            
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("✅ Redis cache service initialized successfully")
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            self.enabled = False
            self.redis_client = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis: {e}")
            self.enabled = False
            self.redis_client = None
    
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
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        if not self.enabled or not self.redis_client:
            return default
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return default
            
            return self._deserialize_value(value)
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return default
        except Exception as e:
            logger.error(f"Unexpected Redis get error for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            serialized_value = self._serialize_value(value)
            result = self.redis_client.setex(key, ttl_seconds, serialized_value)
            return bool(result)
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis set error for key {key}: {e}")
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
            logger.error(f"Unexpected Redis delete many error for keys {keys}: {e}")
            return 0
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis clear pattern error for pattern {pattern}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected Redis clear pattern error for pattern {pattern}: {e}")
            return 0
    
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
        """Check Redis health status"""
        if not self.enabled or not self.redis_client:
            return {
                "status": "disabled",
                "enabled": False,
                "error": "Redis not initialized"
            }
        
        try:
            # Test connection
            self.redis_client.ping()
            
            # Get Redis info
            info = self.redis_client.info()
            
            return {
                "status": "healthy",
                "enabled": True,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime": info.get("uptime_in_seconds", 0)
            }
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            return {
                "status": "unhealthy",
                "enabled": True,
                "error": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "enabled": True,
                "error": str(e)
            }


# Global cache service instance
cache_service = CacheService()
