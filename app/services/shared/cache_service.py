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

logger = logging.getLogger(__name__)


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
