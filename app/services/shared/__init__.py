"""
Shared services and utilities

This module provides centralized access to shared services like database clients,
external APIs, and caching systems used across the application.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional

import openai
import redis

from ..storage.pinecone_client import get_pinecone_index as _get_pinecone_index
from ..storage.supabase_client import get_supabase_client as _get_supabase_client
from .cache_service import VectorCacheService, cache_service
from .cache_warming_service import cache_warmup_service
from .optimized_vector_search import OptimizedVectorSearch
from .vector_search_cache import vector_search_cache

logger = logging.getLogger(__name__)


class ClientManager:
    """Centralized manager for all external service clients"""

    def __init__(self):
        self.openai: Optional[openai.OpenAI] = None
        self.pinecone_index = None
        self.supabase = None
        self.redis: Optional[redis.Redis] = None
        self.vector_cache: Optional[VectorCacheService] = None
        self._initialized = False

    def initialize(self):
        """Initialize all clients"""
        if self._initialized:
            return

        try:
            # Initialize OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai = openai.OpenAI(api_key=openai_key)
                logger.info("✅ OpenAI client initialized")
            else:
                logger.warning("⚠️ OpenAI API key not found")

            # Initialize Pinecone using centralized client
            try:
                self.pinecone_index = _get_pinecone_index()
                logger.info("✅ Pinecone client initialized")
            except Exception as e:
                logger.warning("⚠️ Pinecone initialization failed: %s", e)
                self.pinecone_index = None

            # Initialize Supabase using centralized client
            try:
                self.supabase = _get_supabase_client()
                logger.info("✅ Supabase client initialized")
            except Exception as e:
                logger.error("❌ Failed to initialize Supabase: %s", e)
                self.supabase = None

            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=False,  # We handle encoding manually
                    socket_connect_timeout=2,  # Reduced from 5 to 2 seconds
                    socket_timeout=2,  # Reduced from 5 to 2 seconds
                    retry_on_timeout=False,  # Don't retry on timeout
                    health_check_interval=30,
                )

                # Test connection with timeout (non-blocking)
                # Even if ping fails, we'll create the client and let it fail gracefully on use
                try:
                    self.redis.ping()
                    logger.info("✅ Redis client initialized and connected")
                    redis_available = True
                except Exception as ping_error:
                    logger.warning(
                        "⚠️ Redis client created but connection test failed (non-critical): %s",
                        ping_error,
                    )
                    # Redis client is created but may not be fully connected
                    # This is OK - the server can start and Redis will be checked again on use
                    redis_available = False

                # Initialize vector cache service (even if Redis ping failed)
                # VectorCacheService should handle Redis unavailability gracefully
                cache_enabled = (
                    os.getenv("VECTOR_CACHE_ENABLED", "true").lower() == "true"
                )
                cache_ttl = int(os.getenv("VECTOR_CACHE_TTL", "300"))
                cache_prefix = os.getenv("VECTOR_CACHE_PREFIX", "zaaky:vector:")

                self.vector_cache = VectorCacheService(
                    redis_client=self.redis,
                    ttl_seconds=cache_ttl,
                    key_prefix=cache_prefix,
                    enabled=cache_enabled,
                )
                if redis_available:
                    logger.info(
                        "✅ Vector cache service initialized (TTL: %ss, Enabled: %s)",
                        cache_ttl,
                        cache_enabled,
                    )
                else:
                    logger.info(
                        "⚠️ Vector cache service initialized but Redis may be unavailable (TTL: %ss, Enabled: %s)",
                        cache_ttl,
                        cache_enabled,
                    )

            except Exception as e:
                logger.warning("⚠️ Redis initialization failed: %s", e)
                self.redis = None
                self.vector_cache = None

            self._initialized = True
            logger.info("🎉 All clients initialized successfully")

        except Exception as e:
            logger.error("❌ Client initialization failed: %s", e)
            raise

    def health_check(self, timeout: float = 2.0) -> Dict[str, bool]:
        """
        Check health of all clients with timeout protection

        Args:
            timeout: Maximum time in seconds to wait for each health check (default: 2.0)

        Returns:
            Dictionary mapping client names to health status (True/False)
        """
        health = {}

        def check_with_timeout(check_func, client_name: str, default: bool = False):
            """Execute a health check with timeout using threading"""
            result = {"status": default, "error": None, "completed": False}

            def run_check():
                try:
                    check_result = check_func()
                    result["status"] = (
                        bool(check_result) if check_result is not None else True
                    )
                except Exception as e:
                    result["error"] = str(e)
                    result["status"] = False
                finally:
                    result["completed"] = True

            thread = threading.Thread(target=run_check, daemon=True)
            thread.start()
            thread.join(timeout=timeout)

            if not result["completed"]:
                # Thread is still running, health check timed out
                logger.warning(
                    "Health check for %s timed out after %s seconds (non-critical)",
                    client_name,
                    timeout,
                )
                return False

            if result.get("error"):
                logger.debug(
                    "Health check for %s failed: %s (non-critical)",
                    client_name,
                    result["error"],
                )

            return result["status"]

        # OpenAI health check with timeout
        if self.openai:

            def check_openai():
                try:
                    self.openai.models.list()
                    return True
                except Exception:
                    return False

            health["openai"] = check_with_timeout(check_openai, "openai")
        else:
            health["openai"] = False

        # Pinecone health check with timeout
        if self.pinecone_index:

            def check_pinecone():
                try:
                    self.pinecone_index.query(
                        vector=[0.0] * 1536, top_k=1, include_metadata=False
                    )
                    return True
                except Exception:
                    return False

            health["pinecone"] = check_with_timeout(check_pinecone, "pinecone")
        else:
            health["pinecone"] = False

        # Supabase health check with timeout
        if self.supabase:

            def check_supabase():
                try:
                    self.supabase.table("organizations").select("id").limit(1).execute()
                    return True
                except Exception:
                    return False

            health["supabase"] = check_with_timeout(check_supabase, "supabase")
        else:
            health["supabase"] = False

        # Redis health check with timeout
        if self.redis:

            def check_redis():
                try:
                    self.redis.ping()
                    return True
                except Exception:
                    return False

            health["redis"] = check_with_timeout(check_redis, "redis")
        else:
            health["redis"] = False

        return health


# Global client manager instance
_client_manager: Optional[ClientManager] = None


def get_client_manager() -> ClientManager:
    """Get the global client manager instance"""
    global _client_manager
    if _client_manager is None:
        _client_manager = ClientManager()
        _client_manager.initialize()
    return _client_manager


# Legacy exports for backwards compatibility


def get_openai_client():
    """Get OpenAI client (legacy)"""
    return get_client_manager().openai


def get_pinecone_index():
    """Get Pinecone index (legacy)"""
    # Use centralized client directly
    return _get_pinecone_index()


def get_supabase_client():
    """Get Supabase client (legacy)"""
    return get_client_manager().supabase


def get_redis_client():
    """Get Redis client"""
    return get_client_manager().redis


def get_vector_cache():
    """Get Vector cache service (legacy)"""
    return get_client_manager().vector_cache


def get_vector_search_cache():
    """Get optimized vector search cache service"""
    return vector_search_cache


# Cache service instance
cache_service = None


def get_cache_service():
    """Get cache service instance"""
    global cache_service
    if cache_service is None:
        vector_cache = get_vector_cache()
        if vector_cache:
            cache_service = vector_cache
    return cache_service


__all__ = [
    "ClientManager",
    "get_client_manager",
    "get_openai_client",
    "get_pinecone_index",
    "get_supabase_client",
    "get_redis_client",
    "get_vector_cache",
    "get_vector_search_cache",
    "get_cache_service",
    "vector_search_cache",
    "cache_service",
]
