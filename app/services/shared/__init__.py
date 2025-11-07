"""
Shared services and utilities

This module provides centralized access to shared services like database clients,
external APIs, and caching systems used across the application.
"""

import logging
import os
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
            except Exception as e:
                logger.error("❌ Failed to initialize Supabase: %s", e)
                self.supabase = None
            else:
                supabase_config_error = getattr(self.supabase, "_error", None)
                if supabase_config_error:
                    logger.warning(
                        "⚠️ Supabase configuration not found: %s",
                        supabase_config_error,
                    )
                else:
                    logger.info("✅ Supabase client initialized")

            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=False,  # We handle encoding manually
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Test connection
                self.redis.ping()
                logger.info("✅ Redis client initialized")

                # Initialize vector cache service
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
                logger.info(
                    "✅ Vector cache service initialized (TTL: %ss, Enabled: %s)",
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

    def health_check(self) -> Dict[str, bool]:
        """Check health of all clients"""
        health = {}

        # OpenAI health
        try:
            if self.openai:
                # Simple test call
                self.openai.models.list()
                health["openai"] = True
            else:
                health["openai"] = False
        except Exception:
            health["openai"] = False

        # Pinecone health
        try:
            if self.pinecone_index:
                # Simple query test
                self.pinecone_index.query(
                    vector=[0.0] * 1536, top_k=1, include_metadata=False
                )
                health["pinecone"] = True
            else:
                health["pinecone"] = False
        except Exception:
            health["pinecone"] = False

        # Supabase health
        try:
            if self.supabase:
                # Simple query test
                self.supabase.table("organizations").select("id").limit(1).execute()
                health["supabase"] = True
            else:
                health["supabase"] = False
        except Exception:
            health["supabase"] = False

        # Redis health
        try:
            if self.redis:
                self.redis.ping()
                health["redis"] = True
            else:
                health["redis"] = False
        except Exception:
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
