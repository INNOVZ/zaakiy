"""
Shared services and utilities

This module provides centralized access to shared services like database clients,
external APIs, and caching systems used across the application.
"""

import os
import logging
from typing import Optional, Dict, Any
import openai
import redis
from pinecone import Pinecone
from supabase import create_client, Client

from .cache_service import VectorCacheService

logger = logging.getLogger(__name__)

class ClientManager:
    """Centralized manager for all external service clients"""
    
    def __init__(self):
        self.openai: Optional[openai.OpenAI] = None
        self.pinecone_index = None
        self.supabase: Optional[Client] = None
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
            
            # Initialize Pinecone
            pinecone_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = os.getenv("PINECONE_INDEX")
            if pinecone_key and pinecone_index_name:
                pc = Pinecone(api_key=pinecone_key)
                self.pinecone_index = pc.Index(pinecone_index_name)
                logger.info("✅ Pinecone client initialized")
            else:
                logger.warning("⚠️ Pinecone configuration not found")
            
            # Initialize Supabase
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase client initialized")
            else:
                logger.warning("⚠️ Supabase configuration not found")
            
            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=False,  # We handle encoding manually
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                self.redis.ping()
                logger.info("✅ Redis client initialized")
                
                # Initialize vector cache service
                cache_enabled = os.getenv("VECTOR_CACHE_ENABLED", "true").lower() == "true"
                cache_ttl = int(os.getenv("VECTOR_CACHE_TTL", "300"))
                cache_prefix = os.getenv("VECTOR_CACHE_PREFIX", "zaaky:vector:")
                
                self.vector_cache = VectorCacheService(
                    redis_client=self.redis,
                    ttl_seconds=cache_ttl,
                    key_prefix=cache_prefix,
                    enabled=cache_enabled
                )
                logger.info(f"✅ Vector cache service initialized (TTL: {cache_ttl}s, Enabled: {cache_enabled})")
                
            except Exception as e:
                logger.warning(f"⚠️ Redis initialization failed: {e}")
                self.redis = None
                self.vector_cache = None
            
            self._initialized = True
            logger.info("🎉 All clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Client initialization failed: {e}")
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
                    vector=[0.0] * 1536,
                    top_k=1,
                    include_metadata=False
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
    return get_client_manager().pinecone_index

def get_supabase_client():
    """Get Supabase client (legacy)"""
    return get_client_manager().supabase

def get_redis_client():
    """Get Redis client"""
    return get_client_manager().redis

def get_vector_cache():
    """Get Vector cache service"""
    return get_client_manager().vector_cache

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
    "get_cache_service"
]