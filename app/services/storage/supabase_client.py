import logging

import httpx
from supabase import Client, create_client

from ...config.settings import get_database_config

logger = logging.getLogger(__name__)

# Get configuration
db_config = get_database_config()

# Validate required configuration
if not db_config.supabase_url:
    raise ValueError("SUPABASE_URL environment variable is required")
if not db_config.supabase_service_key:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")

logger.info("Initializing Supabase clients with centralized configuration")

# Create headers for HTTP client
headers = {
    "apikey": db_config.supabase_service_key,
    "Authorization": f"Bearer {db_config.supabase_service_key}",
    "Content-Type": "application/json",
}

logger.info("Supabase client configuration loaded")

# Global variables for lazy initialization
_client = None
_supabase = None


def get_supabase_client() -> Client:
    """Get the Supabase client instance with lazy initialization"""
    global _client, _supabase

    if _supabase is None:
        logger.info("Initializing Supabase clients...")

        # HTTP client for REST API calls
        _client = httpx.AsyncClient(
            base_url=f"{db_config.supabase_url}/rest/v1", headers=headers, timeout=30.0
        )

        # Supabase client for ORM-style operations
        _supabase = create_client(
            db_config.supabase_url, db_config.supabase_service_key
        )

        logger.info("Supabase clients initialized successfully")

    return _supabase


def get_supabase_http_client():
    """Get the HTTP client instance for REST API calls"""
    # Initialize clients if not already done
    get_supabase_client()  # This will initialize both clients
    return _client


def get_connection_stats() -> dict:
    """Get connection statistics for monitoring"""
    # For now, return basic stats since we don't have detailed connection pooling
    current_supabase = get_supabase_client() if _supabase is None else _supabase
    return {
        "pool_size": 1,  # Single client instance
        "total_connections": 1,
        "active_connections": 1 if current_supabase else 0,
        "failed_connections": 0,
        "client_initialized": current_supabase is not None,
    }
