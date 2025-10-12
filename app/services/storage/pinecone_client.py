import logging

from pinecone import Pinecone

from ...config.settings import get_ai_config

logger = logging.getLogger(__name__)

# Get configuration
ai_config = get_ai_config()

# Validate required configuration
if not ai_config.pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is required")
if not ai_config.pinecone_index:
    raise ValueError("PINECONE_INDEX environment variable is required")

logger.info("Pinecone client configuration loaded")

# Global variables for lazy initialization
_pc = None
_index = None


def get_pinecone_index():
    """Get the Pinecone index instance with lazy initialization"""
    global _pc, _index

    if _index is None:
        logger.info("Initializing Pinecone client...")
        _pc = Pinecone(api_key=ai_config.pinecone_api_key)
        _index = _pc.Index(ai_config.pinecone_index)
        logger.info("Pinecone client initialized successfully")

    return _index


def get_connection_stats() -> dict:
    """Get connection statistics for monitoring"""
    # For now, return basic stats since we don't have detailed connection pooling
    current_index = get_pinecone_index() if _index is None else _index
    return {
        "pool_size": 1,  # Single client instance
        "total_connections": 1,
        "active_connections": 1 if current_index else 0,
        "failed_connections": 0,
        "client_initialized": current_index is not None,
    }
