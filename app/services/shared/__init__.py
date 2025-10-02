"""
Shared services module

This module contains shared services and utilities used across multiple domains
including client management, worker scheduling, and common utilities.
"""

# Lazy imports to avoid initialization errors when API keys are not available
def get_client_manager():
    """Get client manager instance (lazy import)"""
    from .client_manager import client_manager
    return client_manager

def get_worker_scheduler():
    """Get worker scheduler class (lazy import)"""
    from .worker_scheduler import IngestionWorkerScheduler
    return IngestionWorkerScheduler

def get_cache_service():
    """Get cache service instance (lazy import)"""
    from .cache_service import cache_service
    return cache_service

# Direct imports for services that don't require API keys
from .cache_service import cache_service

__all__ = [
    "get_client_manager",
    "get_worker_scheduler", 
    "get_cache_service",
    "cache_service"
]
