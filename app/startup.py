"""
Application Startup Script
Initializes cache warming and other startup tasks
"""
import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


async def warm_cache_on_startup():
    """Warm cache with common queries on application startup"""
    try:
        from app.utils.robust_cache import warm_common_queries

        logger.info("🔥 Starting cache warming...")
        await warm_common_queries()
        logger.info("✅ Cache warming completed")

    except Exception as e:
        logger.warning(f"Cache warming failed (non-critical): {e}")


async def initialize_robust_cache():
    """Initialize robust cache service"""
    try:
        from app.utils.robust_cache import get_robust_cache

        robust_cache = get_robust_cache()
        if robust_cache:
            logger.info("✅ Robust cache service initialized")

            # Log initial metrics
            metrics = robust_cache.get_metrics()
            logger.info(f"Cache metrics: {metrics}")
        else:
            logger.warning("⚠️  Cache service not available")

    except Exception as e:
        logger.warning(f"Robust cache initialization failed: {e}")


async def run_startup_tasks():
    """Run all startup tasks"""
    logger.info("🚀 Running startup tasks...")

    # Initialize robust cache
    await initialize_robust_cache()

    # Warm cache
    await warm_cache_on_startup()

    logger.info("✅ Startup tasks completed")


def setup_startup_hook(app):
    """
    Setup startup hook for FastAPI application

    Usage:
        from app.startup import setup_startup_hook

        app = FastAPI()
        setup_startup_hook(app)
    """

    @app.on_event("startup")
    async def startup_event():
        await run_startup_tasks()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Log cache metrics on shutdown"""
        try:
            from app.utils.robust_cache import get_robust_cache

            robust_cache = get_robust_cache()
            if robust_cache:
                metrics = robust_cache.get_metrics()
                logger.info(f"Final cache metrics: {metrics}")
        except Exception as e:
            logger.warning(f"Error logging final metrics: {e}")
