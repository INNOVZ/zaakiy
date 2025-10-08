"""
Cache warming service for proactive cache population
"""
import logging
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from .cache_service import cache_service

logger = logging.getLogger(__name__)


class CacheWarmupService:
    """Service to warm up caches proactively for better performance"""

    def __init__(self):
        self.enabled = True
        self.warmup_in_progress = False
        self.last_warmup = None
        self.warmup_interval_minutes = 30

    async def warm_critical_caches(self) -> Dict[str, Any]:
        """Warm up frequently accessed caches"""
        if self.warmup_in_progress:
            logger.info("Cache warmup already in progress, skipping")
            return {"status": "skipped", "reason": "warmup_in_progress"}

        self.warmup_in_progress = True
        start_time = datetime.utcnow()
        results = {
            "status": "success",
            "start_time": start_time.isoformat(),
            "configurations_warmed": 0,
            "queries_warmed": 0,
            "organizations_processed": 0,
            "errors": []
        }

        try:
            logger.info("🔥 Starting cache warmup process")

            # Get active organizations
            active_orgs = await self._get_active_organizations()
            results["organizations_processed"] = len(active_orgs)

            for org_id in active_orgs:
                try:
                    # Warm up configurations
                    config_count = await self._warm_org_configurations(org_id)
                    results["configurations_warmed"] += config_count

                    # Warm up popular queries
                    query_count = await self._warm_popular_queries(org_id)
                    results["queries_warmed"] += query_count

                    logger.debug(
                        "Warmed cache for org %s: %d configs, %d queries", org_id, config_count, query_count)

                except Exception as e:
                    error_msg = f"Cache warming failed for org {org_id}: {e}"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)

            self.last_warmup = datetime.utcnow()
            duration_seconds = (self.last_warmup - start_time).total_seconds()
            results["duration_seconds"] = duration_seconds
            results["end_time"] = self.last_warmup.isoformat()

            logger.info(
                "✅ Cache warmup completed: %d configs, %d queries in %.2fs",
                results['configurations_warmed'], results['queries_warmed'], duration_seconds
            )

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            logger.error("Cache warmup process failed: %s", e)

        finally:
            self.warmup_in_progress = False

        return results

    async def _get_active_organizations(self) -> List[str]:
        """Get list of active organization IDs"""
        try:
            # This would typically query your database for active organizations
            # For now, we'll use a simple approach based on cache keys

            # You might want to replace this with actual database query
            # Example: SELECT org_id FROM organizations WHERE status = 'active' AND last_activity > NOW() - INTERVAL '7 days'

            # For demonstration, we'll return some sample org IDs
            # In real implementation, query your supabase/database
            active_orgs = []

            # Check if we have any cached data that indicates active orgs
            cache_keys = cache_service.redis_client.keys(
                "*") if cache_service.enabled else []
            org_ids_from_cache = set()

            for key in cache_keys:
                # Extract org IDs from cache keys (assuming they contain org IDs)
                parts = key.split(":")
                for part in parts:
                    if part.startswith("org-") or (part.isdigit() and len(part) > 5):
                        org_ids_from_cache.add(part)

            active_orgs = list(org_ids_from_cache)[
                :10]  # Limit to 10 for safety

            logger.debug(
                "Found %d potentially active organizations", len(active_orgs))
            return active_orgs

        except Exception as e:
            logger.error("Failed to get active organizations: %s", e)
            return []

    async def _warm_org_configurations(self, org_id: str) -> int:
        """Warm up configuration cache for an organization"""
        try:
            # Import here to avoid circular imports
            from ..analytics.context_config import context_config_manager

            # Pre-load default configuration
            await context_config_manager.get_config(org_id)

            # Pre-load any custom configurations if they exist
            config_count = 1

            # You might want to add logic to warm up custom configurations
            # custom_configs = await self._get_custom_configs(org_id)
            # for config_name in custom_configs:
            #     await context_config_manager.get_config(org_id, config_name)
            #     config_count += 1

            return config_count

        except Exception as e:
            logger.warning(
                "Failed to warm configurations for org %s: %s", org_id, e)
            return 0

    async def _warm_popular_queries(self, org_id: str) -> int:
        """Warm up cache for popular queries"""
        try:
            popular_queries = await self._get_popular_queries(org_id)
            warmed_count = 0

            for query_data in popular_queries[:5]:  # Top 5 queries
                try:
                    # This would involve running the actual vector search to populate cache
                    # For now, we'll just simulate cache warming
                    cache_key = f"popular_query:v1:{org_id}:{hash(query_data.get('query', ''))}"

                    # Check if already cached
                    if not cache_service.exists(cache_key):
                        # In real implementation, you'd run the actual search here
                        # search_results = await vector_search_service.search(query_data)
                        # cache_service.set(cache_key, search_results, 1800)  # 30 minutes

                        # For now, just mark as warmed
                        cache_service.set(
                            cache_key, {"warmed": True, "query": query_data.get("query")}, 1800)
                        warmed_count += 1

                except Exception as e:
                    logger.warning("Failed to warm query cache: %s", e)
                    continue

            return warmed_count

        except Exception as e:
            logger.warning(
                "Failed to warm popular queries for org %s: %s", org_id, e)
            return 0

    async def _get_popular_queries(self, org_id: str) -> List[Dict[str, Any]]:
        """Get popular queries for an organization"""
        try:
            # This would typically query your analytics data
            # For now, return some sample popular queries
            return [
                {"query": "How do I reset my password?", "frequency": 45},
                {"query": "What are your business hours?", "frequency": 32},
                {"query": "How can I contact support?", "frequency": 28},
                {"query": "What payment methods do you accept?", "frequency": 22},
                {"query": "How do I cancel my subscription?", "frequency": 18}
            ]

        except Exception as e:
            logger.error(
                "Failed to get popular queries for org %s: %s", org_id, e)
            return []

    async def schedule_warmup(self):
        """Schedule periodic cache warmup"""
        while self.enabled:
            try:
                # Check if warmup is needed
                if self._should_run_warmup():
                    await self.warm_critical_caches()

                # Wait for next interval
                await asyncio.sleep(self.warmup_interval_minutes * 60)

            except Exception as e:
                logger.error("Scheduled cache warmup error: %s", e)
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def _should_run_warmup(self) -> bool:
        """Check if cache warmup should run"""
        if not self.enabled or self.warmup_in_progress:
            return False

        if self.last_warmup is None:
            return True

        time_since_last = datetime.utcnow() - self.last_warmup
        return time_since_last >= timedelta(minutes=self.warmup_interval_minutes)

    def get_warmup_status(self) -> Dict[str, Any]:
        """Get current warmup status"""
        return {
            "enabled": self.enabled,
            "warmup_in_progress": self.warmup_in_progress,
            "last_warmup": self.last_warmup.isoformat() if self.last_warmup else None,
            "warmup_interval_minutes": self.warmup_interval_minutes,
            "next_warmup_due": (
                (self.last_warmup +
                 timedelta(minutes=self.warmup_interval_minutes)).isoformat()
                if self.last_warmup else "immediately"
            )
        }

    def start_background_warmup(self):
        """Start background cache warming task"""
        if not self.enabled:
            logger.info("Cache warming is disabled")
            return

        logger.info("Starting background cache warmup (interval: %d minutes)",
                    self.warmup_interval_minutes)
        asyncio.create_task(self.schedule_warmup())


# Global cache warmup service instance
cache_warmup_service = CacheWarmupService()
