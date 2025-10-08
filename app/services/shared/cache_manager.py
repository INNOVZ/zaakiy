"""
Cache management utilities and monitoring
"""
import logging
from typing import Dict, Any
from .cache_service import cache_service
from .cache_warming_service import cache_warmup_service

logger = logging.getLogger(__name__)


class CacheManager:
    """Centralized cache management and monitoring"""

    def __init__(self):
        self.enabled = True

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive cache system status"""
        try:
            status = {
                "timestamp": "2024-10-08T00:00:00Z",  # This would be current timestamp
                "overall_status": "healthy",
                "cache_service": {},
                "warmup_service": {},
                "recommendations": []
            }

            # Get cache service health
            cache_health = cache_service.health_check()
            status["cache_service"] = cache_health

            # Get warmup service status
            warmup_status = cache_warmup_service.get_warmup_status()
            status["warmup_service"] = warmup_status

            # Determine overall status
            if cache_health.get("status") != "healthy":
                status["overall_status"] = "degraded"
                status["recommendations"].append(
                    "Cache service is not healthy - check Redis connection")

            # Performance recommendations
            cache_metrics = cache_health.get("performance_metrics", {})
            hit_rate = cache_metrics.get("hit_rate", 0)

            if hit_rate < 70:
                status["recommendations"].append(
                    f"Cache hit rate is low ({hit_rate}%) - consider warming more data")

            slow_operations_rate = cache_metrics.get("slow_operations_rate", 0)
            if slow_operations_rate > 10:
                status["recommendations"].append(
                    f"High slow operations rate ({slow_operations_rate}%) - check Redis performance")

            error_rate = cache_metrics.get("error_rate", 0)
            if error_rate > 5:
                status["recommendations"].append(
                    f"High error rate ({error_rate}%) - investigate cache failures")

            # Circuit breaker status
            circuit_breaker = cache_health.get("circuit_breaker", {})
            if circuit_breaker.get("state") == "OPEN":
                status["overall_status"] = "critical"
                status["recommendations"].append(
                    "Cache circuit breaker is OPEN - cache operations are failing")

            return status

        except Exception as e:
            logger.error("Failed to get cache status: %s", e)
            return {
                "timestamp": "2024-10-08T00:00:00Z",
                "overall_status": "error",
                "error": str(e),
                "recommendations": ["Unable to assess cache status - check system health"]
            }

    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """Analyze and optimize cache performance"""
        try:
            results = {
                "optimizations_applied": [],
                "warnings": [],
                "recommendations": []
            }

            cache_health = cache_service.health_check()

            # Check memory usage
            memory_usage = cache_health.get(
                "performance_metrics", {}).get("memory_usage_mb", 0)
            if memory_usage > 500:  # 500MB threshold
                results["warnings"].append(
                    f"High memory usage: {memory_usage}MB")
                results["recommendations"].append(
                    "Consider implementing cache eviction policies")

            # Check connection pool usage
            pool_usage = cache_health.get("performance_metrics", {}).get(
                "connection_pool_usage", 0)
            if pool_usage > 80:
                results["warnings"].append(
                    f"High connection pool usage: {pool_usage}%")
                results["recommendations"].append(
                    "Consider increasing Redis connection pool size")

            # Trigger cache warmup if hit rate is low
            hit_rate = cache_health.get(
                "performance_metrics", {}).get("hit_rate", 0)
            if hit_rate < 60:
                results["recommendations"].append(
                    "Triggering cache warmup to improve hit rate")
                warmup_result = await cache_warmup_service.warm_critical_caches()
                if warmup_result.get("status") == "success":
                    results["optimizations_applied"].append(
                        "Cache warmup completed successfully")
                else:
                    results["warnings"].append("Cache warmup failed")

            return results

        except Exception as e:
            logger.error("Failed to optimize cache performance: %s", e)
            return {
                "error": str(e),
                "optimizations_applied": [],
                "warnings": ["Failed to perform optimization"],
                "recommendations": []
            }

    async def invalidate_organization_cache(self, org_id: str) -> Dict[str, Any]:
        """Invalidate all cache entries for a specific organization"""
        try:
            patterns_to_clear = [
                f"*:v1:{org_id}:*",
                f"conversation_history:v1:{org_id}:*",
                f"vector_retrieval:v1:{org_id}:*",
                f"message:v1:{org_id}:*",
                f"recent_messages:v1:{org_id}:*"
            ]

            total_cleared = 0
            for pattern in patterns_to_clear:
                cleared = cache_service.clear_pattern(pattern)
                total_cleared += cleared
                logger.info(
                    "Cleared %d cache entries with pattern: %s", cleared, pattern)

            return {
                "status": "success",
                "org_id": org_id,
                "total_entries_cleared": total_cleared,
                "patterns_processed": len(patterns_to_clear)
            }

        except Exception as e:
            logger.error(
                "Failed to invalidate organization cache for %s: %s", org_id, e)
            return {
                "status": "error",
                "org_id": org_id,
                "error": str(e),
                "total_entries_cleared": 0
            }

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache usage statistics"""
        try:
            cache_health = cache_service.health_check()

            stats = {
                "cache_service": {
                    "enabled": cache_health.get("enabled", False),
                    "status": cache_health.get("status", "unknown"),
                    "performance": cache_health.get("performance_metrics", {}),
                    "redis_info": cache_health.get("redis_info", {}),
                    "memory_cache": cache_health.get("memory_cache", {}),
                    "circuit_breaker": cache_health.get("circuit_breaker", {})
                },
                "warmup_service": cache_warmup_service.get_warmup_status()
            }

            # Add recommendations based on stats
            recommendations = []
            perf_metrics = stats["cache_service"]["performance"]

            if perf_metrics.get("hit_rate", 0) < 70:
                recommendations.append(
                    "Consider implementing cache warming for frequently accessed data")

            if perf_metrics.get("error_rate", 0) > 2:
                recommendations.append("Investigate source of cache errors")

            if stats["cache_service"]["circuit_breaker"].get("state") != "CLOSED":
                recommendations.append(
                    "Cache circuit breaker is not in normal state")

            stats["recommendations"] = recommendations

            return stats

        except Exception as e:
            logger.error("Failed to get cache statistics: %s", e)
            return {
                "error": str(e),
                "recommendations": ["Unable to retrieve cache statistics"]
            }


# Global cache manager instance
cache_manager = CacheManager()
