"""
Performance Monitoring Utility
Provides tools to monitor and log chat response performance
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log performance metrics for chat operations"""

    def __init__(self):
        self.metrics = {}

    @asynccontextmanager
    async def track_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager to track operation performance

        Usage:
            async with perf_monitor.track_operation("document_retrieval", {"org_id": "123"}):
                # Your code here
                results = await retrieve_documents()
        """
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time * 1000)}"

        try:
            yield operation_id
        finally:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log performance
            log_data = {
                "operation": operation_name,
                "duration_ms": duration_ms,
                "operation_id": operation_id,
            }

            if metadata:
                log_data.update(metadata)

            # Log based on performance thresholds
            if duration_ms > 5000:
                logger.warning(
                    f"SLOW OPERATION: {operation_name} took {duration_ms}ms",
                    extra=log_data,
                )
            elif duration_ms > 2000:
                logger.info(
                    f"Operation {operation_name} took {duration_ms}ms", extra=log_data
                )
            else:
                logger.debug(
                    f"Operation {operation_name} took {duration_ms}ms", extra=log_data
                )

            # Store in metrics
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []

            self.metrics[operation_name].append(
                {
                    "duration_ms": duration_ms,
                    "timestamp": start_time,
                    "metadata": metadata or {},
                }
            )

            # Keep only last 100 metrics per operation to avoid memory bloat
            if len(self.metrics[operation_name]) > 100:
                self.metrics[operation_name] = self.metrics[operation_name][-100:]

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return {
                "operation": operation_name,
                "count": 0,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "p95_ms": 0,
            }

        durations = [m["duration_ms"] for m in self.metrics[operation_name]]
        durations.sort()

        count = len(durations)
        avg_ms = sum(durations) / count if count > 0 else 0
        min_ms = durations[0] if count > 0 else 0
        max_ms = durations[-1] if count > 0 else 0
        p95_index = int(count * 0.95)
        p95_ms = durations[p95_index] if count > 0 else 0

        return {
            "operation": operation_name,
            "count": count,
            "avg_ms": round(avg_ms, 2),
            "min_ms": min_ms,
            "max_ms": max_ms,
            "p95_ms": p95_ms,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked operations"""
        return {
            operation: self.get_operation_stats(operation)
            for operation in self.metrics.keys()
        }

    def clear_metrics(self):
        """Clear all stored metrics"""
        self.metrics = {}
        logger.info("Performance metrics cleared")


# Global instance
performance_monitor = PerformanceMonitor()


# Decorator for easy performance tracking
def track_performance(operation_name: str):
    """
    Decorator to track function performance

    Usage:
        @track_performance("database_query")
        async def fetch_data():
            # Your code here
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with performance_monitor.track_operation(operation_name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
