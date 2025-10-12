"""
Database query optimization utilities

Provides helpers to prevent N+1 queries, optimize joins,
and add query performance monitoring.
"""
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryPerformanceMonitor:
    """Monitor and log slow database queries"""

    def __init__(self, slow_query_threshold_ms: int = 1000):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.query_stats = []

    @contextmanager
    def monitor_query(self, query_name: str, query_params: Dict[str, Any] = None):
        """
        Context manager to monitor query performance

        Usage:
            with monitor.monitor_query("list_uploads", {"org_id": "123"}):
                result = supabase.table("uploads").select("*").execute()
        """
        start_time = time.time()
        error = None

        try:
            yield
        except Exception as e:
            error = e
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Log slow queries
            if duration_ms > self.slow_query_threshold_ms:
                logger.warning(
                    f"Slow query detected: {query_name}",
                    extra={
                        "query_name": query_name,
                        "duration_ms": duration_ms,
                        "params": query_params,
                        "error": str(error) if error else None,
                    },
                )

            # Store stats
            self.query_stats.append(
                {
                    "query_name": query_name,
                    "duration_ms": duration_ms,
                    "params": query_params,
                    "success": error is None,
                    "timestamp": time.time(),
                }
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        if not self.query_stats:
            return {"total_queries": 0, "avg_duration_ms": 0, "slow_queries": 0}

        durations = [s["duration_ms"] for s in self.query_stats]

        return {
            "total_queries": len(self.query_stats),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "slow_queries": sum(
                1 for d in durations if d > self.slow_query_threshold_ms
            ),
            "failed_queries": sum(1 for s in self.query_stats if not s["success"]),
        }

    def reset_stats(self):
        """Reset query statistics"""
        self.query_stats.clear()


# Global query monitor
query_monitor = QueryPerformanceMonitor()


def monitor_query(query_name: str):
    """
    Decorator to monitor query performance

    Usage:
        @monitor_query("get_user_uploads")
        async def get_uploads(user_id: str):
            return supabase.table("uploads").select("*").execute()
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with query_monitor.monitor_query(
                query_name, {"args": args, "kwargs": kwargs}
            ):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with query_monitor.monitor_query(
                query_name, {"args": args, "kwargs": kwargs}
            ):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class QueryOptimizer:
    """Utilities for optimizing database queries"""

    @staticmethod
    def select_fields(fields: List[str]) -> str:
        """
        Create optimized field selection string

        Only select fields you need to reduce data transfer

        Args:
            fields: List of field names to select

        Returns:
            Comma-separated field string
        """
        if not fields:
            return "*"

        # Validate field names (prevent injection)
        import re

        for field in fields:
            if not re.match(r"^[a-zA-Z0-9_]+$", field):
                raise ValueError(f"Invalid field name: {field}")

        return ",".join(fields)

    @staticmethod
    def build_filter(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build optimized filter dictionary

        Args:
            filters: Dictionary of filters

        Returns:
            Validated filter dictionary
        """
        from app.utils.validators import validate_metadata_filter

        # Remove None values
        cleaned = {k: v for k, v in filters.items() if v is not None}

        # Validate if it's a metadata filter
        if any(key in cleaned for key in ["upload_id", "org_id", "type"]):
            return validate_metadata_filter(cleaned)

        return cleaned

    @staticmethod
    def add_index_hints(query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add hints for database index usage

        Args:
            query_params: Query parameters

        Returns:
            Query parameters with index hints
        """
        # For Supabase/PostgreSQL, ensure we're using indexed columns
        indexed_columns = {
            "org_id",
            "user_id",
            "created_at",
            "updated_at",
            "status",
            "type",
            "chatbot_id",
            "upload_id",
        }

        # Prioritize indexed columns in filters
        if "filters" in query_params:
            filters = query_params["filters"]
            # Sort filters to put indexed columns first
            sorted_filters = {}
            for col in indexed_columns:
                if col in filters:
                    sorted_filters[col] = filters[col]
            for col, val in filters.items():
                if col not in indexed_columns:
                    sorted_filters[col] = val
            query_params["filters"] = sorted_filters

        return query_params


class BatchQueryHelper:
    """Helper for batching queries to prevent N+1 problems"""

    @staticmethod
    async def fetch_related_batch(
        table: str, foreign_key: str, ids: List[str], supabase_client
    ) -> Dict[str, List[Dict]]:
        """
        Fetch related records in batch to prevent N+1 queries

        Args:
            table: Table name
            foreign_key: Foreign key column name
            ids: List of IDs to fetch
            supabase_client: Supabase client instance

        Returns:
            Dictionary mapping ID to list of related records
        """
        if not ids:
            return {}

        # Remove duplicates
        unique_ids = list(set(ids))

        # Batch fetch (Supabase supports 'in' operator)
        result = (
            supabase_client.table(table)
            .select("*")
            .in_(foreign_key, unique_ids)
            .execute()
        )

        # Group by foreign key
        grouped = {}
        for record in result.data:
            fk_value = record.get(foreign_key)
            if fk_value not in grouped:
                grouped[fk_value] = []
            grouped[fk_value].append(record)

        return grouped

    @staticmethod
    def batch_process(
        items: List[Any], batch_size: int = 100, processor: Callable = None
    ) -> List[Any]:
        """
        Process items in batches to prevent memory issues

        Args:
            items: List of items to process
            batch_size: Size of each batch
            processor: Function to process each batch

        Returns:
            List of processed results
        """
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            if processor:
                batch_result = processor(batch)
                results.extend(batch_result)
            else:
                results.extend(batch)

        return results


def optimize_supabase_query(
    table: str,
    select_fields: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build optimized Supabase query parameters

    Args:
        table: Table name
        select_fields: Fields to select (None for all)
        filters: Filter conditions
        order_by: Order by column
        limit: Result limit
        offset: Result offset

    Returns:
        Dictionary of query parameters
    """
    optimizer = QueryOptimizer()

    query_params = {
        "table": table,
        "select": optimizer.select_fields(select_fields or []),
        "filters": optimizer.build_filter(filters or {}),
    }

    if order_by:
        query_params["order_by"] = order_by

    if limit:
        query_params["limit"] = min(limit, 1000)  # Cap at 1000

    if offset:
        query_params["offset"] = offset

    return optimizer.add_index_hints(query_params)


# Query performance tips
QUERY_OPTIMIZATION_TIPS = """
Database Query Optimization Best Practices:

1. Always Use Pagination
   - Never fetch all records without limit
   - Use page_size <= 100 for list endpoints
   - Implement cursor-based pagination for large datasets

2. Select Only Needed Fields
   - Use .select("id,name,created_at") instead of .select("*")
   - Reduces data transfer and parsing time
   - Example: supabase.table("uploads").select("id,status,type")

3. Use Indexed Columns in Filters
   - Filter by org_id, user_id, created_at first
   - These columns should have database indexes
   - Example: .eq("org_id", org_id).eq("status", "completed")

4. Avoid N+1 Queries
   - Fetch related data in batch
   - Use joins when possible
   - Example: Use fetch_related_batch() helper

5. Add Query Monitoring
   - Use @monitor_query decorator
   - Log slow queries (>1000ms)
   - Review query_monitor.get_stats() regularly

6. Optimize Ordering
   - Order by indexed columns when possible
   - Use .order("created_at", desc=True) for recent items
   - Avoid ordering by computed fields

7. Use Count Efficiently
   - Only fetch count when needed
   - Use count="exact" sparingly
   - Consider count="estimated" for large tables

8. Batch Operations
   - Use .in_() for multiple IDs
   - Process in batches of 100-1000
   - Example: .in_("id", [id1, id2, id3])

9. Cache Frequently Accessed Data
   - Cache organization settings
   - Cache user permissions
   - Use Redis for shared cache

10. Monitor and Optimize
    - Review slow query logs
    - Add indexes for common filters
    - Use EXPLAIN ANALYZE in PostgreSQL
"""


def print_optimization_tips():
    """Print query optimization tips"""
    print(QUERY_OPTIMIZATION_TIPS)


# Example usage
"""
Example Usage:

1. Monitor Query Performance:
```python
from app.utils.query_optimizer import monitor_query

@monitor_query("list_uploads")
async def list_uploads(org_id: str):
    return supabase.table("uploads").select("*").eq("org_id", org_id).execute()
```

2. Prevent N+1 Queries:
```python
from app.utils.query_optimizer import BatchQueryHelper

# Bad: N+1 query
for upload in uploads:
    messages = supabase.table("messages").eq("upload_id", upload.id).execute()

# Good: Batch query
upload_ids = [u.id for u in uploads]
messages_by_upload = await BatchQueryHelper.fetch_related_batch(
    "messages", "upload_id", upload_ids, supabase
)
```

3. Optimize Field Selection:
```python
from app.utils.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer()
fields = optimizer.select_fields(["id", "name", "status"])
result = supabase.table("uploads").select(fields).execute()
```

4. Get Performance Stats:
```python
from app.utils.query_optimizer import query_monitor

stats = query_monitor.get_stats()
print(f"Average query time: {stats['avg_duration_ms']}ms")
print(f"Slow queries: {stats['slow_queries']}")
```
"""
