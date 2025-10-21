"""
Monitoring endpoints for system health and performance

Provides endpoints to monitor connection pools, query performance,
and system resources.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..services.auth import verify_jwt_token_from_header
from ..services.storage.pinecone_client import (
    get_connection_stats as get_pinecone_stats,
)
from ..services.storage.supabase_client import (
    get_connection_stats as get_supabase_stats,
)
from ..utils.performance_monitor import performance_monitor
from ..utils.query_optimizer import query_monitor

router = APIRouter()


@router.get("/connection-pools")
async def get_connection_pool_stats(user=Depends(verify_jwt_token_from_header)):
    """
    Get connection pool statistics for all database clients

    Requires authentication to prevent information disclosure
    """
    try:
        supabase_stats = get_supabase_stats()
        pinecone_stats = get_pinecone_stats()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "supabase": {
                "pool_size": supabase_stats.get("pool_size", 0),
                "total_connections": supabase_stats.get("total_connections", 0),
                "active_connections": supabase_stats.get("active_connections", 0),
                "failed_connections": supabase_stats.get("failed_connections", 0),
                "utilization": round(
                    supabase_stats.get("active_connections", 0)
                    / max(supabase_stats.get("pool_size", 1), 1)
                    * 100,
                    2,
                ),
            },
            "pinecone": {
                "pool_size": pinecone_stats.get("pool_size", 0),
                "total_connections": pinecone_stats.get("total_connections", 0),
                "active_connections": pinecone_stats.get("active_connections", 0),
                "failed_connections": pinecone_stats.get("failed_connections", 0),
                "utilization": round(
                    pinecone_stats.get("active_connections", 0)
                    / max(pinecone_stats.get("pool_size", 1), 1)
                    * 100,
                    2,
                ),
            },
            "health": {
                "supabase": "healthy"
                if supabase_stats.get("failed_connections", 0) == 0
                else "degraded",
                "pinecone": "healthy"
                if pinecone_stats.get("failed_connections", 0) == 0
                else "degraded",
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get connection pool stats: {str(e)}"
        )


@router.get("/query-performance")
async def get_query_performance(user=Depends(verify_jwt_token_from_header)):
    """
    Get query performance statistics

    Requires authentication to prevent information disclosure
    """
    try:
        stats = query_monitor.get_stats()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queries": {
                "total": stats.get("total_queries", 0),
                "failed": stats.get("failed_queries", 0),
                "slow": stats.get("slow_queries", 0),
            },
            "performance": {
                "avg_duration_ms": round(stats.get("avg_duration_ms", 0), 2),
                "max_duration_ms": round(stats.get("max_duration_ms", 0), 2),
                "min_duration_ms": round(stats.get("min_duration_ms", 0), 2),
            },
            "health": {
                "status": "healthy"
                if stats.get("slow_queries", 0) < 10
                else "degraded",
                "slow_query_threshold_ms": 1000,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get query performance: {str(e)}"
        )


@router.post("/query-performance/reset")
async def reset_query_performance(user=Depends(verify_jwt_token_from_header)):
    """
    Reset query performance statistics

    Requires authentication
    """
    try:
        query_monitor.reset_stats()

        return {
            "success": True,
            "message": "Query performance statistics reset",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset query performance: {str(e)}"
        )


@router.get("/system-health")
async def get_system_health():
    """
    Get overall system health status

    Public endpoint for health checks (no auth required)
    """
    try:
        # Get connection pool stats
        supabase_stats = get_supabase_stats()
        pinecone_stats = get_pinecone_stats()

        # Get query stats
        query_stats = query_monitor.get_stats()

        # Determine overall health
        supabase_healthy = supabase_stats.get("failed_connections", 0) == 0
        pinecone_healthy = pinecone_stats.get("failed_connections", 0) == 0
        queries_healthy = query_stats.get("slow_queries", 0) < 10

        overall_healthy = supabase_healthy and pinecone_healthy and queries_healthy

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "database": "healthy" if supabase_healthy else "degraded",
                "vector_store": "healthy" if pinecone_healthy else "degraded",
                "query_performance": "healthy" if queries_healthy else "degraded",
            },
            "metrics": {
                "supabase_pool_utilization": round(
                    supabase_stats.get("active_connections", 0)
                    / max(supabase_stats.get("pool_size", 1), 1)
                    * 100,
                    2,
                ),
                "pinecone_pool_utilization": round(
                    pinecone_stats.get("active_connections", 0)
                    / max(pinecone_stats.get("pool_size", 1), 1)
                    * 100,
                    2,
                ),
                "avg_query_time_ms": round(query_stats.get("avg_duration_ms", 0), 2),
            },
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


@router.get("/alerts")
async def get_system_alerts(user=Depends(verify_jwt_token_from_header)):
    """
    Get system alerts based on thresholds

    Requires authentication
    """
    try:
        alerts = []

        # Check connection pool utilization
        supabase_stats = get_supabase_stats()
        pinecone_stats = get_pinecone_stats()

        supabase_util = (
            supabase_stats.get("active_connections", 0)
            / max(supabase_stats.get("pool_size", 1), 1)
            * 100
        )

        if supabase_util > 80:
            alerts.append(
                {
                    "severity": "warning",
                    "component": "supabase",
                    "message": f"Supabase connection pool utilization high: {supabase_util:.1f}%",
                    "recommendation": "Consider increasing SUPABASE_POOL_SIZE",
                }
            )

        pinecone_util = (
            pinecone_stats.get("active_connections", 0)
            / max(pinecone_stats.get("pool_size", 1), 1)
            * 100
        )

        if pinecone_util > 80:
            alerts.append(
                {
                    "severity": "warning",
                    "component": "pinecone",
                    "message": f"Pinecone connection pool utilization high: {pinecone_util:.1f}%",
                    "recommendation": "Consider increasing PINECONE_POOL_SIZE",
                }
            )

        # Check query performance
        query_stats = query_monitor.get_stats()

        if query_stats.get("slow_queries", 0) > 10:
            alerts.append(
                {
                    "severity": "warning",
                    "component": "queries",
                    "message": f"High number of slow queries: {query_stats.get('slow_queries', 0)}",
                    "recommendation": "Review slow query logs and add database indexes",
                }
            )

        if query_stats.get("avg_duration_ms", 0) > 500:
            alerts.append(
                {
                    "severity": "info",
                    "component": "queries",
                    "message": f"Average query time elevated: {query_stats.get('avg_duration_ms', 0):.0f}ms",
                    "recommendation": "Consider query optimization",
                }
            )

        # Check failed connections
        if supabase_stats.get("failed_connections", 0) > 0:
            alerts.append(
                {
                    "severity": "error",
                    "component": "supabase",
                    "message": f"Failed connections detected: {supabase_stats.get('failed_connections', 0)}",
                    "recommendation": "Check database connectivity and credentials",
                }
            )

        if pinecone_stats.get("failed_connections", 0) > 0:
            alerts.append(
                {
                    "severity": "error",
                    "component": "pinecone",
                    "message": f"Failed connections detected: {pinecone_stats.get('failed_connections', 0)}",
                    "recommendation": "Check Pinecone API status and credentials",
                }
            )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_count": len(alerts),
            "alerts": alerts,
            "status": "ok" if len(alerts) == 0 else "alerts_present",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get system alerts: {str(e)}"
        )


@router.get("/chat-performance")
async def get_chat_performance(user=Depends(verify_jwt_token_from_header)):
    """
    Get chat service performance statistics

    Returns performance metrics for:
    - Query enhancement
    - Document retrieval
    - Response generation

    Requires authentication
    """
    try:
        stats = performance_monitor.get_all_stats()

        # Calculate overall health
        total_avg = sum(s.get("avg_ms", 0) for s in stats.values()) / max(len(stats), 1)

        health_status = "healthy"
        if total_avg > 3000:
            health_status = "degraded"
        elif total_avg > 5000:
            health_status = "unhealthy"

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": health_status,
            "operations": stats,
            "summary": {
                "total_avg_ms": round(total_avg, 2),
                "operations_tracked": len(stats),
            },
            "recommendations": _get_performance_recommendations(stats),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat performance: {str(e)}"
        )


@router.post("/chat-performance/reset")
async def reset_chat_performance(user=Depends(verify_jwt_token_from_header)):
    """
    Reset chat performance statistics

    Requires authentication
    """
    try:
        performance_monitor.clear_metrics()

        return {
            "success": True,
            "message": "Chat performance statistics reset",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset chat performance: {str(e)}"
        )


def _get_performance_recommendations(stats):
    """Generate performance recommendations based on stats"""
    recommendations = []

    # Check query enhancement performance
    query_stats = stats.get("query_enhancement", {})
    if query_stats.get("avg_ms", 0) > 800:
        recommendations.append(
            {
                "operation": "query_enhancement",
                "issue": "Slow query enhancement",
                "recommendation": "Consider disabling query_rewriting in context config",
                "expected_improvement": "500-1000ms",
            }
        )

    # Check document retrieval performance
    retrieval_stats = stats.get("document_retrieval", {})
    if retrieval_stats.get("avg_ms", 0) > 800:
        recommendations.append(
            {
                "operation": "document_retrieval",
                "issue": "Slow document retrieval",
                "recommendation": "Reduce initial_retrieval_count or enable caching",
                "expected_improvement": "200-400ms",
            }
        )

    # Check response generation performance
    response_stats = stats.get("response_generation", {})
    if response_stats.get("avg_ms", 0) > 2000:
        recommendations.append(
            {
                "operation": "response_generation",
                "issue": "Slow response generation",
                "recommendation": "Use gpt-3.5-turbo instead of gpt-4 or reduce max_tokens",
                "expected_improvement": "500-1000ms",
            }
        )

    return recommendations
