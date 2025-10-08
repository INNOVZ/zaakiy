"""
Cache management endpoints for Redis operations
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from ..services.shared.cache_service import cache_service
from ..services.auth.middleware import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cache"])


@router.get("/health")
async def cache_health():
    """Check Redis cache health status"""
    try:
        health_status = cache_service.health_check()
        return {
            "status": "success",
            "cache": health_status
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        raise HTTPException(status_code=500, detail="Cache health check failed")


@router.get("/stats")
async def cache_stats(user=Depends(get_current_user)):
    """Get cache statistics (admin only)"""
    try:
        if not user.get("role") == "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        health_status = cache_service.health_check()
        
        return {
            "status": "success",
            "cache_stats": {
                "enabled": health_status.get("enabled", False),
                "status": health_status.get("status", "unknown"),
                "redis_version": health_status.get("redis_version", "unknown"),
                "memory_usage": health_status.get("used_memory", "unknown"),
                "connected_clients": health_status.get("connected_clients", 0),
                "uptime_seconds": health_status.get("uptime", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Cache stats retrieval failed")


@router.delete("/clear")
async def clear_cache(
    pattern: str = Query(..., description="Cache key pattern to clear"),
    user=Depends(get_current_user)
):
    """Clear cache entries matching pattern (admin only)"""
    try:
        if not user.get("role") == "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        cleared_count = cache_service.clear_pattern(pattern)
        
        return {
            "status": "success",
            "message": f"Cleared {cleared_count} cache entries matching pattern: {pattern}",
            "cleared_count": cleared_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Cache clear failed")


@router.delete("/clear/org/{org_id}")
async def clear_org_cache(
    org_id: str,
    user=Depends(get_current_user)
):
    """Clear all cache entries for a specific organization"""
    try:
        # Check if user has access to this org
        user_org_id = user.get("org_id")
        if user_org_id != org_id and user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied to this organization")
        
        # Clear organization-specific cache patterns
        patterns = [
            f"context_config:{org_id}:*",
            f"conversation_history:*:{org_id}:*",
            f"org_data:{org_id}:*"
        ]
        
        total_cleared = 0
        for pattern in patterns:
            cleared = cache_service.clear_pattern(pattern)
            total_cleared += cleared
        
        return {
            "status": "success",
            "message": f"Cleared {total_cleared} cache entries for organization: {org_id}",
            "cleared_count": total_cleared
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Organization cache clear failed")


@router.get("/keys")
async def list_cache_keys(
    pattern: str = Query(default="*", description="Pattern to match keys"),
    limit: int = Query(default=100, le=1000, description="Maximum number of keys to return"),
    user=Depends(get_current_user)
):
    """List cache keys matching pattern (admin only)"""
    try:
        if not user.get("role") == "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Note: This is a simplified implementation
        # In production, you might want to use SCAN instead of KEYS for better performance
        keys = cache_service.redis_client.keys(pattern) if cache_service.enabled else []
        
        return {
            "status": "success",
            "keys": keys[:limit],
            "total_count": len(keys),
            "returned_count": min(len(keys), limit)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache keys listing failed: {e}")
        raise HTTPException(status_code=500, detail="Cache keys listing failed")


@router.get("/key/{key}")
async def get_cache_value(
    key: str,
    user=Depends(get_current_user)
):
    """Get value for a specific cache key (admin only)"""
    try:
        if not user.get("role") == "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        value = cache_service.get(key)
        ttl = cache_service.get_ttl(key)
        
        return {
            "status": "success",
            "key": key,
            "value": value,
            "ttl_seconds": ttl,
            "exists": value is not None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache value retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Cache value retrieval failed")


@router.delete("/key/{key}")
async def delete_cache_key(
    key: str,
    user=Depends(get_current_user)
):
    """Delete a specific cache key (admin only)"""
    try:
        if not user.get("role") == "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        deleted = cache_service.delete(key)
        
        return {
            "status": "success",
            "key": key,
            "deleted": deleted
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache key deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Cache key deletion failed")
