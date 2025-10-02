"""
Monitoring and health check endpoints
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from ..services.auth import verify_jwt_token
from ..utils.error_monitoring import error_monitor
from ..utils.error_context import ErrorContextManager
from ..services.shared import get_client_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Check critical services
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "unknown",
                "vector_db": "unknown",
                "ai_service": "unknown"
            }
        }
        
        # Check database connection
        try:
            client_manager = get_client_manager()
            if client_manager.supabase:
                # Simple query to test connection
                response = client_manager.supabase.table("organizations").select("id").limit(1).execute()
                health_status["services"]["database"] = "healthy"
            else:
                health_status["services"]["database"] = "unavailable"
        except Exception as e:
            health_status["services"]["database"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check vector database
        try:
            if client_manager.pinecone_index:
                # Simple query to test connection
                client_manager.pinecone_index.query(
                    vector=[0.0] * 1536,  # Dummy vector
                    top_k=1,
                    include_metadata=False
                )
                health_status["services"]["vector_db"] = "healthy"
            else:
                health_status["services"]["vector_db"] = "unavailable"
        except Exception as e:
            health_status["services"]["vector_db"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check AI service
        try:
            if client_manager.openai:
                # Simple test call
                response = client_manager.openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                health_status["services"]["ai_service"] = "healthy"
            else:
                health_status["services"]["ai_service"] = "unavailable"
        except Exception as e:
            health_status["services"]["ai_service"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/detailed")
async def detailed_health_check(user=Depends(verify_jwt_token)):
    """Detailed health check with system metrics"""
    try:
        # Get error monitoring health
        error_health = error_monitor.get_health_status()
        
        # Get system metrics
        system_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_monitoring": error_health,
            "services": {
                "database": await _check_database_health(),
                "vector_db": await _check_vector_db_health(),
                "ai_service": await _check_ai_service_health(),
                "scraping": await _check_scraping_health()
            }
        }
        
        # Determine overall health
        overall_status = "healthy"
        for service, status in system_metrics["services"].items():
            if status.get("status") == "unhealthy":
                overall_status = "unhealthy"
            elif status.get("status") == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        system_metrics["overall_status"] = overall_status
        
        return system_metrics
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/errors/metrics")
async def get_error_metrics(
    service: str = None,
    hours: int = 24,
    user=Depends(verify_jwt_token)
):
    """Get error metrics for monitoring dashboard"""
    try:
        metrics = error_monitor.get_error_metrics(service=service, hours=hours)
        
        # Convert to serializable format
        serializable_metrics = []
        for metric in metrics:
            serializable_metrics.append({
                "error_type": metric.error_type,
                "count": metric.count,
                "first_occurrence": metric.first_occurrence.isoformat(),
                "last_occurrence": metric.last_occurrence.isoformat(),
                "severity": metric.severity,
                "service": metric.service,
                "category": metric.category
            })
        
        return {
            "metrics": serializable_metrics,
            "summary": {
                "total_errors": sum(m["count"] for m in serializable_metrics),
                "unique_error_types": len(serializable_metrics),
                "time_range_hours": hours,
                "service_filter": service
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get error metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get error metrics: {str(e)}")


@router.get("/errors/alerts")
async def get_active_alerts(user=Depends(verify_jwt_token)):
    """Get currently active alerts"""
    try:
        alerts = error_monitor.get_active_alerts()
        
        # Convert to serializable format
        serializable_alerts = [alert.to_dict() for alert in alerts]
        
        return {
            "alerts": serializable_alerts,
            "summary": {
                "total_alerts": len(serializable_alerts),
                "critical_alerts": len([a for a in serializable_alerts if a["level"] == "critical"]),
                "error_alerts": len([a for a in serializable_alerts if a["level"] == "error"]),
                "warning_alerts": len([a for a in serializable_alerts if a["level"] == "warning"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/errors/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, user=Depends(verify_jwt_token)):
    """Resolve an active alert"""
    try:
        error_monitor.resolve_alert(alert_id)
        return {"message": f"Alert {alert_id} resolved successfully"}
        
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


async def _check_database_health() -> Dict[str, Any]:
    """Check database health and performance"""
    try:
        client_manager = get_client_manager()
        if not client_manager.supabase:
            return {"status": "unavailable", "message": "Supabase client not initialized"}
        
        start_time = datetime.utcnow()
        
        # Test basic query
        response = client_manager.supabase.table("organizations").select("id").limit(1).execute()
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "connection": "active"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "failed"
        }


async def _check_vector_db_health() -> Dict[str, Any]:
    """Check vector database health and performance"""
    try:
        client_manager = get_client_manager()
        if not client_manager.pinecone_index:
            return {"status": "unavailable", "message": "Pinecone index not initialized"}
        
        start_time = datetime.utcnow()
        
        # Test basic query
        response = client_manager.pinecone_index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=1,
            include_metadata=False
        )
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "connection": "active"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "failed"
        }


async def _check_ai_service_health() -> Dict[str, Any]:
    """Check AI service health and performance"""
    try:
        client_manager = get_client_manager()
        if not client_manager.openai:
            return {"status": "unavailable", "message": "OpenAI client not initialized"}
        
        start_time = datetime.utcnow()
        
        # Test basic completion
        response = client_manager.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "model": "gpt-3.5-turbo",
            "connection": "active"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "failed"
        }


async def _check_scraping_health() -> Dict[str, Any]:
    """Check scraping service health"""
    try:
        # Check if scraping services are available
        from services.scraping.web_scraper import SecureWebScraper
        
        scraper = SecureWebScraper()
        
        return {
            "status": "healthy",
            "message": "Scraping service available",
            "features": {
                "ssrf_protection": True,
                "rate_limiting": True,
                "robots_txt": True
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Scraping service unavailable"
        }
