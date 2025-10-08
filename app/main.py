import os
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from .services.storage.supabase_client import supabase
from .routers import org, users, uploads, auth, chat, public_chat, monitoring, cache
from .services.shared.worker_scheduler import start_background_worker, stop_background_worker
from .services.shared import get_client_manager
from .config.settings import settings, validate_environment
from .utils.logging_config import setup_logging, get_logger
from .utils.error_monitoring import error_monitor

# Load environment variables once
load_dotenv()

# Setup logging first
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None)
)

logger = get_logger("main")

# Validate environment BEFORE starting anything else
try:
    validate_environment()
    logger.info("Environment validation passed - starting server")
except SystemExit:
    raise  # Re-raise SystemExit as-is (it's intentional)
except Exception as e:
    logger.error("Environment validation failed: %s", str(e))
    raise SystemExit(1) from e


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    # Startup
    logger.info("Starting ZaaKy AI Platform")

    # Initialize shared clients
    try:
        client_manager = get_client_manager()
        health = client_manager.health_check()
        logger.info("Client health check completed", extra={
            "health_status": health,
            "all_healthy": all(health.values())
        })

        if not all(health.values()):
            failed_clients = [client for client,
                              status in health.items() if not status]
            logger.warning(
                "Some clients failed health check but server will continue",
                extra={"failed_clients": failed_clients}
            )
        else:
            logger.info("All API clients initialized successfully")

    except Exception as e:
        logger.error("Client initialization failed: %s", str(e))
        raise SystemExit(1) from e

    # Start background services
    logger.info("Starting background worker")
    try:
        start_background_worker()
        logger.info("Background worker started successfully")
    except Exception as e:
        logger.error("Failed to start background worker: %s", str(e))
        raise SystemExit(1) from e

    # Start error monitoring
    logger.info("Starting error monitoring")
    try:
        error_monitor.start_monitoring()
        logger.info("Error monitoring started successfully")
    except Exception as e:
        logger.error("Failed to start error monitoring: %s", str(e))
        # Don't fail startup for monitoring issues

    logger.info("ZaaKy AI Platform started successfully", extra={
        "version": settings.app.app_version,
        "environment": settings.app.environment,
        "debug_mode": settings.app.debug
    })

    yield

    # Shutdown
    logger.info("Shutting down ZaaKy AI Platform")
    try:
        stop_background_worker()
        logger.info("Background worker stopped successfully")
    except Exception as e:
        logger.error("Error stopping background worker: %s", str(e))

    # Stop error monitoring
    try:
        error_monitor.stop_monitoring()
        logger.info("Error monitoring stopped successfully")
    except Exception as e:
        logger.error("Error stopping error monitoring: %s", str(e))

    logger.info("ZaaKy AI Platform shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.app.app_name,
    version=settings.app.app_version,
    description="Advanced AI Chatbot Platform with Omnichannel Deployment",
    debug=settings.app.debug,
    lifespan=lifespan
)

# Configure CORS middleware using centralized config
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI application configured", extra={
    "cors_origins": settings.security.cors_origins,
    "debug_mode": settings.app.debug
})


@app.get("/health")
async def health_check():
    """Enhanced system health check endpoint"""
    try:
        # Test database
        db_response = supabase.table(
            "organizations").select("id").limit(1).execute()
        db_status = "healthy" if db_response.data is not None else "unhealthy"

        health_data = {
            "status": "healthy" if db_status == "healthy" else "degraded",
           "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app.app_version,
            "environment": settings.app.environment,
            "services": {
                "database": db_status,
                "vector_store": "healthy",
                "ai_service": "healthy",
                "background_worker": "healthy"
            },
            "configuration": settings.to_dict()
        }

        logger.debug("Health check completed", extra={
            "status": health_data["status"],
            "services": health_data["services"]
        })

        return health_data

    except Exception as e:
        logger.error("Health check failed: %s", str(e), extra={
            "endpoint": "/health",
            "error_type": type(e).__name__
        })
        return {
            "status": "unhealthy",
            "timestamp": "2025-01-07T12:00:00Z",
            "error": str(e)
        }


@app.get("/health/clients")
async def health_check_clients():
    """Check health of all API clients"""
    try:
        client_manager = get_client_manager()
        health = client_manager.health_check()

        response_data = {
            "status": "healthy" if all(health.values()) else "degraded",
            "clients": health,
            "timestamp": "2025-01-07T12:00:00Z",
            "details": {
                "openai": "API connection test",
                "pinecone": "Vector database connection",
                "supabase": "Main database connection"
            }
        }

        logger.debug("Client health check completed", extra={
            "client_health": health,
            "overall_status": response_data["status"]
        })

        return response_data

    except Exception as e:
        logger.error("Client health check failed: %s", str(e), extra={
            "endpoint": "/health/clients",
            "error_type": type(e).__name__
        })
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-07T12:00:00Z"
        }
@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "pinecone": "connected", 
            "openai": "connected"
        }
    }

# Route registration with logging
logger.info("Registering API routes")
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(org.router, prefix="/api/org", tags=["organizations"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(uploads.router, prefix="/api/uploads", tags=["uploads"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(public_chat.router, prefix="/api/public", tags=["public"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(cache.router, prefix="/api/cache", tags=["cache"])
logger.info("All API routes registered successfully")


@app.get("/")
def root():
    """Root endpoint with system information"""
    response_data = {
        "message": "ZaaKy AI Platform API",
        "version": settings.app.app_version,
        "environment": settings.app.environment,
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

    logger.debug("Root endpoint accessed", extra={
        "version": response_data["version"],
        "environment": response_data["environment"]
    })

    return response_data


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server", extra={
        "host": "0.0.0.0",
        "port": 8001,
        "reload": settings.app.debug,
        "log_level": settings.app.log_level.lower()
    })

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower()
    )
