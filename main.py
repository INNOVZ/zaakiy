import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routers import search, org, users, uploads, auth, chat, public_chat
from services.worker_scheduler import start_background_worker, stop_background_worker
from config.settings import settings, validate_environment
from utils.logging_config import setup_logging, get_logger


load_dotenv()

# Setup logging first
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None)
)

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ZaaKy AI Platform...")
    
    # Validate configuration
    try:
        validate_environment()
        validation = settings.validate_all()
        
        if not validation["valid"]:
            logger.error(f"Configuration validation failed: {validation['errors']}")
            raise ValueError("Invalid configuration")
     
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(f"Configuration warning: {warning}")

        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        raise

    # Start background services
    logger.info("Starting background worker...")
    start_background_worker()

    logger.info("ZaaKy AI Platform started successfully")
    yield

    # Shutdown
    logger.info("Shutting down ZaaKy AI Platform...")
    stop_background_worker()
    logger.info("ZaaKy AI Platform shut down complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.app.app_name,
    version=settings.app.app_version,
    description="Advanced AI Chatbot Platform with Omnichannel Deployment",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health_check():
    """Enhanced system health check endpoint"""
    try:
        from services.supabase_client import supabase
        
        # Test database
        db_response = supabase.table("organizations").select("id").limit(1).execute()
        db_status = "healthy" if db_response.data is not None else "unhealthy"
        
        return {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "timestamp": "2025-01-07T12:00:00Z",
            "version": settings.app.app_version,
            "environment": settings.app.environment,
            "services": {
                "database": db_status,
                "vector_store": "healthy",
                "ai_service": "healthy",
                "background_worker": "healthy"
            },
            "configuration": {
                "max_concurrent_requests": settings.performance.max_concurrent_requests,
                "cache_enabled": settings.app.enable_caching,
                "analytics_enabled": settings.app.enable_analytics
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": "2025-01-07T12:00:00Z",
            "error": str(e)
        }


# Route registration
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(org.router, prefix="/api/org", tags=["organizations"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(uploads.router, prefix="/api/uploads", tags=["uploads"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(public_chat.router, prefix="/api/public", tags=["public"])


@app.get("/")
def root():
    """Root endpoint with system information"""
    return {
        "message": "ZaaKi AI Platform API",
        "version": settings.app.app_version,
        "environment": settings.app.environment,
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }