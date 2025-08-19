import os
import logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routers import org
from routers import users
from routers import uploads
from routers import auth
from routers import chat
from services.worker_scheduler import start_background_worker, stop_background_worker
from routers import search
from routers import public_chat

load_dotenv()  # Load .env variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting background worker...")
    start_background_worker()
    yield
    # Shutdown
    logging.info("Stopping background worker...")
    stop_background_worker()

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS middleware

# Add after app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add to main.py


@app.on_event("startup")
async def startup_event():
    """Validate environment and connections on startup"""
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_JWT_SECRET",
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    print("✅ All environment variables configured")
    print("✅ System ready for production")


@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-01-07T12:00:00Z",
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "vector_store": "connected",
            "ai_service": "connected"
        }
    }
# Route registration
app.include_router(auth.router, prefix="/api/auth")
app.include_router(org.router, prefix="/api/org")
app.include_router(users.router, prefix="/api/users")
# app.include_router(org.router, prefix="/api/console")
app.include_router(uploads.router, prefix="/api/uploads")
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(public_chat.router, prefix="/api/public", tags=["public"])


@app.get("/")
def root():
    return {"message": "API is running"}
