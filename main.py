from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routers import org
from routers import users
from routers import uploads
from routers import auth
from services.worker_scheduler import start_background_worker, stop_background_worker
import logging

load_dotenv()  # Load .env variables

# Configure logging
logging.basicConfig(level=logging.INFO)


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

# Route registration
app.include_router(auth.router, prefix="/api/auth")
app.include_router(org.router, prefix="/api/org")
app.include_router(users.router, prefix="/api/users")
# app.include_router(org.router, prefix="/api/console")
app.include_router(uploads.router, prefix="/api/uploads")


@app.get("/")
def root():
    return {"message": "API is running"}
