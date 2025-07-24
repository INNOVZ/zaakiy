from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routers import org
from routers import users
from routers import uploads
from routers import auth


load_dotenv()  # Load .env variables

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
