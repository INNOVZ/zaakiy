"""
Custom CORS middleware for multi-tenant widget deployment
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SmartCORSMiddleware(BaseHTTPMiddleware):
    """
    Smart CORS middleware that:
    - Allows all origins for public widget endpoints (/api/public/*)
    - Restricts origins for authenticated endpoints

    This enables multi-tenant chatbot deployment without manual CORS configuration
    for each tenant domain.
    """

    def __init__(self, app, allowed_origins=None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or [
            "https://zaakiy.vercel.app",
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3002",
        ]

    async def dispatch(self, request: Request, call_next):
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            return self._create_preflight_response(request)

        # Process request
        response = await call_next(request)

        # Add CORS headers based on endpoint type
        origin = request.headers.get("origin")

        # Public widget endpoints - allow any origin
        if self._is_public_endpoint(request.url.path):
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers[
                "Access-Control-Allow-Methods"
            ] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers[
                "Access-Control-Allow-Headers"
            ] = "Content-Type, Authorization, X-User-ID, X-Request-ID, X-Org-ID"
            logger.debug(
                f"Public endpoint CORS: {request.url.path} - Origin: {origin or '*'}"
            )

        # Authenticated endpoints - strict CORS
        elif origin and origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers[
                "Access-Control-Allow-Methods"
            ] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers[
                "Access-Control-Allow-Headers"
            ] = "Content-Type, Authorization, X-User-ID, X-Request-ID, X-Org-ID"
            logger.debug(
                f"Authenticated endpoint CORS: {request.url.path} - Origin: {origin}"
            )

        else:
            # Log if origin is not allowed for authenticated endpoints
            if origin and not self._is_public_endpoint(request.url.path):
                logger.warning(
                    f"Origin {origin} not in allowed list for {request.url.path}"
                )

        return response

    def _is_public_endpoint(self, path: str) -> bool:
        """
        Check if endpoint is a public widget endpoint that should allow all origins
        """
        public_paths = [
            "/api/public/",  # All public API endpoints
            "/api/uploads/avatar/",  # Avatar images for chatbot widgets
            "/health",  # Health check
            "/docs",  # API documentation
            "/openapi.json",  # OpenAPI schema
            "/redoc",  # Alternative API docs
        ]
        return any(path.startswith(p) for p in public_paths)

    def _create_preflight_response(self, request: Request) -> Response:
        """
        Create response for preflight OPTIONS request

        Preflight requests are sent by browsers before actual requests
        to check if CORS allows the request.
        """
        origin = request.headers.get("origin")
        path = request.url.path

        # Allow all origins for public endpoints, restricted for others
        if self._is_public_endpoint(path):
            allowed_origin = origin or "*"
        elif origin in self.allowed_origins:
            allowed_origin = origin
        else:
            allowed_origin = self.allowed_origins[0]  # Default to first allowed

        headers = {
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-User-ID, X-Request-ID, X-Org-ID",
            "Access-Control-Max-Age": "86400",  # 24 hours
        }

        return Response(status_code=200, headers=headers)
