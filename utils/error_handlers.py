"""
Centralized error handling utilities
"""
import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from utils.exceptions import *

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling class"""

    @staticmethod
    def handle_exception(
        e: Exception,
        context: str = "",
        user_message: str = None,
        status_code: int = 500
    ) -> HTTPException:
        """Convert any exception to properly logged HTTPException"""

        # Extract error details
        error_id = f"ERR_{hash(str(e)) % 10000:04d}"

        # Log the full error with context
        logger.error(
            "[%s] %s: %s: %s\nTraceback: %s",
            error_id, context, type(e).__name__, str(e), traceback.format_exc()
        )

        # Determine user-facing message and status code
        if isinstance(e, ZaaKyBaseException):
            user_msg = user_message or e.message
            status = status_code
        elif isinstance(e, ValueError):
            user_msg = user_message or "Invalid input provided"
            status = 400
        elif isinstance(e, PermissionError):
            user_msg = user_message or "Access denied"
            status = 403
        elif isinstance(e, FileNotFoundError):
            user_msg = user_message or "Resource not found"
            status = 404
        elif isinstance(e, ConnectionError):
            user_msg = user_message or "Service temporarily unavailable"
            status = 503
        elif isinstance(e, TimeoutError):
            user_msg = user_message or "Request timeout"
            status = 504
        else:
            user_msg = user_message or "An unexpected error occurred"
            status = 500

        return HTTPException(
            status_code=status,
            detail={
                "error": user_msg,
                "error_id": error_id,
                "error_type": type(e).__name__,
                "timestamp": "2025-01-07T12:00:00Z"
            }
        )

    @staticmethod
    def log_and_raise(
        exception_class: type,
        message: str,
        context: str = "",
        details: dict = None,
        original_exception: Exception = None
    ):
        """Log error and raise custom exception"""

        error_details = details or {}
        if original_exception:
            error_details["original_error"] = str(original_exception)
            error_details["original_type"] = type(original_exception).__name__

        logger.error("%s: %s", context, message, extra=error_details)

        if original_exception:
            raise exception_class(
                message, details=error_details) from original_exception
        else:
            raise exception_class(message, details=error_details)


def create_error_response(
    error: Exception,
    status_code: int = 500,
    include_details: bool = False
) -> JSONResponse:
    """Create standardized error response"""

    error_data = {
        "error": True,
        "message": str(error),
        "timestamp": "2025-01-07T12:00:00Z",
        "type": type(error).__name__
    }

    if include_details and hasattr(error, 'details'):
        error_data["details"] = error.details

    return JSONResponse(
        status_code=status_code,
        content=error_data
    )


# Decorator for automatic error handling
def handle_errors(context: str = "", user_message: str = None):
    """Decorator to automatically handle errors in route functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                raise ErrorHandler.handle_exception(
                    e,
                    context=context or func.__name__,
                    user_message=user_message
                )
        return wrapper
    return decorator
