"""
Centralized error handling utilities with structured logging and recovery
"""
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .error_context import (ErrorCategory, ErrorContext, ErrorContextManager,
                            ErrorSeverity, error_logger)
from .exceptions import *

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling class with structured logging and recovery"""

    @staticmethod
    def handle_exception(
        e: Exception,
        context: str = "",
        user_message: str = None,
        status_code: int = 500,
        service: str = None,
        recovery_strategies: List[Callable] = None,
    ) -> HTTPException:
        """Convert any exception to properly logged HTTPException with recovery attempts"""

        # Create structured error context
        error_context = ErrorContextManager.create_error_context(
            error=e, service=service, additional_context={"operation_context": context}
        )

        # Log error with structured context
        error_logger.log_error(e, error_context, context)

        # Attempt recovery if strategies provided
        if recovery_strategies:
            ErrorHandler._attempt_recovery(e, error_context, recovery_strategies)

        # Determine user-facing message and status code
        user_msg, status = ErrorHandler._determine_user_response(
            e, user_message, status_code
        )

        return HTTPException(
            status_code=status,
            detail={
                "error": user_msg,
                "error_id": error_context.error_id,
                "error_type": type(e).__name__,
                "timestamp": error_context.timestamp.isoformat(),
                "recovery_attempted": error_context.recovery_attempted,
                "recovery_successful": error_context.recovery_successful,
            },
        )

    @staticmethod
    def _attempt_recovery(
        error: Exception, context: ErrorContext, strategies: List[Callable]
    ):
        """Attempt error recovery using provided strategies"""
        for strategy in strategies:
            try:
                result = strategy(error, context)
                if result:
                    error_logger.log_recovery_attempt(
                        context, strategy.__name__, success=True
                    )
                    return True
            except Exception as recovery_error:
                error_logger.log_recovery_attempt(
                    context, strategy.__name__, success=False
                )
                logger.warning(
                    f"Recovery strategy {strategy.__name__} failed: {recovery_error}"
                )

        return False

    @staticmethod
    def _determine_user_response(
        e: Exception, user_message: str = None, status_code: int = 500
    ) -> tuple[str, int]:
        """Determine appropriate user-facing message and status code"""
        if isinstance(e, ZaaKyBaseException):
            return user_message or e.message, status_code
        elif isinstance(e, ValueError):
            return user_message or "Invalid input provided", 400
        elif isinstance(e, PermissionError):
            return user_message or "Access denied", 403
        elif isinstance(e, FileNotFoundError):
            return user_message or "Resource not found", 404
        elif isinstance(e, ConnectionError):
            return user_message or "Service temporarily unavailable", 503
        elif isinstance(e, TimeoutError):
            return user_message or "Request timeout", 504
        else:
            return user_message or "An unexpected error occurred", 500

    @staticmethod
    def log_and_raise(
        exception_class: type,
        message: str,
        context: str = "",
        details: dict = None,
        original_exception: Exception = None,
        service: str = None,
    ):
        """Log error and raise custom exception with structured context"""

        error_details = details or {}
        if original_exception:
            error_details["original_error"] = str(original_exception)
            error_details["original_type"] = type(original_exception).__name__

        # Create structured error context
        error_context = ErrorContextManager.create_error_context(
            error=original_exception or Exception(message),
            service=service,
            additional_context={
                "operation_context": context,
                "custom_details": error_details,
            },
        )

        # Log with structured context
        error_logger.log_error(
            original_exception or Exception(message),
            error_context,
            f"{context}: {message}",
        )

        if original_exception:
            raise exception_class(
                message, details=error_details
            ) from original_exception
        else:
            raise exception_class(message, details=error_details)


def create_error_response(
    error: Exception, status_code: int = 500, include_details: bool = False
) -> JSONResponse:
    """Create standardized error response"""

    error_data = {
        "error": True,
        "message": str(error),
        "timestamp": "2025-01-07T12:00:00Z",
        "type": type(error).__name__,
    }

    if include_details and hasattr(error, "details"):
        error_data["details"] = error.details

    return JSONResponse(status_code=status_code, content=error_data)


# Decorator for automatic error handling
def handle_errors(
    context: str = "",
    user_message: str = None,
    service: str = None,
    recovery_strategies: List[Callable] = None,
):
    """Decorator to automatically handle errors in route functions with structured logging"""

    def decorator(func):
        import functools

        @functools.wraps(func)
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
                    user_message=user_message,
                    service=service,
                    recovery_strategies=recovery_strategies,
                )

        return wrapper

    return decorator


# Circuit breaker for external service calls
class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (
            datetime.now(timezone.utc) - self.last_failure_time
        ).seconds >= self.timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# Retry decorator with exponential backoff
def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator for retrying operations with exponential backoff"""

    def decorator(func):
        import asyncio
        import functools
        import random

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= 0.5 + random.random() * 0.5

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator
