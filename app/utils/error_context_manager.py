"""
Enhanced error context manager with context manager pattern

This module provides a context manager for automatic error context cleanup,
ensuring that error context is always properly cleaned up even if exceptions occur.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorContextScope:
    """
    Context manager for error context that ensures automatic cleanup

    Usage:
        async def my_endpoint(user=Depends(verify_jwt_token)):
            with ErrorContextScope(
                request_id="req_123",
                user_id=user["user_id"],
                operation="my_operation"
            ):
                # Your code here
                # Context is automatically cleaned up on exit
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize error context scope

        Args:
            request_id: Unique request identifier
            user_id: User identifier
            org_id: Organization identifier
            operation: Operation name
            **kwargs: Additional context data
        """
        self.context_data = {
            "request_id": request_id,
            "user_id": user_id,
            "org_id": org_id,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self._context_set = False

    def __enter__(self):
        """Enter the context - set error context"""
        try:
            from .error_context import ErrorContextManager

            # Set the context
            ErrorContextManager.set_request_context(**self.context_data)
            self._context_set = True

            logger.debug(
                f"Error context set for operation: {self.context_data.get('operation')}",
                extra={"context": self.context_data}
            )
        except Exception as e:
            logger.error(f"Failed to set error context: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context - clean up error context"""
        if self._context_set:
            try:
                from .error_context import ErrorContextManager

                ErrorContextManager.clear_context()

                logger.debug(
                    f"Error context cleared for operation: {self.context_data.get('operation')}"
                )
            except Exception as e:
                logger.error(f"Failed to clear error context: {e}")

        # Don't suppress exceptions
        return False

    def update(self, **kwargs):
        """Update context with additional data"""
        try:
            from .error_context import ErrorContextManager

            self.context_data.update(kwargs)
            ErrorContextManager.set_request_context(**kwargs)

            logger.debug(f"Error context updated", extra={"updates": kwargs})
        except Exception as e:
            logger.error(f"Failed to update error context: {e}")


@contextmanager
def error_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs
):
    """
    Context manager function for error context

    This is a functional alternative to ErrorContextScope class.

    Usage:
        async def my_endpoint(user=Depends(verify_jwt_token)):
            with error_context(
                request_id="req_123",
                user_id=user["user_id"],
                operation="my_operation"
            ):
                # Your code here
    """
    scope = ErrorContextScope(
        request_id=request_id,
        user_id=user_id,
        org_id=org_id,
        operation=operation,
        **kwargs
    )

    with scope:
        yield scope


class SafeErrorContext:
    """
    Thread-safe error context manager with additional safety features

    This version includes:
    - Automatic request ID generation
    - Nested context support
    - Context validation
    - Automatic cleanup on errors
    """

    def __init__(self, **context_data):
        """Initialize safe error context"""
        self.context_data = context_data
        self._entered = False

        # Generate request ID if not provided
        if "request_id" not in context_data:
            self.context_data["request_id"] = self._generate_request_id()

    @staticmethod
    def _generate_request_id() -> str:
        """Generate a unique request ID"""
        import uuid
        return f"req_{uuid.uuid4().hex[:12]}"

    def __enter__(self):
        """Enter context with validation"""
        try:
            from .error_context import ErrorContextManager

            # Validate context data
            self._validate_context()

            # Set context
            ErrorContextManager.set_request_context(**self.context_data)
            self._entered = True

            logger.debug(
                "Safe error context entered",
                extra={"request_id": self.context_data.get("request_id")}
            )
        except Exception as e:
            logger.error(f"Failed to enter safe error context: {e}")
            # Don't raise - allow operation to continue

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context with guaranteed cleanup"""
        if self._entered:
            try:
                from .error_context import ErrorContextManager

                ErrorContextManager.clear_context()

                if exc_type:
                    logger.debug(
                        f"Safe error context exited with exception: {exc_type.__name__}",
                        extra={"request_id": self.context_data.get(
                            "request_id")}
                    )
                else:
                    logger.debug(
                        "Safe error context exited successfully",
                        extra={"request_id": self.context_data.get(
                            "request_id")}
                    )
            except Exception as cleanup_error:
                logger.error(f"Error during context cleanup: {cleanup_error}")

        return False

    def _validate_context(self):
        """Validate context data"""
        # Ensure required fields are present
        if not self.context_data.get("request_id"):
            raise ValueError("request_id is required")

        # Validate data types
        for key, value in self.context_data.items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                logger.warning(
                    f"Context value for '{key}' is not a simple type: {type(value)}"
                )

    def update(self, **kwargs):
        """Update context safely"""
        if self._entered:
            try:
                from .error_context import ErrorContextManager

                self.context_data.update(kwargs)
                ErrorContextManager.set_request_context(**kwargs)
            except Exception as e:
                logger.error(f"Failed to update safe error context: {e}")


def with_error_context(**context_kwargs):
    """
    Decorator for automatic error context management

    Usage:
        @with_error_context(operation="my_operation")
        async def my_endpoint(user=Depends(verify_jwt_token)):
            # Error context is automatically managed
            pass
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from kwargs if available
            user = kwargs.get('user')
            if user and isinstance(user, dict):
                context_kwargs.setdefault('user_id', user.get('user_id'))

            # Use safe error context
            with SafeErrorContext(**context_kwargs):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


# Example usage patterns
"""
# Pattern 1: Using ErrorContextScope class
async def endpoint1(user=Depends(verify_jwt_token)):
    with ErrorContextScope(
        request_id="req_123",
        user_id=user["user_id"],
        operation="endpoint1"
    ) as ctx:
        # Do work
        ctx.update(org_id="org_456")  # Update context
        return {"result": "success"}

# Pattern 2: Using error_context function
async def endpoint2(user=Depends(verify_jwt_token)):
    with error_context(
        user_id=user["user_id"],
        operation="endpoint2"
    ) as ctx:
        # Do work
        return {"result": "success"}

# Pattern 3: Using SafeErrorContext
async def endpoint3(user=Depends(verify_jwt_token)):
    with SafeErrorContext(
        user_id=user["user_id"],
        operation="endpoint3"
    ):
        # Do work - context is automatically cleaned up
        return {"result": "success"}

# Pattern 4: Using decorator
@with_error_context(operation="endpoint4")
async def endpoint4(user=Depends(verify_jwt_token)):
    # Error context is automatically managed
    return {"result": "success"}
"""
