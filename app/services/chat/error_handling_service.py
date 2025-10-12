"""
Error Handling Service
Provides centralized error handling, retry logic, and exception management
"""
import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorHandlingServiceError(Exception):
    """Exception for error handling service errors"""


class RetryableError(Exception):
    """Exception that can be retried"""


class NonRetryableError(Exception):
    """Exception that should not be retried"""


class ErrorHandlingService:
    """Centralized error handling and recovery service"""

    def __init__(self, org_id: str, error_monitor=None):
        self.org_id = org_id
        self.error_monitor = error_monitor

        # Retry configuration
        self.retry_config = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "exponential_base": 2,
        }

        # Error severity mapping
        self.severity_mapping = {
            "openai.OpenAIError": "high",
            "ConnectionError": "high",
            "TimeoutError": "medium",
            "ValueError": "low",
            "KeyError": "low",
            "TypeError": "low",
        }

        # Error category mapping
        self.category_mapping = {
            "openai.OpenAIError": "external_service",
            "ConnectionError": "database",
            "TimeoutError": "network",
            "RetrievalError": "retrieval",
            "ContextError": "context_engineering",
            "ResponseGenerationError": "response_generation",
        }

    def record_error(
        self,
        error: Exception,
        service: str = "chat_service",
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record error in monitoring system"""
        try:
            error_type = type(error).__name__
            severity = self.severity_mapping.get(error_type, "medium")
            category = self.category_mapping.get(error_type, "unknown")

            if self.error_monitor:
                self.error_monitor.record_error(
                    error_type=error_type,
                    severity=severity,
                    service=service,
                    category=category,
                )

            # Also log locally
            logger.error(
                "Error recorded: %s - %s in %s (context: %s)",
                error_type,
                str(error),
                service,
                context,
            )

        except Exception as monitor_error:
            logger.warning(
                "Failed to record error in monitoring system: %s", monitor_error
            )

    async def handle_with_retry(
        self,
        operation: Callable,
        *args,
        operation_name: str = "unknown_operation",
        retryable_exceptions: tuple = (ConnectionError, TimeoutError, RetryableError),
        **kwargs,
    ) -> Any:
        """Execute operation with retry logic"""
        last_exception = None

        for attempt in range(self.retry_config["max_attempts"]):
            try:
                return (
                    await operation(*args, **kwargs)
                    if asyncio.iscoroutinefunction(operation)
                    else operation(*args, **kwargs)
                )

            except retryable_exceptions as e:
                last_exception = e

                if attempt == self.retry_config["max_attempts"] - 1:
                    # Last attempt failed
                    self.record_error(
                        error=e,
                        context=f"{operation_name}_final_attempt_failed",
                        metadata={"attempts": attempt + 1},
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.retry_config["base_delay"]
                    * (self.retry_config["exponential_base"] ** attempt),
                    self.retry_config["max_delay"],
                )

                logger.warning(
                    "Retrying %s after %s seconds (attempt %d/%d): %s",
                    operation_name,
                    delay,
                    attempt + 1,
                    self.retry_config["max_attempts"],
                    str(e),
                )

                await asyncio.sleep(delay)

            except NonRetryableError as e:
                # Don't retry these errors
                self.record_error(
                    error=e,
                    context=f"{operation_name}_non_retryable",
                    metadata={"attempts": attempt + 1},
                )
                raise

            except Exception as e:
                # Unknown exception - don't retry to avoid infinite loops
                self.record_error(
                    error=e,
                    context=f"{operation_name}_unknown_error",
                    metadata={"attempts": attempt + 1},
                )
                raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

    def handle_database_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle database-related errors"""
        self.record_error(
            error=error,
            context=f"database_{operation}",
            metadata={"operation": operation},
        )

        return {
            "success": False,
            "error_type": "database_error",
            "message": "Database operation failed. Please try again.",
            "retry_recommended": True,
        }

    def handle_openai_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle OpenAI API errors"""
        self.record_error(
            error=error,
            context=f"openai_{operation}",
            metadata={"operation": operation},
        )

        error_message = str(error).lower()

        if "rate limit" in error_message:
            return {
                "success": False,
                "error_type": "rate_limit",
                "message": "API rate limit reached. Please wait a moment and try again.",
                "retry_recommended": True,
                "suggested_delay": 60,
            }
        elif "quota" in error_message:
            return {
                "success": False,
                "error_type": "quota_exceeded",
                "message": "API quota exceeded. Please contact support.",
                "retry_recommended": False,
            }
        elif "authentication" in error_message or "unauthorized" in error_message:
            return {
                "success": False,
                "error_type": "authentication",
                "message": "API authentication failed. Please contact support.",
                "retry_recommended": False,
            }
        else:
            return {
                "success": False,
                "error_type": "api_error",
                "message": "AI service temporarily unavailable. Please try again.",
                "retry_recommended": True,
            }

    def handle_retrieval_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """Handle document retrieval errors"""
        self.record_error(
            error=error,
            context="document_retrieval",
            metadata={"query": query[:100]},  # Truncate query for logging
        )

        return {
            "success": False,
            "error_type": "retrieval_error",
            "message": "Knowledge retrieval failed. Using fallback approach.",
            "retry_recommended": True,
            "fallback_available": True,
        }

    def handle_context_error(
        self, error: Exception, context_type: str
    ) -> Dict[str, Any]:
        """Handle context engineering errors"""
        self.record_error(
            error=error,
            context=f"context_engineering_{context_type}",
            metadata={"context_type": context_type},
        )

        return {
            "success": False,
            "error_type": "context_error",
            "message": "Context processing failed. Using simplified approach.",
            "retry_recommended": True,
            "fallback_available": True,
        }

    def handle_response_generation_error(
        self, error: Exception, model: str
    ) -> Dict[str, Any]:
        """Handle response generation errors"""
        self.record_error(
            error=error, context="response_generation", metadata={"model": model}
        )

        return {
            "success": False,
            "error_type": "response_generation_error",
            "message": "Response generation failed. Please try rephrasing your question.",
            "retry_recommended": True,
            "fallback_available": True,
        }

    async def create_fallback_response(
        self, error_message: str, include_suggestions: bool = True
    ) -> Dict[str, Any]:
        """Create a fallback response when errors occur"""

        base_response = f"I apologize, but I encountered an issue: {error_message}"

        if include_suggestions:
            suggestions = [
                "Try rephrasing your question",
                "Check if your query is specific enough",
                "Wait a moment and try again",
            ]
            base_response += "\n\nSuggestions:\n" + "\n".join(
                f"• {s}" for s in suggestions
            )

        return {
            "response": base_response,
            "sources": [],
            "conversation_id": None,
            "message_id": None,
            "processing_time_ms": 0,
            "context_quality": {"coverage_score": 0, "relevance_score": 0},
            "config_used": "fallback",
            "is_fallback": True,
            "error_handled": True,
        }

    def with_error_handling(
        self, operation_name: str = None, service: str = "chat_service"
    ):
        """Decorator for automatic error handling"""

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.record_error(
                        error=e,
                        service=service,
                        context=op_name,
                        metadata={"function": func.__name__},
                    )
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.record_error(
                        error=e,
                        service=service,
                        context=op_name,
                        metadata={"function": func.__name__},
                    )
                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics (would integrate with monitoring system)"""
        # This would normally fetch from the monitoring system
        return {
            "total_errors_today": 0,
            "error_rate_percent": 0.0,
            "most_common_errors": [],
            "recovery_rate_percent": 0.0,
            "average_retry_success_rate": 0.0,
        }

    def configure_retry_settings(
        self,
        max_attempts: int = None,
        base_delay: float = None,
        max_delay: float = None,
        exponential_base: int = None,
    ) -> None:
        """Configure retry behavior"""
        if max_attempts is not None:
            self.retry_config["max_attempts"] = max_attempts
        if base_delay is not None:
            self.retry_config["base_delay"] = base_delay
        if max_delay is not None:
            self.retry_config["max_delay"] = max_delay
        if exponential_base is not None:
            self.retry_config["exponential_base"] = exponential_base

        logger.info("Retry configuration updated: %s", self.retry_config)
