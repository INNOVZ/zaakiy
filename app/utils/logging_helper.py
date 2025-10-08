"""
Logging helper utilities for consistent error handling

This module provides utilities to ensure consistent logging across the application,
replacing print() statements with proper logging.
"""

import logging
import functools
from typing import Any, Callable, Optional
from datetime import datetime


def get_module_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module

    Args:
        name: Module name (usually __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with parameters and results

    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger.debug(
                f"Calling {func_name}",
                extra={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )

            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(
                    f"{func_name} failed: {e}",
                    exc_info=True,
                    extra={"function": func_name}
                )
                raise

        return wrapper
    return decorator


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **context
):
    """
    Log a message with additional context

    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **context: Additional context as keyword arguments
    """
    log_func = getattr(logger, level.lower())
    log_func(message, extra=context)


class LoggerAdapter:
    """
    Adapter to provide consistent logging interface

    This helps migrate from print() statements to proper logging
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, message: str, **context):
        """Log info message"""
        self.logger.info(message, extra=context)

    def success(self, message: str, **context):
        """Log success message (info level with success indicator)"""
        self.logger.info(f"✅ {message}", extra={"success": True, **context})

    def warning(self, message: str, **context):
        """Log warning message"""
        self.logger.warning(message, extra=context)

    def error(self, message: str, exc_info: bool = False, **context):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, extra=context)

    def debug(self, message: str, **context):
        """Log debug message"""
        self.logger.debug(message, extra=context)

    def critical(self, message: str, exc_info: bool = True, **context):
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info, extra=context)


def create_logger_adapter(name: str) -> LoggerAdapter:
    """
    Create a logger adapter for a module

    Args:
        name: Module name (usually __name__)

    Returns:
        LoggerAdapter: Configured logger adapter
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger)


# Migration helpers for replacing print() statements

def print_to_logger_info(logger: logging.Logger, message: str):
    """
    Helper to replace print() with logger.info()

    Usage:
        # Old: print(f"[Info] Processing {item}")
        # New: print_to_logger_info(logger, f"Processing {item}")
    """
    logger.info(message)


def print_to_logger_error(logger: logging.Logger, message: str, exc_info: bool = False):
    """
    Helper to replace print() with logger.error()

    Usage:
        # Old: print(f"[Error] Failed: {e}")
        # New: print_to_logger_error(logger, f"Failed: {e}", exc_info=True)
    """
    logger.error(message, exc_info=exc_info)


def print_to_logger_warning(logger: logging.Logger, message: str):
    """
    Helper to replace print() with logger.warning()

    Usage:
        # Old: print(f"[Warning] Issue detected")
        # New: print_to_logger_warning(logger, "Issue detected")
    """
    logger.warning(message)


# Context manager for operation logging

class LogOperation:
    """
    Context manager for logging operations with timing

    Usage:
        with LogOperation(logger, "Processing file"):
            # Do work
            pass
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: str = "info",
        log_success: bool = True,
        log_failure: bool = True
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.log_success = log_success
        self.log_failure = log_failure
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        log_func = getattr(self.logger, self.level)
        log_func(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()

        if exc_type is None and self.log_success:
            self.logger.info(
                f"Completed: {self.operation}",
                extra={"duration_seconds": duration, "success": True}
            )
        elif exc_type is not None and self.log_failure:
            self.logger.error(
                f"Failed: {self.operation} - {exc_val}",
                exc_info=True,
                extra={"duration_seconds": duration, "success": False}
            )

        return False  # Don't suppress exceptions


# Structured logging helpers

def log_api_call(
    logger: logging.Logger,
    method: str,
    url: str,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None
):
    """Log API call with structured data"""
    level = "info" if status_code and status_code < 400 else "error"
    log_func = getattr(logger, level)

    log_func(
        f"{method} {url} - {status_code or 'N/A'}",
        extra={
            "api_call": True,
            "method": method,
            "url": url,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "error": error
        }
    )


def log_database_operation(
    logger: logging.Logger,
    operation: str,
    table: str,
    success: bool,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None
):
    """Log database operation with structured data"""
    level = "info" if success else "error"
    log_func = getattr(logger, level)

    log_func(
        f"DB {operation} on {table} - {'success' if success else 'failed'}",
        extra={
            "database_operation": True,
            "operation": operation,
            "table": table,
            "success": success,
            "duration_ms": duration_ms,
            "error": error
        }
    )


def log_file_operation(
    logger: logging.Logger,
    operation: str,
    file_path: str,
    success: bool,
    file_size: Optional[int] = None,
    error: Optional[str] = None
):
    """Log file operation with structured data"""
    level = "info" if success else "error"
    log_func = getattr(logger, level)

    log_func(
        f"File {operation}: {file_path} - {'success' if success else 'failed'}",
        extra={
            "file_operation": True,
            "operation": operation,
            "file_path": file_path,
            "success": success,
            "file_size": file_size,
            "error": error
        }
    )


# Example usage documentation
"""
MIGRATION GUIDE: From print() to logger

1. Import logger at module level:
   import logging
   logger = logging.getLogger(__name__)

2. Replace print statements:
   
   # OLD
   print(f"[Info] Processing {item}")
   print(f"[Error] Failed: {e}")
   print(f"[Warning] Issue detected")
   
   # NEW
   logger.info(f"Processing {item}")
   logger.error(f"Failed: {e}", exc_info=True)
   logger.warning("Issue detected")

3. Use structured logging:
   
   # Instead of
   print(f"[Info] API call to {url} returned {status}")
   
   # Use
   log_api_call(logger, "GET", url, status_code=200, duration_ms=150)

4. Use context managers for operations:
   
   # Instead of
   print("[Info] Starting processing")
   try:
       process()
       print("[Success] Processing complete")
   except Exception as e:
       print(f"[Error] Processing failed: {e}")
   
   # Use
   with LogOperation(logger, "Processing"):
       process()
"""
