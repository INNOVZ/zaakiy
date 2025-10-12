"""
Enhanced logging configuration with structured logging for ZaaKy AI Platform
"""
import json
import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config.settings import get_app_config

# Optional import for database logging
try:
    from services.storage.supabase_client import supabase
except ImportError:
    supabase = None

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
org_id_var: ContextVar[str] = ContextVar("org_id", default="")
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="")


class StructuredFormatter(logging.Formatter):
    """Enhanced structured JSON formatter with context injection"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON with request context"""

        # Base log structure
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add request context if available
        request_context = {}
        if request_id_var.get():
            request_context["request_id"] = request_id_var.get()
        if user_id_var.get():
            request_context["user_id"] = user_id_var.get()
        if org_id_var.get():
            request_context["org_id"] = org_id_var.get()
        if conversation_id_var.get():
            request_context["conversation_id"] = conversation_id_var.get()

        if request_context:
            log_data["context"] = request_context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add custom fields from record
        custom_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ]:
                custom_fields[key] = value

        if custom_fields:
            log_data["extra"] = custom_fields

        # Add service info
        app_config = get_app_config()
        log_data["service"] = {
            "name": "zaaky-api",
            "version": app_config.app_version,
            "environment": app_config.environment,
        }

        return orjson.dumps(log_data).decode("utf-8")


class ContextFilter(logging.Filter):
    """Filter to add service context to log records"""

    def __init__(self, service_name: str = "zaaky-api"):
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Add service context to log record"""
        record.service_name = self.service_name
        record.hostname = "localhost"  # Could be dynamic
        return True


class DatabaseLogHandler(logging.Handler):
    """Custom handler to store critical logs in database"""

    def _init_supabase(self):
        """Initialize Supabase client for database logging"""
        try:
            if supabase is not None:
                self.supabase = supabase
            else:
                print("Supabase client not available for database logging")
        except Exception as e:
            print(f"Failed to initialize database logging: {e}")
        try:
            self.supabase = supabase
        except Exception as e:
            print(f"Failed to initialize database logging: {e}")

    def emit(self, record: logging.LogRecord):
        """Store critical log entries in database"""
        if not self.supabase:
            return

        try:
            log_entry = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "request_id": request_id_var.get() or None,
                "user_id": user_id_var.get() or None,
                "org_id": org_id_var.get() or None,
                "exception_info": self.format(record) if record.exc_info else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Only log ERROR and CRITICAL to database
            if record.levelno >= logging.ERROR:
                self.supabase.table("system_logs").insert(log_entry).execute()

        except Exception:
            # Never let logging cause application failures
            pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_structured: bool = True,
    enable_database_logging: bool = False,
    service_name: str = "zaaky-api",
) -> None:
    """Setup comprehensive logging configuration"""

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with structured logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Add context filter
    context_filter = ContextFilter(service_name)
    console_handler.addFilter(context_filter)

    if enable_structured:
        console_formatter = StructuredFormatter()
    else:
        # Human-readable format for development
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=100 * 1024 * 1024, backupCount=10  # 100MB
        )
        file_handler.setLevel(numeric_level)
        file_handler.addFilter(context_filter)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Database handler for critical errors
    if enable_database_logging:
        try:
            db_handler = DatabaseLogHandler(level=logging.ERROR)
            root_logger.addHandler(db_handler)
        except Exception as e:
            print(f"Failed to setup database logging: {e}")

    # Configure specific loggers
    configure_logger_levels()

    # Log startup message
    logger = logging.getLogger("logging")
    logger.info(
        "Logging configured - Level: %s, Structured: %s", log_level, enable_structured
    )


def configure_logger_levels() -> None:
    """Configure specific logger levels and behavior"""

    # Third-party library loggers (reduce noise)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("pinecone").setLevel(logging.INFO)
    logging.getLogger("supabase").setLevel(logging.INFO)

    # Application loggers (detailed)
    logging.getLogger("services").setLevel(logging.DEBUG)
    logging.getLogger("routers").setLevel(logging.DEBUG)
    logging.getLogger("main").setLevel(logging.INFO)
    logging.getLogger("chat_service").setLevel(logging.DEBUG)
    logging.getLogger("context_analytics").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with enhanced functionality"""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to log messages"""

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **extra_context,
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.org_id = org_id
        self.conversation_id = conversation_id
        self.extra_context = extra_context

        # Store previous values
        self.prev_request_id = None
        self.prev_user_id = None
        self.prev_org_id = None
        self.prev_conversation_id = None

    def __enter__(self):
        """Set context variables"""
        self.prev_request_id = request_id_var.get()
        self.prev_user_id = user_id_var.get()
        self.prev_org_id = org_id_var.get()
        self.prev_conversation_id = conversation_id_var.get()

        request_id_var.set(self.request_id)
        if self.user_id:
            user_id_var.set(self.user_id)
        if self.org_id:
            org_id_var.set(self.org_id)
        if self.conversation_id:
            conversation_id_var.set(self.conversation_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context values"""
        request_id_var.set(self.prev_request_id)
        user_id_var.set(self.prev_user_id)
        org_id_var.set(self.prev_org_id)
        conversation_id_var.set(self.prev_conversation_id)


class PerformanceLogger:
    """Logger for performance metrics"""

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now(timezone.utc)
        self.logger.info(
            "Starting operation: %s",
            self.operation,
            extra={
                "operation": self.operation,
                "event": "start",
                "timestamp": self.start_time.isoformat(),
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds() * 1000

        log_data = {
            "operation": self.operation,
            "event": "complete",
            "duration_ms": round(duration, 2),
            "success": exc_type is None,
            "timestamp": self.end_time.isoformat(),
        }

        if exc_type:
            log_data["error_type"] = exc_type.__name__
            log_data["error_message"] = str(exc_val)
            self.logger.error(
                "Operation failed: %s (%.2fms)",
                self.operation,
                duration,
                extra=log_data,
            )
        else:
            self.logger.info(
                "Operation completed: %s (%.2fms)",
                self.operation,
                duration,
                extra=log_data,
            )


# Convenience functions for common logging patterns


def log_api_request(method: str, path: str, user_id: str = None, org_id: str = None):
    """Log API request with context"""
    logger = get_logger("api")
    logger.info(
        "API Request: %s %s",
        method,
        path,
        extra={
            "event_type": "api_request",
            "method": method,
            "path": path,
            "user_id": user_id,
            "org_id": org_id,
        },
    )


def log_api_response(method: str, path: str, status_code: int, duration_ms: float):
    """Log API response with performance data"""
    logger = get_logger("api")
    logger.info(
        "API Response: %s %s - %s (%.2fms)",
        method,
        path,
        status_code,
        duration_ms,
        extra={
            "event_type": "api_response",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
        },
    )


def log_chat_interaction(
    org_id: str,
    conversation_id: str,
    message_type: str,
    processing_time_ms: float = None,
    context_quality: float = None,
):
    """Log chat interactions with analytics data"""
    logger = get_logger("chat")

    log_data = {
        "event_type": "chat_interaction",
        "org_id": org_id,
        "conversation_id": conversation_id,
        "message_type": message_type,
    }

    if processing_time_ms is not None:
        log_data["processing_time_ms"] = processing_time_ms
    if context_quality is not None:
        log_data["context_quality"] = context_quality

    logger.info("Chat interaction: %s", message_type, extra=log_data)


def log_vector_operation(
    operation: str, upload_id: str, count: int, duration_ms: float
):
    """Log vector database operations"""
    logger = get_logger("vector_db")
    logger.info(
        "Vector operation: %s",
        operation,
        extra={
            "event_type": "vector_operation",
            "operation": operation,
            "upload_id": upload_id,
            "vector_count": count,
            "duration_ms": duration_ms,
        },
    )


def log_error_with_context(
    logger: logging.Logger,
    message: str,
    error: Exception,
    context: Dict[str, Any] = None,
):
    """Log errors with full context and stack trace"""
    error_data = {
        "event_type": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
    }

    logger.error(message, extra=error_data, exc_info=True)
