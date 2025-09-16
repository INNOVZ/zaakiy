import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class ContextFilter(logging.Filter):
    """Add context information to log records"""

    def filter(self, record):
        # Add timestamp and request context
        record.app_name = "ZaaKy"
        record.environment = os.getenv("ENVIRONMENT", "development")
        return True


class SecurityFilter(logging.Filter):
    """Filter sensitive information from logs"""

    SENSITIVE_PATTERNS = [
        "password", "token", "key", "secret", "auth",
        "api_key", "jwt", "bearer"
    ]

    def filter(self, record):
        # Check if log message contains sensitive information
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            msg_lower = record.msg.lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in msg_lower:
                    record.msg = self._sanitize_message(record.msg, pattern)
        return True

    def _sanitize_message(self, message: str, sensitive_pattern: str) -> str:
        """Replace sensitive information with placeholder"""
        import re
        # Basic sanitization - replace sensitive values
        pattern = rf'{sensitive_pattern}["\s]*[:=]["\s]*[^"\s,}}\]]+'
        return re.sub(pattern, f'{sensitive_pattern}": "***REDACTED***"', message, flags=re.IGNORECASE)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup comprehensive logging configuration"""

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Default log file
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"{log_dir}/zaaky_{timestamp}.log"

    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(app_name)s | %(environment)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "app": "%(app_name)s", "env": "%(environment)s", "logger": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "context_filter": {
                "()": ContextFilter
            },
            "security_filter": {
                "()": SecurityFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "simple",
                "level": log_level,
                "filters": ["context_filter", "security_filter"]
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "formatter": "detailed",
                "level": log_level,
                "filters": ["context_filter", "security_filter"]
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/zaaky_errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "formatter": "json",
                "level": "ERROR",
                "filters": ["context_filter", "security_filter"]
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "zaaky": {
                "handlers": ["console", "file", "error_file"],
                "level": log_level,
                "propagate": False
            },
            "services": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "routers": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }

    logging.config.dictConfig(config)

    # Test logging setup
    logger = logging.getLogger("zaaky.setup")
    logger.info("Logging configuration initialized successfully")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with proper configuration"""
    return logging.getLogger(f"zaaky.{name}")


# Application-specific loggers
def get_service_logger(service_name: str) -> logging.Logger:
    """Get a logger for services"""
    return logging.getLogger(f"services.{service_name}")


def get_router_logger(router_name: str) -> logging.Logger:
    """Get a logger for routers"""
    return logging.getLogger(f"routers.{router_name}")


def log_request_context(logger: logging.Logger, request_id: str, user_id: Optional[str] = None, org_id: Optional[str] = None):
    """Log request context for tracing"""
    context = {
        "request_id": request_id,
        "user_id": user_id,
        "org_id": org_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.info(f"Request context: {context}")


def log_performance_metrics(logger: logging.Logger, operation: str, duration_ms: int, success: bool = True, **kwargs):
    """Log performance metrics"""
    metrics = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger.info(f"Performance metrics: {metrics}")
