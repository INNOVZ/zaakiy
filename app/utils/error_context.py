"""
Error context management for structured logging and error tracking
"""
import uuid
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
org_id_var: ContextVar[Optional[str]] = ContextVar('org_id', default=None)
operation_var: ContextVar[Optional[str]] = ContextVar('operation', default=None)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Structured error context for logging and monitoring"""
    error_id: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    operation: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    timestamp: datetime = None
    service: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_context: Dict[str, Any] = None
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.additional_context is None:
            self.additional_context = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        data = asdict(self)
        # Convert enums to strings
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ErrorContextManager:
    """Manages error context throughout request lifecycle"""

    @staticmethod
    def create_error_context(
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        additional_context: Dict[str, Any] = None,
        service: str = None
    ) -> ErrorContext:
        """Create error context from current request context"""
        
        # Generate unique error ID
        error_id = f"ERR_{uuid.uuid4().hex[:8].upper()}"
        
        # Get current context
        request_id = request_id_var.get()
        user_id = user_id_var.get()
        org_id = org_id_var.get()
        operation = operation_var.get()
        
        # Determine category from exception type
        if category == ErrorCategory.UNKNOWN:
            category = ErrorContextManager._categorize_exception(error)
        
        # Determine severity from exception type
        if severity == ErrorSeverity.MEDIUM:
            severity = ErrorContextManager._determine_severity(error)
        
        return ErrorContext(
            error_id=error_id,
            request_id=request_id,
            user_id=user_id,
            org_id=org_id,
            operation=operation,
            severity=severity,
            category=category,
            service=service,
            additional_context=additional_context or {},
            stack_trace=traceback.format_exc()
        )

    @staticmethod
    def _categorize_exception(error: Exception) -> ErrorCategory:
        """Categorize exception based on type"""
        error_type = type(error).__name__.lower()
        
        if 'auth' in error_type or 'permission' in error_type:
            return ErrorCategory.AUTHENTICATION
        elif 'validation' in error_type or 'value' in error_type:
            return ErrorCategory.VALIDATION
        elif 'database' in error_type or 'sql' in error_type:
            return ErrorCategory.DATABASE
        elif 'connection' in error_type or 'timeout' in error_type:
            return ErrorCategory.NETWORK
        elif 'config' in error_type:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN

    @staticmethod
    def _determine_severity(error: Exception) -> ErrorSeverity:
        """Determine error severity based on type and context"""
        error_type = type(error).__name__.lower()
        
        if 'critical' in error_type or 'fatal' in error_type:
            return ErrorSeverity.CRITICAL
        elif 'auth' in error_type or 'permission' in error_type:
            return ErrorSeverity.HIGH
        elif 'validation' in error_type:
            return ErrorSeverity.MEDIUM
        elif 'timeout' in error_type or 'connection' in error_type:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.MEDIUM

    @staticmethod
    def set_request_context(
        request_id: str = None,
        user_id: str = None,
        org_id: str = None,
        operation: str = None
    ):
        """Set request context variables"""
        if request_id:
            request_id_var.set(request_id)
        if user_id:
            user_id_var.set(user_id)
        if org_id:
            org_id_var.set(org_id)
        if operation:
            operation_var.set(operation)

    @staticmethod
    def clear_context():
        """Clear all context variables"""
        request_id_var.set(None)
        user_id_var.set(None)
        org_id_var.set(None)
        operation_var.set(None)


class StructuredErrorLogger:
    """Structured error logging with context"""

    def __init__(self, logger_name: str = None):
        self.logger = logging.getLogger(logger_name or __name__)

    def log_error(
        self,
        error: Exception,
        context: ErrorContext,
        message: str = None
    ):
        """Log error with structured context"""
        
        log_data = {
            "event_type": "error",
            "error_context": context.to_dict(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "message": message or f"Error in {context.operation or 'unknown operation'}"
        }
        
        # Log at appropriate level based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_data, exc_info=True)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_data, exc_info=True)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_data, exc_info=True)
        else:
            self.logger.info(log_data, exc_info=True)

    def log_recovery_attempt(
        self,
        context: ErrorContext,
        recovery_method: str,
        success: bool
    ):
        """Log error recovery attempt"""
        context.recovery_attempted = True
        context.recovery_successful = success
        
        log_data = {
            "event_type": "error_recovery",
            "error_context": context.to_dict(),
            "recovery_method": recovery_method,
            "recovery_successful": success
        }
        
        if success:
            self.logger.info(log_data)
        else:
            self.logger.warning(log_data)


# Global error logger instance
error_logger = StructuredErrorLogger()
