"""
Custom exception classes for ZaaKy AI Platform
"""


class ZaaKyBaseException(Exception):
    """Base exception for all ZaaKy-specific errors"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(ZaaKyBaseException):
    """Authentication related errors"""

    pass


class AuthorizationError(ZaaKyBaseException):
    """Authorization related errors"""

    pass


class ValidationError(ZaaKyBaseException):
    """Data validation errors"""

    pass


class ServiceUnavailableError(ZaaKyBaseException):
    """External service unavailable errors"""

    pass


class DatabaseError(ZaaKyBaseException):
    """Database operation errors"""

    pass


class VectorStoreError(ZaaKyBaseException):
    """Vector database errors"""

    pass


class AIServiceError(ZaaKyBaseException):
    """AI/OpenAI service errors"""

    pass


class ChatServiceError(ZaaKyBaseException):
    """Chat service specific errors"""

    pass


class FileProcessingError(ZaaKyBaseException):
    """File upload and processing errors"""

    pass


class ConfigurationError(ZaaKyBaseException):
    """Configuration and settings errors"""

    pass
