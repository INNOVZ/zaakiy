"""
Authentication and authorization specific exceptions

Custom exceptions for authentication and authorization errors.
"""

from typing import Optional


class AuthBaseException(Exception):
    """Base exception for authentication and authorization errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class AuthenticationError(AuthBaseException):
    """Raised when authentication fails"""

    def __init__(
        self, message: str = "Authentication failed", error_code: str = "AUTH_FAILED"
    ):
        super().__init__(message, error_code)


class AuthorizationError(AuthBaseException):
    """Raised when authorization fails (user doesn't have required permissions)"""

    def __init__(
        self, message: str = "Authorization failed", error_code: str = "AUTHZ_FAILED"
    ):
        super().__init__(message, error_code)


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired"""

    def __init__(
        self, message: str = "Token has expired", error_code: str = "TOKEN_EXPIRED"
    ):
        super().__init__(message, error_code)


class InvalidTokenError(AuthenticationError):
    """Raised when JWT token is invalid or malformed"""

    def __init__(
        self, message: str = "Invalid token", error_code: str = "INVALID_TOKEN"
    ):
        super().__init__(message, error_code)


class UserNotFoundError(AuthenticationError):
    """Raised when user is not found in the system"""

    def __init__(
        self, message: str = "User not found", error_code: str = "USER_NOT_FOUND"
    ):
        super().__init__(message, error_code)


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user doesn't have required permissions"""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        error_code: str = "INSUFFICIENT_PERMISSIONS",
    ):
        super().__init__(message, error_code)


class OrganizationAccessError(AuthorizationError):
    """Raised when user doesn't have access to organization"""

    def __init__(
        self,
        message: str = "Organization access denied",
        error_code: str = "ORG_ACCESS_DENIED",
    ):
        super().__init__(message, error_code)


class RoleRequiredError(AuthorizationError):
    """Raised when specific role is required but user doesn't have it"""

    def __init__(
        self,
        message: str = "Required role not found",
        error_code: str = "ROLE_REQUIRED",
    ):
        super().__init__(message, error_code)


class UserCreationError(AuthenticationError):
    """Raised when user creation fails"""

    def __init__(
        self,
        message: str = "Failed to create user",
        error_code: str = "USER_CREATION_FAILED",
    ):
        super().__init__(message, error_code)


class OrganizationCreationError(AuthenticationError):
    """Raised when organization creation fails"""

    def __init__(
        self,
        message: str = "Failed to create organization",
        error_code: str = "ORG_CREATION_FAILED",
    ):
        super().__init__(message, error_code)


class ConfigurationError(AuthBaseException):
    """Raised when authentication configuration is invalid"""

    def __init__(
        self,
        message: str = "Authentication configuration error",
        error_code: str = "CONFIG_ERROR",
    ):
        super().__init__(message, error_code)
