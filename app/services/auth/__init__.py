"""
Authentication and authorization services

This module provides centralized authentication and authorization functionality
for the Zentria backend application.
"""

from .jwt_handler import verify_jwt_token, debug_verify_jwt_token
from .user_auth import (
    get_user_with_org,
    sync_user_if_missing,
    create_organization_for_user
)
from .permissions import check_user_permission, check_org_permission
from .middleware import get_current_user, require_auth, CurrentUser, CurrentUserWithOrg
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    TokenExpiredError,
    InvalidTokenError
)

__all__ = [
    # JWT handling
    "verify_jwt_token",
    "debug_verify_jwt_token",
    
    # User authentication
    "get_user_with_org",
    "sync_user_if_missing", 
    "create_organization_for_user",
    
    # Permissions
    "check_user_permission",
    "check_org_permission",
    
    # Middleware
    "get_current_user",
    "require_auth",
    "CurrentUser",
    "CurrentUserWithOrg",
    
    # Exceptions
    "AuthenticationError",
    "AuthorizationError", 
    "TokenExpiredError",
    "InvalidTokenError"
]
