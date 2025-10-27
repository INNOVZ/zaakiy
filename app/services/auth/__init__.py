"""
Authentication and authorization services

This module provides centralized authentication and authorization functionality
for the Zentria backend application with enhanced security features.
"""

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    InvalidTokenError,
    TokenExpiredError,
)
from .jwt_handler import (
    get_jwt_validator_status,
    verify_jwt_token,
    verify_jwt_token_from_header,
)
from .middleware import (
    CurrentUser,
    CurrentUserWithOrg,
    RequireAdmin,
    RequireAdminManagement,
    RequireAdminSignup,
    RequireChatManagement,
    RequireFileManagement,
    RequireOrgAccess,
    RequireOrgManagement,
    RequireOwner,
    RequireSystemManagement,
    RequireUserAccess,
    RequireUserManagement,
    get_current_user,
    get_current_user_with_org,
    require_admin_role,
    require_admin_with_monitoring,
    require_auth,
    require_entity_access,
    require_org_permission,
    require_owner_role,
    require_permission,
    require_role,
    require_user_with_org,
    security_monitor,
)
from .permissions import (
    Permission,
    UserRole,
    check_org_permission,
    check_user_permission,
)
from .user_auth import (
    create_organization_for_user,
    get_user_with_org,
    sync_user_if_missing,
)

__all__ = [
    # JWT handling
    "verify_jwt_token",
    "verify_jwt_token_from_header",
    "get_jwt_validator_status",
    # User authentication
    "get_user_with_org",
    "sync_user_if_missing",
    "create_organization_for_user",
    # Permissions
    "check_user_permission",
    "check_org_permission",
    "Permission",
    "UserRole",
    # Core middleware
    "get_current_user",
    "get_current_user_with_org",
    "require_auth",
    "require_user_with_org",
    # Role-based dependencies
    "require_role",
    "require_admin_role",
    "require_owner_role",
    # Permission-based dependencies
    "require_permission",
    "require_org_permission",
    # Enhanced security
    "require_entity_access",
    "require_admin_with_monitoring",
    "security_monitor",
    # Convenience dependencies
    "CurrentUser",
    "CurrentUserWithOrg",
    "RequireAdmin",
    "RequireOwner",
    "RequireUserManagement",
    "RequireOrgManagement",
    "RequireChatManagement",
    "RequireFileManagement",
    "RequireSystemManagement",
    "RequireAdminSignup",
    "RequireAdminManagement",
    # Factory functions
    "RequireUserAccess",
    "RequireOrgAccess",
    # Exceptions
    "AuthenticationError",
    "AuthorizationError",
    "TokenExpiredError",
    "InvalidTokenError",
]
