"""
Consolidated Authentication and Authorization Middleware
Production-ready security implementation with complete authentication, authorization,
and security monitoring capabilities.

This module combines core JWT authentication with enhanced security features including
entity ownership verification, security monitoring, and suspicious activity tracking.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .exceptions import AuthenticationError, AuthorizationError
from .jwt_handler import verify_jwt_token
from .permissions import (Permission, UserRole, check_org_permission,
                          check_user_permission, get_user_role)
from .user_auth import get_user_by_id, get_user_with_org

logger = logging.getLogger(__name__)

# Security scheme for OpenAPI documentation
security = HTTPBearer()


# ==========================================
# CORE AUTHENTICATION FUNCTIONS
# ==========================================


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        Dict containing user information

    Raises:
        HTTPException: For authentication failures
    """
    try:
        # Pass the raw token directly to verify_jwt_token
        user_info = await verify_jwt_token(credentials.credentials)

        # Get full user data from database
        user_data = await get_user_by_id(user_info["user_id"])

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return {**user_info, "user_data": user_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
        ) from e


async def get_current_user_with_org(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current user with organization data

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        Dict containing user and organization information

    Raises:
        HTTPException: For authentication failures
    """
    try:
        # Get current user
        user_info = await get_current_user(credentials)
        user_id = user_info["user_id"]

        # Get user with organization data
        user_with_org = await get_user_with_org(user_id)

        return {**user_info, "user_with_org": user_with_org}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user organization: {str(e)}",
        ) from e


def require_auth():
    """
    Decorator factory for requiring authentication

    Returns:
        FastAPI dependency for authentication
    """
    return Depends(get_current_user)


def require_user_with_org():
    """
    Decorator factory for requiring user with organization data

    Returns:
        FastAPI dependency for user with organization
    """
    return Depends(get_current_user_with_org)


# ==========================================
# PERMISSION-BASED AUTHORIZATION
# ==========================================


def require_permission(permission: Permission):
    """
    Decorator factory for requiring specific permission

    Args:
        permission: Required permission

    Returns:
        FastAPI dependency for permission check
    """

    async def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_id = current_user["user_id"]

        if not await check_user_permission(user_id, permission):
            user_role = await get_user_role(user_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission.value}, "
                f"User role: {user_role.value if user_role else 'unknown'}",
            )

        return current_user

    return Depends(permission_dependency)


def require_org_permission(permission: Permission):
    """
    Decorator factory for requiring organization-specific permission

    Args:
        permission: Required permission

    Returns:
        FastAPI dependency for organization permission check
    """

    async def org_permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user_with_org)
    ) -> Dict[str, Any]:
        user_id = current_user["user_id"]
        user_data = current_user["user_with_org"]
        org_id = user_data.get("org_id")

        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is not associated with an organization",
            )

        if not await check_org_permission(user_id, org_id, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for organization. "
                f"Required: {permission.value}",
            )

        return current_user

    return Depends(org_permission_dependency)


# ==========================================
# ROLE-BASED AUTHORIZATION
# ==========================================


def require_role(required_role: UserRole):
    """
    Decorator factory for requiring specific user role

    Args:
        required_role: Required user role

    Returns:
        FastAPI dependency for role check
    """

    async def role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_id = current_user["user_id"]
        user_role = await get_user_role(user_id)

        if not user_role or user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {required_role.value}, "
                f"User role: {user_role.value if user_role else 'unknown'}",
            )

        return current_user

    return Depends(role_dependency)


def require_admin_role():
    """
    Decorator factory for requiring admin role (admin or owner)

    Returns:
        FastAPI dependency for admin role check
    """

    async def admin_role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_id = current_user["user_id"]
        user_role = await get_user_role(user_id)

        if not user_role or user_role not in [UserRole.ADMIN, UserRole.OWNER]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required"
            )

        return current_user

    return Depends(admin_role_dependency)


def require_owner_role():
    """
    Decorator factory for requiring owner role

    Returns:
        FastAPI dependency for owner role check
    """
    return require_role(UserRole.OWNER)


# ==========================================
# ENHANCED SECURITY FEATURES
# ==========================================


class SecurityMonitor:
    """Security monitoring and entity ownership verification"""

    def __init__(self):
        self.suspicious_activities = {}  # Track suspicious activities

    async def verify_entity_ownership(
        self, current_user: Dict[str, Any], entity_type: str, entity_id: str
    ) -> bool:
        """
        Verify user owns or has access to the specified entity

        Args:
            current_user: Current authenticated user
            entity_type: Type of entity ("user" or "organization")
            entity_id: ID of the entity to access

        Returns:
            True if user has access, False otherwise
        """

        user_id = current_user["user_id"]
        user_data = current_user.get("user_data", {})

        if entity_type == "user":
            # User can only access their own data
            has_access = user_id == entity_id

        elif entity_type == "organization":
            # User can only access their organization's data
            user_org_id = user_data.get("org_id")
            if not user_org_id:
                has_access = False
            else:
                has_access = user_org_id == entity_id
        else:
            has_access = False

        # Log access attempts for audit
        if not has_access:
            logger.warning(
                "Unauthorized access attempt: user %s tried to access %s %s",
                user_id,
                entity_type,
                entity_id,
                extra={
                    "event_type": "unauthorized_access",
                    "user_id": user_id,
                    "target_entity_type": entity_type,
                    "target_entity_id": entity_id,
                    "timestamp": time.time(),
                },
            )

            # Track suspicious activity
            self._track_suspicious_activity(user_id, "unauthorized_access")

        return has_access

    def _track_suspicious_activity(self, user_id: str, activity_type: str):
        """Track suspicious activities for security monitoring"""
        current_time = time.time()

        if user_id not in self.suspicious_activities:
            self.suspicious_activities[user_id] = {}

        if activity_type not in self.suspicious_activities[user_id]:
            self.suspicious_activities[user_id][activity_type] = []

        # Add timestamp
        self.suspicious_activities[user_id][activity_type].append(current_time)

        # Clean old entries (keep last hour)
        cutoff_time = current_time - 3600
        self.suspicious_activities[user_id][activity_type] = [
            t
            for t in self.suspicious_activities[user_id][activity_type]
            if t > cutoff_time
        ]

        # Alert if too many suspicious activities
        if len(self.suspicious_activities[user_id][activity_type]) > 5:
            logger.error(
                "High suspicious activity detected for user %s: %s",
                user_id,
                activity_type,
                extra={
                    "event_type": "high_suspicious_activity",
                    "user_id": user_id,
                    "activity_type": activity_type,
                    "count": len(self.suspicious_activities[user_id][activity_type]),
                },
            )

    async def require_entity_access(
        self, entity_type: str, entity_id: str, current_user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Require user to have access to the specified entity

        Args:
            entity_type: Type of entity
            entity_id: ID of entity
            current_user: Current authenticated user

        Returns:
            Current user if authorized

        Raises:
            HTTPException: If user doesn't have access
        """

        if not await self.verify_entity_ownership(current_user, entity_type, entity_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: You can only access your own {entity_type} data",
            )

        return current_user

    async def require_admin_with_monitoring(
        self, current_user: Dict[str, Any], endpoint: str
    ) -> Dict[str, Any]:
        """
        Require admin role with security monitoring

        Args:
            current_user: Current user
            endpoint: Endpoint name for monitoring

        Returns:
            Current user if authorized admin

        Raises:
            HTTPException: If not admin
        """

        user_id = current_user["user_id"]
        user_role = await get_user_role(user_id)

        # Check admin role
        if not user_role or user_role not in [UserRole.ADMIN, UserRole.OWNER]:
            logger.warning(
                "Non-admin user %s attempted admin action on %s",
                user_id,
                endpoint,
                extra={
                    "event_type": "unauthorized_admin_attempt",
                    "user_id": user_id,
                    "endpoint": endpoint,
                    "user_role": user_role.value if user_role else "none",
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required"
            )

        # Log admin action
        logger.info(
            "Admin action authorized: user %s accessing %s",
            user_id,
            endpoint,
            extra={
                "event_type": "admin_action_authorized",
                "user_id": user_id,
                "endpoint": endpoint,
                "user_role": user_role.value,
            },
        )

        return current_user


# Global security monitor instance
security_monitor = SecurityMonitor()


# ==========================================
# ENHANCED FASTAPI DEPENDENCIES
# ==========================================


def require_entity_access(entity_type: str, entity_id: str):
    """
    Factory function for entity access dependency

    Args:
        entity_type: Type of entity to check access for
        entity_id: ID of entity to check access for

    Returns:
        FastAPI dependency function
    """

    async def dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        return await security_monitor.require_entity_access(
            entity_type, entity_id, current_user
        )

    return Depends(dependency)


def require_admin_with_monitoring(endpoint: str):
    """
    Factory function for admin access with monitoring

    Args:
        endpoint: Endpoint name for monitoring

    Returns:
        FastAPI dependency function
    """

    async def dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        return await security_monitor.require_admin_with_monitoring(
            current_user, endpoint
        )

    return Depends(dependency)


# ==========================================
# CONVENIENCE DEPENDENCIES
# ==========================================

# Standard auth dependencies
CurrentUser = Depends(get_current_user)
CurrentUserWithOrg = Depends(get_current_user_with_org)
RequireAuth = Depends(get_current_user)
RequireAdmin = require_admin_role()
RequireOwner = require_owner_role()

# Permission-based dependencies
RequireUserManagement = require_permission(Permission.UPDATE_USER)
RequireOrgManagement = require_permission(Permission.UPDATE_ORG)
RequireChatManagement = require_permission(Permission.CREATE_CHAT)
RequireFileManagement = require_permission(Permission.UPLOAD_FILE)
RequireSystemManagement = require_permission(Permission.MANAGE_SYSTEM)


# Entity access dependencies (factory functions)
def RequireUserAccess(entity_id: str):
    """Require access to specific user"""
    return require_entity_access("user", entity_id)


def RequireOrgAccess(entity_id: str):
    """Require access to specific organization"""
    return require_entity_access("organization", entity_id)


# Admin with monitoring dependencies
RequireAdminSignup = require_admin_with_monitoring("signup")
RequireAdminManagement = require_admin_with_monitoring("management")
