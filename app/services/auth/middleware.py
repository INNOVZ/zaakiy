"""
Authentication middleware and FastAPI dependencies

Provides reusable authentication and authorization dependencies for FastAPI routes.
"""

from typing import Dict, Any, Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .jwt_handler import verify_jwt_token
from .user_auth import get_user_with_org, get_user_by_id
from .permissions import (
    UserRole, 
    Permission, 
    get_user_role, 
    check_user_permission,
    require_user_permission,
    check_org_permission,
    require_org_permission
)
from .exceptions import AuthenticationError, AuthorizationError

# Security scheme for OpenAPI documentation
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
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
        # Extract token from credentials
        token = credentials.credentials
        authorization = f"Bearer {token}"
        
        # Verify token and get user info
        user_info = await verify_jwt_token(authorization)
        
        # Get full user data from database
        user_data = await get_user_by_id(user_info["user_id"])
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            **user_info,
            "user_data": user_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        ) from e


async def get_current_user_with_org(
    credentials: HTTPAuthorizationCredentials = Depends(security)
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
        
        return {
            **user_info,
            "user_with_org": user_with_org
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user organization: {str(e)}"
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
                       f"User role: {user_role.value if user_role else 'unknown'}"
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
                detail="User is not associated with an organization"
            )
        
        if not await check_org_permission(user_id, org_id, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for organization. "
                       f"Required: {permission.value}"
            )
        
        return current_user
    
    return Depends(org_permission_dependency)


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
                       f"User role: {user_role.value if user_role else 'unknown'}"
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
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required"
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


# Convenience dependencies for common use cases
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
