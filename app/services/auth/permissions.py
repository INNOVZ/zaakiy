"""
Role-based access control and permissions

Handles user permissions, organization access, and role-based authorization.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from ..storage.supabase_client import get_supabase_http_client
from .exceptions import AuthorizationError

logger = logging.getLogger(__name__)

# Module-level client variable for lazy loading
_client = None


def _get_client():
    """Get HTTP client with lazy initialization"""
    global _client
    if _client is None:
        _client = get_supabase_http_client()
    return _client


# Create a client property that returns the lazily loaded client


class ClientProxy:
    def __getattr__(self, name):
        return getattr(_get_client(), name)

    async def get(self, *args, **kwargs):
        return await _get_client().get(*args, **kwargs)

    async def post(self, *args, **kwargs):
        return await _get_client().post(*args, **kwargs)

    async def patch(self, *args, **kwargs):
        return await _get_client().patch(*args, **kwargs)

    async def delete(self, *args, **kwargs):
        return await _get_client().delete(*args, **kwargs)


client = ClientProxy()


class UserRole(Enum):
    """User roles in the system"""

    ADMIN = "admin"
    USER = "user"
    OWNER = "owner"
    VIEWER = "viewer"


class Permission(Enum):
    """System permissions"""

    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"

    # Organization management
    CREATE_ORG = "create_org"
    READ_ORG = "read_org"
    UPDATE_ORG = "update_org"
    DELETE_ORG = "delete_org"

    # Chat and content
    CREATE_CHAT = "create_chat"
    READ_CHAT = "read_chat"
    UPDATE_CHAT = "update_chat"
    DELETE_CHAT = "delete_chat"

    # File management
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"
    DELETE_FILE = "delete_file"

    # System administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_LOGS = "view_logs"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.OWNER: [
        Permission.CREATE_USER,
        Permission.READ_USER,
        Permission.UPDATE_USER,
        Permission.DELETE_USER,
        Permission.CREATE_ORG,
        Permission.READ_ORG,
        Permission.UPDATE_ORG,
        Permission.DELETE_ORG,
        Permission.CREATE_CHAT,
        Permission.READ_CHAT,
        Permission.UPDATE_CHAT,
        Permission.DELETE_CHAT,
        Permission.UPLOAD_FILE,
        Permission.DOWNLOAD_FILE,
        Permission.DELETE_FILE,
        Permission.MANAGE_SYSTEM,
        Permission.VIEW_LOGS,
    ],
    UserRole.ADMIN: [
        Permission.CREATE_USER,
        Permission.READ_USER,
        Permission.UPDATE_USER,
        Permission.READ_ORG,
        Permission.UPDATE_ORG,
        Permission.CREATE_CHAT,
        Permission.READ_CHAT,
        Permission.UPDATE_CHAT,
        Permission.DELETE_CHAT,
        Permission.UPLOAD_FILE,
        Permission.DOWNLOAD_FILE,
        Permission.DELETE_FILE,
        Permission.VIEW_LOGS,
    ],
    UserRole.USER: [
        Permission.READ_USER,
        Permission.UPDATE_USER,
        Permission.READ_ORG,
        Permission.CREATE_CHAT,
        Permission.READ_CHAT,
        Permission.UPDATE_CHAT,
        Permission.UPLOAD_FILE,
        Permission.DOWNLOAD_FILE,
    ],
    UserRole.VIEWER: [
        Permission.READ_USER,
        Permission.READ_ORG,
        Permission.READ_CHAT,
        Permission.DOWNLOAD_FILE,
    ],
}


async def get_user_role(user_id: str) -> Optional[UserRole]:
    """
    Get user role from database

    Args:
        user_id: User ID

    Returns:
        UserRole enum or None if not found
    """
    try:
        response = await client.get(
            "/users", params={"id": f"eq.{user_id}", "select": "role"}
        )

        if response.status_code == 200 and response.json():
            role_str = response.json()[0].get("role")
            if role_str:
                try:
                    return UserRole(role_str)
                except ValueError:
                    return UserRole.USER  # Default role
        return UserRole.USER  # Default role
    except Exception as e:
        logger.error(
            "Error fetching user role",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        return UserRole.USER  # Default role


async def check_user_permission(user_id: str, permission: Permission) -> bool:
    """
    Check if user has specific permission

    Args:
        user_id: User ID
        permission: Permission to check

    Returns:
        True if user has permission, False otherwise
    """
    try:
        user_role = await get_user_role(user_id)
        if not user_role:
            return False

        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        return permission in user_permissions
    except Exception as e:
        logger.error(
            "Error checking user permission",
            extra={"user_id": user_id, "permission": permission, "error": str(e)},
            exc_info=True
        )
        return False


async def require_user_permission(user_id: str, permission: Permission) -> None:
    """
    Require user to have specific permission, raise exception if not

    Args:
        user_id: User ID
        permission: Permission required

    Raises:
        AuthorizationError: If user doesn't have permission
    """
    if not await check_user_permission(user_id, permission):
        user_role = await get_user_role(user_id)
        raise AuthorizationError(
            f"User {user_id} with role {user_role.value if user_role else 'unknown'} "
            f"does not have permission: {permission.value}"
        )


async def check_org_permission(
    user_id: str, org_id: str, permission: Permission
) -> bool:
    """
    Check if user has permission for specific organization

    Args:
        user_id: User ID
        org_id: Organization ID
        permission: Permission to check

    Returns:
        True if user has permission for the organization, False otherwise
    """
    try:
        # First check if user belongs to the organization
        user_response = await client.get(
            "/users",
            params={
                "id": f"eq.{user_id}",
                "org_id": f"eq.{org_id}",
                "select": "id,role,org_id",
            },
        )

        if not user_response.json():
            return False  # User doesn't belong to organization

        # Check if user has the permission
        return await check_user_permission(user_id, permission)
    except Exception as e:
        logger.error(
            "Error checking organization permission",
            extra={"user_id": user_id, "org_id": org_id, "permission": permission, "error": str(e)},
            exc_info=True
        )
        return False


async def require_org_permission(
    user_id: str, org_id: str, permission: Permission
) -> None:
    """
    Require user to have permission for specific organization

    Args:
        user_id: User ID
        org_id: Organization ID
        permission: Permission required

    Raises:
        AuthorizationError: If user doesn't have permission
    """
    if not await check_org_permission(user_id, org_id, permission):
        raise AuthorizationError(
            f"User {user_id} does not have permission {permission.value} "
            f"for organization {org_id}"
        )


async def get_user_organizations(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all organizations user has access to

    Args:
        user_id: User ID

    Returns:
        List of organization data
    """
    try:
        # Get user's organization
        user_response = await client.get(
            "/users", params={"id": f"eq.{user_id}", "select": "org_id"}
        )

        if not user_response.json():
            return []

        org_id = user_response.json()[0].get("org_id")
        if not org_id:
            return []

        # Get organization details
        org_response = await client.get("/organizations", params={"id": f"eq.{org_id}"})
        if org_response.json():
            return org_response.json()

        return []
    except Exception as e:
        logger.error(
            "Error fetching user organizations",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        return []


async def update_user_role(user_id: str, new_role: UserRole) -> bool:
    """
    Update user role (admin only operation)

    Args:
        user_id: User ID to update
        new_role: New role to assign

    Returns:
        True if update successful, False otherwise
    """
    try:
        response = await client.patch(
            f"/users?id=eq.{user_id}",
            json={"role": new_role.value, "updated_at": "now()"},
        )

        return response.status_code in [200, 204]
    except Exception as e:
        logger.error(
            "Error updating user role",
            extra={"user_id": user_id, "role": role, "error": str(e)},
            exc_info=True
        )
        return False


def get_role_permissions(role: UserRole) -> List[Permission]:
    """
    Get all permissions for a role

    Args:
        role: User role

    Returns:
        List of permissions
    """
    return ROLE_PERMISSIONS.get(role, [])


def is_admin_role(role: UserRole) -> bool:
    """
    Check if role has admin privileges

    Args:
        role: User role

    Returns:
        True if role has admin privileges
    """
    return role in [UserRole.OWNER, UserRole.ADMIN]


def is_owner_role(role: UserRole) -> bool:
    """
    Check if role is owner

    Args:
        role: User role

    Returns:
        True if role is owner
    """
    return role == UserRole.OWNER
