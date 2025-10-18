"""
User authentication and management services

Handles user creation, synchronization, and organization management.
"""


import httpx
import logging
from typing import Any, Dict, Optional

from ..storage.supabase_client import get_supabase_http_client

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


async def get_user_with_org(user_id: str) -> Dict[str, Any]:
    """
    Get user with their associated organization details

    Args:
        user_id: User ID to fetch

    Returns:
        Dict containing user data with organization information

    Raises:
        ValueError: If user is not found
    """
    # First get the user
    user_response = await client.get("/users", params={"id": f"eq.{user_id}"})
    user_data = user_response.json()

    if not user_data:
        raise ValueError("User not found")

    user = user_data[0]

    # Then get the organization if org_id exists
    if user.get("org_id"):
        org_response = await client.get(
            "/organizations", params={"id": f"eq.{user['org_id']}"}
        )
        org_data = org_response.json()

        if org_data:
            user["organization"] = org_data[0]
        else:
            user["organization"] = None
    else:
        user["organization"] = None

    return user


async def sync_user_if_missing(user_id: str, email: str) -> Dict[str, Any]:
    """
    Create a user in the users table if they don't exist

    Args:
        user_id: Supabase user ID
        email: User email address

    Returns:
        Dict containing user data

    Raises:
        RuntimeError: If user creation fails
    """

    logger = logging.getLogger(__name__)

    try:
        logger.info("Syncing user %s with email %s", user_id, email)

        # Check if user already exists
        check = await client.get("/users", params={"id": f"eq.{user_id}"})
        logger.info("User check response: %s", check.status_code)

        if check.status_code == 200 and check.json():
            logger.info("User %s already exists, returning existing user", user_id)
            return check.json()[0]  # User already exists

        logger.info("User %s not found, creating new user and organization", user_id)

        # Create a new organization for the user if not found
        try:
            org_id = await create_organization_for_user(user_id, email)
            logger.info("Created organization %s for user %s", org_id, user_id)
        except Exception as e:
            logger.error(
                "Failed to create organization for user %s: %s", user_id, str(e)
            )
            # Create user without organization - they can be assigned later
            org_id = None

        user_data = {
            "id": user_id,
            "email": email,
            "full_name": "user",
            "org_id": org_id,
        }

        response = await client.post("/users", json=user_data)
        logger.info("User creation response: %s", response.status_code)

        if response.status_code >= 400:
            logger.error(
                "Failed to insert user: %s - %s", response.status_code, response.text
            )
            raise RuntimeError(
                f"Failed to create user: {response.status_code} - {response.text}"
            )

        # Handle empty response body for successful creation
        if response.text.strip():
            logger.info(
                "Successfully created user %s with organization %s", user_id, org_id
            )
            return response.json()
        else:
            # Return the user data we just created since Supabase didn't return it
            logger.info(
                "Successfully created user %s with organization %s (empty response)",
                user_id,
                org_id,
            )
            return user_data

    except Exception as e:
        logger.error("Exception in sync_user_if_missing: %s", str(e))
        raise RuntimeError(f"Failed to sync user {user_id}: {str(e)}") from e


async def create_organization_for_user(user_id: str, email: str) -> str:
    """
    Create a new organization for the user

    Args:
        user_id: User ID
        email: User email address

    Returns:
        Organization ID

    Raises:
        RuntimeError: If organization creation fails
    """

    from ...config.settings import get_database_config

    logger = logging.getLogger(__name__)

    try:
        org_data = {"name": f"Organization for {user_id}", "email": email}
        logger.info("Creating organization for user %s with email %s", user_id, email)

        # Try using the client first
        try:
            org_response = await client.post("/organizations", json=org_data)
            logger.info("Organization creation response: %s", org_response.status_code)
            logger.info("Organization creation response text: %s", org_response.text)
            logger.info(
                "Organization creation response headers: %s", org_response.headers
            )

            # Organization creation response handled below
            if org_response.status_code == 201:
                org_json = org_response.json()
                org_id = org_json["id"]
                logger.info(
                    "Successfully created organization %s for user %s", org_id, user_id
                )
                return org_id
            elif org_response.status_code == 409:
                # Organization with this email already exists, fetch it
                logger.info(
                    "Organization with email %s already exists, fetching existing one",
                    email,
                )
                existing_org_resp = await client.get(
                    "/organizations", params={"email": f"eq.{email}"}
                )

                if existing_org_resp.status_code == 200 and existing_org_resp.json():
                    org_id = existing_org_resp.json()[0]["id"]
                    logger.info(
                        "Found existing organization %s for email %s", org_id, email
                    )
                    return org_id
                else:
                    logger.error(
                        "Failed to fetch existing organization: %s - %s",
                        existing_org_resp.status_code,
                        existing_org_resp.text,
                    )
                    raise RuntimeError("Failed to fetch existing organization")
            else:
                logger.error(
                    "Failed to create organization: %s - %s",
                    org_response.status_code,
                    org_response.text,
                )
                raise RuntimeError(
                    f"Failed to create organization: {org_response.status_code} - {org_response.text}"
                )

        except Exception as client_error:
            logger.warning(
                "Client method failed: %s, trying direct HTTP request",
                str(client_error),
            )

            # Fallback: Use direct HTTP request
            db_config = get_database_config()
            headers = {
                "apikey": db_config.supabase_service_key,
                "Authorization": f"Bearer {db_config.supabase_service_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    f"{db_config.supabase_url}/rest/v1/organizations",
                    json=org_data,
                    headers=headers,
                    timeout=30.0,
                )

                logger.info(
                    "Direct HTTP organization creation response: %s",
                    response.status_code,
                )
                logger.info("Direct HTTP response text: %s", response.text)

                if response.status_code == 201:
                    org_json = response.json()
                    org_id = org_json["id"]
                    logger.info(
                        "Successfully created organization %s via direct HTTP for user %s",
                        org_id,
                        user_id,
                    )
                    return org_id
                elif response.status_code == 409:
                    # Organization with this email already exists, fetch it
                    logger.info(
                        "Organization with email %s already exists, fetching existing one via direct HTTP",
                        email,
                    )
                    existing_resp = await http_client.get(
                        f"{db_config.supabase_url}/rest/v1/organizations",
                        params={"email": f"eq.{email}"},
                        headers=headers,
                        timeout=30.0,
                    )

                    if existing_resp.status_code == 200 and existing_resp.json():
                        org_id = existing_resp.json()[0]["id"]
                        logger.info(
                            "Found existing organization %s for email %s via direct HTTP",
                            org_id,
                            email,
                        )
                        return org_id
                    else:
                        logger.error(
                            "Failed to fetch existing organization via direct HTTP: %s - %s",
                            existing_resp.status_code,
                            existing_resp.text,
                        )
                        raise RuntimeError("Failed to fetch existing organization")
                else:
                    logger.error(
                        "Failed to create organization via direct HTTP: %s - %s",
                        response.status_code,
                        response.text,
                    )
                    raise RuntimeError(
                        f"Failed to create organization: {response.status_code} - {response.text}"
                    )

    except Exception as e:
        logger.error("Exception in create_organization_for_user: %s", str(e))
        raise RuntimeError(
            f"Failed to create organization for user {user_id}: {str(e)}"
        )


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user by ID without organization data

    Args:
        user_id: User ID to fetch

    Returns:
        User data dict or None if not found
    """
    try:
        response = await client.get("/users", params={"id": f"eq.{user_id}"})

        if response.status_code == 200 and response.json():
            return response.json()[0]

        return None
    except Exception as e:
        logger.error(
            "Error fetching user",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        return None


async def update_user_profile(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update user profile information

    Args:
        user_id: User ID to update
        updates: Dict of fields to update

    Returns:
        Updated user data

    Raises:
        RuntimeError: If update fails
    """
    # Add updated_at timestamp
    updates["updated_at"] = "now()"

    response = await client.patch(f"/users?id=eq.{user_id}", json=updates)

    if response.status_code not in [200, 204]:
        raise RuntimeError(
            f"Failed to update user: {response.status_code} - {response.text}"
        )

    # Return updated user data
    return await get_user_by_id(user_id)


async def delete_user(user_id: str) -> bool:
    """
    Delete user account

    Args:
        user_id: User ID to delete

    Returns:
        True if deletion successful, False otherwise
    """
    try:
        response = await client.delete(f"/users?id=eq.{user_id}")
        return response.status_code in [200, 204]
    except Exception as e:
        logger.error(
            "Error deleting user",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        return False
