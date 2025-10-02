"""
User authentication and management services

Handles user creation, synchronization, and organization management.
"""

from typing import Dict, Any, Optional
from ..storage.supabase_client import client


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
            "/organizations", 
            params={"id": f"eq.{user['org_id']}"}
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
    # Check if user already exists
    check = await client.get("/users", params={"id": f"eq.{user_id}"})
    
    if check.status_code == 200 and check.json():
        return check.json()[0]  # User already exists
    
    # Create a new organization for the user if not found
    org_id = await create_organization_for_user(user_id, email)
    
    user_data = {
        "id": user_id,
        "email": email,
        "full_name": "user",
        "org_id": org_id
    }
    
    print(f"Creating user with data: {user_data}")  # Debug log
    response = await client.post("/users", json=user_data)
    
    print(f"User creation status: {response.status_code}")  # Debug log
    print(f"User creation response: {response.text}")  # Debug log
    
    if response.status_code >= 400:
        print("Failed to insert user:", response.status_code, response.text)
        raise RuntimeError(
            f"Failed to create user: {response.status_code} - {response.text}"
        )
    
    # Handle empty response body for successful creation
    if response.text.strip():
        return response.json()
    else:
        # Return the user data we just created since Supabase didn't return it
        return user_data


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
    org_data = {
        "name": f"Organization for {user_id}",
        "email": email
    }
    
    org_response = await client.post("/organizations", json=org_data)
    
    print("Org insert status:", org_response.status_code)
    print("Org insert response:", org_response.text)
    
    if org_response.status_code == 201:
        org_json = org_response.json()
        return org_json["id"]
    elif org_response.status_code == 409:
        # Organization with this email already exists, fetch it
        existing_org_resp = await client.get(
            "/organizations", 
            params={"email": f"eq.{email}"}
        )
        
        if existing_org_resp.status_code == 200 and existing_org_resp.json():
            return existing_org_resp.json()[0]["id"]
        else:
            print(
                "Failed to fetch existing organization:",
                existing_org_resp.status_code, 
                existing_org_resp.text
            )
            raise RuntimeError("Failed to fetch existing organization")
    else:
        print(
            "Failed to create organization:",
            org_response.status_code, 
            org_response.text
        )
        raise RuntimeError("Failed to create a new organization")


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
        print(f"Error fetching user {user_id}: {e}")
        return None


async def update_user_profile(
    user_id: str, 
    updates: Dict[str, Any]
) -> Dict[str, Any]:
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
    
    response = await client.patch(
        f"/users?id=eq.{user_id}",
        json=updates
    )
    
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
        print(f"Error deleting user {user_id}: {e}")
        return False
