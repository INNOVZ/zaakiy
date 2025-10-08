from fastapi import APIRouter, Depends, HTTPException
from ..models import UpdateUserRequest
from ..services.auth import verify_jwt_token, CurrentUser
from ..services.storage.supabase_client import client

router = APIRouter()


@router.post("/init")
async def init_user(user=Depends(verify_jwt_token)):
    # Check if already exists
    existing = await client.get("/users", params={"id": f"eq.{user['user_id']}"})
    if existing.json():
        return {"message": "User already exists"}

    # You can default to assigning org manually or use a placeholder
    new_user = {
        "id": user["user_id"],
        "email": user["email"],
        "org_id": "org-placeholder-id",  # update this logic as needed
        "role": "admin"  # or "user", "owner"
    }

    await client.post("/users", json=new_user)

    return {"message": "User initialized"}


@router.post("/update")
async def update_user(
    request: UpdateUserRequest,
    user=Depends(verify_jwt_token)
):
    """Update user's full name"""
    try:
        user_id = user["user_id"]
        response = await client.patch(
            f"/users?id=eq.{user_id}",
            json={
                "full_name": request.full_name,
                "updated_at": "now()"
            }
        )
        if response.status_code not in [200, 204]:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "success": True,
            "message": "User updated successfully",
            "user": {
                "full_name": request.full_name
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Update user failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user: {str(e)}"
        ) from e


@router.get("/profile")
async def get_user_profile(user=CurrentUser):
    """Get user profile information"""
    try:
        user_id = user["user_id"]

        response = await client.get(
            "/users",
            params={
                "id": f"eq.{user_id}",
                "select": "id,email,full_name,created_at,updated_at"
            }
        )

        if response.status_code != 200 or not response.json():
            raise HTTPException(status_code=404, detail="User not found")

        user_data = response.json()[0]

        return {
            "success": True,
            "user": user_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Get user profile failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user profile: {str(e)}"
        ) from e
