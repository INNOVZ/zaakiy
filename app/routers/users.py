from fastapi import APIRouter, Depends, HTTPException

from ..models import UpdateUserRequest
from ..services.auth import CurrentUser, verify_jwt_token_from_header
from ..services.storage.supabase_client import get_supabase_http_client
from ..utils.logging_config import get_logger
logger = get_logger(__name__)
# Constants
USERS_ENDPOINT = "/users"

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

router = APIRouter()


@router.post("/init")
async def init_user(user=Depends(verify_jwt_token_from_header)):
    # Check if already exists
    existing = await client.get(USERS_ENDPOINT, params={"id": f"eq.{user['user_id']}"})
    if existing.json():
        return {"message": "User already exists"}

    # You can default to assigning org manually or use a placeholder
    new_user = {
        "id": user["user_id"],
        "email": user["email"],
        "org_id": "org-placeholder-id",  # update this logic as needed
        "role": "admin",  # or "user", "owner"
    }

    await client.post(USERS_ENDPOINT, json=new_user)

    return {"message": "User initialized"}


@router.post("/update")
async def update_user(
    request: UpdateUserRequest, user=Depends(verify_jwt_token_from_header)
):
    """Update user's full name"""
    try:
        user_id = user["user_id"]
        response = await client.patch(
            f"{USERS_ENDPOINT}?id=eq.{user_id}",
            json={"full_name": request.full_name, "updated_at": "now()"},
        )
        if response.status_code not in [200, 204]:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "success": True,
            "message": "User updated successfully",
            "user": {"full_name": request.full_name},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update user profile",
            extra={"error": str(e), "user_id": user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to update user: {str(e)}"
        ) from e


@router.get("/profile")
async def get_user_profile(user=CurrentUser):
    """Get user profile information"""
    try:
        user_id = user["user_id"]

        response = await client.get(
            USERS_ENDPOINT,
            params={
                "id": f"eq.{user_id}",
                "select": "id,email,full_name,created_at,updated_at",
            },
        )

        if response.status_code != 200 or not response.json():
            raise HTTPException(status_code=404, detail="User not found")

        user_data = response.json()[0]

        return {"success": True, "user": user_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get user profile",
            extra={"error": str(e), "user_id": user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get user profile: {str(e)}"
        ) from e
