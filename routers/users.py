from fastapi import APIRouter, Depends
from services.supabase_auth import verify_jwt_token
from services.supabase_client import client

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
