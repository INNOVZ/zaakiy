from fastapi import APIRouter, Depends
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.supabase_client import client

router = APIRouter()


@router.get("/info")
async def get_org_info(user=Depends(verify_jwt_token)):
    user_data = await get_user_with_org(user["user_id"])
    return {
        "user": {
            "email": user_data["email"]
        },
        "organization": {
            "name": user_data["org"]["name"],
            "plan_id": user_data["org"]["plan_id"]
        }
    }


@router.get("/chatbots")
async def list_chatbots(user=Depends(verify_jwt_token)):
    user_data = await get_user_with_org(user["user_id"])
    org_id = user_data["org_id"]

    response = await client.get(
        "/chatbots",
        params={
            "select": "*",
            "org_id": f"eq.{org_id}"
        }
    )
    return response.json()
