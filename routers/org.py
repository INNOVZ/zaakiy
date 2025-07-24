from fastapi import APIRouter, Depends
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.supabase_client import client

router = APIRouter()


@router.get("/info")
async def get_org_info(user=Depends(verify_jwt_token)):
    user_data = await get_user_with_org(user["user_id"])

    organization = user_data.get("organization")
    if not organization:
        organization = {"name": "No Organization", "plan_id": None}

    return {
        "user": {
            "email": user_data["email"]
        },
        "organization": {
            "name": organization.get("name", "No Organization"),
            "plan_id": organization.get("plan_id", None)
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
