from fastapi import APIRouter, Depends, HTTPException
from ..models import UpdateOrganizationRequest
from ..services.auth import verify_jwt_token, get_user_with_org
from ..services.storage.supabase_client import client

router = APIRouter()


@router.get("/info")
async def get_org_info(user=Depends(verify_jwt_token)):
    """
    Retrieve organization and user information for the authenticated user.
    """
    user_data = await get_user_with_org(user["user_id"])

    organization = user_data.get("organization")
    if not organization:
        organization = {"name": "No Organization", "plan_id": None}

    return {
        "organization": {
            "email": organization.get("email", "No Email"),
            "business_type": organization.get("business_type", "N/A"),
            "contact_phone": organization.get("contact_phone", "N/A"),
            "name": organization.get("name", "No Organization"),
            "plan_id": organization.get("plan_id", None)
        }
    }


@router.patch("/update")
async def update_organization(
    request: UpdateOrganizationRequest,
    user=Depends(verify_jwt_token)
):
    """Update organization name and email"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data.get("org_id")

        if not org_id:
            raise HTTPException(
                status_code=400, detail="User not associated with an organization")

        # Update organization in database
        response = await client.patch(
            f"/organizations?id=eq.{org_id}",
            json={
                "name": request.name,
                "email": request.email,
                "contact_phone": request.contact_phone,
                "business_type": request.business_type,
                "updated_at": "now()"
            }
        )

        if getattr(response, "status_code", None) not in [200, 204]:
            raise HTTPException(
                status_code=404, detail="Organization not found")

        return {
            "success": True,
            "message": "Organization updated successfully",
            "organization": {
                "name": request.name,
                "email": request.email
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Update organization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update organization: {str(e)}"
        ) from e


@router.get("/chatbots")
async def list_chatbots(user=Depends(verify_jwt_token)):
    """
    Retrieve a list of chatbots associated with the authenticated user's organization.
    """
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
