from fastapi import APIRouter, Depends, HTTPException
from ..models import UpdateOrganizationRequest
from ..services.auth import get_user_with_org, verify_jwt_token_from_header
from ..services.storage.supabase_client import get_supabase_http_client
from ..utils.logging_config import get_logger
logger = get_logger(__name__)

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


@router.get("/info")
async def get_org_info(user=Depends(verify_jwt_token_from_header)):
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
            "plan_id": organization.get("plan_id", None),
        }
    }


@router.patch("/update")
async def update_organization(
    request: UpdateOrganizationRequest, user=Depends(verify_jwt_token_from_header)
):
    """Update organization name and email"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data.get("org_id")

        if not org_id:
            raise HTTPException(
                status_code=400, detail="User not associated with an organization"
            )

        # Update organization in database
        response = await client.patch(
            f"/organizations?id=eq.{org_id}",
            json={
                "name": request.name,
                "email": request.email,
                "contact_phone": request.contact_phone,
                "business_type": request.business_type,
                "updated_at": "now()",
            },
        )

        if getattr(response, "status_code", None) not in [200, 204]:
            raise HTTPException(status_code=404, detail="Organization not found")

        return {
            "success": True,
            "message": "Organization updated successfully",
            "organization": {"name": request.name, "email": request.email},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update organization",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to update organization: {str(e)}"
        ) from e


@router.get("/chatbots")
async def list_chatbots(user=Depends(verify_jwt_token_from_header)):
    """
    Retrieve a list of chatbots associated with the authenticated user's organization.
    """
    user_data = await get_user_with_org(user["user_id"])
    org_id = user_data["org_id"]

    response = await client.get(
        "/chatbots", params={"select": "*", "org_id": f"eq.{org_id}"}
    )
    return response.json()
