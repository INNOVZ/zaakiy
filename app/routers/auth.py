from fastapi import APIRouter, Depends, HTTPException

from ..services.auth import CurrentUser, verify_jwt_token_from_header

router = APIRouter()


@router.get("/me")
def get_user_info(user=CurrentUser):
    """Get current user information"""
    return {"message": "User verified", "user": user}


# Debug endpoints are completely disabled for security
@router.get("/debug-me")
async def debug_endpoint_disabled():
    """Debug endpoint is disabled for security reasons"""
    raise HTTPException(
        status_code=404, detail="Debug endpoints are disabled for security reasons"
    )
