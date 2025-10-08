from fastapi import APIRouter, Depends, HTTPException
from ..services.auth import verify_jwt_token, debug_verify_jwt_token, CurrentUser
from ..config.settings import is_debug_mode

router = APIRouter()


@router.get("/me")
def get_user_info(user=Depends(CurrentUser)):
    """Get current user information"""
    return {"message": "User verified", "user": user}


# Debug endpoint - only available in debug mode
if is_debug_mode():
    @router.get("/debug-me")
    async def debug_get_user_info(user=Depends(debug_verify_jwt_token)):
        """
        Debug endpoint for JWT verification testing

        ⚠️ WARNING: This endpoint is only available in debug mode.
        It exposes detailed user information and should NEVER be enabled in production.
        """
        return {
            "message": "Debug verification successful",
            "user": user,
            "warning": "This is a debug endpoint - not for production use"
        }
else:
    # In production, return 404 for debug endpoints
    @router.get("/debug-me")
    async def debug_endpoint_disabled():
        """Debug endpoint is disabled in production"""
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in production mode"
        )
