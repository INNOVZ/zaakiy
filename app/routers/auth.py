from ..services.auth import verify_jwt_token, debug_verify_jwt_token, CurrentUser
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/me")
def get_user_info(user=CurrentUser):
    return {"message": "User verified", "user": user}
# Add to routers/auth.py


@router.get("/debug-me")
async def debug_get_user_info(user=Depends(debug_verify_jwt_token)):
    return {"message": "Debug verification successful", "user": user}
# @router.get("/me")
# async def get_me(user=Depends(verify_jwt_token)):
#     user_data = await get_user_with_org(user["user_id"])
#     return user_data
