from fastapi import APIRouter, Depends
from services.supabase_auth import verify_jwt_token
# from services.user_service import get_user_with_org

router = APIRouter()


@router.get("/me")
def get_user_info(user=Depends(verify_jwt_token)):
    return {"message": "User verified", "user": user}


# @router.get("/me")
# async def get_me(user=Depends(verify_jwt_token)):
#     user_data = await get_user_with_org(user["user_id"])
#     return user_data
