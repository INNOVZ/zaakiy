

import os
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from fastapi import Header, HTTPException
from dotenv import load_dotenv
from services.user_service import sync_user_if_missing

load_dotenv()


async def verify_jwt_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")

    token = authorization.split(" ")[1]

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")

        payload = jwt.decode(
            token,
            os.getenv("SUPABASE_JWT_SECRET"),
            algorithms=["HS256"],
            audience="authenticated",  # or None if not set in your project
            issuer=f"https://{os.getenv('SUPABASE_PROJECT_ID')}.supabase.co/auth/v1"
        )
        user_id = payload["sub"]
        email = payload.get("email")

        
        if not user_id or not email:
            raise ValueError("Invalid token payload")

        # Sync user to custom users table if missing
        await sync_user_if_missing(user_id, email)

        return {
            "user_id": user_id,
            "email": email
        }


    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except JWTError as exc:
        print("JWT decode error:", exc)
        raise HTTPException(status_code=401, detail="Invalid token") from exc
