
import os
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from fastapi import Header, HTTPException
from dotenv import load_dotenv
from services.user_service import sync_user_if_missing

load_dotenv()


async def verify_jwt_token(authorization: str = Header(...)):
    # Check if the authorization header is present
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")

    try:
        scheme, token = authorization.split(" ", 1)
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
# Add this to services/supabase_auth.py for debugging


async def debug_verify_jwt_token(authorization: str = Header(...)):
    """Debug version of JWT verification"""
    print(f"[Debug] Received authorization header: {authorization[:50]}...")

    if not authorization.startswith("Bearer "):
        print("[Debug] Authorization header doesn't start with 'Bearer '")
        raise HTTPException(status_code=401, detail="Invalid token format")

    try:
        scheme, token = authorization.split(" ", 1)
        print(f"[Debug] Scheme: {scheme}")
        print(f"[Debug] Token length: {len(token)}")
        print(f"[Debug] Token start: {token[:50]}...")

        if scheme.lower() != "bearer":
            print("[Debug] Scheme is not 'bearer'")
            raise ValueError("Invalid scheme")

        # Check environment variables
        jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        project_id = os.getenv("SUPABASE_PROJECT_ID")

        print(f"[Debug] JWT Secret exists: {bool(jwt_secret)}")
        print(f"[Debug] Project ID: {project_id}")

        if not jwt_secret:
            print("[Debug] JWT Secret is missing!")
            raise HTTPException(
                status_code=500, detail="JWT Secret not configured")

        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
            issuer=f"https://{project_id}.supabase.co/auth/v1"
        )

        print(f"[Debug] JWT payload: {payload}")

        user_id = payload.get("sub")
        email = payload.get("email")

        print(f"[Debug] User ID: {user_id}")
        print(f"[Debug] Email: {email}")

        if not user_id:
            raise HTTPException(
                status_code=401, detail="Invalid token payload")

        # Sync user if missing
        user_data = await sync_user_if_missing(user_id, email)
        return {"user_id": user_id, "email": email, "user_data": user_data}

    except ExpiredSignatureError as exc:
        print("[Debug] Token has expired")
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except JWTError as e:
        print(f"[Debug] JWT Error: {e}")
        raise HTTPException(
            status_code=401, detail=f"Invalid token: {str(e)}") from e
    except Exception as e:
        print(f"[Debug] Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {str(e)}") from e
