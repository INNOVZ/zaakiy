"""
JWT token validation and management

Handles JWT token verification, validation, and user extraction from Supabase tokens.
"""

import os
from typing import Any, Dict

from fastapi import Header, HTTPException
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

from .exceptions import AuthenticationError, InvalidTokenError, TokenExpiredError
from .user_auth import sync_user_if_missing


class JWTValidator:
    """JWT token validator for Supabase authentication"""

    def __init__(self):
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        self.project_id = os.getenv("SUPABASE_PROJECT_ID")

        if not self.jwt_secret:
            raise ValueError("SUPABASE_JWT_SECRET environment variable is required")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and extract user information

        Args:
            token: JWT token string

        Returns:
            Dict containing user_id and email

        Raises:
            TokenExpiredError: If token has expired
            InvalidTokenError: If token is invalid or malformed
            AuthenticationError: For other authentication errors
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=f"https://{self.project_id}.supabase.co/auth/v1",
            )

            user_id = payload.get("sub")
            email = payload.get("email")

            if not user_id or not email:
                raise InvalidTokenError(
                    "Invalid token payload: missing user_id or email"
                )

            return {"user_id": user_id, "email": email, "payload": payload}

        except ExpiredSignatureError as e:
            raise TokenExpiredError("Token has expired") from e
        except JWTError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}") from e
        except Exception as e:
            raise AuthenticationError(f"Authentication error: {str(e)}") from e


# Global validator instance
_jwt_validator = JWTValidator()


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    JWT token verification function

    Args:
        token: Raw JWT token string

    Returns:
        Dict containing user_id and email

    Raises:
        HTTPException: For authentication failures
    """
    try:
        # Validate token and extract user info
        user_info = _jwt_validator.validate_token(token)

        # Sync user to database if missing
        await sync_user_if_missing(user_info["user_id"], user_info["email"])

        return {"user_id": user_info["user_id"], "email": user_info["email"]}

    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {str(e)}"
        ) from e


async def verify_jwt_token_from_header(
    authorization: str = Header(...),
) -> Dict[str, Any]:
    """
    FastAPI dependency for JWT token verification from Authorization header

    Args:
        authorization: Authorization header containing Bearer token

    Returns:
        Dict containing user_id and email

    Raises:
        HTTPException: For authentication failures
    """
    # Validate authorization header format
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid token format. Expected 'Bearer <token>'"
        )

    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")

        # Use the main verification function
        return await verify_jwt_token(token)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {str(e)}"
        ) from e


# Debug JWT verification function removed for security
# This function exposed sensitive information and is no longer available


def extract_user_from_token(token: str) -> Dict[str, Any]:
    """
    Extract user information from JWT token without database sync

    Args:
        token: JWT token string

    Returns:
        Dict containing user_id and email

    Raises:
        TokenExpiredError: If token has expired
        InvalidTokenError: If token is invalid
    """
    return _jwt_validator.validate_token(token)


def is_token_valid(token: str) -> bool:
    """
    Check if JWT token is valid without raising exceptions

    Args:
        token: JWT token string

    Returns:
        True if token is valid, False otherwise
    """
    try:
        _jwt_validator.validate_token(token)
        return True
    except AuthenticationError:
        return False
