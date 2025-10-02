"""
JWT token validation and management

Handles JWT token verification, validation, and user extraction from Supabase tokens.
"""

import os
from typing import Dict, Any, Optional
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from fastapi import Header, HTTPException
from dotenv import load_dotenv

from .user_auth import sync_user_if_missing
from .exceptions import (
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError
)

load_dotenv()


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
                issuer=f"https://{self.project_id}.supabase.co/auth/v1"
            )
            
            user_id = payload.get("sub")
            email = payload.get("email")
            
            if not user_id or not email:
                raise InvalidTokenError("Invalid token payload: missing user_id or email")
            
            return {
                "user_id": user_id,
                "email": email,
                "payload": payload
            }
            
        except ExpiredSignatureError as e:
            raise TokenExpiredError("Token has expired") from e
        except JWTError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}") from e
        except Exception as e:
            raise AuthenticationError(f"Authentication error: {str(e)}") from e


# Global validator instance
_jwt_validator = JWTValidator()


async def verify_jwt_token(authorization: str = Header(...)) -> Dict[str, Any]:
    """
    FastAPI dependency for JWT token verification
    
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
            status_code=401, 
            detail="Invalid token format. Expected 'Bearer <token>'"
        )
    
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401, 
                detail="Invalid authorization scheme"
            )
        
        # Validate token and extract user info
        user_info = _jwt_validator.validate_token(token)
        
        # Sync user to database if missing
        await sync_user_if_missing(
            user_info["user_id"], 
            user_info["email"]
        )
        
        return {
            "user_id": user_info["user_id"],
            "email": user_info["email"]
        }
        
    except (TokenExpiredError, InvalidTokenError, AuthenticationError) as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Authentication error: {str(e)}"
        ) from e


async def debug_verify_jwt_token(authorization: str = Header(...)) -> Dict[str, Any]:
    """
    Debug version of JWT verification with detailed logging
    
    Args:
        authorization: Authorization header containing Bearer token
        
    Returns:
        Dict containing user_id, email, and debug information
        
    Raises:
        HTTPException: For authentication failures
    """
    print(f"[Debug] Received authorization header: {authorization[:50]}...")
    
    if not authorization.startswith("Bearer "):
        print("[Debug] Authorization header doesn't start with 'Bearer '")
        raise HTTPException(
            status_code=401, 
            detail="Invalid token format. Expected 'Bearer <token>'"
        )
    
    try:
        scheme, token = authorization.split(" ", 1)
        print(f"[Debug] Scheme: {scheme}")
        print(f"[Debug] Token length: {len(token)}")
        print(f"[Debug] Token start: {token[:50]}...")
        
        if scheme.lower() != "bearer":
            print("[Debug] Scheme is not 'bearer'")
            raise HTTPException(
                status_code=401, 
                detail="Invalid authorization scheme"
            )
        
        # Check environment variables
        jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        project_id = os.getenv("SUPABASE_PROJECT_ID")
        
        print(f"[Debug] JWT Secret exists: {bool(jwt_secret)}")
        print(f"[Debug] Project ID: {project_id}")
        
        if not jwt_secret:
            print("[Debug] JWT Secret is missing!")
            raise HTTPException(
                status_code=500, 
                detail="JWT Secret not configured"
            )
        
        # Validate token
        user_info = _jwt_validator.validate_token(token)
        print(f"[Debug] JWT payload: {user_info.get('payload', {})}")
        print(f"[Debug] User ID: {user_info['user_id']}")
        print(f"[Debug] Email: {user_info['email']}")
        
        # Sync user if missing
        user_data = await sync_user_if_missing(
            user_info["user_id"], 
            user_info["email"]
        )
        
        return {
            "user_id": user_info["user_id"],
            "email": user_info["email"],
            "user_data": user_data
        }
        
    except (TokenExpiredError, InvalidTokenError, AuthenticationError) as e:
        print(f"[Debug] Auth Error: {e}")
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as e:
        print(f"[Debug] Unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Authentication error: {str(e)}"
        ) from e


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
    except (TokenExpiredError, InvalidTokenError, AuthenticationError):
        return False
