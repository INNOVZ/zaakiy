"""
JWT token validation and management

Handles JWT token verification, validation, and user extraction from Supabase tokens.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

from fastapi import Header, HTTPException
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

from ...utils.env_loader import is_test_environment
from .exceptions import AuthenticationError, InvalidTokenError, TokenExpiredError
from .user_auth import sync_user_if_missing

logger = logging.getLogger(__name__)


class JWTValidator:
    """JWT token validator for Supabase authentication"""

    def __init__(self):
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        self.project_id = os.getenv("SUPABASE_PROJECT_ID")

        if not self.jwt_secret:
            raise ValueError("SUPABASE_JWT_SECRET environment variable is required")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token and extract user information."""

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


class DisabledJWTValidator:
    """Fallback validator used when JWT configuration is unavailable."""

    def __init__(self, reason: str):
        self.reason = reason

    def validate_token(
        self, token: str
    ) -> Dict[str, Any]:  # pragma: no cover - simple guard
        raise AuthenticationError(f"JWT validation is disabled: {self.reason}")


ValidatorType = Union[JWTValidator, DisabledJWTValidator]

# Lazy-loaded validator state
_validator: Optional[ValidatorType] = None
_validator_error: Optional[str] = None


def _initialize_validator() -> None:
    """Instantiate the JWT validator, falling back to a disabled stub in tests."""
    global _validator, _validator_error

    if _validator is not None:
        return

    try:
        _validator = JWTValidator()
        _validator_error = None
        logger.info("JWT validator initialized successfully")
    except ValueError as exc:
        error_message = str(exc)
        _validator_error = error_message
        if is_test_environment():
            logger.warning("JWT validator disabled for tests: %s", error_message)
            _validator = DisabledJWTValidator(error_message)
        else:
            logger.error("JWT validator initialization failed: %s", error_message)
            raise


def _get_validator() -> ValidatorType:
    """Return the active validator instance, raising HTTP 503 if unavailable."""
    global _validator

    # Allow re-initialization if we previously had a disabled stub and config is now present
    if isinstance(_validator, DisabledJWTValidator):
        _validator = None

    try:
        _initialize_validator()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable: JWT configuration is missing.",
        ) from exc

    if _validator is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable: JWT validator not initialized.",
        )

    return _validator


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Validate a JWT token and ensure the corresponding user exists."""

    try:
        validator = _get_validator()
        user_info = validator.validate_token(token)
        await sync_user_if_missing(user_info["user_id"], user_info["email"])

        return {"user_id": user_info["user_id"], "email": user_info["email"]}

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {str(e)}"
        ) from e


async def verify_jwt_token_from_header(
    authorization: str = Header(...),
) -> Dict[str, Any]:
    """FastAPI dependency for JWT token verification from the Authorization header."""

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
    """Extract user information from a JWT token without database sync."""

    validator = _get_validator()
    return validator.validate_token(token)


def is_token_valid(token: str) -> bool:
    """
    Check if JWT token is valid without raising exceptions

    Args:
        token: JWT token string

    Returns:
        True if token is valid, False otherwise
    """
    try:
        validator = _get_validator()
        validator.validate_token(token)
        return True
    except (AuthenticationError, HTTPException):
        return False


def get_jwt_validator_status() -> Dict[str, Any]:
    """Report readiness information for operational diagnostics."""
    try:
        _initialize_validator()
    except ValueError as exc:
        return {
            "ready": False,
            "configured": False,
            "error": str(exc),
        }

    if isinstance(_validator, DisabledJWTValidator):
        return {
            "ready": False,
            "configured": False,
            "error": _validator_error,
        }

    return {"ready": True, "configured": True, "error": None}
