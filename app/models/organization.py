"""Organization and user management models."""

from typing import Optional
from pydantic import BaseModel, field_validator


class UpdateOrganizationRequest(BaseModel):
    """Request model for updating organization details."""

    name: str
    email: str
    contact_phone: Optional[str] = None
    business_type: Optional[str] = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate organization name"""
        if not v or not isinstance(v, str):
            raise ValueError("Organization name must be provided")

        v = v.strip()

        if len(v) < 2:
            raise ValueError("Organization name must be at least 2 characters")

        if len(v) > 200:
            raise ValueError("Organization name too long (max 200 characters)")

        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Validate email format"""
        if not v or not isinstance(v, str):
            raise ValueError("Email must be provided")

        v = v.strip().lower()

        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError("Invalid email format")

        if len(v) > 255:
            raise ValueError("Email too long (max 255 characters)")

        return v

    @field_validator('contact_phone')
    @classmethod
    def validate_phone(cls, v):
        """Validate contact phone"""
        if v is not None:
            v = v.strip()
            if len(v) > 20:
                raise ValueError("Phone number too long (max 20 characters)")
        return v

    @field_validator('business_type')
    @classmethod
    def validate_business_type(cls, v):
        """Validate business type"""
        if v is not None:
            v = v.strip()
            if len(v) > 100:
                raise ValueError("Business type too long (max 100 characters)")
        return v


class UpdateUserRequest(BaseModel):
    """Request model for updating user details."""

    full_name: str

    @field_validator('full_name')
    @classmethod
    def validate_full_name(cls, v):
        """Validate full name"""
        if not v or not isinstance(v, str):
            raise ValueError("Full name must be provided")

        v = v.strip()

        if len(v) < 2:
            raise ValueError("Full name must be at least 2 characters")

        if len(v) > 100:
            raise ValueError("Full name too long (max 100 characters)")

        return v
