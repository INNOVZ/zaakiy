"""WhatsApp request and response models."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WhatsAppWebhookRequest(BaseModel):
    """Model for incoming Twilio WhatsApp webhook."""

    MessageSid: str
    AccountSid: str
    From: str  # Phone number with whatsapp: prefix
    To: str  # Your Twilio phone number
    Body: Optional[str] = None
    NumMedia: Optional[str] = "0"
    MediaUrl0: Optional[str] = None
    MediaUrl1: Optional[str] = None
    MediaUrl2: Optional[str] = None
    MediaUrl3: Optional[str] = None
    DateSent: Optional[str] = None


class WhatsAppConfigRequest(BaseModel):
    """Request model for configuring WhatsApp."""

    provider_type: str = "platform"  # 'platform' or 'tenant_managed'
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Sensitive: Twilio auth token (never expose in logs or responses)",
    )
    twilio_phone_number: str
    is_active: bool = True

    @field_validator("provider_type")
    @classmethod
    def validate_provider_type(cls, v):
        """Validate provider type"""
        if v not in ["platform", "tenant_managed"]:
            raise ValueError("provider_type must be 'platform' or 'tenant_managed'")
        return v

    @field_validator("twilio_phone_number")
    @classmethod
    def validate_phone_number(cls, v):
        """Validate phone number format"""
        if not v:
            raise ValueError("twilio_phone_number is required")
        # Remove whatsapp: prefix if present
        v = v.replace("whatsapp:", "")
        # Basic E.164 validation
        if not v.startswith("+"):
            raise ValueError("Phone number must be in E.164 format (e.g., +1234567890)")
        return v

    @field_validator("twilio_account_sid")
    @classmethod
    def validate_account_sid(cls, v):
        """Validate account SID if provided"""
        if v is not None and len(v) == 0:
            raise ValueError("twilio_account_sid cannot be empty if provided")
        return v

    @field_validator("twilio_auth_token")
    @classmethod
    def validate_auth_token(cls, v):
        """Validate auth token if provided (sensitive field)"""
        if v is not None and len(v) == 0:
            raise ValueError("twilio_auth_token cannot be empty if provided")
        return v

    @model_validator(mode="after")
    def validate_tenant_managed_credentials(self):
        """
        Cross-field validation: If provider_type is 'tenant_managed',
        both twilio_account_sid and twilio_auth_token must be provided.
        """
        if self.provider_type == "tenant_managed":
            if not self.twilio_account_sid:
                raise ValueError(
                    "twilio_account_sid is required when provider_type is 'tenant_managed'"
                )
            if not self.twilio_auth_token:
                raise ValueError(
                    "twilio_auth_token is required when provider_type is 'tenant_managed'"
                )
        elif self.provider_type == "platform":
            # For platform-managed, credentials should not be provided
            if self.twilio_account_sid or self.twilio_auth_token:
                raise ValueError(
                    "twilio_account_sid and twilio_auth_token should not be provided "
                    "when provider_type is 'platform'"
                )
        return self


class WhatsAppConfigResponse(BaseModel):
    """Response model for WhatsApp configuration."""

    id: str
    org_id: str
    provider_type: str
    twilio_phone_number: str
    is_active: bool
    webhook_url: Optional[str] = None
    created_at: str
    updated_at: str


class WhatsAppStatusResponse(BaseModel):
    """Response model for WhatsApp status."""

    is_configured: bool
    is_active: bool
    provider_type: Optional[str] = None
    phone_number: Optional[str] = None
    message: Optional[str] = None
