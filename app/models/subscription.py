"""Subscription and billing models for user/organization onboarding."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator


class SubscriptionPlan(str, Enum):
    """Available subscription plans."""

    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription status options."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class Channel(str, Enum):
    """Supported communication channels."""

    WEBSITE = "website"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    INSTAGRAM = "instagram"
    API = "api"
    MOBILE_APP = "mobile_app"


class PlanFeatures(BaseModel):
    """Features and limits for each subscription plan."""

    name: str
    monthly_token_limit: int
    price_per_month: float
    max_chatbots: int
    max_documents_per_chatbot: int
    priority_support: bool
    custom_branding: bool
    api_access: bool
    analytics_retention_days: int


class OnboardingRequest(BaseModel):
    """Request model for user/organization onboarding."""

    # Entity type
    entity_type: str  # "user" or "organization"

    # User fields (required for both user and organization)
    full_name: str
    email: str

    # Organization fields (required only for organization)
    organization_name: Optional[str] = None
    contact_phone: Optional[str] = None
    business_type: Optional[str] = None

    # Subscription
    selected_plan: SubscriptionPlan

    # Required password for Supabase auth user creation
    password: str

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v):
        """Validate entity type"""
        if v not in ["user", "organization"]:
            raise ValueError("Entity type must be 'user' or 'organization'")
        return v

    @field_validator("full_name")
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

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Validate email format"""
        if not v or not isinstance(v, str):
            raise ValueError("Email must be provided")

        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        if len(v) > 255:
            raise ValueError("Email too long (max 255 characters)")

        return v

    @field_validator("organization_name")
    @classmethod
    def validate_organization_name(cls, v, info):
        """Validate organization name when entity_type is organization"""
        entity_type = info.data.get("entity_type")

        if entity_type == "organization":
            if not v or not isinstance(v, str):
                raise ValueError(
                    "Organization name is required for organization signup"
                )

            v = v.strip()
            if len(v) < 2:
                raise ValueError("Organization name must be at least 2 characters")
            if len(v) > 200:
                raise ValueError("Organization name too long (max 200 characters)")

        return v

    @field_validator("contact_phone")
    @classmethod
    def validate_phone(cls, v):
        """Validate contact phone"""
        if v is not None:
            v = v.strip()
            if len(v) > 20:
                raise ValueError("Phone number too long (max 20 characters)")
        return v

    @field_validator("business_type")
    @classmethod
    def validate_business_type(cls, v):
        """Validate business type"""
        if v is not None:
            v = v.strip()
            if len(v) > 100:
                raise ValueError("Business type too long (max 100 characters)")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        """Validate password strength"""
        if not v or not isinstance(v, str):
            raise ValueError("Password must be provided")

        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        if len(v) > 128:
            raise ValueError("Password too long (max 128 characters)")

        # Check for at least one uppercase, one lowercase, and one number
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            raise ValueError(
                "Password must contain at least one uppercase letter, one lowercase letter, and one number"
            )

        return v


class OnboardingResponse(BaseModel):
    """Response model for successful onboarding."""

    success: bool
    message: str
    entity_id: str
    entity_type: str
    subscription_id: str
    plan: SubscriptionPlan
    tokens_remaining: int
    tokens_limit: int
    email_confirmation_required: bool = True
    email_sent_to: Optional[str] = None


class SubscriptionUsage(BaseModel):
    """Model for tracking subscription usage."""

    subscription_id: str
    tokens_used_this_month: int
    tokens_remaining: int
    monthly_limit: int
    usage_percentage: float
    reset_date: datetime


class TokenUsageRequest(BaseModel):
    """Request model for token usage tracking."""

    entity_id: str
    entity_type: str
    tokens_consumed: int
    operation_type: str  # "chat", "document_processing", etc.
    channel: Optional[Channel] = None  # Channel where the operation occurred
    chatbot_id: Optional[str] = None  # Specific chatbot used
    session_id: Optional[str] = None  # Session identifier for tracking
    # End-user identifier (for analytics)
    user_identifier: Optional[str] = None

    @field_validator("tokens_consumed")
    @classmethod
    def validate_tokens_consumed(cls, v):
        """Validate tokens consumed"""
        if v < 0:
            raise ValueError("Tokens consumed cannot be negative")
        if v > 10000:  # Reasonable upper limit per operation
            raise ValueError("Tokens consumed seems too high for single operation")
        return v

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v):
        """Validate entity type"""
        if v not in ["user", "organization"]:
            raise ValueError("Entity type must be 'user' or 'organization'")
        return v


class SubscriptionUpdateRequest(BaseModel):
    """Request model for updating subscription plan."""

    entity_id: str
    entity_type: str
    new_plan: SubscriptionPlan

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v):
        """Validate entity type"""
        if v not in ["user", "organization"]:
            raise ValueError("Entity type must be 'user' or 'organization'")
        return v


class ChannelUsageStats(BaseModel):
    """Channel-specific usage statistics."""

    channel: Channel
    tokens_used: int
    message_count: int
    unique_users: int
    avg_tokens_per_message: float
    peak_usage_hour: Optional[int] = None


class SubscriptionAnalytics(BaseModel):
    """Comprehensive subscription analytics."""

    subscription_id: str
    entity_id: str
    entity_type: str
    plan: SubscriptionPlan

    # Overall usage
    total_tokens_used: int
    total_tokens_limit: int
    usage_percentage: float

    # Channel breakdown
    channel_usage: List[ChannelUsageStats]

    # Time-based analytics
    daily_usage: Dict[str, int]  # date -> tokens
    hourly_distribution: Dict[int, int]  # hour -> tokens

    # Performance metrics
    most_active_channel: Channel
    least_active_channel: Channel
    growth_rate: float  # percentage change from previous period

    # Billing info
    billing_cycle_start: datetime
    billing_cycle_end: datetime
    days_remaining: int


class ChannelLimits(BaseModel):
    """Channel-specific limits and configurations."""

    channel: Channel
    enabled: bool = True
    rate_limit_per_minute: int = 60
    max_message_length: int = 4000
    custom_token_multiplier: float = 1.0  # Adjust token cost per channel
    priority_level: int = 1  # 1=highest, 5=lowest


class EnhancedPlanFeatures(BaseModel):
    """Enhanced plan features with channel support."""

    name: str
    monthly_token_limit: int
    price_per_month: float
    max_chatbots: int
    max_documents_per_chatbot: int
    priority_support: bool
    custom_branding: bool
    api_access: bool
    analytics_retention_days: int

    # Channel-specific features
    supported_channels: List[Channel]
    channel_limits: Dict[Channel, ChannelLimits]
    concurrent_conversations: int
    webhook_support: bool
    white_label_options: bool


# Channel-specific configurations
DEFAULT_CHANNEL_LIMITS = {
    Channel.WEBSITE: ChannelLimits(
        channel=Channel.WEBSITE,
        enabled=True,
        rate_limit_per_minute=60,
        max_message_length=4000,
        custom_token_multiplier=1.0,
        priority_level=1,
    ),
    Channel.WHATSAPP: ChannelLimits(
        channel=Channel.WHATSAPP,
        enabled=True,
        rate_limit_per_minute=30,
        max_message_length=1600,  # WhatsApp limit
        custom_token_multiplier=1.2,  # Slightly higher cost
        priority_level=1,
    ),
    Channel.MESSENGER: ChannelLimits(
        channel=Channel.MESSENGER,
        enabled=True,
        rate_limit_per_minute=40,
        max_message_length=2000,
        custom_token_multiplier=1.1,
        priority_level=2,
    ),
    Channel.INSTAGRAM: ChannelLimits(
        channel=Channel.INSTAGRAM,
        enabled=True,
        rate_limit_per_minute=25,
        max_message_length=1000,
        custom_token_multiplier=1.3,
        priority_level=2,
    ),
    Channel.API: ChannelLimits(
        channel=Channel.API,
        enabled=True,
        rate_limit_per_minute=100,
        max_message_length=8000,
        custom_token_multiplier=0.9,  # Lower cost for API
        priority_level=1,
    ),
    Channel.MOBILE_APP: ChannelLimits(
        channel=Channel.MOBILE_APP,
        enabled=True,
        rate_limit_per_minute=80,
        max_message_length=4000,
        custom_token_multiplier=1.0,
        priority_level=1,
    ),
}

# Enhanced plan configurations with channel support
SUBSCRIPTION_PLANS = {
    SubscriptionPlan.BASIC: EnhancedPlanFeatures(
        name="Basic Plan",
        monthly_token_limit=10000,
        price_per_month=29.99,
        max_chatbots=3,
        max_documents_per_chatbot=50,
        priority_support=False,
        custom_branding=False,
        api_access=False,
        analytics_retention_days=30,
        supported_channels=[Channel.WEBSITE, Channel.WHATSAPP],
        channel_limits={
            Channel.WEBSITE: DEFAULT_CHANNEL_LIMITS[Channel.WEBSITE],
            Channel.WHATSAPP: ChannelLimits(
                channel=Channel.WHATSAPP,
                enabled=True,
                rate_limit_per_minute=20,  # Lower limit for basic
                max_message_length=1600,
                custom_token_multiplier=1.2,
                priority_level=3,
            ),
        },
        concurrent_conversations=10,
        webhook_support=False,
        white_label_options=False,
    ),
    SubscriptionPlan.PROFESSIONAL: EnhancedPlanFeatures(
        name="Professional Plan",
        monthly_token_limit=50000,
        price_per_month=99.99,
        max_chatbots=10,
        max_documents_per_chatbot=200,
        priority_support=True,
        custom_branding=True,
        api_access=True,
        analytics_retention_days=90,
        supported_channels=[
            Channel.WEBSITE,
            Channel.WHATSAPP,
            Channel.MESSENGER,
            Channel.API,
        ],
        channel_limits={
            Channel.WEBSITE: DEFAULT_CHANNEL_LIMITS[Channel.WEBSITE],
            Channel.WHATSAPP: DEFAULT_CHANNEL_LIMITS[Channel.WHATSAPP],
            Channel.MESSENGER: DEFAULT_CHANNEL_LIMITS[Channel.MESSENGER],
            Channel.API: DEFAULT_CHANNEL_LIMITS[Channel.API],
        },
        concurrent_conversations=50,
        webhook_support=True,
        white_label_options=True,
    ),
    SubscriptionPlan.ENTERPRISE: EnhancedPlanFeatures(
        name="Enterprise Plan",
        monthly_token_limit=200000,
        price_per_month=299.99,
        max_chatbots=50,
        max_documents_per_chatbot=1000,
        priority_support=True,
        custom_branding=True,
        api_access=True,
        analytics_retention_days=365,
        supported_channels=list(Channel),  # All channels
        channel_limits=DEFAULT_CHANNEL_LIMITS,
        concurrent_conversations=200,
        webhook_support=True,
        white_label_options=True,
    ),
}
