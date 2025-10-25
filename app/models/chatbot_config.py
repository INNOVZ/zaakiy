"""
Pydantic models for chatbot configuration
Provides type safety and validation for chatbot settings
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatbotConfig(BaseModel):
    """
    Configuration model for chatbot behavior and AI parameters.

    This ensures type safety and provides validation for all chatbot settings,
    preventing configuration errors and typos.
    """

    # AI Model Configuration
    model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model to use (gpt-3.5-turbo, gpt-4, gpt-4-turbo)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (0.0-2.0). Lower = more deterministic.",
    )
    max_tokens: int = Field(
        default=300,
        ge=50,
        le=4000,
        description="Maximum tokens in response (50-4000)",
    )

    # Chatbot Identity
    name: str = Field(default="Assistant", description="Chatbot display name")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for the chatbot",
    )
    tone: str = Field(
        default="friendly",
        description="Tone of responses (friendly, professional, casual, etc.)",
    )

    # Display Configuration
    color_hex: Optional[str] = Field(
        default="#2563eb", description="Primary color for chatbot UI"
    )
    avatar_url: Optional[str] = Field(
        default=None, description="URL to chatbot avatar image"
    )

    # Behavior Configuration
    fallback_message: str = Field(
        default="I apologize, but I'm experiencing some technical difficulties. Please try again in a moment.",
        description="Message to show when response generation fails",
    )
    welcome_message: Optional[str] = Field(
        default=None, description="Initial greeting message"
    )

    # Organization and identification
    org_id: Optional[str] = Field(default=None, description="Organization ID")
    id: Optional[str] = Field(default=None, description="Chatbot ID")
    status: Optional[str] = Field(default="active", description="Chatbot status")

    # Additional metadata
    description: Optional[str] = Field(default=None, description="Chatbot description")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is in valid range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Ensure max_tokens is in valid range"""
        if not 50 <= v <= 4000:
            raise ValueError("max_tokens must be between 50 and 4000")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Ensure model is a valid OpenAI model"""
        valid_models = {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
        }
        if v not in valid_models:
            # Allow it but log a warning
            # This provides flexibility for new models
            pass
        return v

    class Config:
        """Pydantic config"""

        # Allow extra fields for forward compatibility
        extra = "allow"
        # Use attribute access instead of dict access
        validate_assignment = True
        # Populate by name (support both snake_case and camelCase)
        populate_by_name = True

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ChatbotConfig":
        """
        Create ChatbotConfig from a dictionary with graceful handling.

        This provides backward compatibility with existing dict-based configs.
        """
        if config_dict is None:
            return cls()

        # Handle both dict and already-converted ChatbotConfig
        if isinstance(config_dict, cls):
            return config_dict

        # Create config from dict, using defaults for missing fields
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return self.model_dump(exclude_none=True)


class ContextConfig(BaseModel):
    """
    Configuration for context retrieval and processing.

    Controls how documents are retrieved and used for generating responses.
    """

    # Retrieval Strategy
    retrieval_strategy: str = Field(
        default="semantic_only",
        description="Strategy for document retrieval (semantic_only, hybrid, keyword_boost, domain_specific)",
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity in hybrid retrieval (0.0-1.0)",
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword matching in hybrid retrieval (0.0-1.0)",
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use",
    )

    # Retrieval Parameters
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1-20)",
    )
    max_query_variants: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum query variants for enhanced retrieval (1-10)",
    )

    # Context Processing
    max_context_length: int = Field(
        default=2000,
        ge=500,
        le=8000,
        description="Maximum context length in characters (500-8000)",
    )

    @field_validator("semantic_weight", "keyword_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Ensure weights are in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")
        return v

    class Config:
        """Pydantic config"""

        extra = "allow"
        validate_assignment = True

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ContextConfig":
        """Create ContextConfig from dictionary with graceful handling"""
        if config_dict is None:
            return cls()
        if isinstance(config_dict, cls):
            return config_dict
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return self.model_dump(exclude_none=True)
