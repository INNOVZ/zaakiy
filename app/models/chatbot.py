"""Chatbot configuration models."""

from typing import Optional

from pydantic import BaseModel, field_validator

from ..utils.validators import (sanitize_text_input, validate_chatbot_name,
                                validate_hex_color, validate_max_tokens,
                                validate_temperature)


class CreateChatbotRequest(BaseModel):
    """Request model for creating a chatbot."""

    name: str
    description: Optional[str] = None
    color_hex: Optional[str] = "#3B82F6"
    tone: Optional[str] = "helpful"
    behavior: Optional[str] = "Be helpful and informative"
    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = "Hello! How can I help you today?"
    fallback_message: Optional[str] = "I'm sorry, I don't have information about that."
    ai_model_config: Optional[dict] = None
    is_active: Optional[bool] = True
    avatar_url: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate chatbot name"""
        try:
            return validate_chatbot_name(v, min_length=2, max_length=100)
        except Exception as e:
            raise ValueError(str(e)) from e

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Validate description length"""
        if v is not None:
            v = sanitize_text_input(v, max_length=500)
            if len(v) > 500:
                raise ValueError("Description too long (max 500 characters)")
        return v

    @field_validator("color_hex")
    @classmethod
    def validate_color(cls, v):
        """Validate hex color"""
        if v is not None:
            try:
                return validate_hex_color(v)
            except Exception as e:
                raise ValueError(str(e)) from e
        return v

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v):
        """Validate tone"""
        if v is not None:
            allowed_tones = ["helpful", "professional", "friendly", "casual", "formal"]
            v = v.strip().lower()
            if v not in allowed_tones:
                raise ValueError(f"Tone must be one of: {', '.join(allowed_tones)}")
        return v

    @field_validator(
        "behavior", "system_prompt", "greeting_message", "fallback_message"
    )
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields"""
        if v is not None:
            v = sanitize_text_input(v, max_length=2000)
            if len(v) > 2000:
                raise ValueError("Text field too long (max 2000 characters)")
        return v

    @field_validator("ai_model_config")
    @classmethod
    def validate_model_config(cls, v):
        """Validate AI model configuration"""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("ai_model_config must be a dictionary")

        cls._validate_temperature_field(v)
        cls._validate_max_tokens_field(v)
        cls._validate_model_field(v)

        return v

    @classmethod
    def _validate_temperature_field(cls, config_dict):
        """Validate temperature field in model config"""
        if "temperature" not in config_dict:
            return

        try:
            config_dict["temperature"] = validate_temperature(
                config_dict["temperature"]
            )
        except Exception as e:
            raise ValueError(f"Invalid temperature: {str(e)}") from e

    @classmethod
    def _validate_max_tokens_field(cls, config_dict):
        """Validate max_tokens field in model config"""
        if "max_tokens" not in config_dict:
            return

        try:
            config_dict["max_tokens"] = validate_max_tokens(config_dict["max_tokens"])
        except Exception as e:
            raise ValueError(f"Invalid max_tokens: {str(e)}") from e

    @classmethod
    def _validate_model_field(cls, config_dict):
        """Validate model field in model config"""
        if "model" not in config_dict:
            return

        allowed_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        if config_dict["model"] not in allowed_models:
            raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")

    @field_validator("avatar_url")
    @classmethod
    def validate_avatar_url(cls, v):
        """Validate avatar URL"""
        if v is not None:
            v = v.strip()
            
            # Validate URL format
            if len(v) > 500:
                raise ValueError("Avatar URL too long")
            if not v.startswith(("http://", "https://")):
                raise ValueError("Avatar URL must start with http:// or https://")
        return v


class UpdateChatbotRequest(BaseModel):
    """Request model for updating a chatbot."""

    name: Optional[str] = None
    description: Optional[str] = None
    color_hex: Optional[str] = None
    tone: Optional[str] = None
    behavior: Optional[str] = None
    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = None
    fallback_message: Optional[str] = None
    ai_model_config: Optional[dict] = None
    is_active: Optional[bool] = None
    avatar_url: Optional[str] = None
