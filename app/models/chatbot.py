"""Chatbot configuration models."""

from typing import Optional
from pydantic import BaseModel, field_validator
from ..utils.validators import (
    validate_chatbot_name,
    validate_hex_color,
    validate_temperature,
    validate_max_tokens,
    sanitize_text_input
)


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

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate chatbot name"""
        try:
            return validate_chatbot_name(v, min_length=2, max_length=100)
        except Exception as e:
            raise ValueError(str(e)) from e

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        """Validate description length"""
        if v is not None:
            v = sanitize_text_input(v, max_length=500)
            if len(v) > 500:
                raise ValueError("Description too long (max 500 characters)")
        return v

    @field_validator('color_hex')
    @classmethod
    def validate_color(cls, v):
        """Validate hex color"""
        if v is not None:
            try:
                return validate_hex_color(v)
            except Exception as e:
                raise ValueError(str(e)) from e
        return v

    @field_validator('tone')
    @classmethod
    def validate_tone(cls, v):
        """Validate tone"""
        if v is not None:
            allowed_tones = ['helpful', 'professional',
                             'friendly', 'casual', 'formal']
            v = v.strip().lower()
            if v not in allowed_tones:
                raise ValueError(
                    f"Tone must be one of: {', '.join(allowed_tones)}")
        return v

    @field_validator('behavior', 'system_prompt', 'greeting_message', 'fallback_message')
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields"""
        if v is not None:
            v = sanitize_text_input(v, max_length=2000)
            if len(v) > 2000:
                raise ValueError("Text field too long (max 2000 characters)")
        return v

    @field_validator('ai_model_config')
    @classmethod
    def validate_model_config(cls, v):
        """Validate AI model configuration"""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("ai_model_config must be a dictionary")

            # Validate temperature if present
            if 'temperature' in v:
                try:
                    v['temperature'] = validate_temperature(v['temperature'])
                except Exception as e:
                    raise ValueError(f"Invalid temperature: {str(e)}") from e

            # Validate max_tokens if present
            if 'max_tokens' in v:
                try:
                    v['max_tokens'] = validate_max_tokens(v['max_tokens'])
                except Exception as e:
                    raise ValueError(f"Invalid max_tokens: {str(e)}") from e

            # Validate model name
            if 'model' in v:
                allowed_models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']
                if v['model'] not in allowed_models:
                    raise ValueError(
                        f"Model must be one of: {', '.join(allowed_models)}")

        return v

    @field_validator('avatar_url')
    @classmethod
    def validate_avatar_url(cls, v):
        """Validate avatar URL"""
        if v is not None:
            v = v.strip()
            if len(v) > 500:
                raise ValueError("Avatar URL too long")
            if not v.startswith(('http://', 'https://')):
                raise ValueError(
                    "Avatar URL must start with http:// or https://")
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
