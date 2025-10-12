"""Public chat models for embedded chatbots."""

from typing import List, Optional

from pydantic import BaseModel, field_validator


class PublicChatRequest(BaseModel):
    """Request model for public chat endpoint."""

    message: str
    chatbot_id: str
    session_id: Optional[str] = None
    user_identifier: Optional[str] = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Validate message length and content"""
        if not v or not isinstance(v, str):
            raise ValueError("Message must be a non-empty string")

        v = v.strip()

        if len(v) < 1:
            raise ValueError("Message cannot be empty")

        if len(v) > 4000:
            raise ValueError("Message too long (max 4000 characters)")

        return v

    @field_validator("chatbot_id")
    @classmethod
    def validate_chatbot_id(cls, v):
        """Validate chatbot_id format"""
        if not v or not isinstance(v, str):
            raise ValueError("Chatbot ID must be provided")

        v = v.strip()

        if len(v) == 0:
            raise ValueError("Chatbot ID cannot be empty")

        if len(v) > 100:
            raise ValueError("Chatbot ID too long")

        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id format"""
        if v is not None:
            v = v.strip()
            if len(v) > 100:
                raise ValueError("Session ID too long")
        return v

    @field_validator("user_identifier")
    @classmethod
    def validate_user_identifier(cls, v):
        """Validate user_identifier"""
        if v is not None:
            v = v.strip()
            if len(v) > 100:
                raise ValueError("User identifier too long")
        return v


class PublicChatResponse(BaseModel):
    """Response model for public chat endpoint."""

    response: str
    product_links: List[dict] = []
    chatbot: dict
    session_id: str
