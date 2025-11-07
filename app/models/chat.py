"""Chat request and response models."""

from typing import List, Optional

from pydantic import BaseModel, field_validator

from ..utils.validators import validate_message_length


class ChatRequest(BaseModel):
    """Request model for chat messages."""

    message: str
    chatbot_id: Optional[str] = None
    conversation_id: Optional[str] = None
    channel: Optional[str] = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Validate message length and content"""
        try:
            return validate_message_length(v, min_length=1, max_length=4000)
        except Exception as e:
            raise ValueError(str(e))

    @field_validator("chatbot_id")
    @classmethod
    def validate_chatbot_id(cls, v):
        """Validate chatbot_id format"""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                raise ValueError("Chatbot ID cannot be empty")
            if len(v) > 100:
                raise ValueError("Chatbot ID too long")
        return v

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v):
        """Validate conversation_id format"""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                raise ValueError("Conversation ID cannot be empty")
            if len(v) > 100:
                raise ValueError("Conversation ID too long")
        return v


class ChatResponse(BaseModel):
    """Response model for chat messages."""

    response: str
    sources: List[str] = []
    product_links: List[dict] = []
    chatbot_config: dict
    conversation_id: str
    message_id: Optional[str] = None
    processing_time_ms: int = 0
    context_quality: dict = {}
