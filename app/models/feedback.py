"""Feedback request models."""

from typing import Optional
from pydantic import BaseModel, field_validator
from ..utils.validators import validate_rating, sanitize_text_input


class FeedbackRequest(BaseModel):
    """Request model for user feedback on chat messages."""

    message_id: str
    rating: int  # 1 for thumbs up, -1 for thumbs down
    feedback_text: Optional[str] = None

    @field_validator('message_id')
    @classmethod
    def validate_message_id(cls, v):
        """Validate message ID"""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Message ID cannot be empty")
        if len(v) > 100:
            raise ValueError("Message ID too long")
        return v

    @field_validator('rating')
    @classmethod
    def validate_rating_value(cls, v):
        """Validate rating value"""
        try:
            return validate_rating(v)
        except Exception as e:
            raise ValueError(str(e))

    @field_validator('feedback_text')
    @classmethod
    def validate_feedback(cls, v):
        """Validate feedback text"""
        if v is not None:
            v = sanitize_text_input(v, max_length=1000)
            if len(v) > 1000:
                raise ValueError(
                    "Feedback text too long (max 1000 characters)")
        return v
