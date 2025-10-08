"""Context configuration models."""

from pydantic import BaseModel


class ContextConfigRequest(BaseModel):
    """Request model for context configuration updates."""

    config_updates: dict
