"""Upload and search request models."""

from typing import List, Optional

from pydantic import BaseModel, field_validator

from ..utils.validators import validate_top_k, validate_url


class URLIngestRequest(BaseModel):
    """Request model for ingesting a URL."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v):
        """Validate URL format and security"""
        try:
            return validate_url(v, allow_localhost=False)
        except Exception as e:
            raise ValueError(str(e))


class UpdateRequest(BaseModel):
    """Request model for updating an upload with a new URL."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v):
        """Validate URL format and security"""
        try:
            return validate_url(v, allow_localhost=False)
        except Exception as e:
            raise ValueError(str(e))


class SearchRequest(BaseModel):
    """Request model for searching uploads."""

    query: str
    top_k: int = 5
    filter_upload_ids: Optional[List[str]] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate search query"""
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string")

        v = v.strip()

        if len(v) < 1:
            raise ValueError("Query cannot be empty")

        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters)")

        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k_value(cls, v):
        """Validate top_k parameter"""
        try:
            return validate_top_k(v, min_value=1, max_value=100)
        except Exception as e:
            raise ValueError(str(e))

    @field_validator("filter_upload_ids")
    @classmethod
    def validate_upload_ids(cls, v):
        """Validate upload IDs list"""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("filter_upload_ids must be a list")

            if len(v) > 50:
                raise ValueError("Too many upload IDs (max 50)")

            for upload_id in v:
                if not isinstance(upload_id, str) or len(upload_id.strip()) == 0:
                    raise ValueError("Invalid upload ID in filter list")

        return v
