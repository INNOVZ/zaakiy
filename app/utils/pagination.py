"""
Pagination utilities for API endpoints

Provides standardized pagination helpers to prevent memory issues
and improve API performance.
"""
from typing import Dict, Any, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Standard pagination parameters"""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100,
                           description="Items per page")

    @validator('page')
    def validate_page(cls, v):
        if v < 1:
            raise ValueError("Page must be >= 1")
        return v

    @validator('page_size')
    def validate_page_size(cls, v):
        if v < 1:
            raise ValueError("Page size must be >= 1")
        if v > 100:
            raise ValueError("Page size cannot exceed 100")
        return v

    @property
    def offset(self) -> int:
        """Calculate offset for database query"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Get limit for database query"""
        return self.page_size


class PaginationMeta(BaseModel):
    """Pagination metadata for responses"""

    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_items: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")

    @classmethod
    def from_params(
        cls,
        params: PaginationParams,
        total_items: int
    ) -> "PaginationMeta":
        """Create pagination metadata from params and total count"""
        total_pages = (total_items + params.page_size - 1) // params.page_size

        return cls(
            page=params.page,
            page_size=params.page_size,
            total_items=total_items,
            total_pages=max(1, total_pages),  # At least 1 page
            has_next=params.page < total_pages,
            has_prev=params.page > 1
        )


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""

    items: List[T] = Field(description="List of items for current page")
    pagination: PaginationMeta = Field(description="Pagination metadata")


def validate_pagination_params(
    page: int = 1,
    page_size: int = 20,
    max_page_size: int = 100
) -> PaginationParams:
    """
    Validate and create pagination parameters

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        max_page_size: Maximum allowed page size

    Returns:
        Validated PaginationParams

    Raises:
        HTTPException: If parameters are invalid
    """
    try:
        # Enforce max page size
        if page_size > max_page_size:
            raise ValueError(f"Page size cannot exceed {max_page_size}")

        return PaginationParams(page=page, page_size=page_size)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def create_pagination_meta(
    page: int,
    page_size: int,
    total_items: int
) -> Dict[str, Any]:
    """
    Create pagination metadata dictionary

    Args:
        page: Current page number
        page_size: Items per page
        total_items: Total number of items

    Returns:
        Dictionary with pagination metadata
    """
    total_pages = (total_items + page_size -
                   1) // page_size if total_items > 0 else 1

    return {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }


def apply_supabase_pagination(
    query,
    params: PaginationParams
):
    """
    Apply pagination to a Supabase query

    Args:
        query: Supabase query builder
        params: Pagination parameters

    Returns:
        Query with pagination applied
    """
    return query.range(
        params.offset,
        params.offset + params.page_size - 1
    )


class CursorPaginationParams(BaseModel):
    """Cursor-based pagination parameters (for large datasets)"""

    cursor: Optional[str] = Field(
        default=None, description="Cursor for next page")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")

    @validator('limit')
    def validate_limit(cls, v):
        if v < 1:
            raise ValueError("Limit must be >= 1")
        if v > 100:
            raise ValueError("Limit cannot exceed 100")
        return v


class CursorPaginationMeta(BaseModel):
    """Cursor-based pagination metadata"""

    next_cursor: Optional[str] = Field(description="Cursor for next page")
    has_more: bool = Field(description="Whether there are more items")
    limit: int = Field(description="Items per page")


def encode_cursor(value: Any) -> str:
    """
    Encode a value as a cursor

    Args:
        value: Value to encode (typically timestamp or ID)

    Returns:
        Base64-encoded cursor string
    """
    import base64
    import json

    cursor_data = {"value": str(value)}
    json_str = json.dumps(cursor_data)
    return base64.b64encode(json_str.encode()).decode()


def decode_cursor(cursor: str) -> Any:
    """
    Decode a cursor value

    Args:
        cursor: Base64-encoded cursor string

    Returns:
        Decoded cursor value

    Raises:
        ValueError: If cursor is invalid
    """
    import base64
    import json

    try:
        json_str = base64.b64decode(cursor.encode()).decode()
        cursor_data = json.loads(json_str)
        return cursor_data["value"]
    except Exception as e:
        raise ValueError(f"Invalid cursor: {str(e)}")


# Pagination configuration presets
PAGINATION_PRESETS = {
    "small": {"default_page_size": 10, "max_page_size": 50},
    "medium": {"default_page_size": 20, "max_page_size": 100},
    "large": {"default_page_size": 50, "max_page_size": 200},
}


def get_pagination_preset(preset: str = "medium") -> Dict[str, int]:
    """
    Get pagination configuration preset

    Args:
        preset: Preset name (small, medium, large)

    Returns:
        Dictionary with default_page_size and max_page_size
    """
    return PAGINATION_PRESETS.get(preset, PAGINATION_PRESETS["medium"])


# Example usage documentation
"""
Example Usage:

1. Basic Pagination:
```python
from app.utils.pagination import validate_pagination_params, create_pagination_meta

@router.get("/items")
async def list_items(page: int = 1, page_size: int = 20):
    # Validate parameters
    params = validate_pagination_params(page, page_size)
    
    # Query with pagination
    offset = params.offset
    limit = params.page_size
    
    items = db.query().offset(offset).limit(limit).all()
    total = db.query().count()
    
    # Create response
    return {
        "items": items,
        "pagination": create_pagination_meta(page, page_size, total)
    }
```

2. With Supabase:
```python
from app.utils.pagination import PaginationParams, apply_supabase_pagination

@router.get("/uploads")
async def list_uploads(page: int = 1, page_size: int = 20):
    params = PaginationParams(page=page, page_size=page_size)
    
    query = supabase.table("uploads").select("*", count="exact")
    query = apply_supabase_pagination(query, params)
    
    result = query.execute()
    
    return {
        "items": result.data,
        "pagination": PaginationMeta.from_params(params, result.count)
    }
```

3. Cursor-Based Pagination (for real-time data):
```python
from app.utils.pagination import CursorPaginationParams, encode_cursor

@router.get("/messages")
async def list_messages(cursor: str = None, limit: int = 20):
    params = CursorPaginationParams(cursor=cursor, limit=limit)
    
    query = db.query()
    
    if params.cursor:
        last_id = decode_cursor(params.cursor)
        query = query.filter(Message.id > last_id)
    
    messages = query.limit(params.limit + 1).all()
    
    has_more = len(messages) > params.limit
    items = messages[:params.limit]
    
    next_cursor = None
    if has_more:
        next_cursor = encode_cursor(items[-1].id)
    
    return {
        "items": items,
        "pagination": {
            "next_cursor": next_cursor,
            "has_more": has_more,
            "limit": params.limit
        }
    }
```
"""
