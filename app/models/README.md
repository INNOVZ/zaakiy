# Models Package

This package contains Pydantic models for request/response validation and data structures.

## Structure

```
app/models/
├── __init__.py          # Package exports
├── chat.py              # Chat request/response models
├── chatbot.py           # Chatbot configuration models
├── upload.py            # Upload and search models
├── feedback.py          # Feedback models
├── public.py            # Public chat models
├── context.py           # Context configuration models
└── organization.py      # Organization and user models
```

## Models Overview

### Chat Models (`chat.py`)

- **ChatRequest**: Validates chat messages with conversation context
- **ChatResponse**: Structured chat response with sources and metadata

### Chatbot Models (`chatbot.py`)

- **CreateChatbotRequest**: Validates chatbot creation with full configuration
- **UpdateChatbotRequest**: Partial updates for chatbot settings

### Upload Models (`upload.py`)

- **URLIngestRequest**: URL validation for web scraping
- **UpdateRequest**: Update existing uploads
- **SearchRequest**: Search with filters and pagination

### Feedback Models (`feedback.py`)

- **FeedbackRequest**: User feedback on chat messages (thumbs up/down)

### Public Models (`public.py`)

- **PublicChatRequest**: Public-facing chat endpoint requests
- **PublicChatResponse**: Public chat responses

### Context Models (`context.py`)

- **ContextConfigRequest**: Context configuration updates

### Organization Models (`organization.py`)

- **UpdateOrganizationRequest**: Update organization details (name, email, phone, business type)
- **UpdateUserRequest**: Update user profile information

## Usage

Import models from the package:

```python
from app.models import ChatRequest, ChatResponse, CreateChatbotRequest

# Use in router endpoints
@router.post("/chat")
async def chat(request: ChatRequest):
    # Request is automatically validated
    pass
```

## Pydantic V2

All models use Pydantic V2 syntax with `@field_validator` decorators:

```python
from pydantic import BaseModel, field_validator

class MyModel(BaseModel):
    name: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        # Validation logic
        return v
```

## Benefits

1. **Centralized Validation**: All validation logic in one place
2. **Reusability**: Models can be used across multiple routers
3. **Type Safety**: Better IDE support and type hints
4. **Maintainability**: Easier to update validation rules
5. **Testing**: Models can be tested independently
6. **Documentation**: Auto-generated API docs from models
