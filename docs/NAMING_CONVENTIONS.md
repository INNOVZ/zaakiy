# Naming Conventions Guide

## Overview

This document establishes consistent naming conventions across the ZaaKy AI Platform codebase based on the Supabase database schema.

## 🎯 Core Principles

1. **Consistency** - Use the same name for the same concept everywhere
2. **Clarity** - Names should be self-explanatory
3. **Database Alignment** - Match database column names in code
4. **Snake Case** - Use snake_case for variables, functions, and database fields
5. **Camel Case** - Use camelCase only for frontend/JavaScript

## 📊 Database Schema Reference

Based on your Supabase schema, here are the standard field names:

### Organizations Table

```sql
organizations
├── id (uuid)
├── name (text)
├── email (text)
├── created_at (timestamp)
└── updated_at (timestamp)
```

**Standard naming:** `org_id` (NOT `organization_id`)

### Chatbots Table

```sql
chatbots
├── id (uuid)
├── org_id (uuid) ← Foreign key
├── name (text)
├── description (text)
├── color_hex (text)
├── tone (text)
├── behavior (text)
├── system_prompt (text)
├── greeting_message (text)
├── fallback_message (text)
├── model_config (jsonb)
├── chain_status (text)
├── avatar_url (text)
├── trained_at (timestamp)
├── created_at (timestamp)
└── updated_at (timestamp)
```

**Standard naming:** `chatbot_id` (NOT `chatbot_config`, `bot_id`, or `assistant_id`)

### Uploads Table

```sql
uploads
├── id (uuid)
├── org_id (uuid) ← Foreign key
├── type (text)
├── source (text)
├── pinecone_namespace (text)
├── status (text)
├── error_message (text)
├── created_at (timestamp)
└── updated_at (timestamp)
```

**Standard naming:** `upload_id` (NOT `file_id` or `document_id`)

### Messages Table

```sql
messages
├── id (uuid)
├── conversation_id (uuid) ← Foreign key
├── role (text)
├── content (text)
├── token_count (int)
├── processing_time_ms (int)
├── created_at (timestamp)
└── updated_at (timestamp)
```

**Standard naming:** `message_id` (NOT `msg_id`)

### Conversations Table

```sql
conversations
├── id (uuid)
├── chatbot_id (uuid) ← Foreign key
├── org_id (uuid) ← Foreign key
├── session_id (text)
├── status (text)
├── metadata (jsonb)
├── created_at (timestamp)
└── updated_at (timestamp)
```

**Standard naming:** `conversation_id` (NOT `conv_id` or `session_id`)

## 🔤 Naming Standards

### Primary Keys

Always use `{table_name}_id` format:

```python
# ✅ CORRECT
org_id: str
chatbot_id: str
upload_id: str
message_id: str
conversation_id: str
user_id: str

# ❌ INCORRECT
organization_id: str  # Too long
bot_id: str          # Ambiguous
file_id: str         # Wrong context
msg_id: str          # Abbreviated
conv_id: str         # Abbreviated
```

### Foreign Keys

Use the same name as the referenced primary key:

```python
# ✅ CORRECT - Matches primary key name
class Upload:
    id: str
    org_id: str  # References organizations.id

class Chatbot:
    id: str
    org_id: str  # References organizations.id

# ❌ INCORRECT
class Upload:
    id: str
    organization_id: str  # Doesn't match database
```

### Configuration Objects

Use `{entity}_config` for configuration dictionaries:

```python
# ✅ CORRECT
model_config: dict  # AI model configuration
chatbot_config: dict  # Full chatbot object from database
context_config: dict  # Context engineering configuration

# ❌ INCORRECT
chatbot: dict  # Ambiguous - is it ID or config?
bot_settings: dict  # Inconsistent naming
```

### Function Parameters

Be explicit about what you're passing:

```python
# ✅ CORRECT - Clear what each parameter is
async def create_chatbot(
    org_id: str,           # Organization ID
    name: str,             # Chatbot name
    model_config: dict     # Model configuration
) -> dict:
    pass

async def get_chatbot(
    chatbot_id: str        # Chatbot ID
) -> dict:
    pass

# ❌ INCORRECT - Ambiguous
async def create_chatbot(
    org: str,              # Is this ID or object?
    chatbot: dict          # Is this config or full object?
) -> dict:
    pass
```

### Variable Names

#### IDs (strings)

```python
# ✅ CORRECT
org_id = "123e4567-e89b-12d3-a456-426614174000"
chatbot_id = "123e4567-e89b-12d3-a456-426614174001"
upload_id = "123e4567-e89b-12d3-a456-426614174002"

# ❌ INCORRECT
organization_id = "..."  # Too verbose
bot_id = "..."          # Ambiguous
file_id = "..."         # Wrong context
```

#### Objects (dictionaries)

```python
# ✅ CORRECT
chatbot_config = {      # Full chatbot object from database
    "id": "...",
    "name": "...",
    "model_config": {...}
}

model_config = {        # AI model configuration
    "model": "gpt-4",
    "temperature": 0.7
}

# ❌ INCORRECT
chatbot = {...}         # Ambiguous
bot_config = {...}      # Inconsistent
settings = {...}        # Too generic
```

#### Lists

```python
# ✅ CORRECT
chatbots: List[dict]
uploads: List[dict]
messages: List[dict]

# ❌ INCORRECT
chatbot_list: List[dict]  # Redundant
all_chatbots: List[dict]  # Redundant prefix
```

## 📝 Code Examples

### Database Queries

```python
# ✅ CORRECT
async def get_chatbot(chatbot_id: str, org_id: str) -> dict:
    """Get chatbot by ID"""
    result = supabase.table("chatbots").select("*").eq(
        "id", chatbot_id
    ).eq(
        "org_id", org_id
    ).execute()

    return result.data[0] if result.data else None

# ❌ INCORRECT
async def get_chatbot(bot_id: str, organization_id: str) -> dict:
    """Inconsistent naming"""
    result = supabase.table("chatbots").select("*").eq(
        "id", bot_id  # Doesn't match database
    ).eq(
        "org_id", organization_id  # Doesn't match database
    ).execute()
```

### API Endpoints

```python
# ✅ CORRECT
@router.get("/chatbots/{chatbot_id}")
async def get_chatbot(
    chatbot_id: str,
    user=Depends(verify_jwt_token)
):
    user_data = await get_user_with_org(user["user_id"])
    org_id = user_data["org_id"]

    chatbot_config = await fetch_chatbot(chatbot_id, org_id)
    return chatbot_config

# ❌ INCORRECT
@router.get("/chatbots/{bot_id}")
async def get_chatbot(
    bot_id: str,  # Inconsistent
    user=Depends(verify_jwt_token)
):
    user_data = await get_user_with_org(user["user_id"])
    organization_id = user_data["org_id"]  # Inconsistent
```

### Service Classes

```python
# ✅ CORRECT
class ChatService:
    def __init__(self, org_id: str, chatbot_config: dict):
        self.org_id = org_id
        self.chatbot_id = chatbot_config["id"]
        self.chatbot_config = chatbot_config

    async def send_message(
        self,
        message: str,
        conversation_id: str
    ) -> dict:
        pass

# ❌ INCORRECT
class ChatService:
    def __init__(self, organization_id: str, chatbot: dict):
        self.org = organization_id  # Inconsistent
        self.bot_id = chatbot["id"]  # Inconsistent
```

### Pydantic Models

```python
# ✅ CORRECT
from pydantic import BaseModel

class ChatbotCreate(BaseModel):
    name: str
    org_id: str
    model_config: dict

class ChatbotResponse(BaseModel):
    id: str
    org_id: str
    name: str
    model_config: dict
    created_at: datetime

# ❌ INCORRECT
class ChatbotCreate(BaseModel):
    name: str
    organization_id: str  # Doesn't match database
    config: dict          # Too generic
```

## 🔄 Migration Guide

### Finding Inconsistencies

```bash
# Search for inconsistent naming
grep -r "organization_id" app/
grep -r "bot_id" app/
grep -r "file_id" app/ --include="*.py"
```

### Refactoring Steps

1. **Identify all occurrences**

   ```bash
   grep -r "organization_id" app/ > inconsistencies.txt
   ```

2. **Replace systematically**

   ```python
   # Before
   def process_upload(organization_id: str, file_id: str):
       pass

   # After
   def process_upload(org_id: str, upload_id: str):
       pass
   ```

3. **Update tests**

   ```python
   # Before
   def test_upload(organization_id="test-org"):
       pass

   # After
   def test_upload(org_id="test-org"):
       pass
   ```

4. **Update documentation**

## 🎨 Frontend Naming (JavaScript/TypeScript)

### API Responses (camelCase)

```typescript
// ✅ CORRECT - Match backend snake_case in responses
interface ChatbotResponse {
  id: string;
  org_id: string; // Keep as snake_case from API
  name: string;
  model_config: object;
  created_at: string;
}

// Or convert to camelCase
interface Chatbot {
  id: string;
  orgId: string; // Converted from org_id
  name: string;
  modelConfig: object; // Converted from model_config
  createdAt: string; // Converted from created_at
}
```

### Component Props (camelCase)

```typescript
// ✅ CORRECT
interface ChatbotWidgetProps {
  chatbotId: string;
  orgId: string;
  onMessage: (message: string) => void;
}

// ❌ INCORRECT
interface ChatbotWidgetProps {
  chatbot_id: string; // Use camelCase in frontend
  org_id: string; // Use camelCase in frontend
}
```

## 📋 Quick Reference

### Standard Names

| Concept         | Database Column               | Python Variable   | TypeScript Variable |
| --------------- | ----------------------------- | ----------------- | ------------------- |
| Organization ID | `org_id`                      | `org_id`          | `orgId`             |
| Chatbot ID      | `id` (in chatbots table)      | `chatbot_id`      | `chatbotId`         |
| Upload ID       | `id` (in uploads table)       | `upload_id`       | `uploadId`          |
| Message ID      | `id` (in messages table)      | `message_id`      | `messageId`         |
| Conversation ID | `id` (in conversations table) | `conversation_id` | `conversationId`    |
| User ID         | `id` (in users table)         | `user_id`         | `userId`            |

### Configuration Objects

| Type                | Python Name       | Description                             |
| ------------------- | ----------------- | --------------------------------------- |
| Full chatbot object | `chatbot_config`  | Complete chatbot from database          |
| AI model settings   | `model_config`    | Model configuration (temperature, etc.) |
| Context settings    | `context_config`  | Context engineering configuration       |
| Upload metadata     | `upload_metadata` | Upload-specific metadata                |

## ✅ Validation Checklist

Before committing code, verify:

- [ ] All IDs use `{entity}_id` format
- [ ] No use of `organization_id` (use `org_id`)
- [ ] No use of `bot_id` (use `chatbot_id`)
- [ ] No use of `file_id` in upload context (use `upload_id`)
- [ ] Configuration objects use `{entity}_config` format
- [ ] Database queries match column names exactly
- [ ] Function parameters are clearly named
- [ ] Pydantic models match database schema
- [ ] Frontend uses camelCase consistently
- [ ] Documentation uses standard names

## 🔍 Code Review Guidelines

When reviewing code, check for:

1. **Consistency with database schema**

   - Do variable names match database columns?
   - Are foreign keys named consistently?

2. **Clarity**

   - Is it clear whether a variable is an ID or an object?
   - Are configuration objects clearly labeled?

3. **No abbreviations**

   - No `org` instead of `org_id`
   - No `bot` instead of `chatbot_id`
   - No `msg` instead of `message_id`

4. **Proper typing**
   - Are IDs typed as `str`?
   - Are configs typed as `dict` or proper models?

---

**Last Updated**: 2025-01-08
**Version**: 1.0
**Status**: Standard ✅
