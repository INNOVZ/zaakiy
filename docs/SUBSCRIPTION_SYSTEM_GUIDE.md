# Multi-Channel Subscription System Implementation Guide

This guide explains how to use the enhanced subscription-based onboarding system with channel-specific token tracking for your omnichannel RAG AI chat platform.

## Overview

The subscription system provides:

- **Three subscription plans**: Basic, Professional, Enterprise
- **Multi-channel support**: Website, WhatsApp, Messenger, Instagram, API, Mobile App
- **Channel-specific token tracking** and analytics
- **Token-based usage limits** with channel multipliers
- **User and Organization onboarding** with plan selection
- **Real-time token tracking** and validation
- **Comprehensive analytics** and performance insights
- **Automatic billing cycle management**

## Subscription Plans

### Basic Plan ($29.99/month)

- 10,000 tokens per month
- 3 chatbots maximum
- 50 documents per chatbot
- Standard support
- 30-day analytics retention

### Professional Plan ($99.99/month)

- 50,000 tokens per month
- 10 chatbots maximum
- 200 documents per chatbot
- Priority support
- Custom branding
- API access
- 90-day analytics retention

### Enterprise Plan ($299.99/month)

- 200,000 tokens per month
- 50 chatbots maximum
- 1,000 documents per chatbot
- Priority support
- Custom branding
- API access
- 365-day analytics retention

## Supported Channels

The system supports multiple communication channels with channel-specific configurations:

### 🌐 Website Chat

- **Use Cases**: Customer support, lead generation, FAQ
- **Token Multiplier**: 1.0x (base rate)
- **Rate Limit**: 60 messages/minute
- **Max Message Length**: 4,000 characters

### 📱 WhatsApp Business

- **Use Cases**: Customer service, order updates, marketing
- **Token Multiplier**: 1.2x (slightly higher cost)
- **Rate Limit**: 30 messages/minute
- **Max Message Length**: 1,600 characters

### 💬 Facebook Messenger

- **Use Cases**: Social commerce, customer support, engagement
- **Token Multiplier**: 1.1x
- **Rate Limit**: 40 messages/minute
- **Max Message Length**: 2,000 characters

### 📸 Instagram Direct

- **Use Cases**: Brand engagement, product inquiries, support
- **Token Multiplier**: 1.3x
- **Rate Limit**: 25 messages/minute
- **Max Message Length**: 1,000 characters

### 🔌 REST API

- **Use Cases**: Custom apps, system integration, automation
- **Token Multiplier**: 0.9x (lower cost for API)
- **Rate Limit**: 100 messages/minute
- **Max Message Length**: 8,000 characters

### 📲 Mobile App

- **Use Cases**: In-app support, user onboarding, feature guidance
- **Token Multiplier**: 1.0x
- **Rate Limit**: 80 messages/minute
- **Max Message Length**: 4,000 characters

## Channel-Specific Plan Features

### Basic Plan

- **Supported Channels**: Website, WhatsApp
- **Concurrent Conversations**: 10
- **Webhook Support**: No

### Professional Plan

- **Supported Channels**: Website, WhatsApp, Messenger, API
- **Concurrent Conversations**: 50
- **Webhook Support**: Yes
- **White Label Options**: Yes

### Enterprise Plan

- **Supported Channels**: All channels
- **Concurrent Conversations**: 200
- **Webhook Support**: Yes
- **White Label Options**: Yes
- **Advanced Analytics**: Yes

## Database Setup

1. **Run the migration script** in your Supabase SQL editor:

   ```sql
   -- Copy and paste the contents of database_migration_subscriptions.sql
   ```

2. **Verify tables were created**:
   - `subscriptions` - Main subscription records
   - `token_usage_logs` - Token usage tracking
   - `subscription_plans` - Plan configurations

## API Endpoints

### 1. Get Available Plans

```http
GET /api/onboarding/plans
```

**Response:**

```json
{
  "success": true,
  "plans": {
    "basic": {
      "name": "Basic Plan",
      "monthly_token_limit": 10000,
      "price_per_month": 29.99,
      "max_chatbots": 3,
      "max_documents_per_chatbot": 50,
      "priority_support": false,
      "custom_branding": false,
      "api_access": false,
      "analytics_retention_days": 30
    }
    // ... other plans
  }
}
```

### 2. User/Organization Signup

```http
POST /api/onboarding/signup
```

**Request Body (User):**

```json
{
  "entity_type": "user",
  "full_name": "John Doe",
  "email": "john@example.com",
  "selected_plan": "professional"
}
```

**Request Body (Organization):**

```json
{
  "entity_type": "organization",
  "full_name": "John Doe",
  "email": "admin@company.com",
  "organization_name": "Acme Corp",
  "contact_phone": "+1-555-0123",
  "business_type": "Technology",
  "selected_plan": "enterprise"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Organization successfully created with Enterprise Plan",
  "entity_id": "uuid-here",
  "entity_type": "organization",
  "subscription_id": "subscription-uuid",
  "plan": "enterprise",
  "tokens_remaining": 200000,
  "tokens_limit": 200000
}
```

### 3. Check Subscription Status

```http
GET /api/onboarding/subscription/{entity_type}/{entity_id}
```

**Response:**

```json
{
  "success": true,
  "subscription_id": "uuid-here",
  "tokens_used_this_month": 15000,
  "tokens_remaining": 185000,
  "monthly_limit": 200000,
  "usage_percentage": 7.5,
  "reset_date": "2025-02-07T12:00:00Z"
}
```

### 4. Consume Tokens

```http
POST /api/onboarding/tokens/consume
```

**Request Body:**

```json
{
  "entity_id": "uuid-here",
  "entity_type": "organization",
  "tokens_consumed": 150,
  "operation_type": "chat"
}
```

**Response (Success):**

```json
{
  "success": true,
  "tokens_consumed": 150,
  "tokens_remaining": 184850,
  "monthly_limit": 200000,
  "usage_percentage": 7.58
}
```

**Response (Insufficient Tokens):**

```json
{
  "detail": {
    "error": "Insufficient tokens",
    "tokens_requested": 150,
    "tokens_available": 50,
    "monthly_limit": 200000,
    "reset_date": "2025-02-07T12:00:00Z"
  }
}
```

### 5. Check Token Availability

```http
GET /api/onboarding/tokens/check/{entity_type}/{entity_id}/{required_tokens}
```

**Response:**

```json
{
  "success": true,
  "has_enough_tokens": true,
  "tokens_required": 150,
  "tokens_available": 184850,
  "can_proceed": true
}
```

### 6. Get Subscription Analytics

```http
GET /api/onboarding/analytics/{subscription_id}?days_back=30
```

**Response:**

```json
{
  "success": true,
  "analytics": {
    "subscription_id": "uuid-here",
    "entity_id": "uuid-here",
    "entity_type": "organization",
    "plan": "enterprise",
    "total_tokens_used": 15000,
    "total_tokens_limit": 200000,
    "usage_percentage": 7.5,
    "channel_usage": [
      {
        "channel": "website",
        "tokens_used": 8000,
        "message_count": 120,
        "unique_users": 45,
        "avg_tokens_per_message": 66.67,
        "peak_usage_hour": 14
      },
      {
        "channel": "whatsapp",
        "tokens_used": 7000,
        "message_count": 95,
        "unique_users": 38,
        "avg_tokens_per_message": 73.68,
        "peak_usage_hour": 10
      }
    ],
    "daily_usage": {
      "2025-01-01": 500,
      "2025-01-02": 750,
      "2025-01-03": 600
    },
    "hourly_distribution": {
      "0": 50,
      "1": 20,
      "2": 10,
      "9": 800,
      "10": 1200,
      "14": 1500
    },
    "most_active_channel": "website",
    "least_active_channel": "instagram",
    "growth_rate": 15.5,
    "billing_cycle_start": "2025-01-01T00:00:00Z",
    "billing_cycle_end": "2025-01-31T23:59:59Z",
    "days_remaining": 24
  }
}
```

### 7. Get Channel Performance Comparison

```http
GET /api/onboarding/analytics/{subscription_id}/channels/comparison?days_back=30
```

**Response:**

```json
{
  "success": true,
  "comparison": {
    "website": {
      "tokens_used": 8000,
      "message_count": 120,
      "unique_users": 45,
      "avg_tokens_per_message": 66.67,
      "usage_share_percentage": 53.33,
      "efficiency_score": 95.2,
      "peak_usage_hour": 14,
      "performance_rating": "Excellent"
    },
    "whatsapp": {
      "tokens_used": 7000,
      "message_count": 95,
      "unique_users": 38,
      "avg_tokens_per_message": 73.68,
      "usage_share_percentage": 46.67,
      "efficiency_score": 88.5,
      "peak_usage_hour": 10,
      "performance_rating": "Good"
    }
  },
  "analysis_period_days": 30
}
```

### 8. Get Channel Trends

```http
GET /api/onboarding/analytics/{subscription_id}/channels/{channel}/trends?days_back=30
```

**Response:**

```json
{
  "success": true,
  "channel": "whatsapp",
  "trends": {
    "daily_tokens": [
      { "date": "2025-01-01", "value": 200 },
      { "date": "2025-01-02", "value": 250 },
      { "date": "2025-01-03", "value": 180 }
    ],
    "daily_messages": [
      { "date": "2025-01-01", "value": 3 },
      { "date": "2025-01-02", "value": 4 },
      { "date": "2025-01-03", "value": 2 }
    ],
    "daily_users": [
      { "date": "2025-01-01", "value": 2 },
      { "date": "2025-01-02", "value": 3 },
      { "date": "2025-01-03", "value": 2 }
    ],
    "trend_direction": "increasing",
    "average_daily_tokens": 210
  },
  "analysis_period_days": 30
}
```

### 9. Get Supported Channels

```http
GET /api/onboarding/channels
```

**Response:**

```json
{
  "success": true,
  "channels": {
    "website": {
      "name": "Website Chat",
      "description": "Embedded chat widget on websites",
      "icon": "🌐",
      "typical_use_cases": ["Customer support", "Lead generation", "FAQ"]
    },
    "whatsapp": {
      "name": "WhatsApp Business",
      "description": "WhatsApp Business API integration",
      "icon": "📱",
      "typical_use_cases": ["Customer service", "Order updates", "Marketing"]
    }
  },
  "total_channels": 6
}
```

### 10. Get Channel Configurations

```http
GET /api/onboarding/subscription/{subscription_id}/channels/config
```

**Response:**

```json
{
  "success": true,
  "subscription_id": "uuid-here",
  "configurations": {
    "website": {
      "enabled": true,
      "rate_limit_per_minute": 60,
      "max_message_length": 4000,
      "custom_token_multiplier": 1.0,
      "priority_level": 1,
      "webhook_url": null,
      "custom_settings": {},
      "created_at": "2025-01-07T12:00:00Z",
      "updated_at": "2025-01-07T12:00:00Z"
    },
    "whatsapp": {
      "enabled": true,
      "rate_limit_per_minute": 30,
      "max_message_length": 1600,
      "custom_token_multiplier": 1.2,
      "priority_level": 1,
      "webhook_url": "https://api.example.com/webhook",
      "custom_settings": { "business_account_id": "123456789" },
      "created_at": "2025-01-07T12:00:00Z",
      "updated_at": "2025-01-07T12:00:00Z"
    }
  }
}
```

### 11. Update Channel Configuration

```http
PUT /api/onboarding/subscription/{subscription_id}/channels/{channel}/config
```

**Request Body:**

```json
{
  "enabled": true,
  "rate_limit_per_minute": 45,
  "webhook_url": "https://api.example.com/new-webhook",
  "custom_settings": {
    "business_account_id": "123456789",
    "auto_reply_enabled": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "message": "Channel configuration updated for whatsapp",
  "subscription_id": "uuid-here",
  "channel": "whatsapp",
  "updated_fields": [
    "enabled",
    "rate_limit_per_minute",
    "webhook_url",
    "custom_settings"
  ]
}
```

## Integration with Chat System

### 1. Update Chat Endpoint

Modify your existing chat endpoints to validate tokens before processing:

```python
from app.services.auth.billing_middleware import TokenValidationMiddleware, estimate_tokens_for_operation

@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    supabase: Client = Depends(get_supabase_client)
):
    # Initialize token middleware
    token_middleware = TokenValidationMiddleware(supabase)

    # Extract entity info from request/auth
    entity_id = request.entity_id  # or from JWT token
    entity_type = request.entity_type  # "user" or "organization"

    # Estimate tokens for this operation
    estimated_tokens = estimate_tokens_for_operation(
        "chat",
        message_length=len(request.message)
    )

    # Validate and consume tokens
    await token_middleware.validate_and_consume_tokens(
        entity_id=entity_id,
        entity_type=entity_type,
        estimated_tokens=estimated_tokens,
        operation_type="chat"
    )

    # Process chat normally
    response = await process_chat(request)
    return response
```

### 2. Update Document Upload

```python
@router.post("/upload")
async def upload_document(
    file: UploadFile,
    entity_id: str,
    entity_type: str,
    supabase: Client = Depends(get_supabase_client)
):
    token_middleware = TokenValidationMiddleware(supabase)

    # Estimate tokens based on file size
    estimated_tokens = estimate_tokens_for_operation(
        "document_upload",
        document_size=file.size
    )

    # Validate tokens before processing
    await token_middleware.validate_and_consume_tokens(
        entity_id=entity_id,
        entity_type=entity_type,
        estimated_tokens=estimated_tokens,
        operation_type="document_processing"
    )

    # Process upload
    result = await process_upload(file)
    return result
```

## Frontend Integration

### 1. Onboarding Flow

```javascript
// Get available plans
const getPlans = async () => {
  const response = await fetch("/api/onboarding/plans");
  const data = await response.json();
  return data.plans;
};

// Submit signup
const signup = async (formData) => {
  const response = await fetch("/api/onboarding/signup", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  return await response.json();
};

// Example usage
const handleSignup = async (formData) => {
  try {
    const result = await signup({
      entity_type: "organization",
      full_name: "John Doe",
      email: "admin@company.com",
      organization_name: "Acme Corp",
      selected_plan: "professional",
    });

    console.log("Signup successful:", result);
    // Redirect to dashboard or next step
  } catch (error) {
    console.error("Signup failed:", error.message);
  }
};
```

### 2. Token Usage Display

```javascript
// Get subscription status
const getSubscriptionStatus = async (entityType, entityId) => {
  const response = await fetch(
    `/api/onboarding/subscription/${entityType}/${entityId}`
  );
  return await response.json();
};

// Display usage in UI
const TokenUsageWidget = ({ entityId, entityType }) => {
  const [usage, setUsage] = useState(null);

  useEffect(() => {
    const fetchUsage = async () => {
      const data = await getSubscriptionStatus(entityType, entityId);
      setUsage(data);
    };

    fetchUsage();
    // Refresh every 30 seconds
    const interval = setInterval(fetchUsage, 30000);
    return () => clearInterval(interval);
  }, [entityId, entityType]);

  if (!usage) return <div>Loading...</div>;

  return (
    <div className="token-usage-widget">
      <h3>Token Usage</h3>
      <div className="usage-bar">
        <div
          className="usage-fill"
          style={{ width: `${usage.usage_percentage}%` }}
        />
      </div>
      <p>
        {usage.tokens_used_this_month.toLocaleString()} /{" "}
        {usage.monthly_limit.toLocaleString()} tokens used
      </p>
      <p>Resets: {new Date(usage.reset_date).toLocaleDateString()}</p>
    </div>
  );
};
```

### 3. Handle Token Errors

```javascript
// Wrapper for API calls that consume tokens
const apiCallWithTokenHandling = async (apiCall) => {
  try {
    return await apiCall();
  } catch (error) {
    if (error.status === 402) {
      // Payment required - insufficient tokens
      const errorData = error.detail;

      // Show upgrade modal or token limit message
      showTokenLimitModal({
        tokensAvailable: errorData.tokens_available,
        tokensRequired: errorData.tokens_requested,
        resetDate: errorData.reset_date,
      });

      return null;
    }
    throw error;
  }
};

// Example chat with token handling
const sendChatMessage = async (message, entityId, entityType) => {
  return apiCallWithTokenHandling(async () => {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Entity-ID": entityId,
        "X-Entity-Type": entityType,
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const error = await response.json();
      const apiError = new Error(error.detail);
      apiError.status = response.status;
      apiError.detail = error.detail;
      throw apiError;
    }

    return await response.json();
  });
};
```

## Token Estimation Guidelines

The system uses these base estimates (adjust based on your actual usage):

- **Chat response**: 100 tokens + message length/4
- **Document upload**: 500 tokens + document size/100
- **Document analysis**: 200 tokens
- **Web scraping**: 300 tokens
- **Embedding generation**: 50 tokens

You can customize these in the `estimate_tokens_for_operation` function.

## Monitoring and Analytics

### 1. Usage Analytics Query

```sql
-- Monthly usage by plan
SELECT
    s.plan,
    COUNT(*) as subscribers,
    AVG(s.tokens_used_this_month) as avg_tokens_used,
    SUM(s.tokens_used_this_month) as total_tokens_used,
    AVG(s.tokens_used_this_month::FLOAT / s.monthly_token_limit * 100) as avg_usage_percentage
FROM subscriptions s
WHERE s.status = 'active'
GROUP BY s.plan;

-- Top token consumers
SELECT
    s.entity_id,
    s.entity_type,
    s.plan,
    s.tokens_used_this_month,
    s.monthly_token_limit,
    s.tokens_used_this_month::FLOAT / s.monthly_token_limit * 100 as usage_percentage
FROM subscriptions s
WHERE s.status = 'active'
ORDER BY s.tokens_used_this_month DESC
LIMIT 10;
```

### 2. Set Up Alerts

Create alerts for:

- Users approaching token limits (>90% usage)
- Failed token consumption attempts
- Subscription renewals needed

## Security Considerations

1. **Rate Limiting**: Implement rate limiting on token consumption endpoints
2. **Authentication**: Ensure proper authentication for all subscription endpoints
3. **Input Validation**: Validate all input parameters
4. **Audit Logging**: Log all subscription changes and token usage
5. **Data Privacy**: Ensure compliance with data protection regulations

## Testing

Run the included tests to verify the implementation:

```bash
# Test subscription service
python -m pytest tests/test_subscription_service.py

# Test onboarding endpoints
python -m pytest tests/test_onboarding_api.py

# Test token middleware
python -m pytest tests/test_token_middleware.py
```

## Troubleshooting

### Common Issues

1. **"Subscriptions table not found"**

   - Run the database migration script
   - Check Supabase connection

2. **"Insufficient tokens" errors**

   - Check current usage with `/subscription/{entity_type}/{entity_id}`
   - Verify token estimation is accurate

3. **Email already exists**

   - Check both users and organizations tables
   - Implement proper email validation

4. **Token consumption fails**
   - Verify entity_id and entity_type are correct
   - Check subscription status is 'active'

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("app.services.subscription").setLevel(logging.DEBUG)
```

This will provide detailed logs of all subscription operations.
