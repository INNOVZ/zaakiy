# WhatsApp Business API Integration with Twilio

This guide explains how to integrate WhatsApp Business API using Twilio with your ZaaKy AI Platform.

## Overview

The WhatsApp integration allows your chatbots to:

- Receive incoming WhatsApp messages from customers
- Send automated responses via WhatsApp
- Track token usage per WhatsApp conversation
- Integrate seamlessly with your existing subscription system

## Prerequisites

1. **Twilio Account**: Sign up at [twilio.com](https://www.twilio.com)
2. **WhatsApp Business API Access**: Request access from Twilio
3. **Twilio Phone Number**: Get a WhatsApp-enabled phone number from Twilio
4. **Database Tables**: Ensure `whatsapp_configurations` table exists (from schema)

## Database Schema

The integration uses the following database tables:

### `whatsapp_configurations`

Stores Twilio credentials and configuration per organization:

- `id` (UUID)
- `org_id` (UUID) - Organization ID
- `provider_type` (text) - "twilio"
- `twilio_account_sid` (text) - Twilio Account SID
- `twilio_auth_token` (text) - Twilio Auth Token (encrypted in production)
- `twilio_phone_number` (text) - WhatsApp-enabled phone number
- `webhook_url` (text) - Webhook URL for receiving messages
- `is_active` (bool) - Whether configuration is active
- `created_at`, `updated_at` (timestamptz)

### `token_usage_logs`

Tracks token consumption per WhatsApp message:

- `subscription_id` (UUID)
- `tokens_consumed` (int)
- `channel` (varchar) - "whatsapp"
- `chatbot_id` (UUID)
- `session_id` (varchar) - WhatsApp phone number
- `user_identifier` (varchar) - Customer phone number

## Setup Instructions

### 1. Install Dependencies

The Twilio SDK is already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configure Twilio Webhook

1. Log into your Twilio Console
2. Navigate to Phone Numbers → Manage → Active Numbers
3. Select your WhatsApp-enabled number
4. Under "Messaging Configuration", set the webhook URL to:
   ```
   https://your-domain.com/api/whatsapp/webhook
   ```
5. Set HTTP method to `POST`
6. Save the configuration

### 3. Configure WhatsApp in Your Application

#### Using the API Endpoint

```bash
POST /api/whatsapp/config
Authorization: Bearer <your-jwt-token>
Content-Type: application/json

{
  "twilio_account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "twilio_auth_token": "your_auth_token_here",
  "twilio_phone_number": "+1234567890",
  "webhook_url": "https://your-domain.com/api/whatsapp/webhook",
  "is_active": true
}
```

#### Response:

```json
{
  "success": true,
  "message": "WhatsApp configuration created",
  "config_id": "uuid-here"
}
```

### 4. Validate Configuration

```bash
GET /api/whatsapp/validate
Authorization: Bearer <your-jwt-token>
```

This will test the Twilio connection and return validation results.

## API Endpoints

### Webhook Endpoint (Public)

**POST** `/api/whatsapp/webhook`

This endpoint receives incoming WhatsApp messages from Twilio. It's automatically called by Twilio when a message is received.

**Note**: This endpoint should be publicly accessible and doesn't require authentication (Twilio validates via signature).

### Configuration Endpoints (Authenticated)

#### Get Configuration

**GET** `/api/whatsapp/config`

- Returns current WhatsApp configuration for your organization
- Requires authentication

#### Update Configuration

**POST** `/api/whatsapp/config`

- Creates or updates WhatsApp configuration
- Requires authentication
- Body: `WhatsAppConfigRequest`

#### Validate Configuration

**GET** `/api/whatsapp/validate`

- Tests Twilio connection and validates configuration
- Requires authentication

### Send Message Endpoint (Authenticated)

**POST** `/api/whatsapp/send`

- Sends a WhatsApp message to a customer
- Requires authentication
- Body: `WhatsAppSendRequest`

Example:

```json
{
  "to": "+1234567890",
  "message": "Hello! How can I help you today?",
  "chatbot_id": "optional-chatbot-id"
}
```

## How It Works

### Incoming Messages Flow

1. Customer sends WhatsApp message → Twilio
2. Twilio calls `/api/whatsapp/webhook` with message data
3. Webhook handler:
   - Identifies organization by Twilio Account SID
   - Finds active chatbot for the organization
   - Processes message through ChatService
   - Generates AI response
   - Sends response back via Twilio
   - Logs token usage

### Outgoing Messages Flow

1. Application calls `/api/whatsapp/send`
2. WhatsAppService:
   - Validates organization configuration
   - Checks token availability
   - Sends message via Twilio API
   - Consumes tokens from subscription
   - Logs message in database

## Token Consumption

WhatsApp messages consume tokens from your subscription:

- **Token Multiplier**: 1.2x (WhatsApp has slightly higher cost)
- **Rate Limit**: 30 messages/minute (configurable per plan)
- **Message Length**: Max 1600 characters (WhatsApp limit)

Token consumption is tracked in:

- `token_usage_logs` table
- `channel_usage_analytics` table (daily aggregates)
- Subscription `tokens_used_this_month` counter

## Integration with Chat Service

The WhatsApp integration seamlessly integrates with your existing chat service:

1. **Uses Same Chatbots**: WhatsApp messages are processed by the same chatbots configured in your dashboard
2. **Same AI Models**: Uses the same OpenAI models and configurations
3. **Context Preservation**: Maintains conversation context per WhatsApp number
4. **Document Retrieval**: Uses the same RAG system and document knowledge base

## Session Management

WhatsApp conversations use phone numbers as session identifiers:

- Format: `whatsapp_{phone_number}` (e.g., `whatsapp_1234567890`)
- Each phone number maintains its own conversation history
- Context is preserved across multiple messages

## Error Handling

The service handles various error scenarios:

- **Invalid Configuration**: Returns clear error messages
- **Insufficient Tokens**: Returns 402 Payment Required with usage details
- **Twilio API Errors**: Logs errors and returns user-friendly messages
- **Chatbot Not Found**: Falls back gracefully or returns error

## Security Considerations

1. **Webhook Validation**: Implement Twilio signature validation in production
2. **Auth Token Storage**: Store Twilio auth tokens encrypted in production
3. **Rate Limiting**: WhatsApp endpoints are rate-limited
4. **Input Validation**: All phone numbers and messages are validated
5. **Token Limits**: Enforced per subscription plan

## Testing

### Test Webhook Locally

Use ngrok or similar tool to expose local server:

```bash
ngrok http 8001
```

Then set Twilio webhook URL to:

```
https://your-ngrok-url.ngrok.io/api/whatsapp/webhook
```

### Test Message Sending

```bash
curl -X POST http://localhost:8001/api/whatsapp/send \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "to": "+1234567890",
    "message": "Test message",
    "chatbot_id": "your-chatbot-id"
  }'
```

## Troubleshooting

### Messages Not Received

1. Check Twilio webhook URL is correct
2. Verify webhook endpoint is publicly accessible
3. Check Twilio console for webhook delivery logs
4. Verify `is_active` is `true` in database

### Messages Not Sent

1. Validate Twilio credentials are correct
2. Check token availability in subscription
3. Verify phone number format (E.164: +1234567890)
4. Check Twilio account status and balance

### Token Consumption Issues

1. Verify subscription is active
2. Check `channel_configurations` table for WhatsApp settings
3. Review `token_usage_logs` for consumption records
4. Ensure entity_id and entity_type are correct

## Production Checklist

- [ ] Enable Twilio signature validation
- [ ] Encrypt Twilio auth tokens in database
- [ ] Set up proper webhook URL (HTTPS)
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerts
- [ ] Test with real WhatsApp numbers
- [ ] Configure error notifications
- [ ] Set up token usage alerts

## Support

For issues or questions:

1. Check Twilio console for API errors
2. Review application logs for detailed errors
3. Verify database configuration is correct
4. Test with Twilio's test credentials first

## Next Steps

1. Configure your Twilio account
2. Set up webhook URL
3. Test with a WhatsApp number
4. Monitor token usage
5. Configure channel-specific settings in subscription dashboard
