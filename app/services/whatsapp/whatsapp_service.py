"""
WhatsApp Business API Service using Twilio
Handles sending and receiving WhatsApp messages via Twilio
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioClient

from ...models.subscription import Channel
from ..auth.billing_middleware import TokenValidationMiddleware
from ..storage.supabase_client import get_supabase_client
from .whatsapp_config_cache import get_config_cache
from .whatsapp_rate_limiter import RateLimitExceeded, WhatsAppRateLimiter

logger = logging.getLogger(__name__)


class WhatsAppServiceError(Exception):
    """Base exception for WhatsApp service errors"""


class WhatsAppConfigurationError(WhatsAppServiceError):
    """Exception for WhatsApp configuration errors"""


class WhatsAppService:
    """Service for handling WhatsApp Business API via Twilio"""

    def __init__(self, org_id: str, redis_client=None):
        """
        Initialize WhatsApp service for an organization

        Args:
            org_id: Organization ID
            redis_client: Optional Redis client for rate limiting and caching
        """
        self.org_id = org_id
        self.supabase = get_supabase_client()
        self.token_middleware = TokenValidationMiddleware(self.supabase)
        self._twilio_client: Optional[TwilioClient] = None
        self._config: Optional[Dict[str, Any]] = None

        # Initialize rate limiter
        self.rate_limiter = WhatsAppRateLimiter(redis_client=redis_client)

        # Initialize configuration cache
        self.config_cache = get_config_cache(redis_client=redis_client)

    async def _get_whatsapp_config(self) -> Dict[str, Any]:
        """
        Get WhatsApp configuration from cache or database

        Uses multi-layer caching strategy:
        1. Instance cache (self._config)
        2. Memory/Redis cache (shared across instances via shared cache service)
        3. Database (fallback)
        """
        # Layer 1: Check instance cache (fastest)
        if self._config:
            return self._config

        # Layer 2: Check shared cache (memory + Redis via shared cache service)
        cached_config = await self.config_cache.get(self.org_id)
        if cached_config:
            self._config = cached_config
            logger.debug(
                f"✅ WhatsApp config loaded from cache for org {self.org_id}",
                extra={"org_id": self.org_id, "source": "cache"},
            )
            return self._config

        # Layer 3: Fetch from database (cache miss)
        try:
            logger.debug(
                f"🔍 Fetching WhatsApp config from database for org {self.org_id}",
                extra={"org_id": self.org_id, "source": "database"},
            )

            response = (
                self.supabase.table("whatsapp_configurations")
                .select("*")
                .eq("org_id", self.org_id)
                .eq("is_active", True)
                .execute()
            )

            if not response.data or len(response.data) == 0:
                raise WhatsAppConfigurationError(
                    f"No active WhatsApp configuration found for org {self.org_id}"
                )

            self._config = response.data[0]

            # Store in cache for future requests (async)
            await self.config_cache.set(self.org_id, self._config)

            logger.info(
                f"💾 WhatsApp config fetched from DB and cached for org {self.org_id}",
                extra={"org_id": self.org_id, "source": "database"},
            )

            return self._config

        except Exception as e:
            logger.error(
                "Failed to get WhatsApp configuration",
                extra={"org_id": self.org_id, "error": str(e)},
                exc_info=True,
            )
            raise WhatsAppConfigurationError(
                f"Failed to load WhatsApp config: {e}"
            ) from e

    async def _get_twilio_client(self) -> TwilioClient:
        """Get or create Twilio client"""
        if self._twilio_client:
            return self._twilio_client

        config = await self._get_whatsapp_config()

        account_sid = config.get("twilio_account_sid")
        auth_token = config.get("twilio_auth_token")

        if not account_sid or not auth_token:
            raise WhatsAppConfigurationError(
                "Twilio credentials not configured. Please set account_sid and auth_token."
            )

        try:
            self._twilio_client = TwilioClient(account_sid, auth_token)
            return self._twilio_client
        except Exception as e:
            logger.error(
                "Failed to initialize Twilio client",
                extra={"org_id": self.org_id, "error": str(e)},
                exc_info=True,
            )
            raise WhatsAppConfigurationError(
                f"Failed to initialize Twilio client: {e}"
            ) from e

    async def send_message(
        self,
        to: str,
        message: str,
        chatbot_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        requesting_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send WhatsApp message via Twilio

        Args:
            to: Recipient WhatsApp number (E.164 format, e.g., +1234567890)
            message: Message content (max 1600 chars for WhatsApp)
            chatbot_id: Optional chatbot ID for tracking
            session_id: Optional session ID for conversation tracking
            entity_id: Entity ID for token consumption (user or org)
            entity_type: Entity type ('user' or 'organization')

        Returns:
            Dict with message details and status
        """
        try:
            # Ensure we actually resolve the async config fetch
            config = await self._get_whatsapp_config()
            from_number = config.get("twilio_phone_number")

            if not from_number:
                raise WhatsAppConfigurationError("Twilio phone number not configured")

            # Validate message length (WhatsApp limit is 1600 chars)
            if len(message) > 1600:
                message = message[:1597] + "..."

            # Validate phone number format (basic check)
            if not to.startswith("+"):
                raise ValueError(
                    "Phone number must be in E.164 format (e.g., +1234567890)"
                )

            # Get Twilio client
            client = await self._get_twilio_client()

            # Send message via Twilio WhatsApp API
            twilio_message = client.messages.create(
                body=message,
                from_=f"whatsapp:{from_number}",
                to=f"whatsapp:{to}",
            )

            # Estimate tokens consumed (message length + response)
            estimated_tokens = len(message.split()) * 1.3  # Rough estimate
            if entity_id and entity_type:
                # Consume tokens if entity info provided
                requester = requesting_user_id or entity_id
                try:
                    await self.token_middleware.validate_and_consume_tokens(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        estimated_tokens=int(estimated_tokens),
                        requesting_user_id=requester,
                        operation_type="whatsapp_message_sent",
                        channel=Channel.WHATSAPP,
                        chatbot_id=chatbot_id,
                        session_id=session_id,
                        user_identifier=to,
                    )
                except Exception as token_error:
                    logger.warning(
                        "Failed to consume tokens for WhatsApp message",
                        extra={"error": str(token_error)},
                    )

            # Log message in database
            await self._log_message(
                from_number=from_number,
                customer_number=to,
                message=message,
                twilio_sid=twilio_message.sid,
                chatbot_id=chatbot_id,
                session_id=session_id,
                direction="outbound",
                tokens_consumed=int(estimated_tokens),
            )

            logger.info(
                "WhatsApp message sent successfully",
                extra={
                    "org_id": self.org_id,
                    "to": to,
                    "message_sid": twilio_message.sid,
                    "chatbot_id": chatbot_id,
                },
            )

            return {
                "success": True,
                "message_sid": twilio_message.sid,
                "status": twilio_message.status,
                "to": to,
                "from": from_number,
                "message": message,
            }

        except TwilioRestException as e:
            logger.error(
                "Twilio API error sending WhatsApp message",
                extra={
                    "org_id": self.org_id,
                    "to": to,
                    "error_code": e.code,
                    "error_message": e.msg,
                },
                exc_info=True,
            )
            raise WhatsAppServiceError(f"Twilio API error: {e.msg}") from e

        except Exception as e:
            logger.error(
                "Failed to send WhatsApp message",
                extra={"org_id": self.org_id, "to": to, "error": str(e)},
                exc_info=True,
            )
            raise WhatsAppServiceError(f"Failed to send message: {e}") from e

    async def _log_message(
        self,
        *,
        customer_number: str,
        message: str,
        twilio_sid: str,
        direction: str,
        from_number: Optional[str] = None,
        chatbot_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tokens_consumed: int = 0,
    ):
        """Log WhatsApp message in database"""
        try:
            # Get subscription ID for this org
            subscription_response = (
                self.supabase.table("subscriptions")
                .select("id")
                .eq("entity_id", self.org_id)
                .eq("entity_type", "organization")
                .eq("status", "active")
                .limit(1)
                .execute()
            )

            subscription_id = None
            if subscription_response.data:
                subscription_id = subscription_response.data[0]["id"]

            # Log in token_usage_logs if subscription exists
            if subscription_id:
                log_data = {
                    "subscription_id": subscription_id,
                    "tokens_consumed": max(0, tokens_consumed),
                    "operation_type": "whatsapp_message",
                    "channel": "whatsapp",
                    "chatbot_id": chatbot_id,
                    "session_id": session_id,
                    "user_identifier": customer_number,
                    "metadata": {
                        "twilio_sid": twilio_sid,
                        "from_number": from_number,
                        "direction": direction,
                        "message_length": len(message),
                    },
                }

                self.supabase.table("token_usage_logs").insert(log_data).execute()

        except Exception as e:
            logger.warning(
                "Failed to log WhatsApp message",
                extra={"error": str(e)},
            )

    async def process_incoming_message(
        self,
        from_number: str,
        message_body: str,
        twilio_sid: str,
        chatbot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming WhatsApp message and generate response

        Args:
            from_number: Sender WhatsApp number
            message_body: Message content
            twilio_sid: Twilio message SID
            chatbot_id: Optional chatbot ID to route to

        Returns:
            Dict with response details
        """
        try:
            config = await self._get_whatsapp_config()
            twilio_number = config.get("twilio_phone_number")
            # Get active chatbot for this org
            if not chatbot_id:
                chatbot_response = (
                    self.supabase.table("chatbots")
                    .select("id")
                    .eq("org_id", self.org_id)
                    .eq("chain_status", "active")
                    .limit(1)
                    .execute()
                )

                if chatbot_response.data:
                    chatbot_id = chatbot_response.data[0]["id"]
                else:
                    raise WhatsAppServiceError(
                        f"No active chatbot found for org {self.org_id}"
                    )

            # Process message through chat service
            from ..chat.chat_service import ChatService

            chatbot_config_response = (
                self.supabase.table("chatbots")
                .select("*")
                .eq("id", chatbot_id)
                .execute()
            )

            if not chatbot_config_response.data:
                raise WhatsAppServiceError(f"Chatbot {chatbot_id} not found")

            chatbot_config = chatbot_config_response.data[0]

            # Initialize chat service
            chat_service = ChatService(
                org_id=self.org_id,
                chatbot_config=chatbot_config,
                entity_id=self.org_id,
                entity_type="organization",
            )

            # Generate session ID from phone number
            session_id = f"whatsapp_{from_number.replace('+', '').replace('-', '').replace(' ', '')}"

            # RATE LIMITING: Check if user is within rate limits
            # Strip 'whatsapp:' prefix for rate limiting
            clean_phone = from_number.replace("whatsapp:", "").strip()

            allowed, error_msg, retry_after = self.rate_limiter.check_rate_limit(
                phone_number=clean_phone, org_id=self.org_id
            )

            if not allowed:
                # Rate limit exceeded - send friendly message to user
                logger.warning(f"⚠️ Rate limit exceeded for {clean_phone}: {error_msg}")

                # Send rate limit message to user
                friendly_message = (
                    f"Hey! You're sending messages a bit too quickly 😅\n\n"
                    f"Please wait {retry_after} seconds before sending another message.\n\n"
                    f"This helps us provide the best service to everyone! 🙏"
                )

                try:
                    await self.send_message(
                        to=clean_phone,
                        message=friendly_message,
                        chatbot_id=chatbot_id,
                        session_id=session_id,
                        entity_id=self.org_id,
                        entity_type="organization",
                        requesting_user_id=self.org_id,
                    )
                except Exception as send_error:
                    logger.error(f"Failed to send rate limit message: {send_error}")

                return {
                    "success": False,
                    "error": "rate_limit_exceeded",
                    "message": error_msg,
                    "retry_after": retry_after,
                }

            # Log incoming message after determining session
            await self._log_message(
                customer_number=from_number,
                from_number=twilio_number,
                message=message_body,
                twilio_sid=twilio_sid,
                chatbot_id=chatbot_id,
                session_id=session_id,
                direction="inbound",
                tokens_consumed=0,
            )

            # IMMEDIATE ACKNOWLEDGMENT: Send quick response for better UX
            # This makes the bot feel responsive (< 2 seconds) while processing the full answer

            # Determine appropriate acknowledgment message based on query
            ack_messages = [
                "Let me check that for you... ",
                "Looking that up for you... ",
                "One moment please... ",
                "Checking our knowledge base... ",
            ]

            # Use different message based on message length/complexity
            if len(message_body) > 100:
                ack_message = "Let me look into that for you... "
            elif any(
                word in message_body.lower()
                for word in ["price", "pricing", "cost", "plan"]
            ):
                ack_message = "Checking pricing for you... "
            elif any(
                word in message_body.lower()
                for word in ["contact", "phone", "email", "reach"]
            ):
                ack_message = "Getting contact details..."
            elif any(
                word in message_body.lower()
                for word in ["demo", "book", "schedule", "appointment"]
            ):
                ack_message = "Finding booking options for you... 📅"
            else:
                ack_message = "Let me check that for you... "

            # Send immediate acknowledgment (don't wait for it to complete)
            # This runs in the background while we process the full response
            asyncio.create_task(
                self.send_message(
                    to=clean_phone,
                    message=ack_message,
                    chatbot_id=chatbot_id,
                    session_id=session_id,
                    entity_id=self.org_id,
                    entity_type="organization",
                    requesting_user_id=self.org_id,
                )
            )

            logger.info(
                f"⚡ Sent immediate acknowledgment to {clean_phone}: '{ack_message}'"
            )

            # Process message (this takes time - 5-20 seconds)
            chat_response = await chat_service.process_message(
                message=message_body,
                session_id=session_id,
                channel=Channel.WHATSAPP,
                end_user_identifier=from_number,
                requesting_user_id=self.org_id,
            )

            response_text = chat_response.get(
                "response", "I'm sorry, I couldn't process that message."
            )

            # WHATSAPP FORMATTING: Structure response for better readability
            # Import formatter
            from .whatsapp_response_formatter import format_whatsapp_response

            # Get intent and context for better formatting
            intent_result = chat_response.get("intent")
            intent_type = None
            if intent_result and isinstance(intent_result, dict):
                intent_type = intent_result.get("primary_intent")

            # Get context data for rich formatting (contact info, products, etc.)
            context_data = {
                "contact_info": chat_response.get("contact_info", {}),
                "product_links": chat_response.get("product_links", []),
                "sources": chat_response.get("sources", []),
            }

            # Format response for WhatsApp
            formatted_response = format_whatsapp_response(
                response_text, context_data=context_data, intent_type=intent_type
            )

            logger.info(
                f"📱 Formatted WhatsApp response (intent: {intent_type}, "
                f"original_length: {len(response_text)}, "
                f"formatted_length: {len(formatted_response)})"
            )

            # Strip 'whatsapp:' prefix from phone number if present
            # Twilio sends numbers as 'whatsapp:+1234567890', we need just '+1234567890'
            clean_phone_number = from_number.replace("whatsapp:", "").strip()

            # Send formatted response back via WhatsApp
            send_result = await self.send_message(
                to=clean_phone_number,
                message=formatted_response,  # Use formatted response
                chatbot_id=chatbot_id,
                session_id=session_id,
                entity_id=self.org_id,
                entity_type="organization",
                requesting_user_id=self.org_id,
            )

            return {
                "success": True,
                "response_sent": True,
                "response_text": response_text,
                "message_sid": send_result.get("message_sid"),
                "chat_response": chat_response,
            }

        except Exception as e:
            logger.error(
                "Failed to process incoming WhatsApp message",
                extra={
                    "org_id": self.org_id,
                    "from_number": from_number,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise WhatsAppServiceError(f"Failed to process message: {e}") from e

    async def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate WhatsApp configuration

        Returns:
            Dict with validation status and details
        """
        try:
            config = await self._get_whatsapp_config()
            client = await self._get_twilio_client()

            # Test Twilio connection by fetching account info
            account = client.api.accounts(config.get("twilio_account_sid")).fetch()

            return {
                "valid": True,
                "account_status": account.status,
                "phone_number": config.get("twilio_phone_number"),
                "is_active": config.get("is_active"),
            }

        except Exception as e:
            logger.error(
                "WhatsApp configuration validation failed",
                extra={"org_id": self.org_id, "error": str(e)},
            )
            return {
                "valid": False,
                "error": str(e),
            }

    async def invalidate_config_cache(self) -> None:
        """
        Invalidate cached configuration for this organization

        Should be called when:
        - WhatsApp configuration is updated
        - Twilio credentials are changed
        - Configuration is deactivated
        """
        # Clear instance cache
        self._config = None
        self._twilio_client = None

        # Clear shared cache (async)
        await self.config_cache.invalidate(self.org_id)

        logger.info(
            f"♻️ WhatsApp config cache invalidated for org {self.org_id}",
            extra={"org_id": self.org_id},
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics

        Returns:
            Dictionary with cache hit/miss rates and performance metrics
        """
        return self.config_cache.get_stats()
