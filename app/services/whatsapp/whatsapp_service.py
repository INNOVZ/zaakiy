"""
WhatsApp Business API Service using Twilio
Handles sending and receiving WhatsApp messages via Twilio
"""
import logging
from typing import Any, Dict, Optional

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioClient

from ...models.subscription import Channel
from ..auth.billing_middleware import TokenValidationMiddleware
from ..storage.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class WhatsAppServiceError(Exception):
    """Base exception for WhatsApp service errors"""


class WhatsAppConfigurationError(WhatsAppServiceError):
    """Exception for WhatsApp configuration errors"""


class WhatsAppService:
    """Service for handling WhatsApp Business API via Twilio"""

    def __init__(self, org_id: str):
        """
        Initialize WhatsApp service for an organization

        Args:
            org_id: Organization ID
        """
        self.org_id = org_id
        self.supabase = get_supabase_client()
        self.token_middleware = TokenValidationMiddleware(self.supabase)
        self._twilio_client: Optional[TwilioClient] = None
        self._config: Optional[Dict[str, Any]] = None

    def _get_whatsapp_config(self) -> Dict[str, Any]:
        """Get WhatsApp configuration from database"""
        if self._config:
            return self._config

        try:
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

    def _get_twilio_client(self) -> TwilioClient:
        """Get or create Twilio client"""
        if self._twilio_client:
            return self._twilio_client

        config = self._get_whatsapp_config()

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
            config = self._get_whatsapp_config()
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
            client = self._get_twilio_client()

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
            config = self._get_whatsapp_config()
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

            # Process message
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

            # Send response back via WhatsApp
            send_result = await self.send_message(
                to=from_number,
                message=response_text,
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

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate WhatsApp configuration

        Returns:
            Dict with validation status and details
        """
        try:
            config = self._get_whatsapp_config()
            client = self._get_twilio_client()

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
