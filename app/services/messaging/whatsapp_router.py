"""WhatsApp message router - bridges webhooks to chat service."""

import logging
from typing import Any, Dict, List, Optional

from app.models.subscription import Channel
from app.services.chat.chat_service import ChatService
from app.services.chat.conversation_manager import ConversationManager
from app.services.storage.supabase_client import get_supabase_client

from .provider_factory import ProviderFactory
from .whatsapp_formatter import WhatsAppFormatter

logger = logging.getLogger(__name__)


def _mask_phone_number(phone_number: Optional[str]) -> str:
    """
    Mask phone number for logging to protect PII.

    Args:
        phone_number: Phone number in E.164 format (e.g., +1234567890)

    Returns:
        Masked phone number (e.g., +1234***7890)
    """
    if not phone_number:
        return "unknown"
    # Keep first 4 and last 4 digits, mask the rest
    if len(phone_number) > 8:
        return phone_number[:4] + "***" + phone_number[-4:]
    return "***" + phone_number[-4:] if len(phone_number) > 4 else "***"


class WhatsAppRouter:
    """Routes WhatsApp messages to chat service and sends responses back."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.formatter = WhatsAppFormatter()

    async def process_incoming_message(
        self,
        phone_number: str,
        message_body: str,
        media_urls: Optional[List[str]] = None,
        org_id: Optional[str] = None,
        chatbot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming WhatsApp message and generate response.

        Args:
            phone_number: Sender phone number (E.164 format)
            message_body: Message text content
            media_urls: Optional list of media URLs
            org_id: Organization ID (if known, otherwise will be determined)
            chatbot_id: Optional chatbot ID to route to specific chatbot

        Returns:
            Dict with response message and metadata
        """
        try:
            # Determine org_id if not provided or validate if provided
            # Critical: Always validate org_id to prevent incorrect org lookup
            if not org_id:
                org_id = await self._get_org_id_from_phone(phone_number)
                if not org_id:
                    masked_phone = _mask_phone_number(phone_number)
                    logger.warning(
                        "Could not determine org_id for phone number %s", masked_phone
                    )
                    return {
                        "success": False,
                        "error": "Organization not found for this phone number",
                    }
            else:
                # Validate provided org_id exists and is active
                if not await self._validate_org_id(org_id):
                    masked_phone = _mask_phone_number(phone_number)
                    logger.warning(
                        "Invalid or inactive org_id %s provided for phone number %s",
                        org_id,
                        masked_phone,
                    )
                    return {
                        "success": False,
                        "error": "Invalid organization configuration",
                    }

            # Get chatbot for organization
            if not chatbot_id:
                chatbot_id = await self._get_default_chatbot(org_id)
                if not chatbot_id:
                    logger.warning("No active chatbot found for org %s", org_id)
                    return {
                        "success": False,
                        "error": "No active chatbot configured",
                    }

            # Get chatbot config
            chatbot_config = await self._get_chatbot_config(org_id, chatbot_id)
            if not chatbot_config:
                return {"success": False, "error": "Chatbot not found"}

            # Create session ID for WhatsApp (phone number-based)
            session_id = f"whatsapp:{phone_number}"

            # Initialize chat service
            chat_service = ChatService(
                org_id=org_id,
                chatbot_config=chatbot_config,
                entity_id=org_id,
                entity_type="organization",
            )

            # Process message through chat service with channel
            result = await chat_service.chat(
                message=message_body,
                session_id=session_id,
                chatbot_id=chatbot_id,
                channel=Channel.WHATSAPP,
            )

            # Format response for WhatsApp
            response_text = self.formatter.format_message(
                result.get("response", ""), max_length=1600
            )

            # Get provider to send response
            provider = await ProviderFactory.get_provider(org_id)
            if not provider:
                logger.error("No messaging provider available for org %s", org_id)
                return {
                    "success": False,
                    "error": "Messaging provider not configured",
                }

            # Send response via provider
            # Critical: media_urls parameter is for incoming media from sender,
            # not for outgoing response. Do not pass incoming media to send_message.
            # If response needs media, it should be generated separately.
            send_result = await provider.send_message(
                to=phone_number,
                message=response_text,
                media_urls=None,  # Response is text-only, incoming media not forwarded
            )

            return {
                "success": True,
                "response": response_text,
                "message_id": send_result.get("message_id"),
                "conversation_id": result.get("conversation_id"),
            }

        except Exception as e:
            # Avoid leaking exception details in response - use generic error message
            # Log full details internally for debugging
            masked_phone = (
                _mask_phone_number(phone_number) if phone_number else "unknown"
            )
            logger.error(
                "Failed to process WhatsApp message",
                extra={
                    "phone_number_masked": masked_phone,
                    "org_id": org_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            # Return generic error message to prevent information leakage
            return {
                "success": False,
                "error": "An error occurred while processing your message. Please try again.",
            }

    async def _get_org_id_from_phone(self, phone_number: str) -> Optional[str]:
        """
        Get organization ID from phone number.

        Note: This method receives the sender's phone number, but org lookup
        should be based on the receiving Twilio phone number (the "To" field).
        The webhook handler should determine org_id using get_provider_for_phone_number
        with the receiving phone number before calling this method.

        Args:
            phone_number: Phone number (sender's number - may not be reliable for org lookup)

        Returns:
            Organization ID or None if not found
        """
        try:
            # Note: This lookup by sender phone is not ideal - org should be determined
            # by the receiving Twilio phone number. This is a fallback.
            # First try exact match
            response = (
                self.supabase.table("whatsapp_configurations")
                .select("org_id")
                .eq("twilio_phone_number", phone_number)
                .eq("is_active", True)
                .execute()
            )

            if response.data and len(response.data) > 0:
                return response.data[0]["org_id"]

            # If using platform phone number, we need additional routing logic
            # For now, if platform phone number matches env var, we can't determine org
            # This would require phone number routing table or other logic
            # Return None to indicate we couldn't determine org
            masked_phone = _mask_phone_number(phone_number)
            logger.warning(
                "No org found for phone number %s - may be using platform phone without org routing",
                masked_phone,
            )
            return None

        except Exception as e:
            masked_phone = _mask_phone_number(phone_number)
            logger.error(
                "Failed to get org_id from phone",
                extra={
                    "phone_number_masked": masked_phone,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return None

    async def _get_default_chatbot(self, org_id: str) -> Optional[str]:
        """Get default active chatbot for organization."""
        try:
            response = (
                self.supabase.table("chatbots")
                .select("id")
                .eq("org_id", org_id)
                .eq("chain_status", "active")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if response.data and len(response.data) > 0:
                return response.data[0]["id"]

            return None

        except Exception as e:
            logger.error(
                "Failed to get default chatbot",
                extra={"org_id": org_id, "error_type": type(e).__name__},
                exc_info=True,
            )
            return None

    async def _get_chatbot_config(self, org_id: str, chatbot_id: str) -> Optional[Dict]:
        """Get chatbot configuration."""
        try:
            response = (
                self.supabase.table("chatbots")
                .select("*")
                .eq("id", chatbot_id)
                .eq("org_id", org_id)
                .execute()
            )

            if response.data and len(response.data) > 0:
                return response.data[0]

            return None

        except Exception as e:
            logger.error(
                "Failed to get chatbot config",
                extra={
                    "org_id": org_id,
                    "chatbot_id": chatbot_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return None

    async def _validate_org_id(self, org_id: str) -> bool:
        """
        Validate that org_id exists and has active WhatsApp configuration.

        Args:
            org_id: Organization ID to validate

        Returns:
            True if org_id is valid and active, False otherwise
        """
        try:
            response = (
                self.supabase.table("whatsapp_configurations")
                .select("org_id")
                .eq("org_id", org_id)
                .eq("is_active", True)
                .execute()
            )
            return response.data and len(response.data) > 0
        except Exception as e:
            logger.error(
                "Failed to validate org_id",
                extra={"org_id": org_id, "error_type": type(e).__name__},
                exc_info=True,
            )
            return False
