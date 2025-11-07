"""Twilio messaging provider implementation."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

from ..base_provider import MessagingProvider

logger = logging.getLogger(__name__)


class TwilioProvider(MessagingProvider):
    """Twilio implementation of MessagingProvider."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        phone_number: str,
    ):
        """
        Initialize Twilio provider.

        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            phone_number: Twilio WhatsApp phone number (E.164 format)
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self.client = TwilioClient(account_sid, auth_token)
        self.validator = RequestValidator(auth_token)
        logger.info("TwilioProvider initialized with account SID: %s", account_sid[:10])

    async def send_message(
        self,
        to: str,
        message: str,
        media_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Send a message via Twilio WhatsApp."""
        try:
            # Ensure phone number is in E.164 format
            if not to.startswith("whatsapp:"):
                to = f"whatsapp:{to}"

            # Format sender phone number
            from_number = (
                self.phone_number
                if self.phone_number.startswith("whatsapp:")
                else f"whatsapp:{self.phone_number}"
            )

            # Send message
            message_kwargs = {
                "body": message,
                "from_": from_number,
                "to": to,
            }

            # Add media if provided
            if media_urls and len(media_urls) > 0:
                # Twilio supports one media URL per message
                message_kwargs["media_url"] = media_urls[0]

            twilio_message = await asyncio.to_thread(
                self.client.messages.create, **message_kwargs
            )

            logger.info(
                "Message sent via Twilio",
                extra={
                    "message_sid": twilio_message.sid,
                    "to": to,
                    "status": twilio_message.status,
                },
            )

            return {
                "message_id": twilio_message.sid,
                "status": twilio_message.status,
                "to": to,
                "from": from_number,
                "timestamp": twilio_message.date_created.isoformat()
                if twilio_message.date_created
                else datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(
                "Failed to send Twilio message",
                extra={"error": str(e), "to": to},
                exc_info=True,
            )
            raise

    async def validate_webhook(
        self, request_body: str, signature: str, url: str
    ) -> bool:
        """Validate Twilio webhook signature."""
        try:
            return await asyncio.to_thread(
                self.validator.validate, url, request_body, signature
            )
        except Exception as e:
            logger.warning("Webhook validation failed: %s", str(e))
            return False

    def parse_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Twilio webhook payload."""
        try:
            # Extract phone number (remove whatsapp: prefix if present)
            from_number = payload.get("From", "")
            if from_number.startswith("whatsapp:"):
                from_number = from_number.replace("whatsapp:", "")

            # Extract message body
            message_body = payload.get("Body", "")
            if not message_body:
                # Check for media messages
                message_body = payload.get("MediaUrl0", "")

            # Extract media URLs
            media_urls = []
            num_media_raw = payload.get("NumMedia", 0)
            try:
                num_media = int(num_media_raw)
            except (TypeError, ValueError):
                logger.warning("Invalid NumMedia value: %s", num_media_raw)
                num_media = 0
            for i in range(num_media):
                media_url = payload.get(f"MediaUrl{i}", "")
                if media_url:
                    media_urls.append(media_url)

            return {
                "phone_number": from_number,
                "message_body": message_body,
                "media_urls": media_urls,
                "message_id": payload.get("MessageSid", ""),
                "timestamp": payload.get("DateSent", ""),
                "account_sid": payload.get("AccountSid", ""),
                "raw_payload": payload,
            }

        except Exception as e:
            logger.error("Failed to parse Twilio webhook: %s", str(e), exc_info=True)
            raise

    def format_message(self, text: str, max_length: int = 1600) -> str:
        """
        Format message for WhatsApp (truncate if needed).

        WhatsApp supports up to 4096 characters, but we use 1600 as default
        for better compatibility and cost management.
        """
        if len(text) > max_length:
            truncated = text[: max_length - 3] + "..."
            logger.debug(
                "Message truncated from %d to %d characters", len(text), len(truncated)
            )
            return truncated
        return text

    async def get_message_status(self, message_id: str) -> Dict[str, Any]:
        """Get message delivery status from Twilio."""
        try:

            def _fetch_message():
                return self.client.messages(message_id).fetch()

            message = await asyncio.to_thread(_fetch_message)
            return {
                "message_id": message.sid,
                "status": message.status,
                "date_sent": message.date_sent.isoformat()
                if message.date_sent
                else None,
                "error_code": message.error_code,
                "error_message": message.error_message,
            }
        except Exception as e:
            logger.error(
                "Failed to get Twilio message status: %s", str(e), exc_info=True
            )
            raise
