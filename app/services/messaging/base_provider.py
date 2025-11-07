"""Base messaging provider interface for WhatsApp integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MessagingProvider(ABC):
    """Abstract base class for messaging providers (Twilio, WhatsApp Business API, etc.)"""

    @abstractmethod
    async def send_message(
        self,
        to: str,
        message: str,
        media_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message via the messaging provider.

        Args:
            to: Recipient phone number (E.164 format)
            message: Message text content
            media_urls: Optional list of media URLs to attach

        Returns:
            Dict with message_id and status
        """
        pass

    @abstractmethod
    async def validate_webhook(
        self, request_body: str, signature: str, url: str
    ) -> bool:
        """
        Validate webhook signature from the messaging provider.

        Args:
            request_body: Raw request body
            signature: Signature from request headers
            url: Webhook URL

        Returns:
            True if signature is valid
        """
        pass

    @abstractmethod
    def parse_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse incoming webhook payload into standardized format.

        Args:
            payload: Raw webhook payload

        Returns:
            Dict with:
                - phone_number: Sender phone number
                - message_body: Message text
                - media_urls: List of media URLs
                - message_id: Provider message ID
                - timestamp: Message timestamp
        """
        pass

    @abstractmethod
    def format_message(self, text: str, max_length: int = 1600) -> str:
        """
        Format message text for the messaging provider.

        Args:
            text: Message text to format
            max_length: Maximum message length

        Returns:
            Formatted message text
        """
        pass

    @abstractmethod
    async def get_message_status(self, message_id: str) -> Dict[str, Any]:
        """
        Get delivery status of a sent message.

        Args:
            message_id: Provider message ID

        Returns:
            Dict with status information
        """
        pass
