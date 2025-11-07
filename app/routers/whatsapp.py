"""WhatsApp webhook router for receiving messages from Twilio."""

import logging
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from twilio.twiml.messaging_response import MessagingResponse

from ..models.whatsapp import WhatsAppWebhookRequest
from ..services.auth.user_auth import get_user_with_org
from ..services.messaging.provider_factory import ProviderFactory
from ..services.messaging.whatsapp_router import WhatsAppRouter
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


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


router = APIRouter()


@router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Receive incoming WhatsApp messages from Twilio.

    This endpoint:
    1. Validates Twilio webhook signature
    2. Parses incoming message
    3. Routes to chat service
    4. Sends response back via Twilio

    Note: Twilio webhook validation is done via POST request with signature validation.
    """
    try:
        # Fix request body consumption issue: Read body once and parse manually
        # Twilio sends form-encoded data, so we need to parse it from the raw body
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")

        # Parse form-encoded data manually (since body is already consumed)
        # Format: key1=value1&key2=value2
        parsed_qs = parse_qs(body_str, keep_blank_values=True)
        # Convert to dict with single values (parse_qs returns lists)
        payload = {k: v[0] if v else "" for k, v in parsed_qs.items()}

        # Get webhook signature from headers
        signature = request.headers.get("X-Twilio-Signature", "")

        # Get webhook URL
        webhook_url = str(request.url)

        # Get provider to validate webhook
        # We need to determine which org this is for by looking at the "To" field
        # (the Twilio phone number that received the message)
        to_number = payload.get("To", "")
        if to_number.startswith("whatsapp:"):
            to_number = to_number.replace("whatsapp:", "")

        # Get provider and org_id based on the receiving phone number
        provider, org_id = await ProviderFactory.get_provider_for_phone_number(
            to_number
        )

        if not provider:
            masked_to = _mask_phone_number(to_number)
            logger.warning("No provider found for phone number %s", masked_to)
            # Return TwiML response indicating error
            response = MessagingResponse()
            response.message("Sorry, WhatsApp service is not configured.")
            return response.to_xml()

        # Validate webhook signature
        is_valid = await provider.validate_webhook(body_str, signature, webhook_url)
        if not is_valid:
            logger.warning("Invalid Twilio webhook signature")
            raise HTTPException(status_code=403, detail="Invalid webhook signature")

        # Parse webhook payload
        parsed_data = provider.parse_webhook(payload)

        phone_number = parsed_data["phone_number"]
        message_body = parsed_data["message_body"]
        media_urls = parsed_data.get("media_urls", [])

        if not message_body and not media_urls:
            masked_phone = _mask_phone_number(phone_number)
            logger.warning("Empty message received from %s", masked_phone)
            # Return empty response
            response = MessagingResponse()
            return response.to_xml()

        # Privacy: Mask phone number in logs
        masked_phone = _mask_phone_number(phone_number)
        logger.info(
            "Received WhatsApp message",
            extra={
                "phone_number_masked": masked_phone,
                "message_length": len(message_body),
                "has_media": len(media_urls) > 0,
                "org_id": org_id,
            },
        )

        # Route message through WhatsApp router
        router_service = WhatsAppRouter()
        result = await router_service.process_incoming_message(
            phone_number=phone_number,
            message_body=message_body,
            media_urls=media_urls if media_urls else None,
            org_id=org_id,
        )

        if not result.get("success"):
            masked_phone = _mask_phone_number(phone_number)
            # Log error internally but don't expose details to user
            # Note: exc_info=True only used in exception handlers, not for regular error logging
            logger.error(
                "Failed to process WhatsApp message",
                extra={
                    "phone_number_masked": masked_phone,
                    "org_id": org_id,
                    "error_type": "processing_error",
                },
            )

            # Avoid exposing internal error details to users - use generic message
            response = MessagingResponse()
            response.message("Sorry, I encountered an error. Please try again.")
            return response.to_xml()

        # Success - TwiML response is not needed since we already sent via API
        # But we return a minimal response to acknowledge receipt
        response = MessagingResponse()
        return response.to_xml()

    except HTTPException:
        raise
    except Exception as e:
        # Avoid exposing internal error details - log internally only
        logger.error(
            "Error processing WhatsApp webhook",
            extra={
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        # Return generic error response
        response = MessagingResponse()
        response.message("Sorry, I encountered an error. Please try again later.")
        return response.to_xml()
