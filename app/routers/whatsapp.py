"""
WhatsApp Business API Router
Handles WhatsApp webhooks and message management via Twilio
"""
import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from twilio.request_validator import RequestValidator  # type: ignore

from ..services.auth import get_user_with_org, verify_jwt_token_from_header
from ..services.storage.supabase_client import get_supabase_client
from ..services.whatsapp import WhatsAppService, WhatsAppServiceError
from ..utils.logging_config import get_logger
from ..utils.rate_limiter import get_rate_limit_config, rate_limit

logger = get_logger(__name__)

router = APIRouter()


class WhatsAppConfigRequest(BaseModel):
    """Request model for WhatsApp configuration"""

    twilio_account_sid: str = Field(..., description="Twilio Account SID")
    twilio_auth_token: str = Field(..., description="Twilio Auth Token")
    twilio_phone_number: str = Field(
        ..., description="Twilio WhatsApp-enabled phone number"
    )
    webhook_url: Optional[str] = Field(
        None, description="Webhook URL for receiving messages"
    )
    is_active: bool = Field(True, description="Whether the configuration is active")


class WhatsAppSendRequest(BaseModel):
    """Request model for sending WhatsApp message"""

    to: str = Field(..., description="Recipient WhatsApp number (E.164 format)")
    message: str = Field(..., description="Message content (max 1600 chars)")
    chatbot_id: Optional[str] = Field(None, description="Chatbot ID for routing")


class WhatsAppSendResponse(BaseModel):
    """Response model for sending WhatsApp message"""

    success: bool
    message_sid: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


async def get_org_id_from_user(user: Dict[str, str]) -> str:
    """Extract org_id from authenticated user"""
    try:
        user_with_org = await get_user_with_org(user["user_id"])
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found") from None
    except Exception as exc:
        logger.error(
            "Failed to load user data: %s", exc, extra={"user_id": user["user_id"]}
        )
        raise HTTPException(
            status_code=500, detail="Failed to load user information"
        ) from exc

    org_id = user_with_org.get("org_id") or user_with_org.get("organization", {}).get(
        "id"
    )
    if not org_id:
        raise HTTPException(
            status_code=400,
            detail="User is not associated with an organization",
        )

    return org_id


def _verify_twilio_signature(
    request_url: str,
    form_data: Dict[str, str],
    signature: Optional[str],
    auth_token: str,
) -> bool:
    """Verify Twilio webhook signature using request validator."""
    if not signature:
        logger.warning(
            "Missing Twilio signature header",
            extra={"event": "twilio_signature_missing"},
        )
        return False

    validator = RequestValidator(auth_token)
    return validator.validate(request_url, form_data, signature)


@router.post("/webhook", tags=["whatsapp"])
async def whatsapp_webhook(
    request: Request,
    x_twilio_signature: Optional[str] = Header(None, alias="X-Twilio-Signature"),
):
    """
    Twilio webhook endpoint for receiving WhatsApp messages.

    This endpoint receives incoming WhatsApp messages from Twilio and processes them
    through the chat service.
    """
    form = await request.form()
    form_dict = {key: value for key, value in form.multi_items()}

    account_sid = form_dict.get("AccountSid")
    from_number = form_dict.get("From")
    message_sid = form_dict.get("MessageSid")
    message_body = form_dict.get("Body", "")

    if not account_sid or not from_number or not message_sid:
        logger.warning(
            "Invalid WhatsApp webhook payload received",
            extra={
                "account_sid": account_sid,
                "from_number": from_number,
                "message_sid": message_sid,
            },
        )
        return {"status": "ignored", "message": "Missing required fields"}

    try:
        supabase = get_supabase_client()

        config_response = (
            supabase.table("whatsapp_configurations")
            .select("org_id, twilio_auth_token")
            .eq("twilio_account_sid", account_sid)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )

        if not config_response.data:
            logger.warning(
                "Received webhook for unknown Twilio account",
                extra={"account_sid": account_sid},
            )
            return {"status": "ignored", "message": "Unknown account"}

        config = config_response.data[0]
        auth_token = config.get("twilio_auth_token")
        if not auth_token:
            logger.error(
                "Twilio configuration missing auth token",
                extra={"org_id": config.get("org_id")},
            )
            return {"status": "error", "message": "Configuration incomplete"}

        request_url = str(request.url)
        if not _verify_twilio_signature(
            request_url, form_dict, x_twilio_signature, auth_token
        ):
            logger.warning(
                "Twilio signature validation failed",
                extra={"org_id": config.get("org_id"), "account_sid": account_sid},
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        org_id = config["org_id"]
        whatsapp_service = WhatsAppService(org_id=org_id)

        result = await whatsapp_service.process_incoming_message(
            from_number=from_number,
            message_body=message_body,
            twilio_sid=message_sid,
        )

        logger.info(
            "WhatsApp webhook processed successfully",
            extra={
                "org_id": org_id,
                "from": from_number,
                "message_sid": message_sid,
            },
        )

        return {
            "status": "processed",
            "response_sent": result.get("response_sent", False),
        }

    except WhatsAppServiceError as exc:
        logger.error(
            "WhatsApp service error processing webhook",
            extra={
                "error": str(exc),
                "from": from_number,
                "message_sid": message_sid,
            },
            exc_info=True,
        )
        return {"status": "error", "message": str(exc)}

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "Unexpected error processing WhatsApp webhook",
            extra={"error": str(exc), "from": from_number},
            exc_info=True,
        )
        return {"status": "error", "message": "Internal server error"}


@router.post("/config", tags=["whatsapp"])
@rate_limit(**get_rate_limit_config("whatsapp"))
async def configure_whatsapp(
    config: WhatsAppConfigRequest,
    user=Depends(verify_jwt_token_from_header),
):
    """
    Configure WhatsApp/Twilio settings for an organization

    Requires authentication and organization membership.
    """
    try:
        org_id = await get_org_id_from_user(user)
        supabase = get_supabase_client()

        # Check if configuration already exists
        existing_response = (
            supabase.table("whatsapp_configurations")
            .select("id")
            .eq("org_id", org_id)
            .execute()
        )

        config_data = {
            "org_id": org_id,
            "provider_type": "twilio",
            "twilio_account_sid": config.twilio_account_sid,
            "twilio_auth_token": config.twilio_auth_token,
            "twilio_phone_number": config.twilio_phone_number,
            "webhook_url": config.webhook_url,
            "is_active": config.is_active,
        }

        if existing_response.data:
            # Update existing configuration
            config_id = existing_response.data[0]["id"]
            response = (
                supabase.table("whatsapp_configurations")
                .update(config_data)
                .eq("id", config_id)
                .execute()
            )

            logger.info(
                "WhatsApp configuration updated",
                extra={"org_id": org_id, "config_id": config_id},
            )

            return {
                "success": True,
                "message": "WhatsApp configuration updated",
                "config_id": config_id,
            }
        else:
            # Create new configuration
            response = (
                supabase.table("whatsapp_configurations").insert(config_data).execute()
            )

            if response.data:
                config_id = response.data[0]["id"]
                logger.info(
                    "WhatsApp configuration created",
                    extra={"org_id": org_id, "config_id": config_id},
                )

                return {
                    "success": True,
                    "message": "WhatsApp configuration created",
                    "config_id": config_id,
                }
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to create WhatsApp configuration"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to configure WhatsApp",
            extra={"error": str(e), "user_id": user.get("user_id")},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to configure WhatsApp: {str(e)}"
        ) from e


@router.get("/config", tags=["whatsapp"])
async def get_whatsapp_config(
    user=Depends(verify_jwt_token_from_header),
):
    """Get WhatsApp configuration for the user's organization"""
    try:
        org_id = await get_org_id_from_user(user)
        supabase = get_supabase_client()

        response = (
            supabase.table("whatsapp_configurations")
            .select("*")
            .eq("org_id", org_id)
            .execute()
        )

        if not response.data:
            return {
                "success": False,
                "message": "No WhatsApp configuration found",
                "config": None,
            }

        config = response.data[0]

        # Don't return sensitive auth token
        config.pop("twilio_auth_token", None)

        return {
            "success": True,
            "config": config,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get WhatsApp configuration",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get WhatsApp configuration"
        ) from e


@router.post("/send", tags=["whatsapp"])
@rate_limit(**get_rate_limit_config("whatsapp"))
async def send_whatsapp_message(
    request: WhatsAppSendRequest,
    user=Depends(verify_jwt_token_from_header),
):
    """
    Send WhatsApp message via Twilio

    Requires authentication and organization membership.
    """
    try:
        org_id = await get_org_id_from_user(user)
        whatsapp_service = WhatsAppService(org_id=org_id)

        result = await whatsapp_service.send_message(
            to=request.to,
            message=request.message,
            chatbot_id=request.chatbot_id,
            entity_id=org_id,
            entity_type="organization",
            requesting_user_id=user["user_id"],
        )

        return WhatsAppSendResponse(
            success=True,
            message_sid=result.get("message_sid"),
            status=result.get("status"),
        )

    except WhatsAppServiceError as e:
        logger.error(
            "WhatsApp service error sending message",
            extra={"error": str(e)},
            exc_info=True,
        )
        return WhatsAppSendResponse(
            success=False,
            error=str(e),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to send WhatsApp message",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to send WhatsApp message: {str(e)}"
        ) from e


@router.get("/validate", tags=["whatsapp"])
async def validate_whatsapp_config(
    user=Depends(verify_jwt_token_from_header),
):
    """Validate WhatsApp configuration and test Twilio connection"""
    try:
        org_id = await get_org_id_from_user(user)
        whatsapp_service = WhatsAppService(org_id=org_id)

        validation_result = whatsapp_service.validate_configuration()

        return {
            "success": validation_result.get("valid", False),
            "validation": validation_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to validate WhatsApp configuration",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to validate WhatsApp configuration"
        ) from e
