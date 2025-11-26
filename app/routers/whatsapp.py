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
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data.get("org_id")

        if not org_id:
            logger.warning(
                "User has no organization", extra={"user_id": user["user_id"]}
            )
            raise HTTPException(
                status_code=400,
                detail="User is not associated with an organization. Please contact support.",
            )

        return org_id
    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(
            "User not found", extra={"user_id": user.get("user_id"), "error": str(exc)}
        )
        raise HTTPException(status_code=404, detail="User not found") from exc
    except Exception as exc:
        logger.error(
            "Failed to load user data",
            extra={"user_id": user.get("user_id"), "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to load user information: {str(exc)}"
        ) from exc


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


@router.get("/webhook", tags=["whatsapp"])
async def whatsapp_webhook_verification():
    """
    GET endpoint for WhatsApp webhook verification.

    Twilio may send GET requests to verify the webhook URL is accessible.
    This endpoint returns a simple success response.
    """
    return {
        "status": "ok",
        "message": "WhatsApp webhook endpoint is active",
        "service": "twilio_whatsapp",
        "methods": ["GET", "POST"],
    }


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

        # Validate required fields
        if not config.twilio_account_sid:
            raise HTTPException(
                status_code=400, detail="Twilio Account SID is required"
            )
        if not config.twilio_auth_token:
            raise HTTPException(status_code=400, detail="Twilio Auth Token is required")
        if not config.twilio_phone_number:
            raise HTTPException(
                status_code=400, detail="Twilio Phone Number is required"
            )

        # Check if configuration already exists
        try:
            existing_response = (
                supabase.table("whatsapp_configurations")
                .select("id")
                .eq("org_id", org_id)
                .execute()
            )
        except Exception as db_error:
            error_msg = str(db_error)
            logger.error(
                "Database error checking existing config",
                extra={"org_id": org_id, "error": error_msg},
                exc_info=True,
            )
            # Check if table doesn't exist
            if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail="WhatsApp configurations table not found. Please run the database migration: database_migration_whatsapp.sql",
                ) from db_error
            # Check for RLS/permission issues
            if "permission" in error_msg.lower() or "policy" in error_msg.lower():
                raise HTTPException(
                    status_code=403,
                    detail="Permission denied. Please ensure you have admin access to your organization.",
                ) from db_error
            raise HTTPException(
                status_code=500,
                detail=f"Database error while checking configuration: {error_msg}",
            ) from db_error

        # If activating this config, deactivate all other configs for this org first
        # This handles the unique constraint: only one active config per org
        if config.is_active:
            try:
                supabase.table("whatsapp_configurations").update(
                    {"is_active": False}
                ).eq("org_id", org_id).eq("is_active", True).execute()
            except Exception as deactivate_error:
                logger.warning(
                    "Failed to deactivate other configs (may not exist)",
                    extra={"org_id": org_id, "error": str(deactivate_error)},
                )

        # Ensure provider_type matches database constraint exactly
        # The constraint allows: 'twilio' or 'whatsapp_business'
        provider_type = "twilio"  # Explicitly set to lowercase 'twilio'

        config_data = {
            "org_id": org_id,
            "provider_type": provider_type,
            "twilio_account_sid": config.twilio_account_sid.strip(),
            "twilio_auth_token": config.twilio_auth_token.strip(),
            "twilio_phone_number": config.twilio_phone_number.strip(),
            "webhook_url": config.webhook_url.strip() if config.webhook_url else None,
            "is_active": config.is_active,
        }

        if existing_response.data and len(existing_response.data) > 0:
            # Update existing configuration
            config_id = existing_response.data[0]["id"]
            try:
                response = (
                    supabase.table("whatsapp_configurations")
                    .update(config_data)
                    .eq("id", config_id)
                    .execute()
                )

                if not response.data:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to update WhatsApp configuration",
                    )

                logger.info(
                    "WhatsApp configuration updated",
                    extra={"org_id": org_id, "config_id": config_id},
                )

                # Invalidate cache after configuration update
                try:
                    whatsapp_service = WhatsAppService(org_id=org_id)
                    await whatsapp_service.invalidate_config_cache()
                    logger.info(
                        "Cache invalidated after config update",
                        extra={"org_id": org_id},
                    )
                except Exception as cache_error:
                    # Don't fail the request if cache invalidation fails
                    logger.warning(
                        "Failed to invalidate cache after config update",
                        extra={"org_id": org_id, "error": str(cache_error)},
                    )

                return {
                    "success": True,
                    "message": "WhatsApp configuration updated successfully",
                    "config_id": config_id,
                }
            except Exception as update_error:
                error_msg = str(update_error)
                # Try to extract more detailed error from Supabase response
                if hasattr(update_error, "message"):
                    error_msg = update_error.message
                elif hasattr(update_error, "args") and update_error.args:
                    error_msg = str(update_error.args[0])

                logger.error(
                    "Database error updating config",
                    extra={
                        "org_id": org_id,
                        "config_id": config_id,
                        "error": error_msg,
                        "error_type": type(update_error).__name__,
                    },
                    exc_info=True,
                )
                # Provide more specific error messages
                if (
                    "permission" in error_msg.lower()
                    or "policy" in error_msg.lower()
                    or "row-level security" in error_msg.lower()
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="Permission denied. Please ensure you have admin access to your organization.",
                    ) from update_error
                if (
                    "constraint" in error_msg.lower()
                    or "unique" in error_msg.lower()
                    or "duplicate" in error_msg.lower()
                ):
                    # Check for provider_type constraint violation specifically
                    if "provider_type" in error_msg.lower():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Database constraint violation: The provider_type constraint in your database may not match the expected values ('twilio' or 'whatsapp_business'). Please run the fix script: database_fix_whatsapp_provider_type_constraint.sql. Original error: {error_msg}",
                        ) from update_error
                    raise HTTPException(
                        status_code=400,
                        detail=f"Configuration constraint violation: {error_msg}",
                    ) from update_error
                if (
                    "does not exist" in error_msg.lower()
                    or "relation" in error_msg.lower()
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="WhatsApp configurations table not found. Please run the database migration: database_migration_whatsapp.sql",
                    ) from update_error
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to update WhatsApp configuration: {error_msg}",
                ) from update_error
        else:
            # Create new configuration
            try:
                response = (
                    supabase.table("whatsapp_configurations")
                    .insert(config_data)
                    .execute()
                )

                if not response.data or len(response.data) == 0:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to create WhatsApp configuration",
                    )

                config_id = response.data[0]["id"]
                logger.info(
                    "WhatsApp configuration created",
                    extra={"org_id": org_id, "config_id": config_id},
                )

                # Invalidate cache after configuration creation
                try:
                    whatsapp_service = WhatsAppService(org_id=org_id)
                    await whatsapp_service.invalidate_config_cache()
                    logger.info(
                        "Cache invalidated after config creation",
                        extra={"org_id": org_id},
                    )
                except Exception as cache_error:
                    # Don't fail the request if cache invalidation fails
                    logger.warning(
                        "Failed to invalidate cache after config creation",
                        extra={"org_id": org_id, "error": str(cache_error)},
                    )

                return {
                    "success": True,
                    "message": "WhatsApp configuration created successfully",
                    "config_id": config_id,
                }
            except Exception as insert_error:
                error_msg = str(insert_error)
                # Try to extract more detailed error from Supabase response
                if hasattr(insert_error, "message"):
                    error_msg = insert_error.message
                elif hasattr(insert_error, "args") and insert_error.args:
                    error_msg = str(insert_error.args[0])

                logger.error(
                    "Database error creating config",
                    extra={
                        "org_id": org_id,
                        "error": error_msg,
                        "error_type": type(insert_error).__name__,
                    },
                    exc_info=True,
                )
                # Provide more specific error messages
                if (
                    "permission" in error_msg.lower()
                    or "policy" in error_msg.lower()
                    or "row-level security" in error_msg.lower()
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="Permission denied. Please ensure you have admin access to your organization.",
                    ) from insert_error
                if (
                    "constraint" in error_msg.lower()
                    or "unique" in error_msg.lower()
                    or "duplicate" in error_msg.lower()
                ):
                    # Check for provider_type constraint violation specifically
                    if "provider_type" in error_msg.lower():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Database constraint violation: The provider_type constraint in your database may not match the expected values ('twilio' or 'whatsapp_business'). Please run the fix script: database_fix_whatsapp_provider_type_constraint.sql. Original error: {error_msg}",
                        ) from insert_error
                    raise HTTPException(
                        status_code=400,
                        detail=f"Configuration constraint violation: {error_msg}",
                    ) from insert_error
                if (
                    "does not exist" in error_msg.lower()
                    or "relation" in error_msg.lower()
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="WhatsApp configurations table not found. Please run the database migration: database_migration_whatsapp.sql",
                    ) from insert_error
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create WhatsApp configuration: {error_msg}",
                ) from insert_error

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to configure WhatsApp",
            extra={
                "error": str(e),
                "user_id": user.get("user_id"),
                "error_type": type(e).__name__,
            },
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

        validation_result = await whatsapp_service.validate_configuration()

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


@router.get("/cache/stats", tags=["whatsapp"])
async def get_cache_stats(
    user=Depends(verify_jwt_token_from_header),
):
    """
    Get WhatsApp configuration cache statistics

    Returns cache performance metrics including:
    - Hit/miss rates
    - Memory cache size
    - Redis cache hits
    - Total requests served
    """
    try:
        org_id = await get_org_id_from_user(user)
        whatsapp_service = WhatsAppService(org_id=org_id)

        stats = whatsapp_service.get_cache_stats()

        return {
            "success": True,
            "org_id": org_id,
            "cache_stats": stats,
            "performance_summary": {
                "hit_rate": f"{stats.get('hit_rate_percent', 0):.2f}%",
                "total_requests": stats.get("total_requests", 0),
                "memory_cache_utilization": f"{stats.get('memory_cache_size', 0)}/{stats.get('max_cache_size', 0)}",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get cache statistics",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get cache statistics"
        ) from e


@router.post("/cache/invalidate", tags=["whatsapp"])
async def invalidate_cache(
    user=Depends(verify_jwt_token_from_header),
):
    """
    Manually invalidate WhatsApp configuration cache for the organization

    Useful when:
    - Configuration was updated externally
    - Cache needs to be refreshed immediately
    - Debugging cache-related issues
    """
    try:
        org_id = await get_org_id_from_user(user)
        whatsapp_service = WhatsAppService(org_id=org_id)

        await whatsapp_service.invalidate_config_cache()

        logger.info(
            "Cache manually invalidated",
            extra={"org_id": org_id, "user_id": user.get("user_id")},
        )

        return {
            "success": True,
            "message": "Cache invalidated successfully",
            "org_id": org_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to invalidate cache",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to invalidate cache") from e
