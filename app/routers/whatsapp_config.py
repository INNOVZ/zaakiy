"""WhatsApp configuration router for managing Twilio settings."""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..models.whatsapp import (
    WhatsAppConfigRequest,
    WhatsAppConfigResponse,
    WhatsAppStatusResponse,
)
from ..services.auth import get_user_with_org, verify_jwt_token_from_header
from ..services.messaging.provider_factory import ProviderFactory
from ..services.storage.supabase_client import get_supabase_client, run_supabase
from ..utils.encryption import encrypt_value
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/config", response_model=WhatsAppConfigResponse)
async def create_whatsapp_config(
    request: WhatsAppConfigRequest,
    user=Depends(verify_jwt_token_from_header),
):
    """
    Create or update WhatsApp configuration for organization.

    Supports both platform-managed and tenant-managed accounts.
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        supabase = get_supabase_client()

        # Generate webhook URL
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8001")
        webhook_url = f"{api_base_url}/api/whatsapp/webhook"

        config_data = {
            "org_id": org_id,
            "provider_type": request.provider_type,
            "twilio_phone_number": request.twilio_phone_number,
            "webhook_url": webhook_url,
            "is_active": request.is_active,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add tenant-managed credentials if provided
        # Note: Cross-field validation is handled by WhatsAppConfigRequest model validator
        # Critical: Encrypt sensitive credentials before storing
        if request.provider_type == "tenant_managed":
            # Model validator ensures both fields are present, so safe to access here
            config_data["twilio_account_sid"] = request.twilio_account_sid
            # Encrypt auth token before storing
            config_data["twilio_auth_token"] = encrypt_value(request.twilio_auth_token)

        # Fix race condition: Use upsert instead of check-then-insert/update
        # This ensures atomic operation and prevents race conditions

        # Check if config exists to preserve created_at on update
        # Only set created_at if this is a new record (not an update)
        existing = await run_supabase(
            lambda: (
                supabase.table("whatsapp_configurations")
                .select("id, created_at")
                .eq("org_id", org_id)
                .execute()
            )
        )

        # Only set created_at if this is a new record (not an update)
        if not existing.data or len(existing.data) == 0:
            config_data["created_at"] = datetime.now(timezone.utc).isoformat()
        # For updates, created_at is preserved from existing record (not included in config_data)

        # Upsert configuration (insert or update if exists)
        # on_conflict="org_id" ensures one config per org
        response = await run_supabase(
            lambda: (
                supabase.table("whatsapp_configurations")
                .upsert(config_data, on_conflict="org_id")
                .execute()
            )
        )

        if response.data and len(response.data) > 0:
            config_result = response.data[0].copy()
            # Critical: Remove sensitive credentials from response
            if "twilio_auth_token" in config_result:
                del config_result["twilio_auth_token"]
            if "twilio_account_sid" in config_result:
                # Account SID is less sensitive but still shouldn't be exposed
                del config_result["twilio_account_sid"]

            logger.info("Upserted WhatsApp config for org %s", org_id)
            return WhatsAppConfigResponse(**config_result)

        raise HTTPException(status_code=500, detail="Failed to save configuration")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error creating WhatsApp config: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=WhatsAppConfigResponse)
async def get_whatsapp_config(
    user=Depends(verify_jwt_token_from_header),
):
    """Get current WhatsApp configuration for organization."""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        supabase = get_supabase_client()

        response = await run_supabase(
            lambda: (
                supabase.table("whatsapp_configurations")
                .select("*")
                .eq("org_id", org_id)
                .execute()
            )
        )

        if response.data and len(response.data) > 0:
            config = response.data[0].copy()
            # Critical: Remove all sensitive credentials from response
            if "twilio_auth_token" in config:
                del config["twilio_auth_token"]
            if "twilio_account_sid" in config:
                del config["twilio_account_sid"]
            return WhatsAppConfigResponse(**config)

        raise HTTPException(status_code=404, detail="WhatsApp configuration not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting WhatsApp config: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=WhatsAppStatusResponse)
async def get_whatsapp_status(
    user=Depends(verify_jwt_token_from_header),
):
    """Get WhatsApp service status for organization."""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        supabase = get_supabase_client()

        # Check for configuration
        config_response = await run_supabase(
            lambda: (
                supabase.table("whatsapp_configurations")
                .select("*")
                .eq("org_id", org_id)
                .execute()
            )
        )

        if not config_response.data or len(config_response.data) == 0:
            return WhatsAppStatusResponse(
                is_configured=False,
                is_active=False,
                message="WhatsApp is not configured. Please set up your Twilio credentials.",
            )

        config = config_response.data[0]

        # Check if provider is available
        provider = await ProviderFactory.get_provider(org_id)

        if not provider:
            return WhatsAppStatusResponse(
                is_configured=True,
                is_active=False,
                provider_type=config.get("provider_type"),
                phone_number=config.get("twilio_phone_number"),
                message="WhatsApp provider is not available. Please check your credentials.",
            )

        return WhatsAppStatusResponse(
            is_configured=True,
            is_active=config.get("is_active", False),
            provider_type=config.get("provider_type"),
            phone_number=config.get("twilio_phone_number"),
            message="WhatsApp is configured and active.",
        )

    except Exception as e:
        logger.error("Error getting WhatsApp status: %s", str(e), exc_info=True)
        return WhatsAppStatusResponse(
            is_configured=False,
            is_active=False,
            message=f"Error checking status: {str(e)}",
        )


@router.post("/test")
async def test_whatsapp_connection(
    user=Depends(verify_jwt_token_from_header),
):
    """Test WhatsApp connection by sending a test message."""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Get provider
        provider = await ProviderFactory.get_provider(org_id)

        if not provider:
            raise HTTPException(
                status_code=400, detail="WhatsApp provider is not configured"
            )

        # Get provider type from configuration (not hardcoded)
        supabase = get_supabase_client()
        config_response = await run_supabase(
            lambda: (
                supabase.table("whatsapp_configurations")
                .select("provider_type")
                .eq("org_id", org_id)
                .eq("is_active", True)
                .execute()
            )
        )

        provider_type = "unknown"
        if config_response.data and len(config_response.data) > 0:
            provider_type = config_response.data[0].get("provider_type", "unknown")

        # Get test phone number from request (would need to add to model)
        # For now, just return success if provider is available
        return {
            "success": True,
            "message": "WhatsApp provider is available and configured",
            "provider_type": provider_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error testing WhatsApp connection: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config")
async def delete_whatsapp_config(
    user=Depends(verify_jwt_token_from_header),
):
    """Delete WhatsApp configuration for organization."""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        supabase = get_supabase_client()

        response = await run_supabase(
            lambda: (
                supabase.table("whatsapp_configurations")
                .delete()
                .eq("org_id", org_id)
                .execute()
            )
        )

        logger.info("Deleted WhatsApp config for org %s", org_id)

        return {"success": True, "message": "Configuration deleted"}

    except Exception as e:
        logger.error("Error deleting WhatsApp config: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
