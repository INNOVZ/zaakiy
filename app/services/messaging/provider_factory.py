"""Factory for creating messaging providers based on organization configuration."""

import logging
import os
from typing import Optional, Tuple

from ..storage.supabase_client import get_supabase_client
from .base_provider import MessagingProvider
from .providers.twilio_provider import TwilioProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating messaging providers."""

    @staticmethod
    async def get_provider(org_id: str) -> Optional[MessagingProvider]:
        """
        Get messaging provider for an organization.

        Supports both platform-managed and tenant-managed accounts.
        Falls back to platform account if no org-specific config exists.

        """
        try:
            supabase = get_supabase_client()

            # Get organization's WhatsApp configuration
            response = (
                supabase.table("whatsapp_configurations")
                .select("*")
                .eq("org_id", org_id)
                .eq("is_active", True)
                .execute()
            )

            config = None
            if response.data and len(response.data) > 0:
                config = response.data[0]

            # Determine which credentials to use
            if config and config.get("provider_type") == "tenant_managed":
                # Use tenant's own Twilio account
                account_sid = config.get("twilio_account_sid")
                auth_token_encrypted = config.get("twilio_auth_token")
                phone_number = config.get("twilio_phone_number")

                if not all([account_sid, auth_token_encrypted, phone_number]):
                    logger.warning(
                        "Tenant-managed config incomplete for org %s", org_id
                    )
                    return None

                # Decrypt auth token if it's encrypted
                from ...utils.encryption import decrypt_value, is_encrypted

                if is_encrypted(auth_token_encrypted):
                    try:
                        auth_token = decrypt_value(auth_token_encrypted)
                    except Exception as e:
                        logger.error(
                            "Failed to decrypt auth token for org %s: %s",
                            org_id,
                            str(e),
                            exc_info=True,
                        )
                        return None
                else:
                    # Backward compatibility: if not encrypted, use as-is
                    auth_token = auth_token_encrypted

                # Validate the decrypted auth token before use
                if not auth_token or len(auth_token.strip()) == 0:
                    logger.error("Decrypted auth token is empty for org %s", org_id)
                    return None

                # Basic validation: Twilio auth tokens are typically 32 characters
                # Allow some flexibility but reject obviously invalid tokens
                if len(auth_token) < 10:
                    logger.warning(
                        "Auth token seems too short for org %s (length: %d)",
                        org_id,
                        len(auth_token),
                    )
                    # Don't fail, but log warning

                logger.info("Using tenant-managed Twilio account for org %s", org_id)
                return TwilioProvider(
                    account_sid=account_sid,
                    auth_token=auth_token,
                    phone_number=phone_number,
                )

            else:
                # Use platform-managed account (default)
                account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                platform_phone = os.getenv("TWILIO_WHATSAPP_PHONE_NUMBER")

                if not all([account_sid, auth_token, platform_phone]):
                    logger.warning(
                        "Platform Twilio credentials not configured in environment for org %s",
                        org_id,
                    )
                    return None

                # If org has a phone number configured, verify it belongs to platform account
                # before using it (security: prevent orgs from using arbitrary phone numbers)
                phone_number = platform_phone  # Default to platform phone
                if config and config.get("twilio_phone_number"):
                    org_phone = config.get("twilio_phone_number")
                    # Security: Verify phone number ownership
                    # Only allow org-specific phone numbers if they match platform phone
                    # or if explicitly verified to belong to platform account
                    if org_phone != platform_phone:
                        # Security risk: Org is trying to use a different phone number
                        # with platform-managed account. This should be verified.
                        logger.warning(
                            "Org %s configured phone number %s differs from platform phone %s. "
                            "Phone number ownership not verified - using platform phone for security.",
                            org_id,
                            org_phone,
                            platform_phone,
                        )
                        # Security: Use platform phone instead of unverified org phone
                        phone_number = platform_phone
                    else:
                        # Phone number matches platform phone - safe to use
                        phone_number = org_phone

                logger.info("Using platform-managed Twilio account for org %s", org_id)
                return TwilioProvider(
                    account_sid=account_sid,
                    auth_token=auth_token,
                    phone_number=phone_number,
                )

        except Exception as e:
            logger.error(
                "Failed to get messaging provider for org %s: %s",
                org_id,
                str(e),
                exc_info=True,
            )
            # Consistent return value: always return None on failure
            return None

    @staticmethod
    async def get_provider_for_phone_number(
        phone_number: str,
    ) -> Tuple[Optional[MessagingProvider], Optional[str]]:
        """
        Get messaging provider based on phone number.

        This is useful when receiving webhooks - we need to determine
        which organization the phone number belongs to.

        Args:
            phone_number: WhatsApp phone number (E.164 format)

        Returns:
            Tuple of (MessagingProvider, org_id) or (None, None)
        """
        try:
            supabase = get_supabase_client()

            # Find org by phone number
            response = (
                supabase.table("whatsapp_configurations")
                .select(
                    "org_id, provider_type, twilio_account_sid, twilio_auth_token, twilio_phone_number"
                )
                .eq("twilio_phone_number", phone_number)
                .eq("is_active", True)
                .execute()
            )

            if response.data and len(response.data) > 0:
                config = response.data[0]
                org_id = config["org_id"]

                # Get provider using org_id
                provider = await ProviderFactory.get_provider(org_id)
                return provider, org_id

            # If no org-specific config, try to find org by matching phone pattern
            # This is a fallback - ideally phone numbers should be in config
            logger.warning(
                "No WhatsApp configuration found for phone number %s", phone_number
            )

            # Return platform provider as fallback
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            platform_phone = os.getenv("TWILIO_WHATSAPP_PHONE_NUMBER")

            if all([account_sid, auth_token, platform_phone]):
                # Verify phone number matches platform phone before using fallback
                if phone_number == platform_phone:
                    logger.info(
                        "Using platform provider as fallback for phone number %s",
                        phone_number,
                    )
                    provider = TwilioProvider(
                        account_sid=account_sid,
                        auth_token=auth_token,
                        phone_number=platform_phone,
                    )
                    # Return None for org_id since we couldn't determine it
                    return provider, None
                else:
                    logger.warning(
                        "Phone number %s does not match platform phone %s - cannot use fallback",
                        phone_number,
                        platform_phone,
                    )
                    return None, None
            else:
                logger.warning(
                    "Platform Twilio credentials not configured - cannot create fallback provider for phone number %s",
                    phone_number,
                )
                return None, None

        except Exception as e:
            logger.error(
                "Failed to get provider for phone number %s: %s",
                phone_number,
                str(e),
                exc_info=True,
            )
            return None, None
