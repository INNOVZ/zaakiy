"""Middleware for token usage validation and rate limiting."""

import logging
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status
from supabase import Client

from app.models.subscription import Channel, TokenUsageRequest
from app.services.subscription import SubscriptionService

logger = logging.getLogger(__name__)


class TokenValidationMiddleware:
    """Middleware for validating token usage before AI operations."""

    def __init__(self, supabase_client: Client):
        self.subscription_service = SubscriptionService(supabase_client)

    async def validate_and_consume_tokens(
        self,
        entity_id: str,
        entity_type: str,
        estimated_tokens: int,
        requesting_user_id: str,
        operation_type: str = "chat",
        channel: Optional[Channel] = None,
        chatbot_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
    ) -> bool:
        """
        Validate token availability and consume tokens for an operation.

        Args:
            entity_id: Entity ID (user or organization)
            entity_type: "user" or "organization"
            estimated_tokens: Estimated tokens for the operation
            requesting_user_id: ID of user requesting the operation (for authorization)
            operation_type: Type of operation (chat, document_processing, etc.)
            channel: Channel where the operation occurred
            chatbot_id: Specific chatbot used
            session_id: Session identifier for tracking
            user_identifier: End-user identifier (for analytics)

        Returns:
            True if tokens were successfully consumed

        Raises:
            HTTPException: If insufficient tokens or validation fails
        """
        try:
            # Check token availability first
            (
                has_enough,
                available,
            ) = await self.subscription_service.check_token_availability(
                entity_id, entity_type, estimated_tokens, requesting_user_id
            )

            if not has_enough:
                usage = await self.subscription_service.get_subscription_usage(
                    entity_id, entity_type, requesting_user_id
                )

                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail={
                        "error": "Insufficient tokens for this operation",
                        "tokens_required": estimated_tokens,
                        "tokens_available": available,
                        "monthly_limit": usage.monthly_limit,
                        "usage_percentage": round(usage.usage_percentage, 2),
                        "reset_date": usage.reset_date.isoformat(),
                        "upgrade_message": "Please upgrade your plan or wait for token reset",
                    },
                )

            # Apply channel-specific token multiplier if applicable
            final_tokens = estimated_tokens
            if channel:
                # Get channel configuration to apply multiplier
                try:
                    # Get subscription first
                    subscription_result = (
                        self.subscription_service.supabase.table("subscriptions")
                        .select("id")
                        .eq("entity_id", entity_id)
                        .eq("entity_type", entity_type)
                        .eq("status", "active")
                        .execute()
                    )

                    if subscription_result.data:
                        subscription_id = subscription_result.data[0]["id"]

                        # Get channel configuration
                        config_result = (
                            self.subscription_service.supabase.table(
                                "channel_configurations"
                            )
                            .select("custom_token_multiplier")
                            .eq("subscription_id", subscription_id)
                            .eq("channel", channel.value)
                            .execute()
                        )

                        if config_result.data:
                            multiplier = float(
                                config_result.data[0]["custom_token_multiplier"]
                            )
                            final_tokens = int(estimated_tokens * multiplier)
                            logger.info(
                                "Applied %s multiplier %s: %d -> %d tokens",
                                channel.value,
                                multiplier,
                                estimated_tokens,
                                final_tokens,
                            )

                except Exception as e:
                    logger.warning("Failed to apply channel multiplier: %s", str(e))
                    # Continue with original token count

            # Consume the tokens
            request = TokenUsageRequest(
                entity_id=entity_id,
                entity_type=entity_type,
                tokens_consumed=final_tokens,
                operation_type=operation_type,
                channel=channel,
                chatbot_id=chatbot_id,
                session_id=session_id,
                user_identifier=user_identifier,
            )

            success = await self.subscription_service.consume_tokens(
                request, requesting_user_id
            )

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Failed to consume tokens. Please try again.",
                )

            logger.info(
                "Successfully consumed %d tokens for %s %s (%s) via %s channel",
                final_tokens,
                entity_type,
                entity_id,
                operation_type,
                channel.value if channel else "unknown",
            )

            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Token validation failed: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token validation failed",
            ) from e

    async def check_tokens_only(
        self,
        entity_id: str,
        entity_type: str,
        required_tokens: int,
        requesting_user_id: Optional[str] = None,
    ) -> Tuple[bool, dict]:
        """
        Check token availability without consuming them.

        Args:
            entity_id: Entity ID
            entity_type: "user" or "organization"
            required_tokens: Required tokens for operation
            requesting_user_id: ID of user making the request (optional)

        Returns:
            Tuple of (has_enough_tokens, usage_info)
        """
        try:
            # Some callers may omit requesting_user_id; ensure we pass a safe value to the service.
            requester = requesting_user_id or ""

            (
                has_enough,
                available,
            ) = await self.subscription_service.check_token_availability(
                entity_id, entity_type, required_tokens, requester
            )

            usage = await self.subscription_service.get_subscription_usage(
                entity_id, entity_type, requester
            )

            usage_info = {
                "has_enough_tokens": has_enough,
                "tokens_required": required_tokens,
                "tokens_available": available,
                "monthly_limit": usage.monthly_limit,
                "usage_percentage": round(usage.usage_percentage, 2),
                "reset_date": usage.reset_date.isoformat(),
            }

            return has_enough, usage_info

        except Exception as e:
            logger.error("Token check failed: %s", str(e))
            return False, {"error": "Failed to check token availability"}

    async def get_entity_from_request(
        self, request: Request
    ) -> Optional[Tuple[str, str]]:
        """
        Extract entity information from request headers or authentication.

        This is a placeholder - you'll need to implement based on your auth system.

        Args:
            request: FastAPI request object

        Returns:
            Tuple of (entity_id, entity_type) or None if not found
        """
        try:
            # Example implementation - adjust based on your auth system
            entity_id = request.headers.get("X-Entity-ID")
            entity_type = request.headers.get("X-Entity-Type")

            if entity_id and entity_type and entity_type in ["user", "organization"]:
                return entity_id, entity_type

            # Alternative: Extract from JWT token or session
            # auth_header = request.headers.get("Authorization")
            # if auth_header:
            #     # Decode JWT and extract entity info
            #     pass

            return None

        except Exception as e:
            logger.error("Failed to extract entity from request: %s", str(e))
            return None


def estimate_tokens_for_operation(operation_type: str, **kwargs) -> int:
    """
    Estimate token usage for different operations.

    Args:
        operation_type: Type of operation
        **kwargs: Operation-specific parameters

    Returns:
        Estimated token count
    """
    # Base estimates - adjust based on your actual usage patterns
    estimates = {
        "chat": 100,  # Base chat response
        "document_upload": 500,  # Document processing
        "document_analysis": 200,  # Document analysis
        "web_scraping": 300,  # Web scraping operation
        "embedding_generation": 50,  # Text embedding
    }

    base_estimate = estimates.get(operation_type, 100)

    # Adjust based on content length if provided
    if "message_length" in kwargs:
        # Rough estimate: 1 token per 4 characters
        message_tokens = kwargs["message_length"] // 4
        base_estimate += message_tokens

    if "document_size" in kwargs:
        # Estimate based on document size (bytes)
        doc_tokens = kwargs["document_size"] // 100  # Rough estimate
        base_estimate += doc_tokens

    # Add safety margin
    return int(base_estimate * 1.2)


# Decorator for protecting endpoints with token validation
def require_tokens(estimated_tokens: int = None, operation_type: str = "chat"):
    """
    Decorator to protect endpoints with token validation.

    Args:
        estimated_tokens: Fixed token estimate, or None to calculate dynamically
        operation_type: Type of operation for logging
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This is a simplified example - you'll need to adapt based on your endpoint structure
            # In practice, you'd extract the request and entity info from the function parameters

            # For now, this serves as a template for how to structure token validation
            logger.info("Token validation required for %s", operation_type)

            # Call the original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator
