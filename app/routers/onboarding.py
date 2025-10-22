"""
Robust Onboarding and Subscription Management API Endpoints
Production-ready implementation that handles all edge cases gracefully
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.models.subscription import (Channel, OnboardingRequest, OnboardingResponse,
                                     SubscriptionPlan, TokenUsageRequest)
from app.services.auth import CurrentUser
from app.services.storage.supabase_client import get_supabase_client
from app.services.subscription import SubscriptionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])


def get_subscription_service(
    supabase: Client = Depends(get_supabase_client),
) -> SubscriptionService:
    """Dependency to get subscription service."""
    return SubscriptionService(supabase)


@router.get("/plans")
async def get_subscription_plans(
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    Get all available subscription plans with features and pricing.
    Public endpoint - no authentication required.

    Returns:
        Dictionary of all subscription plans with their features
    """
    try:
        plans = await subscription_service.get_all_plans()
        return {"success": True, "plans": plans}
    except Exception as e:
        logger.error(f"Failed to get subscription plans: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription plans",
        )


@router.get("/plans/{plan_name}")
async def get_plan_details(
    plan_name: str,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    Get details for a specific subscription plan.
    Public endpoint - no authentication required.

    Args:
        plan_name: Name of the subscription plan (basic, professional, enterprise)

    Returns:
        Plan details with features and pricing
    """
    try:
        # Validate plan name
        try:
            plan = SubscriptionPlan(plan_name.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan name. Available plans: {[p.value for p in SubscriptionPlan]}",
            )

        features = await subscription_service.get_plan_features(plan)
        return {"success": True, "plan": plan.value, "features": features}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plan details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve plan details",
        )


@router.post("/admin/signup", response_model=OnboardingResponse)
async def admin_signup_entity(
    request: OnboardingRequest,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> OnboardingResponse:
    """
    Admin-controlled signup process for creating users/organizations with predefined plans.

    NOTE: This endpoint currently has no authentication for demo purposes.
    In production, add proper admin authentication.

    Args:
        request: Onboarding request with entity details and plan selection

    Returns:
        OnboardingResponse with entity and subscription details
    """
    try:
        logger.info(f"Starting onboarding for {request.entity_type}: {request.email}")

        # Check if email already exists
        await _check_email_availability(request.email, subscription_service.supabase)

        # For now, use a dummy admin ID since we don't have admin auth
        # Generate a proper UUID for the admin user
        import uuid

        admin_user_id = str(uuid.uuid4())

        # Process onboarding
        response = await subscription_service.onboard_entity(request, admin_user_id)

        logger.info(f"Onboarding completed successfully for {response.entity_id}")
        return response

    except ValueError as e:
        logger.warning(f"Validation error during onboarding: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Onboarding failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Onboarding process failed. Please try again.",
        )


@router.get("/subscription/{entity_type}/{entity_id}")
async def get_subscription_status(
    entity_type: str,
    entity_id: str,
    # Require authentication but no special permissions
    current_user: dict = CurrentUser,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    ROBUST: Get current subscription status and usage for an entity.

    This endpoint:
    - Requires authentication
    - Verifies user can access the requested entity
    - Gracefully handles missing subscriptions
    - Returns helpful messages for all scenarios

    Args:
        entity_type: "user" or "organization"
        entity_id: Entity ID
        current_user: Current authenticated user

    Returns:
        Subscription status with token usage information or helpful message
    """
    try:
        # Validate entity type
        if entity_type not in ["user", "organization"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity type must be 'user' or 'organization'",
            )

        # ROBUST: Verify user can access this entity
        if not await _verify_entity_access(current_user, entity_type, entity_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access your own subscription data",
            )

        # ROBUST: Try to get subscription usage with proper error handling
        try:
            usage = await subscription_service.get_subscription_usage(
                entity_id, entity_type, current_user["user_id"]
            )

            # Get plan name from subscription
            subscription_result = (
                subscription_service.supabase.table("subscriptions")
                .select("plan")
                .eq("id", usage.subscription_id)
                .execute()
            )
            plan_name = "Basic Plan"  # default
            if subscription_result.data:
                plan = subscription_result.data[0]["plan"]
                # Map plan to display name
                plan_names = {
                    "basic": "Basic Plan",
                    "professional": "Professional Plan",
                    "enterprise": "Enterprise Plan",
                }
                plan_name = plan_names.get(plan, plan.title() + " Plan")

            return {
                "success": True,
                "has_subscription": True,
                "subscription_id": usage.subscription_id,
                "tokens_used_this_month": usage.tokens_used_this_month,
                "tokens_remaining": usage.tokens_remaining,
                "monthly_limit": usage.monthly_limit,
                "usage_percentage": round(usage.usage_percentage, 2),
                "reset_date": usage.reset_date.isoformat(),
                "plan_name": plan_name,
            }

        except Exception as usage_error:
            if "No active subscription found" in str(usage_error):
                # No subscription found - return appropriate response
                logger.info(f"No subscription found for {entity_type} {entity_id}")
                return {
                    "success": True,
                    "has_subscription": False,
                    "message": "No active subscription found. Please contact admin to set up a subscription.",
                    "subscription_id": None,
                    "tokens_used_this_month": 0,
                    "tokens_remaining": 0,
                    "monthly_limit": 0,
                    "usage_percentage": 0,
                    "reset_date": None,
                }
            else:
                # Some other error occurred
                logger.error(f"Error getting subscription usage: {str(usage_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve subscription status",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription status",
        )


@router.post("/tokens/consume")
async def consume_tokens(
    request: TokenUsageRequest,
    current_user: dict = CurrentUser,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    ROBUST: Consume tokens from entity's subscription.

    This endpoint:
    - Requires authentication
    - Verifies user can consume tokens for the entity
    - Handles missing subscriptions gracefully
    - Provides helpful error messages

    Args:
        request: Token usage request
        current_user: Current authenticated user

    Returns:
        Success status and remaining token information
    """
    try:
        # ROBUST: Verify user can consume tokens for this entity
        if not await _verify_entity_access(
            current_user, request.entity_type, request.entity_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only consume tokens for your own account",
            )

        # ROBUST: Try to consume tokens with proper error handling
        try:
            success = await subscription_service.consume_tokens(
                request, current_user["user_id"]
            )

            if not success:
                # Get current usage to provide helpful error message
                try:
                    usage = await subscription_service.get_subscription_usage(
                        request.entity_id, request.entity_type, current_user["user_id"]
                    )
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail={
                            "error": "Insufficient tokens",
                            "tokens_requested": request.tokens_consumed,
                            "tokens_available": usage.tokens_remaining,
                            "monthly_limit": usage.monthly_limit,
                            "reset_date": usage.reset_date.isoformat(),
                        },
                    )
                except Exception:
                    # If we can't get usage info, return generic error
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail="Insufficient tokens or no active subscription",
                    )

            # Get updated usage
            usage = await subscription_service.get_subscription_usage(
                request.entity_id, request.entity_type, current_user["user_id"]
            )

            return {
                "success": True,
                "tokens_consumed": request.tokens_consumed,
                "tokens_remaining": usage.tokens_remaining,
                "monthly_limit": usage.monthly_limit,
                "usage_percentage": round(usage.usage_percentage, 2),
            }

        except HTTPException:
            raise
        except Exception as consume_error:
            if "No active subscription found" in str(consume_error):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No active subscription found. Please contact admin to set up a subscription.",
                )
            else:
                logger.error(f"Error consuming tokens: {str(consume_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to process token consumption",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to consume tokens: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process token consumption",
        )


@router.get("/tokens/check/{entity_type}/{entity_id}/{required_tokens}")
async def check_token_availability(
    entity_type: str,
    entity_id: str,
    required_tokens: int,
    current_user: dict = CurrentUser,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    ROBUST: Check if entity has enough tokens available for an operation.

    Args:
        entity_type: "user" or "organization"
        entity_id: Entity ID
        required_tokens: Number of tokens required
        current_user: Current authenticated user

    Returns:
        Token availability status
    """
    try:
        # Validate inputs
        if entity_type not in ["user", "organization"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity type must be 'user' or 'organization'",
            )

        if required_tokens < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Required tokens must be non-negative",
            )

        # ROBUST: Verify user can check tokens for this entity
        if not await _verify_entity_access(current_user, entity_type, entity_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only check tokens for your own account",
            )

        # ROBUST: Check token availability with proper error handling
        try:
            has_enough, available = await subscription_service.check_token_availability(
                entity_id, entity_type, required_tokens, current_user["user_id"]
            )

            return {
                "success": True,
                "has_enough_tokens": has_enough,
                "tokens_required": required_tokens,
                "tokens_available": available,
                "can_proceed": has_enough,
            }

        except Exception as check_error:
            if "No active subscription found" in str(check_error):
                return {
                    "success": True,
                    "has_enough_tokens": False,
                    "tokens_required": required_tokens,
                    "tokens_available": 0,
                    "can_proceed": False,
                    "message": "No active subscription found",
                }
            else:
                logger.error(f"Error checking token availability: {str(check_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to check token availability",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check token availability: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check token availability",
        )


async def _verify_entity_access(
    current_user: dict, entity_type: str, entity_id: str
) -> bool:
    """
    ROBUST: Verify user has access to the specified entity.

    This function handles all the edge cases and provides proper logging.

    Args:
        current_user: Current authenticated user
        entity_type: Type of entity ("user" or "organization")
        entity_id: ID of the entity to access

    Returns:
        True if user has access, False otherwise
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            logger.warning("User ID not found in current_user")
            return False

        user_data = current_user.get("user_data", {})

        if entity_type == "user":
            # User can only access their own data
            has_access = user_id == entity_id
            if not has_access:
                logger.warning(f"User {user_id} tried to access user {entity_id}")
            return has_access

        elif entity_type == "organization":
            # User can only access their organization's data
            user_org_id = user_data.get("org_id")
            if not user_org_id:
                logger.info(
                    f"User {user_id} has no organization, cannot access org {entity_id}"
                )
                return False

            has_access = user_org_id == entity_id
            if not has_access:
                logger.warning(
                    f"User {user_id} (org: {user_org_id}) tried to access org {entity_id}"
                )
            return has_access

        else:
            logger.error(f"Invalid entity_type: {entity_type}")
            return False

    except Exception as e:
        logger.error(f"Error verifying entity access: {str(e)}")
        return False


async def _check_email_availability(email: str, supabase: Client) -> None:
    """
    Check if email is already registered.

    Args:
        email: Email to check
        supabase: Supabase client

    Raises:
        HTTPException: If email is already registered
    """
    try:
        # Check users table
        user_result = supabase.table("users").select("id").eq("email", email).execute()
        if user_result.data:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email is already registered as a user",
            )

        # Check organizations table
        org_result = (
            supabase.table("organizations").select("id").eq("email", email).execute()
        )
        if org_result.data:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email is already registered as an organization",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check email availability: {str(e)}")
        # Don't block onboarding if we can't check email availability
        pass


@router.get("/channels")
async def get_supported_channels() -> Dict:
    """
    Get list of all supported channels.
    Public endpoint - no authentication required.

    Returns:
        List of supported channels with descriptions
    """
    try:
        channels = {
            Channel.WEBSITE.value: {
                "name": "Website Chat",
                "description": "Embedded chat widget on websites",
                "icon": "🌐",
                "typical_use_cases": ["Customer support", "Lead generation", "FAQ"],
            },
            Channel.WHATSAPP.value: {
                "name": "WhatsApp Business",
                "description": "WhatsApp Business API integration",
                "icon": "📱",
                "typical_use_cases": ["Customer service", "Order updates", "Marketing"],
            },
            Channel.MESSENGER.value: {
                "name": "Facebook Messenger",
                "description": "Facebook Messenger platform integration",
                "icon": "💬",
                "typical_use_cases": [
                    "Social commerce",
                    "Customer support",
                    "Engagement",
                ],
            },
            Channel.INSTAGRAM.value: {
                "name": "Instagram Direct",
                "description": "Instagram Direct Messages integration",
                "icon": "📸",
                "typical_use_cases": [
                    "Brand engagement",
                    "Product inquiries",
                    "Support",
                ],
            },
            Channel.API.value: {
                "name": "REST API",
                "description": "Direct API integration for custom applications",
                "icon": "🔌",
                "typical_use_cases": [
                    "Custom apps",
                    "System integration",
                    "Automation",
                ],
            },
            Channel.MOBILE_APP.value: {
                "name": "Mobile App",
                "description": "Native mobile application integration",
                "icon": "📲",
                "typical_use_cases": [
                    "In-app support",
                    "User onboarding",
                    "Feature guidance",
                ],
            },
        }

        return {"success": True, "channels": channels, "total_channels": len(channels)}

    except Exception as e:
        logger.error(f"Failed to get supported channels: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supported channels",
        )


# ==========================================
# SUBSCRIPTION ANALYTICS ENDPOINTS
# ==========================================


@router.get("/analytics/{subscription_id}")
async def get_subscription_analytics(
    subscription_id: str,
    days_back: int = 30,
    current_user: dict = CurrentUser,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    Get comprehensive subscription analytics including token usage, channel performance,
    and usage trends for the specified subscription.

    Args:
        subscription_id: ID of the subscription to analyze
        days_back: Number of days to look back for analytics (default: 30)
        current_user: Current authenticated user

    Returns:
        Comprehensive analytics data including:
        - Subscription overview (plan, limits, usage)
        - Token usage trends
        - Channel performance
        - Daily/hourly usage patterns
    """
    try:
        # Verify user has access to this subscription
        if not await _verify_subscription_access(current_user, subscription_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access analytics for your own subscription",
            )

        # Get subscription details
        subscription_result = (
            subscription_service.supabase.table("subscriptions")
            .select("*")
            .eq("id", subscription_id)
            .execute()
        )

        if not subscription_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found"
            )

        subscription = subscription_result.data[0]

        # Get plan details
        plan_result = (
            subscription_service.supabase.table("subscription_plans")
            .select("*")
            .eq("plan_name", subscription["plan"])
            .execute()
        )
        plan_details = plan_result.data[0] if plan_result.data else {}

        # Get token usage logs for the period
        from datetime import datetime, timedelta

        start_date = datetime.utcnow() - timedelta(days=days_back)

        usage_logs_result = (
            subscription_service.supabase.table("token_usage_logs")
            .select("*")
            .eq("subscription_id", subscription_id)
            .gte("timestamp", start_date.isoformat())
            .execute()
        )

        # Get channel analytics
        channel_analytics_result = (
            subscription_service.supabase.table("channel_usage_analytics")
            .select("*")
            .eq("subscription_id", subscription_id)
            .gte("date", start_date.date().isoformat())
            .execute()
        )

        # Process analytics data
        analytics_data = await _process_subscription_analytics(
            subscription,
            plan_details,
            usage_logs_result.data or [],
            channel_analytics_result.data or [],
            days_back,
        )

        return {"success": True, "analytics": analytics_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription analytics",
        )


@router.get("/analytics/{subscription_id}/channels/comparison")
async def get_channel_comparison(
    subscription_id: str,
    days_back: int = 30,
    current_user: dict = CurrentUser,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    Get channel performance comparison for a subscription.

    Returns:
        Channel comparison data with performance metrics
    """
    try:
        # Verify access
        if not await _verify_subscription_access(current_user, subscription_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        # Get channel analytics
        from datetime import datetime, timedelta

        start_date = datetime.utcnow() - timedelta(days=days_back)

        channel_analytics_result = (
            subscription_service.supabase.table("channel_usage_analytics")
            .select("*")
            .eq("subscription_id", subscription_id)
            .gte("date", start_date.date().isoformat())
            .execute()
        )

        # Process channel comparison
        comparison_data = await _process_channel_comparison(
            channel_analytics_result.data or []
        )

        return {"success": True, "comparison": comparison_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get channel comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve channel comparison",
        )


@router.get("/analytics/{subscription_id}/channels/{channel}/trends")
async def get_channel_trends(
    subscription_id: str,
    channel: str,
    days_back: int = 30,
    current_user: dict = CurrentUser,
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> Dict:
    """
    Get usage trends for a specific channel.

    Returns:
        Daily trends data for the specified channel
    """
    try:
        # SECURITY: Validate days_back to prevent injection attacks
        # Limit to reasonable range (1-365 days) to prevent resource exhaustion
        if days_back < 1:
            days_back = 1
        elif days_back > 365:
            days_back = 365
        
        # Verify access
        if not await _verify_subscription_access(current_user, subscription_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        # Get channel trends
        from datetime import datetime, timedelta

        start_date = datetime.utcnow() - timedelta(days=days_back)

        trends_result = (
            subscription_service.supabase.table("channel_usage_analytics")
            .select("*")
            .eq("subscription_id", subscription_id)
            .eq("channel", channel)
            .gte("date", start_date.date().isoformat())
            .order("date")
            .execute()
        )

        # Process trends
        trends_data = await _process_channel_trends(trends_result.data or [], days_back)

        return {"success": True, "trends": trends_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get channel trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve channel trends",
        )


# ==========================================
# ANALYTICS HELPER FUNCTIONS
# ==========================================


async def _verify_subscription_access(current_user: dict, subscription_id: str) -> bool:
    """
    Verify user has access to the specified subscription.

    Args:
        current_user: Current authenticated user
        subscription_id: ID of the subscription to access

    Returns:
        True if user has access, False otherwise
    """
    try:
        user_id = current_user.get("user_id")
        user_data = current_user.get("user_data", {})

        # Get subscription details
        supabase = get_supabase_client()
        subscription_result = (
            supabase.table("subscriptions")
            .select("entity_id, entity_type")
            .eq("id", subscription_id)
            .execute()
        )

        if not subscription_result.data:
            return False

        subscription = subscription_result.data[0]
        entity_id = subscription["entity_id"]
        entity_type = subscription["entity_type"]

        # Check access based on entity type
        if entity_type == "user":
            return user_id == entity_id
        elif entity_type == "organization":
            user_org_id = user_data.get("org_id")
            return user_org_id == entity_id

        return False

    except Exception as e:
        logger.error(f"Error verifying subscription access: {str(e)}")
        return False


async def _process_subscription_analytics(
    subscription, plan_details, usage_logs, channel_analytics, days_back
):
    """
    Process raw data into comprehensive analytics.

    Args:
        subscription: Subscription record
        plan_details: Plan details record
        usage_logs: List of token usage logs
        channel_analytics: List of channel analytics records
        days_back: Number of days to analyze

    Returns:
        Processed analytics data
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    # Basic subscription info
    total_tokens_used = subscription["tokens_used_this_month"]
    monthly_limit = subscription["monthly_token_limit"]
    usage_percentage = (
        (total_tokens_used / monthly_limit * 100) if monthly_limit > 0 else 0
    )

    # Process daily usage
    daily_usage = defaultdict(int)
    hourly_distribution = defaultdict(int)

    for log in usage_logs:
        timestamp = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
        daily_usage[timestamp.date().isoformat()] += log["tokens_consumed"]
        hourly_distribution[timestamp.hour] += log["tokens_consumed"]

    # Process channel usage
    channel_usage = []
    channel_totals = defaultdict(lambda: {"tokens": 0, "messages": 0, "users": set()})

    for analytics in channel_analytics:
        channel = analytics["channel"]
        channel_totals[channel]["tokens"] += analytics["tokens_used"]
        channel_totals[channel]["messages"] += analytics["message_count"]
        channel_totals[channel]["users"].add(analytics.get("unique_users", 0))

    for channel, totals in channel_totals.items():
        channel_usage.append(
            {
                "channel": channel,
                "tokens_used": totals["tokens"],
                "message_count": totals["messages"],
                "unique_users": len(totals["users"]),
                "avg_tokens_per_message": totals["tokens"] / totals["messages"]
                if totals["messages"] > 0
                else 0,
                "usage_share_percentage": (totals["tokens"] / total_tokens_used * 100)
                if total_tokens_used > 0
                else 0,
            }
        )

    # Calculate growth rate (comparing first half vs second half of period)
    sorted_daily = sorted(daily_usage.items())
    if len(sorted_daily) > 7:
        mid_point = len(sorted_daily) // 2
        first_half = sum(usage for _, usage in sorted_daily[:mid_point])
        second_half = sum(usage for _, usage in sorted_daily[mid_point:])
        growth_rate = (
            ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
        )
    else:
        growth_rate = 0

    # Find most/least active channels
    most_active_channel = (
        max(channel_usage, key=lambda x: x["tokens_used"])["channel"]
        if channel_usage
        else None
    )
    least_active_channel = (
        min(channel_usage, key=lambda x: x["tokens_used"])["channel"]
        if channel_usage
        else None
    )

    return {
        "subscription_id": subscription["id"],
        "entity_id": subscription["entity_id"],
        "entity_type": subscription["entity_type"],
        "plan": subscription["plan"],
        "plan_name": plan_details.get("display_name", subscription["plan"]),
        "total_tokens_used": total_tokens_used,
        "total_tokens_limit": monthly_limit,
        "usage_percentage": round(usage_percentage, 2),
        "tokens_remaining": max(0, monthly_limit - total_tokens_used),
        "channel_usage": channel_usage,
        "daily_usage": dict(daily_usage),
        "hourly_distribution": dict(hourly_distribution),
        "most_active_channel": most_active_channel,
        "least_active_channel": least_active_channel,
        "growth_rate": round(growth_rate, 2),
        "billing_cycle_start": subscription["billing_cycle_start"],
        "billing_cycle_end": subscription["billing_cycle_end"],
        "days_remaining": max(
            0,
            (
                datetime.fromisoformat(
                    subscription["billing_cycle_end"].replace("Z", "+00:00")
                )
                - datetime.utcnow()
            ).days,
        ),
        "status": subscription["status"],
        "analytics_period_days": days_back,
    }


async def _process_channel_comparison(channel_analytics):
    """
    Process channel analytics into comparison data.

    Args:
        channel_analytics: List of channel analytics records

    Returns:
        Channel comparison data
    """
    from collections import defaultdict

    channel_totals = defaultdict(
        lambda: {
            "tokens_used": 0,
            "message_count": 0,
            "unique_users": set(),
            "avg_response_time_ms": 0,
            "error_count": 0,
            "days_active": set(),
        }
    )

    for analytics in channel_analytics:
        channel = analytics["channel"]
        channel_totals[channel]["tokens_used"] += analytics["tokens_used"]
        channel_totals[channel]["message_count"] += analytics["message_count"]
        channel_totals[channel]["unique_users"].add(analytics.get("unique_users", 0))
        channel_totals[channel]["avg_response_time_ms"] += analytics.get(
            "avg_response_time_ms", 0
        )
        channel_totals[channel]["error_count"] += analytics.get("error_count", 0)
        channel_totals[channel]["days_active"].add(analytics["date"])

    # Calculate totals for percentages
    total_tokens = sum(data["tokens_used"] for data in channel_totals.values())
    total_messages = sum(data["message_count"] for data in channel_totals.values())

    comparison = {}
    for channel, data in channel_totals.items():
        avg_response_time = (
            data["avg_response_time_ms"] / len(data["days_active"])
            if data["days_active"]
            else 0
        )
        efficiency_score = (
            (data["tokens_used"] / data["message_count"])
            if data["message_count"] > 0
            else 0
        )

        comparison[channel] = {
            "tokens_used": data["tokens_used"],
            "message_count": data["message_count"],
            "unique_users": len(data["unique_users"]),
            "avg_tokens_per_message": data["tokens_used"] / data["message_count"]
            if data["message_count"] > 0
            else 0,
            "usage_share_percentage": (data["tokens_used"] / total_tokens * 100)
            if total_tokens > 0
            else 0,
            "efficiency_score": round(efficiency_score, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "error_count": data["error_count"],
            "error_rate": (data["error_count"] / data["message_count"] * 100)
            if data["message_count"] > 0
            else 0,
            "days_active": len(data["days_active"]),
            "performance_rating": _calculate_performance_rating(
                efficiency_score, avg_response_time, data["error_count"]
            ),
        }

    return comparison


async def _process_channel_trends(trends_data, days_back):
    """
    Process channel trends data.

    Args:
        trends_data: List of channel analytics records
        days_back: Number of days to analyze (must be validated before calling)

    Returns:
        Processed trends data
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    # SECURITY: Additional safeguard - ensure days_back is within safe bounds
    # This should already be validated by the caller, but defense in depth
    days_back = max(1, min(int(days_back), 365))

    # Initialize daily data
    daily_tokens = []
    daily_messages = []
    daily_users = []

    # Group by date
    date_data = defaultdict(lambda: {"tokens": 0, "messages": 0, "users": set()})

    for record in trends_data:
        date = record["date"]
        date_data[date]["tokens"] += record["tokens_used"]
        date_data[date]["messages"] += record["message_count"]
        date_data[date]["users"].add(record.get("unique_users", 0))

    # Convert to arrays - now safe from injection
    for i in range(days_back):
        date = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
        data = date_data.get(date, {"tokens": 0, "messages": 0, "users": set()})

        daily_tokens.append({"date": date, "value": data["tokens"]})
        daily_messages.append({"date": date, "value": data["messages"]})
        daily_users.append({"date": date, "value": len(data["users"])})

    # Calculate trend direction
    if len(daily_tokens) >= 7:
        recent_avg = sum(item["value"] for item in daily_tokens[:7]) / 7
        older_avg = (
            sum(item["value"] for item in daily_tokens[7:14]) / 7
            if len(daily_tokens) >= 14
            else recent_avg
        )

        if recent_avg > older_avg * 1.1:
            trend_direction = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
    else:
        trend_direction = "stable"

    average_daily_tokens = (
        sum(item["value"] for item in daily_tokens) / len(daily_tokens)
        if daily_tokens
        else 0
    )

    return {
        "daily_tokens": daily_tokens,
        "daily_messages": daily_messages,
        "daily_users": daily_users,
        "trend_direction": trend_direction,
        "average_daily_tokens": round(average_daily_tokens, 2),
    }


def _calculate_performance_rating(efficiency_score, avg_response_time, error_count):
    """
    Calculate performance rating based on metrics.

    Args:
        efficiency_score: Tokens per message
        avg_response_time: Average response time in ms
        error_count: Number of errors

    Returns:
        Performance rating string
    """
    if error_count == 0 and avg_response_time < 2000 and efficiency_score < 200:
        return "excellent"
    elif error_count < 5 and avg_response_time < 5000 and efficiency_score < 300:
        return "good"
    elif error_count < 10 and avg_response_time < 10000:
        return "fair"
    else:
        return "needs_improvement"
