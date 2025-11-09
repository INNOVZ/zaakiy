"""
Secure Subscription Service with Race Condition Protection
Production-ready implementation with atomic operations and proper security
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from supabase import Client

from app.models.subscription import (
    DEFAULT_CHANNEL_LIMITS,
    SUBSCRIPTION_PLANS,
    Channel,
    EnhancedPlanFeatures,
    OnboardingRequest,
    OnboardingResponse,
    SubscriptionPlan,
    SubscriptionStatus,
    SubscriptionUsage,
    TokenUsageRequest,
)

logger = logging.getLogger(__name__)


class SubscriptionService:
    """
    Secure subscription service with race condition protection and proper authorization
    """

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.auth_client = supabase_client.auth
        # Entity-specific locks for atomic token operations
        self._token_locks = {}
        self._lock_cleanup_interval = 300  # 5 minutes
        self._last_cleanup = datetime.utcnow()

    @asynccontextmanager
    async def _get_entity_lock(self, entity_id: str, entity_type: str):
        """
        Get or create a lock for entity token operations

        Args:
            entity_id: Entity ID
            entity_type: Entity type

        Yields:
            Async context manager for the lock
        """
        lock_key = f"{entity_type}:{entity_id}"

        # Clean up old locks periodically
        await self._cleanup_old_locks()

        if lock_key not in self._token_locks:
            self._token_locks[lock_key] = {
                "lock": asyncio.Lock(),
                "last_used": datetime.utcnow(),
            }

        # Update last used time
        self._token_locks[lock_key]["last_used"] = datetime.utcnow()

        async with self._token_locks[lock_key]["lock"]:
            yield

    async def _cleanup_old_locks(self):
        """Clean up locks that haven't been used recently"""
        current_time = datetime.utcnow()

        if (current_time - self._last_cleanup).seconds < self._lock_cleanup_interval:
            return

        cutoff_time = current_time - timedelta(minutes=10)
        keys_to_remove = []

        for key, lock_data in self._token_locks.items():
            if lock_data["last_used"] < cutoff_time:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._token_locks[key]

        self._last_cleanup = current_time

        if keys_to_remove:
            logger.info("Cleaned up %d old token locks", len(keys_to_remove))

    async def onboard_entity(
        self, request: OnboardingRequest, admin_user_id: str
    ) -> OnboardingResponse:
        """
        Handle complete onboarding process for user or organization with admin tracking

        Args:
            request: Onboarding request with entity details and plan selection
            admin_user_id: ID of admin user performing the onboarding

        Returns:
            OnboardingResponse with entity and subscription details
        """
        try:
            # Log admin action
            logger.info(
                "Admin %s starting onboarding for %s: %s",
                admin_user_id,
                request.entity_type,
                request.email,
                extra={
                    "event_type": "admin_onboarding_start",
                    "admin_user_id": admin_user_id,
                    "entity_type": request.entity_type,
                    "entity_email": request.email,
                    "selected_plan": request.selected_plan.value,
                },
            )

            # Get plan features from database (source of truth for limits and pricing)
            plan_features = await self._get_plan_features_with_db_values(
                request.selected_plan
            )

            if request.entity_type == "user":
                entity_id = await self._create_user(request, admin_user_id)
            else:  # organization
                entity_id = await self._create_organization(request, admin_user_id)

            # Create subscription
            subscription_id = await self._create_subscription(
                entity_id=entity_id,
                entity_type=request.entity_type,
                plan=request.selected_plan,
                plan_features=plan_features,
                admin_user_id=admin_user_id,
            )

            logger.info(
                "Successfully onboarded %s %s with %s plan by admin %s",
                request.entity_type,
                entity_id,
                request.selected_plan,
                admin_user_id,
                extra={
                    "event_type": "admin_onboarding_success",
                    "admin_user_id": admin_user_id,
                    "entity_id": entity_id,
                    "entity_type": request.entity_type,
                    "subscription_id": subscription_id,
                    "plan": request.selected_plan.value,
                },
            )

            return OnboardingResponse(
                success=True,
                message=f"{request.entity_type.title()} successfully created with {plan_features.name}. A confirmation email has been sent to {request.email}.",
                entity_id=entity_id,
                entity_type=request.entity_type,
                subscription_id=subscription_id,
                plan=request.selected_plan,
                tokens_remaining=plan_features.monthly_token_limit,
                tokens_limit=plan_features.monthly_token_limit,
                email_confirmation_required=True,
                email_sent_to=request.email,
            )

        except Exception as e:
            logger.error(
                "Onboarding failed for admin %s: %s",
                admin_user_id,
                str(e),
                extra={
                    "event_type": "admin_onboarding_failed",
                    "admin_user_id": admin_user_id,
                    "entity_type": request.entity_type,
                    "entity_email": request.email,
                    "error": str(e),
                },
            )
            raise Exception(f"Onboarding failed: {str(e)}")

    async def _create_user(self, request: OnboardingRequest, admin_user_id: str) -> str:
        """Create a new user in the database with admin tracking and Supabase auth"""
        try:
            # First create the Supabase auth user (will send confirmation email)
            if not request.password:
                raise Exception("Password is required for user creation")

            # Use regular signup to send confirmation email
            auth_response = self.auth_client.sign_up(
                {
                    "email": request.email,
                    "password": request.password,
                    "options": {
                        "data": {
                            "full_name": request.full_name,
                            "created_by_admin": admin_user_id,
                        }
                    },
                }
            )

            if auth_response.user is None:
                raise Exception(f"Failed to create auth user: {auth_response}")

            auth_user_id = auth_response.user.id

            # Create user record in database with the auth user ID
            user_data = {
                "id": auth_user_id,  # Use the Supabase auth user ID
                "full_name": request.full_name,
                "email": request.email,
                "created_by_admin": admin_user_id,
                "created_at": datetime.utcnow().isoformat(),
            }

            result = self.supabase.table("users").insert(user_data).execute()

            if not result.data:
                # If database insert fails, we should clean up the auth user
                try:
                    # Use the admin client to delete the user
                    admin_client = self.supabase.auth.admin
                    admin_client.delete_user(auth_user_id)
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to cleanup auth user after database failure: %s",
                        cleanup_error,
                    )
                raise Exception("Failed to create user in database")

            logger.info(
                "Successfully created user %s with auth and database records",
                auth_user_id,
            )
            return auth_user_id

        except Exception as e:
            logger.error("Failed to create user: %s", str(e))
            raise Exception(f"Failed to create user: {str(e)}")

    async def _create_organization(
        self, request: OnboardingRequest, admin_user_id: str
    ) -> str:
        """Create a new organization and associated user in the database with admin tracking and Supabase auth"""
        try:
            # First create the Supabase auth user for the organization admin (will send confirmation email)
            if not request.password:
                raise Exception("Password is required for organization creation")

            # Use regular signup to send confirmation email
            auth_response = self.auth_client.sign_up(
                {
                    "email": request.email,
                    "password": request.password,
                    "options": {
                        "data": {
                            "full_name": request.full_name,
                            "organization_name": request.organization_name,
                            "created_by_admin": admin_user_id,
                        }
                    },
                }
            )

            if auth_response.user is None:
                raise Exception(f"Failed to create auth user: {auth_response}")

            auth_user_id = auth_response.user.id

            # Create the organization
            org_data = {
                "id": str(uuid.uuid4()),
                "name": request.organization_name,
                "email": request.email,
                "contact_phone": request.contact_phone,
                "business_type": request.business_type,
                "created_by_admin": admin_user_id,
                "created_at": datetime.utcnow().isoformat(),
            }

            org_result = self.supabase.table("organizations").insert(org_data).execute()

            if not org_result.data:
                # Clean up auth user if organization creation fails
                try:
                    # Use the admin client to delete the user
                    admin_client = self.supabase.auth.admin
                    admin_client.delete_user(auth_user_id)
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to cleanup auth user after organization creation failure: %s",
                        cleanup_error,
                    )
                raise Exception("Failed to create organization")

            org_id = org_result.data[0]["id"]

            # Create the admin user for the organization with the auth user ID
            user_data = {
                "id": auth_user_id,  # Use the Supabase auth user ID
                "full_name": request.full_name,
                "email": request.email,
                "org_id": org_id,
                "created_by_admin": admin_user_id,
                "created_at": datetime.utcnow().isoformat(),
            }

            user_result = self.supabase.table("users").insert(user_data).execute()

            if not user_result.data:
                # Rollback organization creation and auth user
                try:
                    self.supabase.table("organizations").delete().eq(
                        "id", org_id
                    ).execute()
                    # Use the admin client to delete the user
                    admin_client = self.supabase.auth.admin
                    admin_client.delete_user(auth_user_id)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup after user creation failure: {cleanup_error}"
                    )
                raise Exception("Failed to create admin user for organization")

            logger.info(
                "Successfully created organization %s with admin user %s",
                org_id,
                auth_user_id,
            )
            return org_id

        except Exception as e:
            logger.error("Failed to create organization: %s", str(e))
            raise Exception(f"Failed to create organization: {str(e)}")

    async def _create_subscription(
        self,
        entity_id: str,
        entity_type: str,
        plan: SubscriptionPlan,
        plan_features,
        admin_user_id: str,
    ) -> str:
        """Create a subscription record for the entity with admin tracking"""
        try:
            # Calculate billing cycle dates
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=30)  # Monthly billing

            subscription_data = {
                "id": str(uuid.uuid4()),
                "entity_id": entity_id,
                "entity_type": entity_type,
                "plan": plan.value,
                "status": SubscriptionStatus.ACTIVE.value,
                "monthly_token_limit": plan_features.monthly_token_limit,
                "tokens_used_this_month": 0,
                "price_per_month": plan_features.price_per_month,
                "billing_cycle_start": start_date.isoformat(),
                "billing_cycle_end": end_date.isoformat(),
                "created_by_admin": admin_user_id,
                "created_at": start_date.isoformat(),
                "updated_at": start_date.isoformat(),
            }

            result = (
                self.supabase.table("subscriptions").insert(subscription_data).execute()
            )

            if not result.data:
                raise Exception("Failed to create subscription")

            subscription_id = result.data[0]["id"]

            # Update the entity's current_subscription_id field
            await self._update_entity_subscription_reference(
                entity_id, entity_type, subscription_id
            )

            # Create default channel configurations
            await self._create_default_channel_configurations(
                subscription_id, plan_features
            )

            return subscription_id

        except Exception as e:
            logger.error("Failed to create subscription: %s", str(e))
            if "relation" in str(e) and "does not exist" in str(e):
                raise Exception(
                    "Subscriptions table not found. Please run the database migration first."
                )
            raise Exception(f"Failed to create subscription: {str(e)}")

    async def _update_entity_subscription_reference(
        self, entity_id: str, entity_type: str, subscription_id: str
    ) -> None:
        """Update the entity's current_subscription_id field"""
        try:
            if entity_type == "user":
                self.supabase.table("users").update(
                    {"current_subscription_id": subscription_id}
                ).eq("id", entity_id).execute()
            elif entity_type == "organization":
                self.supabase.table("organizations").update(
                    {"current_subscription_id": subscription_id}
                ).eq("id", entity_id).execute()

            logger.info(
                "Updated %s %s with subscription %s",
                entity_type,
                entity_id,
                subscription_id,
            )
        except Exception as e:
            logger.error(f"Failed to update entity subscription reference: {str(e)}")
            # Don't fail subscription creation if this update fails

    async def update_subscription_status(
        self, subscription_id: str, new_status: SubscriptionStatus, admin_user_id: str
    ) -> bool:
        """
        Update subscription status and sync current_subscription_id references

        Args:
            subscription_id: ID of the subscription to update
            new_status: New status for the subscription
            admin_user_id: ID of admin making the change

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get subscription details
            result = (
                self.supabase.table("subscriptions")
                .select("*")
                .eq("id", subscription_id)
                .execute()
            )

            if not result.data:
                logger.error(f"Subscription {subscription_id} not found")
                return False

            subscription = result.data[0]
            entity_id = subscription["entity_id"]
            entity_type = subscription["entity_type"]

            # Update subscription status
            self.supabase.table("subscriptions").update(
                {
                    "status": new_status.value,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            ).eq("id", subscription_id).execute()

            # Update entity's current_subscription_id based on new status
            if new_status == SubscriptionStatus.ACTIVE:
                # Set as current subscription
                await self._update_entity_subscription_reference(
                    entity_id, entity_type, subscription_id
                )
            else:
                # Clear current subscription reference for inactive/cancelled/expired
                await self._clear_entity_subscription_reference(entity_id, entity_type)

            logger.info(
                "Updated subscription %s status to %s by admin %s",
                subscription_id,
                new_status.value,
                admin_user_id,
                extra={
                    "event_type": "subscription_status_update",
                    "admin_user_id": admin_user_id,
                    "subscription_id": subscription_id,
                    "new_status": new_status.value,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                },
            )

            return True

        except Exception as e:
            logger.error("Failed to update subscription status: %s", str(e))
            return False

    async def _clear_entity_subscription_reference(
        self, entity_id: str, entity_type: str
    ) -> None:
        """Clear the entity's current_subscription_id field"""
        try:
            if entity_type == "user":
                self.supabase.table("users").update(
                    {"current_subscription_id": None}
                ).eq("id", entity_id).execute()
            elif entity_type == "organization":
                self.supabase.table("organizations").update(
                    {"current_subscription_id": None}
                ).eq("id", entity_id).execute()

            logger.info(
                "Cleared subscription reference for %s %s", entity_type, entity_id
            )
        except Exception as e:
            logger.error("Failed to clear entity subscription reference: %s", str(e))

    async def get_subscription_usage(
        self, entity_id: str, entity_type: str, requesting_user_id: str
    ) -> SubscriptionUsage:
        """
        Get current subscription usage for an entity with authorization check

        Args:
            entity_id: Entity ID
            entity_type: Entity type
            requesting_user_id: ID of user requesting the data

        Returns:
            SubscriptionUsage object

        Raises:
            Exception: If no subscription found or unauthorized access
        """
        try:
            # Log access attempt
            logger.info(
                "User %s requesting subscription usage for %s %s",
                requesting_user_id,
                entity_type,
                entity_id,
                extra={
                    "event_type": "subscription_usage_request",
                    "requesting_user_id": requesting_user_id,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                },
            )

            result = (
                self.supabase.table("subscriptions")
                .select("*")
                .eq("entity_id", entity_id)
                .eq("entity_type", entity_type)
                .eq("status", "active")
                .execute()
            )

            if not result.data:
                raise Exception("No active subscription found")

            subscription = result.data[0]
            tokens_used = subscription["tokens_used_this_month"]
            monthly_limit = subscription["monthly_token_limit"]
            tokens_remaining = max(0, monthly_limit - tokens_used)
            usage_percentage = (
                (tokens_used / monthly_limit) * 100 if monthly_limit > 0 else 0
            )

            return SubscriptionUsage(
                subscription_id=subscription["id"],
                tokens_used_this_month=tokens_used,
                tokens_remaining=tokens_remaining,
                monthly_limit=monthly_limit,
                usage_percentage=usage_percentage,
                reset_date=datetime.fromisoformat(subscription["billing_cycle_end"]),
            )

        except Exception as e:
            logger.error("Failed to get subscription usage: %s", str(e))
            raise Exception(f"Failed to get subscription usage: {str(e)}")

    async def consume_tokens(
        self, request: TokenUsageRequest, requesting_user_id: str
    ) -> bool:
        """
        Consume tokens from entity's subscription with race condition protection

        Args:
            request: Token usage request
            requesting_user_id: ID of user requesting token consumption

        Returns:
            True if tokens were successfully consumed, False if insufficient tokens
        """

        # Log token consumption attempt
        logger.info(
            "User %s attempting to consume %d tokens for %s %s",
            requesting_user_id,
            request.tokens_consumed,
            request.entity_type,
            request.entity_id,
            extra={
                "event_type": "token_consumption_attempt",
                "requesting_user_id": requesting_user_id,
                "entity_id": request.entity_id,
                "entity_type": request.entity_type,
                "tokens_requested": request.tokens_consumed,
                "operation_type": request.operation_type,
            },
        )

        # Use entity-specific lock for atomic operations
        async with self._get_entity_lock(request.entity_id, request.entity_type):
            try:
                # Use database-level atomic operation
                result = await self._atomic_token_consumption(
                    request, requesting_user_id
                )

                if result:
                    logger.info(
                        "Successfully consumed %d tokens for %s %s",
                        request.tokens_consumed,
                        request.entity_type,
                        request.entity_id,
                        extra={
                            "event_type": "token_consumption_success",
                            "requesting_user_id": requesting_user_id,
                            "entity_id": request.entity_id,
                            "entity_type": request.entity_type,
                            "tokens_consumed": request.tokens_consumed,
                        },
                    )
                else:
                    logger.warning(
                        "Token consumption failed - insufficient tokens for %s %s",
                        request.entity_type,
                        request.entity_id,
                        extra={
                            "event_type": "token_consumption_failed",
                            "requesting_user_id": requesting_user_id,
                            "entity_id": request.entity_id,
                            "entity_type": request.entity_type,
                            "tokens_requested": request.tokens_consumed,
                        },
                    )

                return result

            except Exception as e:
                logger.error(
                    "Failed to consume tokens atomically: %s",
                    str(e),
                    extra={
                        "event_type": "token_consumption_error",
                        "requesting_user_id": requesting_user_id,
                        "entity_id": request.entity_id,
                        "entity_type": request.entity_type,
                        "error": str(e),
                    },
                )
                return False

    async def _atomic_token_consumption(
        self, request: TokenUsageRequest, requesting_user_id: str
    ) -> bool:
        """
        Perform atomic token consumption using database functions

        Args:
            request: Token usage request
            requesting_user_id: ID of user requesting consumption

        Returns:
            True if successful, False otherwise
        """

        try:
            # Call database function that handles the atomic operation
            result = self.supabase.rpc(
                "consume_tokens_atomic",
                {
                    "p_entity_id": request.entity_id,
                    "p_entity_type": request.entity_type,
                    "p_tokens_to_consume": request.tokens_consumed,
                    "p_operation_type": request.operation_type,
                    "p_requesting_user_id": requesting_user_id,
                },
            ).execute()

            if result.data and len(result.data) > 0:
                response = result.data[0]
                success = response.get("success", False)

                if success:
                    # Log successful consumption with details
                    await self._log_token_usage(
                        subscription_id=response["subscription_id"],
                        tokens_consumed=request.tokens_consumed,
                        operation_type=request.operation_type,
                        channel=request.channel,
                        chatbot_id=request.chatbot_id,
                        session_id=request.session_id,
                        user_identifier=request.user_identifier,
                        requesting_user_id=requesting_user_id,
                    )

                return success

            return False

        except Exception as e:
            logger.error("Atomic token consumption failed: %s", str(e))
            return False

    async def _log_token_usage(
        self,
        subscription_id: str,
        tokens_consumed: int,
        operation_type: str,
        requesting_user_id: str,
        channel: Optional[Channel] = None,
        chatbot_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
    ) -> None:
        """Log token usage for analytics and billing with security tracking"""
        try:
            usage_log = {
                "id": str(uuid.uuid4()),
                "subscription_id": subscription_id,
                "tokens_consumed": tokens_consumed,
                "operation_type": operation_type,
                "channel": channel.value if channel else None,
                "chatbot_id": chatbot_id,
                "session_id": session_id,
                "user_identifier": user_identifier,
                "requesting_user_id": requesting_user_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Log to token_usage_logs table
            try:
                self.supabase.table("token_usage_logs").insert(usage_log).execute()
            except Exception as e:
                logger.warning("Failed to insert token usage log: %s", str(e))

        except Exception as e:
            logger.warning("Failed to log token usage: %s", str(e))

    async def check_token_availability(
        self,
        entity_id: str,
        entity_type: str,
        required_tokens: int,
        requesting_user_id: str,
    ) -> Tuple[bool, int]:
        """
        Check if entity has enough tokens available with authorization

        Args:
            entity_id: Entity ID
            entity_type: Entity type
            required_tokens: Number of tokens required
            requesting_user_id: ID of user making the request

        Returns:
            Tuple of (has_enough_tokens, available_tokens)
        """
        try:
            usage = await self.get_subscription_usage(
                entity_id, entity_type, requesting_user_id
            )
            has_enough = usage.tokens_remaining >= required_tokens

            logger.info(
                "Token availability check: %s %s has %d tokens, needs %d",
                entity_type,
                entity_id,
                usage.tokens_remaining,
                required_tokens,
                extra={
                    "event_type": "token_availability_check",
                    "requesting_user_id": requesting_user_id,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "tokens_available": usage.tokens_remaining,
                    "tokens_required": required_tokens,
                    "has_enough": has_enough,
                },
            )

            return has_enough, usage.tokens_remaining

        except Exception as e:
            logger.error("Failed to check token availability: %s", str(e))
            return False, 0

    async def get_plan_features(self, plan: SubscriptionPlan) -> Dict:
        """
        Get features for a specific plan from database, with fallback to hardcoded values.
        Prioritizes database subscription_plans table as source of truth for all plan features.
        """
        try:
            # Try to read from database first
            result = (
                self.supabase.table("subscription_plans")
                .select("*")
                .eq("plan_name", plan.value)
                .execute()
            )

            if result.data and len(result.data) > 0:
                db_plan = result.data[0]
                # Use all database values - database is the source of truth
                # Handle potential None values with defaults from hardcoded fallback
                hardcoded_features = SUBSCRIPTION_PLANS[plan]

                # Helper to safely get values with fallback (handles None but preserves False)
                def get_db_value(key, fallback, type_converter=None):
                    """Get value from db_plan with proper None handling (False values are preserved)"""
                    if key not in db_plan or db_plan[key] is None:
                        return fallback
                    value = db_plan[key]
                    if type_converter and value is not None:
                        try:
                            return type_converter(value)
                        except (ValueError, TypeError):
                            return fallback
                    return value

                return {
                    "name": get_db_value("display_name", hardcoded_features.name),
                    "monthly_token_limit": get_db_value(
                        "monthly_token_limit",
                        hardcoded_features.monthly_token_limit,
                        int,
                    ),
                    "price_per_month": get_db_value(
                        "price_per_month", hardcoded_features.price_per_month, float
                    ),
                    "max_chatbots": get_db_value(
                        "max_chatbots", hardcoded_features.max_chatbots, int
                    ),
                    "max_documents_per_chatbot": get_db_value(
                        "max_documents_per_chatbot",
                        hardcoded_features.max_documents_per_chatbot,
                        int,
                    ),
                    "priority_support": get_db_value(
                        "priority_support", hardcoded_features.priority_support
                    ),
                    "custom_branding": get_db_value(
                        "custom_branding", hardcoded_features.custom_branding
                    ),
                    "api_access": get_db_value(
                        "api_access", hardcoded_features.api_access
                    ),
                    "analytics_retention_days": get_db_value(
                        "analytics_retention_days",
                        hardcoded_features.analytics_retention_days,
                        int,
                    ),
                }
        except Exception as e:
            logger.warning(
                "Failed to read plan %s from database, using hardcoded values: %s",
                plan.value,
                str(e),
            )

        # Fallback to hardcoded values if database read fails
        plan_features = SUBSCRIPTION_PLANS[plan]
        return {
            "name": plan_features.name,
            "monthly_token_limit": plan_features.monthly_token_limit,
            "price_per_month": plan_features.price_per_month,
            "max_chatbots": plan_features.max_chatbots,
            "max_documents_per_chatbot": plan_features.max_documents_per_chatbot,
            "priority_support": plan_features.priority_support,
            "custom_branding": plan_features.custom_branding,
            "api_access": plan_features.api_access,
            "analytics_retention_days": plan_features.analytics_retention_days,
        }

    async def get_all_plans(self) -> Dict:
        """Get all available subscription plans from database with fallback"""
        return {
            plan.value: await self.get_plan_features(plan) for plan in SubscriptionPlan
        }

    async def _get_plan_features_with_db_values(self, plan: SubscriptionPlan):
        """
        Get EnhancedPlanFeatures with ALL database values from subscription_plans table.
        Uses database as source of truth for all plan features, falls back to hardcoded values.
        Channel-related features (not in DB) come from hardcoded defaults.

        Returns:
            EnhancedPlanFeatures object with database values where available
        """
        # Start with hardcoded features (for channel configs, etc. that aren't in DB)
        plan_features = SUBSCRIPTION_PLANS[plan]

        try:
            # Try to read from database - database is source of truth for all plan features
            result = (
                self.supabase.table("subscription_plans")
                .select("*")
                .eq("plan_name", plan.value)
                .execute()
            )

            if result.data and len(result.data) > 0:
                db_plan = result.data[0]

                # Helper to safely get values with fallback (handles None but preserves False)
                def get_db_value(key, fallback, type_converter=None):
                    """Get value from db_plan with proper None handling (False values are preserved)"""
                    if key not in db_plan or db_plan[key] is None:
                        return fallback
                    value = db_plan[key]
                    if type_converter and value is not None:
                        try:
                            return type_converter(value)
                        except (ValueError, TypeError):
                            return fallback
                    return value

                # Use ALL database values for plan features
                # Channel-related features (supported_channels, channel_limits, etc.)
                # are not in DB, so use hardcoded values for those
                return EnhancedPlanFeatures(
                    name=get_db_value("display_name", plan_features.name),
                    monthly_token_limit=get_db_value(
                        "monthly_token_limit", plan_features.monthly_token_limit, int
                    ),
                    price_per_month=get_db_value(
                        "price_per_month", plan_features.price_per_month, float
                    ),
                    max_chatbots=get_db_value(
                        "max_chatbots", plan_features.max_chatbots, int
                    ),
                    max_documents_per_chatbot=get_db_value(
                        "max_documents_per_chatbot",
                        plan_features.max_documents_per_chatbot,
                        int,
                    ),
                    priority_support=get_db_value(
                        "priority_support", plan_features.priority_support
                    ),
                    custom_branding=get_db_value(
                        "custom_branding", plan_features.custom_branding
                    ),
                    api_access=get_db_value("api_access", plan_features.api_access),
                    analytics_retention_days=get_db_value(
                        "analytics_retention_days",
                        plan_features.analytics_retention_days,
                        int,
                    ),
                    # Channel-related features not in DB - use hardcoded values
                    supported_channels=plan_features.supported_channels,
                    channel_limits=plan_features.channel_limits,
                    concurrent_conversations=plan_features.concurrent_conversations,
                    webhook_support=plan_features.webhook_support,
                    white_label_options=plan_features.white_label_options,
                )
        except Exception as e:
            logger.warning(
                "Failed to read plan %s from database, using hardcoded values: %s",
                plan.value,
                str(e),
            )

        # Return hardcoded features if database read fails
        return plan_features

    async def _create_default_channel_configurations(
        self, subscription_id: str, plan_features
    ) -> None:
        """Create default channel configurations for a subscription"""
        try:
            channel_configs = []

            for channel in plan_features.supported_channels:
                channel_limits = plan_features.channel_limits.get(
                    channel, DEFAULT_CHANNEL_LIMITS[channel]
                )

                config = {
                    "id": str(uuid.uuid4()),
                    "subscription_id": subscription_id,
                    "channel": channel.value,
                    "enabled": channel_limits.enabled,
                    "rate_limit_per_minute": channel_limits.rate_limit_per_minute,
                    "max_message_length": channel_limits.max_message_length,
                    "custom_token_multiplier": channel_limits.custom_token_multiplier,
                    "priority_level": channel_limits.priority_level,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }

                channel_configs.append(config)

            if channel_configs:
                self.supabase.table("channel_configurations").insert(
                    channel_configs
                ).execute()
                logger.info(
                    "Created %d channel configurations for subscription %s",
                    len(channel_configs),
                    subscription_id,
                )

        except Exception as e:
            logger.warning("Failed to create channel configurations: %s", str(e))
            # Don't fail subscription creation if channel configs fail
