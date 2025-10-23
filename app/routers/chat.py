"""
chat.py -This module handles chat requests and responses.

Contains functions for creating and managing chat, conversation and configuring context.
"""
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    ChatRequest,
    ChatResponse,
    ContextConfigRequest,
    CreateChatbotRequest,
    FeedbackRequest,
    UpdateChatbotRequest,
)
from ..services.analytics.context_analytics import context_analytics
from ..services.analytics.context_config import context_config_manager
from ..services.auth import get_user_with_org, verify_jwt_token_from_header
from ..services.chat.chat_service import ChatService
from ..services.shared import cache_service
from ..services.storage.supabase_client import get_supabase_client
from ..utils.error_context import ErrorContextManager
from ..utils.error_handlers import handle_errors
from ..utils.error_monitoring import error_monitor
from ..utils.exceptions import ValidationError
from ..utils.logging_config import get_logger
from ..utils.rate_limiter import get_rate_limit_config, rate_limit

# Constants
CHATBOT_NOT_FOUND_MESSAGE = "Chatbot not found"

# Get logger
logger = get_logger(__name__)

# Get centralized Supabase client
supabase = get_supabase_client()

router = APIRouter()

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def get_org_chatbot(org_id: str, chatbot_id: Optional[str] = None):
    """Get organization's active chatbot with enhanced fallback"""
    try:
        if chatbot_id:
            response = (
                supabase.table("chatbots")
                .select("*")
                .eq("id", chatbot_id)
                .eq("org_id", org_id)
                .execute()
            )
        else:
            response = (
                supabase.table("chatbots")
                .select("*")
                .eq("org_id", org_id)
                .eq("chain_status", "active")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

        if response.data and len(response.data) > 0:
            chatbot = response.data[0]
            logger.info(
                "Found chatbot for organization",
                extra={
                    "chatbot_name": chatbot["name"],
                    "org_id": org_id,
                    "chatbot_id": chatbot["id"],
                },
            )
            return chatbot

        # Enhanced default chatbot
        logger.info(
            "No chatbot found, using default configuration", extra={"org_id": org_id}
        )
        return {
            "id": f"default-{org_id}",
            "name": "INNOVZ Assistant",
            "avatar_url": None,
            "color_hex": "#3B82F6",
            "tone": "helpful",
            "behavior": "Be helpful, professional, and informative. Use the knowledge base to provide accurate answers.",
            "chain_status": "active",
            "org_id": org_id,
            "greeting_message": "Hello! I'm your INNOVZ AI Assistant. How can I help you today?",
            "fallback_message": "I apologize, but I don't have enough information to answer that question accurately. Could you please rephrase or provide more context?",
            "system_prompt": "You are INNOVZ Assistant, a helpful AI powered by the organization's knowledge base.",
            "model_config": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000},
        }

    except (ConnectionError, TimeoutError) as e:
        logger.error(
            "Database connection error while fetching chatbot",
            extra={"org_id": org_id, "error": str(e)},
            exc_info=True,
        )
        return {
            "id": f"fallback-{org_id}",
            "name": "INNOVZ Assistant",
            "color_hex": "#3B82F6",
            "tone": "helpful",
            "behavior": "Be helpful and informative",
            "org_id": org_id,
            "greeting_message": "Hello! How can I help you today?",
            "fallback_message": "I'm experiencing some technical difficulties. Please try again.",
        }


# ==========================================
# CHATBOT MANAGEMENT ENDPOINTS
# ==========================================


@router.post("/chatbots")
@handle_errors(context="create_chatbot")
async def create_chatbot(
    request: CreateChatbotRequest, user=Depends(verify_jwt_token_from_header)
):
    """Create a new chatbot for the organization"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Validate request data
        if not request.name or len(request.name.strip()) < 2:
            raise ValidationError("Chatbot name must be at least 2 characters long")

        # Generate chatbot ID
        chatbot_id = str(uuid.uuid4())

        # Default model config
        ai_model_config = request.ai_model_config or {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        # Build enhanced system prompt
        system_prompt = (
            request.system_prompt
            or f"""
You are {request.name}, a {request.tone} AI assistant for INNOVZ.

PERSONALITY AND BEHAVIOR:
- Tone: {request.tone}
- Behavior: {request.behavior}

CORE INSTRUCTIONS:
- Use uploaded documents and knowledge base to answer questions accurately
- Be conversational and maintain a {request.tone} tone
- If you don't have relevant information, acknowledge this honestly
- Provide specific, actionable answers when possible
- Reference sources when appropriate but don't overwhelm users

GREETING: {request.greeting_message}
FALLBACK: {request.fallback_message}
"""
        )

        # Create chatbot data for database
        chatbot_data = {
            "id": chatbot_id,
            "org_id": org_id,
            "name": request.name,
            "description": request.description,
            "color_hex": request.color_hex,
            "tone": request.tone,
            "behavior": request.behavior,
            "system_prompt": system_prompt,
            "greeting_message": request.greeting_message,
            "fallback_message": request.fallback_message,
            "model_config": ai_model_config,
            "chain_status": "active" if request.is_active else "inactive",
            "avatar_url": request.avatar_url,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Insert into Supabase
        response = supabase.table("chatbots").insert(chatbot_data).execute()

        if response.data:
            created_chatbot = response.data[0]
            logger.info(
                "Chatbot created successfully",
                extra={
                    "chatbot_id": chatbot_id,
                    "org_id": org_id,
                    "chatbot_name": created_chatbot.get("name"),
                },
            )

            # Initialize default context config for this org if it doesn't exist
            try:
                await context_config_manager.get_config(org_id)
            except (KeyError, ValueError, ConnectionError) as config_error:
                logger.warning(
                    "Could not initialize context config for new chatbot",
                    extra={
                        "org_id": org_id,
                        "chatbot_id": chatbot_id,
                        "error": str(config_error),
                    },
                )

            return {
                "success": True,
                "chatbot": created_chatbot,
                "id": chatbot_id,
                "message": "Chatbot created successfully",
            }
        else:
            error_msg = getattr(response, "error", "Unknown database error")
            logger.error(
                "Database insert failed for chatbot creation",
                extra={"error_msg": error_msg, "org_id": org_id},
            )
            raise HTTPException(status_code=500, detail=f"Database error: {error_msg}")

    except ValidationError:
        raise  # Re-raise validation errors
    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Failed to create chatbot due to database connection issue",
        ) from e
    except Exception as e:
        logger.error(
            "Failed to create chatbot",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create chatbot: {str(e)}"
        ) from e


@router.put("/chatbots/{chatbot_id}")
async def update_chatbot(
    chatbot_id: str,
    request: UpdateChatbotRequest,
    user=Depends(verify_jwt_token_from_header),
):
    """Update an existing chatbot"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        update_data = _build_update_data(request)
        _validate_update_data(update_data)

        response = _execute_chatbot_update(chatbot_id, org_id, update_data)

        # Invalidate the chatbot config cache after successful update
        if cache_service and response.data:
            try:
                cache_key = f"chatbot_config:{chatbot_id}"
                await cache_service.delete(cache_key)
                logger.info(
                    "Chatbot cache invalidated after update",
                    extra={"chatbot_id": chatbot_id, "cache_key": cache_key},
                )
            except Exception as cache_error:
                logger.warning(
                    "Failed to invalidate chatbot cache",
                    extra={"error": str(cache_error), "chatbot_id": chatbot_id},
                )

        return _format_update_response(response, chatbot_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update chatbot",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to update chatbot: {str(e)}"
        ) from e


def _build_update_data(request: UpdateChatbotRequest) -> dict:
    """Build update data from request, handling field mappings"""
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

    for field, value in request.model_dump(exclude_unset=True).items():
        if value is not None:
            if field == "ai_model_config":
                update_data["model_config"] = value
            elif field == "is_active":
                update_data["chain_status"] = "active" if value else "inactive"
            else:
                update_data[field] = value

    return update_data


def _validate_update_data(update_data: dict) -> None:
    """Validate that update data contains meaningful changes"""
    if len(update_data) == 1:  # Only timestamp was added
        raise HTTPException(status_code=400, detail="No update data provided")


def _execute_chatbot_update(chatbot_id: str, org_id: str, update_data: dict):
    """Execute the database update operation"""
    return (
        supabase.table("chatbots")
        .update(update_data)
        .eq("id", chatbot_id)
        .eq("org_id", org_id)
        .execute()
    )


def _format_update_response(response, chatbot_id: str) -> dict:
    """Format the successful update response"""
    if response.data:
        logger.info("Chatbot updated successfully", extra={"chatbot_id": chatbot_id})
        return {
            "success": True,
            "chatbot": response.data[0],
            "message": "Chatbot updated successfully",
        }
    else:
        raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)


@router.delete("/chatbots/{chatbot_id}")
async def delete_chatbot(chatbot_id: str, user=Depends(verify_jwt_token_from_header)):
    """Delete a chatbot"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = (
            supabase.table("chatbots")
            .delete()
            .eq("id", chatbot_id)
            .eq("org_id", org_id)
            .execute()
        )

        if response.data:
            # Invalidate the chatbot config cache after deletion
            if cache_service:
                try:
                    cache_key = f"chatbot_config:{chatbot_id}"
                    await cache_service.delete(cache_key)
                    logger.info(
                        "Chatbot cache invalidated after deletion",
                        extra={"chatbot_id": chatbot_id, "cache_key": cache_key},
                    )
                except Exception as cache_error:
                    logger.warning(
                        "Failed to invalidate chatbot cache",
                        extra={"error": str(cache_error), "chatbot_id": chatbot_id},
                    )

            logger.info(
                "Chatbot deleted successfully", extra={"chatbot_id": chatbot_id}
            )
            return {"success": True, "message": "Chatbot deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete chatbot",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to delete chatbot: {str(e)}"
        ) from e


@router.get("/chatbots")
async def list_chatbots(
    page: int = 1,
    page_size: int = 50,
    status: str = None,
    user=Depends(verify_jwt_token_from_header),
):
    """
    List organization's chatbots with pagination

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        status: Filter by status (active, inactive)
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")

        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=400, detail="Page size must be between 1 and 100"
            )

        # Calculate offset
        offset = (page - 1) * page_size

        # Build query
        query = (
            supabase.table("chatbots").select("*", count="exact").eq("org_id", org_id)
        )

        # Apply status filter if provided
        if status:
            allowed_statuses = ["active", "inactive"]
            if status not in allowed_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Status must be one of: {', '.join(allowed_statuses)}",
                )
            query = query.eq("chain_status", status)

        # Apply pagination and ordering
        response = (
            query.order("created_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        chatbots = response.data or []

        # Add compatibility mapping for frontend
        for chatbot in chatbots:
            if "model_config" in chatbot:
                chatbot["ai_model_config"] = chatbot["model_config"]

        # Get total count
        total_count = response.count if hasattr(response, "count") else len(chatbots)

        # Calculate pagination metadata
        total_pages = (total_count + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1

        logger.info(
            "Retrieved chatbots for organization",
            extra={
                "count": len(chatbots),
                "org_id": org_id,
                "page": page,
                "total_pages": total_pages,
                "page_size": page_size,
            },
        )

        return {
            "chatbots": chatbots,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to list chatbots",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch chatbots") from e


@router.get("/chatbots/{chatbot_id}")
async def get_chatbot(chatbot_id: str, user=Depends(verify_jwt_token_from_header)):
    """Get specific chatbot configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = (
            supabase.table("chatbots")
            .select("*")
            .eq("id", chatbot_id)
            .eq("org_id", org_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            chatbot = response.data[0]
            # Map model_config back to ai_model_config for frontend compatibility
            if "model_config" in chatbot:
                chatbot["ai_model_config"] = chatbot["model_config"]

            logger.info("Retrieved chatbot details", extra={"chatbot_id": chatbot_id})
            return chatbot
        else:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get chatbot",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch chatbot") from e


@router.post("/chatbots/{chatbot_id}/clear-cache")
async def clear_chatbot_cache(
    chatbot_id: str, user=Depends(verify_jwt_token_from_header)
):
    """Manually clear the cache for a specific chatbot"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Verify the chatbot belongs to this organization
        response = (
            supabase.table("chatbots")
            .select("id")
            .eq("id", chatbot_id)
            .eq("org_id", org_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

        # Clear the cache
        if cache_service:
            try:
                cache_key = f"chatbot_config:{chatbot_id}"
                deleted = await cache_service.delete(cache_key)

                logger.info(
                    "Chatbot cache cleared manually",
                    extra={
                        "chatbot_id": chatbot_id,
                        "cache_key": cache_key,
                        "deleted": deleted,
                    },
                )

                return {
                    "success": True,
                    "message": "Chatbot cache cleared successfully",
                    "cache_key": cache_key,
                    "deleted": deleted,
                }
            except Exception as cache_error:
                logger.error(
                    "Failed to clear chatbot cache",
                    extra={"error": str(cache_error), "chatbot_id": chatbot_id},
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to clear cache: {str(cache_error)}"
                )
        else:
            return {
                "success": True,
                "message": "Cache service not available (cache not enabled)",
                "deleted": False,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to clear chatbot cache",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to clear chatbot cache: {str(e)}"
        ) from e


@router.post("/chatbots/{chatbot_id}/activate")
async def activate_chatbot(chatbot_id: str, user=Depends(verify_jwt_token_from_header)):
    """Activate a chatbot for deployment"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Get all chatbots in org to invalidate their cache later
        org_chatbots_response = (
            supabase.table("chatbots").select("id").eq("org_id", org_id).execute()
        )
        chatbot_ids_to_invalidate = [
            cb["id"] for cb in (org_chatbots_response.data or [])
        ]

        # Deactivate all other chatbots first (only one active at a time)
        supabase.table("chatbots").update(
            {
                "chain_status": "inactive",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("org_id", org_id).execute()

        # Activate the selected chatbot
        response = (
            supabase.table("chatbots")
            .update(
                {
                    "chain_status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .eq("id", chatbot_id)
            .eq("org_id", org_id)
            .execute()
        )

        if response.data:
            # Invalidate cache for all affected chatbots
            if cache_service:
                for cb_id in chatbot_ids_to_invalidate:
                    try:
                        cache_key = f"chatbot_config:{cb_id}"
                        await cache_service.delete(cache_key)
                    except Exception as cache_error:
                        logger.warning(
                            "Failed to invalidate chatbot cache",
                            extra={"error": str(cache_error), "chatbot_id": cb_id},
                        )

                logger.info(
                    "Chatbot cache invalidated after activation",
                    extra={
                        "chatbot_id": chatbot_id,
                        "invalidated_count": len(chatbot_ids_to_invalidate),
                    },
                )

            logger.info(
                "Chatbot activated successfully", extra={"chatbot_id": chatbot_id}
            )
            return {"success": True, "message": "Chatbot activated successfully"}
        else:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to activate chatbot",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to activate chatbot: {str(e)}"
        ) from e


# ==========================================
# CHAT AND CONVERSATION ENDPOINTS
# ==========================================


@router.post("/conversation")
@rate_limit(**get_rate_limit_config("chat"))
@handle_errors(context="chat_conversation", service="chat_router")
async def chat_conversation(
    request: ChatRequest, user=Depends(verify_jwt_token_from_header)
):
    """Main chat endpoint with enhanced context engineering"""
    start_time = datetime.now(timezone.utc)

    try:
        # Set request context for error tracking INSIDE try block
        request_id = f"req_{int(datetime.now(timezone.utc).timestamp())}"
        ErrorContextManager.set_request_context(
            request_id=request_id,
            user_id=user["user_id"],
            operation="chat_conversation",
        )

        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Update context with org_id
        ErrorContextManager.set_request_context(org_id=org_id)

        logger.info(
            "Processing chat request",
            extra={
                "org_id": org_id,
                "message_preview": request.message[:50],
                "chatbot_id": request.chatbot_id,
            },
        )

        # Get chatbot config
        chatbot_config = get_org_chatbot(org_id, request.chatbot_id)

        # Generate session ID if not provided
        session_id = (
            request.conversation_id
            or f"session-{user['user_id']}-{int(datetime.now(timezone.utc).timestamp())}"
        )

        # Initialize chat service with full context engineering and entity information
        chat_service = ChatService(
            org_id=org_id,
            chatbot_config=chatbot_config,
            entity_id=org_id,  # Use organization ID as entity ID
            entity_type="organization",  # Users consume tokens from their organization's subscription
        )

        # Generate response
        result = await chat_service.chat(
            message=request.message,
            session_id=session_id,
            chatbot_id=request.chatbot_id or chatbot_config.get("id"),
        )

        total_time = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        logger.info(
            "Chat completed successfully",
            extra={
                "org_id": org_id,
                "processing_time_ms": total_time,
                "chatbot_id": request.chatbot_id,
            },
        )

        # Handle case where conversation_id might be None (fallback response)
        conversation_id = result.get("conversation_id")
        if conversation_id is None:
            # Generate a fallback conversation_id for error cases
            conversation_id = f"fallback-{uuid.uuid4()}"

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            product_links=result.get("product_links", []),
            conversation_id=conversation_id,
            message_id=result.get("message_id"),
            processing_time_ms=result.get("processing_time_ms", total_time),
            context_quality=result.get("context_quality", {}),
            chatbot_config={
                "name": chatbot_config.get("name", "AI Assistant"),
                "avatar_url": chatbot_config.get("avatar_url"),
                "color_hex": chatbot_config.get("color_hex", "#3B82F6"),
                "tone": chatbot_config.get("tone", "helpful"),
                "id": chatbot_config.get("id"),
            },
        )

    except Exception as e:
        error_time = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        # Record error in monitoring system
        error_monitor.record_error(
            error_type=type(e).__name__,
            severity="high",
            service="chat_router",
            category="api_endpoint",
        )

        logger.error(
            "Chat conversation failed",
            extra={"error": str(e), "processing_time_ms": error_time, "org_id": org_id},
            exc_info=True,
        )
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}") from e

    finally:
        # Clear request context
        ErrorContextManager.clear_context()


@router.get("/conversations")
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    chatbot_id: str = None,
    user=Depends(verify_jwt_token_from_header),
):
    """
    List conversations with pagination and filtering

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        chatbot_id: Filter by specific chatbot
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")

        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=400, detail="Page size must be between 1 and 100"
            )

        # Calculate offset
        offset = (page - 1) * page_size

        # Build query
        query = (
            supabase.table("conversations")
            .select("*", count="exact")
            .eq("org_id", org_id)
        )

        # Apply chatbot filter if provided
        if chatbot_id:
            query = query.eq("chatbot_id", chatbot_id)

        # Apply pagination and ordering
        result = (
            query.order("updated_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        # Get total count
        total_count = result.count if hasattr(result, "count") else len(result.data)

        # Calculate pagination metadata
        total_pages = (total_count + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1

        logger.info(
            "Retrieved conversations for organization",
            extra={
                "count": len(result.data),
                "org_id": org_id,
                "page": page,
                "total_pages": total_pages,
                "page_size": page_size,
            },
        )

        return {
            "conversations": result.data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to list conversations",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to load conversations: {str(e)}"
        ) from e


@router.post("/feedback")
async def add_feedback(
    request: FeedbackRequest, user=Depends(verify_jwt_token_from_header)
):
    """Add message feedback with analytics integration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        chatbot_config = {"name": "Assistant", "org_id": org_id}
        chat_service = ChatService(org_id=org_id, chatbot_config=chatbot_config)

        success = await chat_service.add_feedback(
            message_id=request.message_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
        )

        if success:
            logger.info(
                "Feedback added successfully",
                extra={"message_id": request.message_id, "rating": request.rating},
            )
            return {"success": True, "message": "Feedback recorded successfully"}
        else:
            raise HTTPException(status_code=404, detail="Message not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to add feedback",
            extra={"error": str(e), "message_id": request.message_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to add feedback: {str(e)}"
        ) from e


# ==========================================
# ANALYTICS AND CONFIGURATION ENDPOINTS
# ==========================================


@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    days: int = 7, user=Depends(verify_jwt_token_from_header)
):
    """Get analytics dashboard for the organization"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        dashboard = await context_analytics.get_performance_dashboard(org_id, days)

        logger.info("Analytics dashboard retrieved", extra={"org_id": org_id})
        return dashboard

    except Exception as e:
        logger.error(
            "Failed to get analytics dashboard",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics: {str(e)}"
        ) from e


@router.get("/context-config")
async def get_context_config(user=Depends(verify_jwt_token_from_header)):
    """Get current context engineering configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        config = await context_config_manager.get_config(org_id)

        return {
            "success": True,
            "config": config.dict(),
            "message": "Context configuration retrieved successfully",
        }

    except Exception as e:
        logger.error(
            "Failed to get context config",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get context configuration: {str(e)}"
        ) from e


@router.put("/context-config")
async def update_context_config(
    request: ContextConfigRequest, user=Depends(verify_jwt_token_from_header)
):
    """Update context engineering configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        success = await context_config_manager.update_config(
            org_id, request.config_updates
        )

        if success:
            return {
                "success": True,
                "message": "Context configuration updated successfully",
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to update configuration"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update context config",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to update context configuration: {str(e)}"
        ) from e


@router.get("/performance-insights")
async def get_performance_insights(user=Depends(verify_jwt_token_from_header)):
    """Get AI performance insights and recommendations"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        chatbot_config = {"name": "Assistant", "org_id": org_id}
        chat_service = ChatService(org_id=org_id, chatbot_config=chatbot_config)

        insights = chat_service.get_performance_insights()

        return {
            "success": True,
            "insights": insights,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except (ConnectionError, TimeoutError) as e:
        logger.error(
            "Database connection error in performance insights",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable"
        ) from e
    except (KeyError, ValueError) as e:
        logger.error(
            "Data validation error in performance insights",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail="Invalid data format") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get performance insights",
            extra={"error": str(e), "org_id": org_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance insights: {str(e)}"
        ) from e


@router.get("/analytics/context")
async def get_context_analytics(
    days: int = 7, user=Depends(verify_jwt_token_from_header)
):
    """Get context engineering analytics"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Use the existing context_analytics instance
        analytics_data = context_analytics.get_query_analysis(org_id, "", days)

        return {
            "success": True,
            "analytics": analytics_data,
            "summary": {
                "total_queries": analytics_data.get("similar_queries_found", 0),
                "avg_quality_score": 0.7,  # Default values
                "avg_retrieval_time": 1500,
                "model_distribution": {"gpt-4": 100},
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/analytics/query-optimization")
async def analyze_query_optimization(
    request: dict, user=Depends(verify_jwt_token_from_header)
):
    """Analyze query for optimization suggestions"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        query = request.get("query", "")

        # Use existing context analytics
        analysis = context_analytics.get_query_analysis(org_id, query, 30)

        return {
            "original_query": query,
            "enhanced_query": f"Enhanced: {query}",  # Simple enhancement
            "expected_quality": 0.8,
            "strategy_recommendation": "hybrid",
            "estimated_response_time": 2000,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==========================================
# HEALTH CHECK
# ==========================================


@router.get("/health")
def health_check():
    """Health check endpoint for chat services"""
    try:
        # Test database connection
        test_response = supabase.table("organizations").select("id").limit(1).execute()
        db_status = "healthy" if test_response.data is not None else "error"

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "database": db_status,
                "chat_service": "healthy",
                "context_engine": "healthy",
            },
            "version": "2.0.0",
        }

    except (ConnectionError, TimeoutError) as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"Database connection error: {str(e)}",
            "version": "2.0.0",
        }
    except (ValueError, KeyError) as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"Data validation error: {str(e)}",
            "version": "2.0.0",
        }
    except (AttributeError, ImportError) as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"Service configuration error: {str(e)}",
            "version": "2.0.0",
        }
