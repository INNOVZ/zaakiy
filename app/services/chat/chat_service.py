"""
Main Chat Service to implement the AI chat functionality
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import openai

from app.models.subscription import Channel
from app.services.analytics.context_config import context_config_manager
from app.utils.error_monitoring import error_monitor
from app.utils.performance_monitor import performance_monitor

from .service_factory import (
    get_analytics_service,
    get_chat_utilities,
    get_conversation_manager,
    get_document_retrieval_service,
    get_error_handling_service,
    get_response_generation_service,
)

logger = logging.getLogger(__name__)


class ChatServiceError(Exception):
    """Base exception for chat service errors"""


class RetrievalError(ChatServiceError):
    """Exception for retrieval-related errors"""


class ContextError(ChatServiceError):
    """Exception for context engineering errors"""


class ResponseGenerationError(ChatServiceError):
    """Exception for response generation errors"""


class ChatService:
    """
    Main chat service orchestrator that coordinates modular services.
    """

    def __init__(
        self,
        org_id: str,
        chatbot_config: dict,
        entity_id: str = None,
        entity_type: str = None,
    ):
        self.org_id = org_id
        self.namespace = f"org-{org_id}"
        self.chatbot_config = chatbot_config
        self.entity_id = entity_id  # User ID or Organization ID for token consumption
        self.entity_type = entity_type  # "user" or "organization"

        # Use lazy property access for clients (non-blocking initialization)
        # Clients are singletons, so we just store references to getter functions
        # This avoids blocking initialization and makes failures recoverable
        self._supabase = None
        self._index = None
        self._openai_client = None
        self._token_middleware = None

        # Store initialization errors for graceful handling
        self._init_errors = {}

        # Context engineering config will be loaded per request
        self.context_config = None

        # Initialize retrieval config with defaults (will be updated per request)
        # EMERGENCY MODE: Minimal docs for speed
        self.retrieval_config = {"initial": 3, "rerank": 2, "final": 2}
        self.max_context_length = 2000  # Reduced for speed
        self._background_tasks = set()

        # Initialize modular services (uses lazy-loaded clients)
        self._initialize_services()

    @property
    def supabase(self):
        """Lazy-load Supabase client (singleton, non-blocking)"""
        if self._supabase is None:
            try:
                from ..storage.supabase_client import get_supabase_client

                self._supabase = get_supabase_client()
            except Exception as e:
                logger.warning("Failed to get Supabase client: %s", e)
                raise ChatServiceError(f"Supabase client unavailable: {e}") from e
        return self._supabase

    @property
    def index(self):
        """Lazy-load Pinecone index (singleton, non-blocking)"""
        if self._index is None:
            try:
                from ..storage.pinecone_client import get_pinecone_index

                self._index = get_pinecone_index()
            except Exception as e:
                self._init_errors["pinecone"] = e
                logger.warning("Failed to get Pinecone index: %s", e)
                # Don't raise - document retrieval will handle this gracefully
                return None
        return self._index

    @property
    def openai_client(self):
        """Lazy-load OpenAI client (shared singleton, non-blocking)"""
        if self._openai_client is None:
            try:
                # Use shared client manager instead of creating new client
                from ..shared import get_openai_client

                self._openai_client = get_openai_client()
                if self._openai_client is None:
                    # Fallback: try direct initialization if shared client not available
                    import os

                    openai_key = os.getenv("OPENAI_API_KEY")
                    if openai_key:
                        self._openai_client = openai.OpenAI(api_key=openai_key)
                    else:
                        logger.warning(
                            "OpenAI API key not found - some features may be unavailable"
                        )
            except Exception as e:
                self._init_errors["openai"] = e
                logger.warning("Failed to get OpenAI client: %s", e)
                # Don't raise - validation happens when actually used
                return None
        return self._openai_client

    @property
    def token_middleware(self):
        """Lazy-load token middleware (cached per supabase client)"""
        if self._token_middleware is None:
            try:
                from ..auth.billing_middleware import TokenValidationMiddleware

                # Use property to get supabase (lazy-loaded)
                self._token_middleware = TokenValidationMiddleware(self.supabase)
            except Exception as e:
                logger.warning("Failed to initialize token middleware: %s", e)
                raise ChatServiceError(f"Token middleware unavailable: {e}") from e
        return self._token_middleware

    def _initialize_services(self):
        """
        Initialize all modular services using the service factory.
        Services are cached and reused to avoid expensive re-initialization.
        Clients are lazy-loaded, so validation happens when actually used.
        """
        try:
            # Conversation management service (shared per org, safe to cache)
            self.conversation_manager = get_conversation_manager(org_id=self.org_id)

            # Per-request services (fresh instances to avoid context leakage)
            self.document_retrieval = get_document_retrieval_service(
                org_id=self.org_id, context_config=None
            )
            self.response_generator = get_response_generation_service(
                org_id=self.org_id,
                chatbot_config=self.chatbot_config,
                context_config=None,
            )
            self.analytics = get_analytics_service(
                org_id=self.org_id, context_config=None
            )

            # Stateless utilities and shared error handler
            self.utilities = get_chat_utilities()
            self.error_handler = get_error_handling_service(org_id=self.org_id)

            logger.info(
                "✅ ChatService dependencies initialized for org %s", self.org_id
            )

        except Exception as e:
            logger.error("Failed to initialize modular services: %s", e)
            raise ChatServiceError(f"Service initialization failed: {e}") from e

    async def process_message(
        self,
        message: str,
        session_id: str,
        channel: Optional[Channel] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for processing chat messages.
        Coordinates all modular services to handle the request.
        """
        start_time = time.time()

        # Input validation
        validation_result = self.utilities.validate_message_content(message)
        if not validation_result["valid"]:
            return await self.error_handler.create_fallback_response(
                error_message="Invalid message: "
                + "; ".join(validation_result["errors"])
            )

        try:
            # EMERGENCY FAST MODE: Skip vector search for simple queries
            from .fast_chat_service import FastChatMode

            fast_response = await FastChatMode.get_fast_response(
                message, self.chatbot_config
            )
            if fast_response:
                logger.info("⚡ Using fast mode - no vector search needed")
                return fast_response
            # OPTIMIZATION: Parallelize independent operations
            # Load context config and get/create conversation in parallel
            config_task = self._load_context_config()
            # Prepare metadata for conversation
            conversation_metadata = {}
            if channel:
                conversation_metadata["channel"] = channel.value

            conversation_task = self.conversation_manager.get_or_create_conversation(
                session_id=session_id,
                chatbot_id=self.chatbot_config.get("id"),
                channel=channel.value if channel else None,
                metadata=conversation_metadata if conversation_metadata else None,
            )

            # Wait for both to complete
            await config_task
            conversation = await conversation_task

            # Update services with current context config
            self._update_services_config()

            # OPTIMIZATION: Parallelize adding user message and getting history
            # First add the user message, then get history
            add_message_task = self.conversation_manager.add_message(
                conversation_id=conversation["id"], role="user", content=message
            )

            # Get conversation history (before the current message)
            history_task = self.conversation_manager.get_conversation_history(
                conversation_id=conversation["id"], limit=10
            )

            # Wait for both operations
            await add_message_task
            history = await history_task

            # Step 4: Enhanced query processing and document retrieval (with performance tracking)
            async with performance_monitor.track_operation(
                "query_enhancement", {"org_id": self.org_id}
            ):
                enhanced_queries = await self.response_generator.enhance_query(
                    message, history
                )

            async with performance_monitor.track_operation(
                "document_retrieval",
                {"org_id": self.org_id, "query_count": len(enhanced_queries)},
            ):
                try:
                    logger.info(
                        "Starting document retrieval",
                        extra={
                            "org_id": self.org_id,
                            "namespace": self.namespace,
                            "query_count": len(enhanced_queries),
                            "queries_preview": [q[:50] for q in enhanced_queries[:3]],
                            "has_pinecone_index": self.index is not None,
                        },
                    )
                    documents = await self.document_retrieval.retrieve_documents(
                        queries=enhanced_queries
                    )
                    logger.info(
                        "Document retrieval completed",
                        extra={
                            "org_id": self.org_id,
                            "documents_retrieved": len(documents),
                            "has_documents": len(documents) > 0,
                        },
                    )
                except Exception as retrieval_error:
                    # If retrieval fails, continue with empty documents (fallback mode)
                    logger.error(
                        "Document retrieval failed, continuing without context",
                        extra={
                            "org_id": self.org_id,
                            "namespace": self.namespace,
                            "error": str(retrieval_error),
                            "error_type": type(retrieval_error).__name__,
                            "has_pinecone_index": self.index is not None,
                        },
                        exc_info=True,
                    )
                    documents = []

            # Step 5: Generate response with context (with performance tracking)
            async with performance_monitor.track_operation(
                "response_generation", {"org_id": self.org_id}
            ):
                response_data = (
                    await self.response_generator.generate_enhanced_response(
                        message=message,
                        conversation_history=history,
                        retrieved_documents=documents,
                    )
                )

            # Step 5.5: Consume tokens if entity information is available
            if self.entity_id and self.entity_type and "tokens_used" in response_data:
                try:
                    # Use provided channel or default to website
                    if channel is None:
                        channel = Channel.WEBSITE

                    # Consume tokens using the middleware
                    # The middleware creates TokenUsageRequest internally from these parameters
                    await self.token_middleware.validate_and_consume_tokens(
                        entity_id=self.entity_id,
                        entity_type=self.entity_type,
                        estimated_tokens=response_data["tokens_used"],
                        requesting_user_id=self.entity_id,  # Entity ID is the requesting user
                        operation_type="chat",
                        channel=channel,
                        chatbot_id=self.chatbot_config.get("id"),
                        session_id=session_id,
                        user_identifier=self.entity_id,
                    )

                    logger.info(
                        "Consumed %s tokens for %s %s",
                        response_data["tokens_used"],
                        self.entity_type,
                        self.entity_id,
                    )

                except Exception as e:
                    logger.warning("Failed to consume tokens: %s", str(e))
                    # Don't fail the chat request if token consumption fails

            # OPTIMIZATION: Save assistant message first (needed for analytics)
            processing_time = int((time.time() - start_time) * 1000)
            assistant_message = await self.conversation_manager.add_message(
                conversation_id=conversation["id"],
                role="assistant",
                content=response_data["response"],
                metadata={
                    "sources": response_data.get("sources", []),
                    "context_quality": response_data.get("context_quality", {}),
                    "processing_time_ms": processing_time,
                },
            )

            # OPTIMIZATION: Log analytics asynchronously (don't wait for it)
            # This doesn't block the response
            # Save task to prevent premature garbage collection
            analytics_task = asyncio.create_task(
                self.analytics.log_analytics(
                    conversation_id=conversation["id"],
                    message_id=assistant_message["id"],
                    query_original=message,
                    response_data=response_data,
                    processing_time=processing_time,
                )
            )
            self._track_background_task(analytics_task)

            # Step 8: Return comprehensive response
            return {
                "response": response_data["response"],
                "sources": response_data.get("sources", []),
                "conversation_id": conversation["id"],
                "message_id": assistant_message["id"],
                "processing_time_ms": processing_time,
                "context_quality": response_data.get("context_quality", {}),
                "config_used": self.context_config.config_name
                if self.context_config
                else "default",
            }

        except openai.OpenAIError as e:
            error_info = self.error_handler.handle_openai_error(e, "process_message")
            return await self.error_handler.create_fallback_response(
                error_info["message"]
            )

        except ConnectionError as e:
            error_info = self.error_handler.handle_database_error(e, "process_message")
            return await self.error_handler.create_fallback_response(
                error_info["message"]
            )

        except Exception as e:
            # Log the actual error for debugging
            logger.error(
                "Unexpected error in process_message: %s",
                str(e),
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "session_id": session_id,
                    "message_length": len(message),
                },
            )

            # Record unexpected error
            self.error_handler.record_error(
                error=e,
                context="process_message_unexpected",
                metadata={"message_length": len(message), "session_id": session_id},
            )

            # Return more informative error message
            error_detail = (
                f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
            )
            return await self.error_handler.create_fallback_response(
                f"An unexpected error occurred ({error_detail}). Please try again."
            )

    async def _load_context_config(self):
        """Load context configuration for the organization"""
        try:
            self.context_config = await context_config_manager.get_config(self.org_id)
            logger.debug("Context config loaded for org %s", self.org_id)
        except Exception as e:
            logger.warning(
                "Failed to load context config for org %s: %s", self.org_id, e
            )
            # Use default config - create a basic config object
            self.context_config = type(
                "DefaultConfig",
                (),
                {
                    "config_name": "default",
                    "model_tier": "standard",
                    "business_context": "",
                    "specialized_instructions": "",
                    "retrieval_settings": {},
                },
            )()

    def _update_services_config(self):
        """Update modular services with current context configuration"""
        if self.context_config:
            # Update response generator
            self.response_generator.context_config = self.context_config

            # Update analytics service
            self.analytics.context_config = self.context_config

            # Update document retrieval service
            self.document_retrieval.context_config = self.context_config

            # Update retrieval configuration based on context config
            if hasattr(self.context_config, "retrieval_settings"):
                self.retrieval_config.update(self.context_config.retrieval_settings)

    # ==========================================
    # CONVENIENCE METHODS FOR EXTERNAL ACCESS
    # ==========================================

    async def get_recent_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversations for this organization"""
        return await self.conversation_manager.get_recent_conversations(limit=limit)

    async def add_feedback(
        self, message_id: str, rating: int, feedback_text: Optional[str] = None
    ) -> bool:
        """Add user feedback"""
        return self.analytics.track_user_feedback(
            message_id=message_id, rating=rating, feedback_text=feedback_text
        )

    def get_performance_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get performance insights for this organization"""
        return self.analytics.get_performance_insights(days=days)

    async def update_context_config(self, updates: Dict[str, Any]) -> bool:
        """Update context engineering configuration"""
        try:
            success = await context_config_manager.update_config(self.org_id, updates)
            if success:
                # Reload configuration
                await self._load_context_config()
                self._update_services_config()
            return success
        except Exception as e:
            self.error_handler.record_error(
                error=e, context="update_context_config", metadata={"updates": updates}
            )
            return False

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all modular services"""
        status = {
            "main_service": "operational",
            "conversation_manager": "operational"
            if self.conversation_manager
            else "failed",
            "document_retrieval": "operational"
            if self.document_retrieval
            else "failed",
            "response_generator": "operational"
            if self.response_generator
            else "failed",
            "analytics": "operational" if self.analytics else "failed",
            "utilities": "operational" if self.utilities else "failed",
            "error_handler": "operational" if self.error_handler else "failed",
            "context_config_loaded": bool(self.context_config),
            "org_id": self.org_id,
            "namespace": self.namespace,
        }

        # Overall health check
        all_operational = all(
            status[key] == "operational"
            for key in [
                "conversation_manager",
                "document_retrieval",
                "response_generator",
                "analytics",
                "utilities",
                "error_handler",
            ]
        )
        status["overall_health"] = "healthy" if all_operational else "degraded"

        return status

    # ==========================================
    # BACKWARD COMPATIBILITY METHODS
    # ==========================================

    async def chat(
        self,
        message: str,
        session_id: str,
        chatbot_id: Optional[str] = None,
        channel: Optional[Channel] = None,
    ) -> Dict[str, Any]:
        """Backward compatibility method that wraps process_message"""
        # chatbot_id is ignored since chatbot_config is set during initialization
        return await self.process_message(
            message=message, session_id=session_id, channel=channel
        )

    def _track_background_task(self, task: asyncio.Task):
        """Keep track of detached tasks so failures surface in logs."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._log_background_task_error)

    @staticmethod
    def _log_background_task_error(task: asyncio.Task):
        """Log any exception raised by a detached analytics task."""
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return
        if exception:
            logger.warning("Background task failed: %s", exception)
