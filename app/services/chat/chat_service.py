"""
Main Chat Service Orchestrator
Lightweight coordinator that integrates all modular chat services
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import openai

from app.services.analytics.context_config import context_config_manager
from app.utils.error_monitoring import error_monitor
from app.utils.performance_monitor import performance_monitor

from .analytics_service import AnalyticsService
from .chat_utilities import ChatUtilities

# Import services
from .conversation_manager import ConversationManager
from .document_retrieval_service import DocumentRetrievalService
from .error_handling_service import ErrorHandlingService
from .response_generation_service import ResponseGenerationService

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

    This class has been refactored from a monolithic 1355-line file into a clean
    orchestrator pattern that delegates to specialized services for maintainability.
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

        # Initialize clients with lazy loading (avoid blocking initialization)
        try:
            import os

            from ..storage.pinecone_client import get_pinecone_index
            from ..storage.supabase_client import get_supabase_client

            # Initialize clients on demand
            self.supabase = get_supabase_client()
            self.index = get_pinecone_index()

            # Initialize OpenAI client
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            else:
                self.openai_client = None
                logger.warning("OpenAI API key not found")

            # Initialize billing middleware for token consumption
            from ..auth.billing_middleware import TokenValidationMiddleware

            self.token_middleware = TokenValidationMiddleware(self.supabase)

            logger.info(
                "✅ ChatService initialized with direct clients for org %s", org_id
            )

        except Exception as e:
            error_monitor.record_error(
                error_type="ChatServiceInitializationError",
                severity="critical",
                service="chat_service",
                category="initialization",
            )
            raise ChatServiceError(f"Service initialization failed: {e}") from e

        # Context engineering config will be loaded per request
        self.context_config = None

        # Initialize retrieval config with defaults (will be updated per request)
        # EMERGENCY MODE: Minimal docs for speed
        self.retrieval_config = {"initial": 3, "rerank": 2, "final": 2}
        self.max_context_length = 2000  # Reduced for speed

        # Initialize modular services
        self._initialize_services()

    def _initialize_services(self):
        """Initialize all modular services"""
        try:
            # Validate critical dependencies
            if self.openai_client is None:
                raise ChatServiceError(
                    "OpenAI client not initialized. Please check OPENAI_API_KEY environment variable."
                )

            if self.index is None:
                logger.warning(
                    "Pinecone index not initialized - document retrieval may fail"
                )

            # Conversation management service
            self.conversation_manager = ConversationManager(
                org_id=self.org_id, supabase_client=self.supabase
            )

            # Document retrieval service
            self.document_retrieval = DocumentRetrievalService(
                org_id=self.org_id,
                openai_client=self.openai_client,
                pinecone_index=self.index,
                context_config=None,  # Will be set per request
            )

            # Response generation service
            self.response_generator = ResponseGenerationService(
                org_id=self.org_id,
                openai_client=self.openai_client,
                context_config=None,  # Will be set per request
                chatbot_config=self.chatbot_config,
            )

            # Analytics service
            self.analytics = AnalyticsService(
                org_id=self.org_id,
                supabase_client=self.supabase,
                context_config=None,  # Will be set per request
            )

            # Utilities service
            self.utilities = ChatUtilities()

            # Error handling service
            self.error_handler = ErrorHandlingService(
                org_id=self.org_id, error_monitor=error_monitor
            )

            logger.info("✅ All modular services initialized for org %s", self.org_id)

        except Exception as e:
            logger.error("Failed to initialize modular services: %s", e)
            raise ChatServiceError(f"Service initialization failed: {e}") from e

    async def process_message(
        self,
        message: str,
        session_id: str,
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
            conversation_task = self.conversation_manager.get_or_create_conversation(
                session_id=session_id, chatbot_id=self.chatbot_config.get("id")
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
                    documents = await self.document_retrieval.retrieve_documents(
                        queries=enhanced_queries
                    )
                except Exception as retrieval_error:
                    # If retrieval fails, continue with empty documents (fallback mode)
                    logger.warning(
                        "Document retrieval failed, continuing without context: %s",
                        str(retrieval_error),
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
                    from app.models.subscription import Channel, TokenUsageRequest

                    # Determine channel based on context (default to website for now)
                    channel = (
                        Channel.WEBSITE
                    )  # Could be enhanced to detect actual channel

                    token_request = TokenUsageRequest(
                        entity_id=self.entity_id,
                        entity_type=self.entity_type,
                        tokens_consumed=response_data["tokens_used"],
                        operation_type="chat",
                        channel=channel,
                        chatbot_id=self.chatbot_config.get("id"),
                        session_id=session_id,
                        user_identifier=self.entity_id,  # Use entity_id as user identifier
                    )

                    # Consume tokens using the middleware
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
        self, message: str, session_id: str, chatbot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Backward compatibility method that wraps process_message"""
        # chatbot_id is ignored since chatbot_config is set during initialization
        return await self.process_message(message=message, session_id=session_id)
