"""
Service Factory for Chat Services

Provides helper functions for constructing chat-related services while
reusing shared clients (Supabase, Pinecone, OpenAI). Stateful services
that keep per-request configuration are returned as fresh instances to
prevent cross-request data leakage.
"""
import logging
from threading import Lock
from typing import Dict, Optional, Tuple

from ..storage.pinecone_client import get_pinecone_index
from ..storage.supabase_client import get_supabase_client
from .analytics_service import AnalyticsService
from .chat_utilities import ChatUtilities
from .conversation_manager import ConversationManager
from .document_retrieval_service import DocumentRetrievalService
from .error_handling_service import ErrorHandlingService
from .response_generation_service import ResponseGenerationService

logger = logging.getLogger(__name__)

# Thread synchronization for global cache access
_cache_lock = Lock()
_clients_lock = Lock()

# Stateless singletons
_utilities_cache: Optional[ChatUtilities] = None

# Per-org shared services that do not mutate per-request state
_conversation_manager_cache: Dict[str, ConversationManager] = {}
_error_handler_cache: Dict[str, ErrorHandlingService] = {}

# Shared clients (singletons provided by their respective modules)
_shared_supabase = None
_shared_pinecone_index = None
_shared_openai_client = None


def _get_shared_clients() -> Tuple:
    """
    Get or initialize shared clients (Supabase, Pinecone index, OpenAI).
    These helpers already provide singleton semantics, so we simply cache
    the references for reuse across service builders.
    Thread-safe initialization with double-checked locking pattern.
    """
    global _shared_supabase, _shared_pinecone_index, _shared_openai_client

    # Double-checked locking pattern for thread-safe initialization
    if (
        _shared_supabase is None
        or _shared_pinecone_index is None
        or _shared_openai_client is None
    ):
        with _clients_lock:
            # Check again inside lock (double-checked locking)
            if _shared_supabase is None:
                _shared_supabase = get_supabase_client()

            if _shared_pinecone_index is None:
                _shared_pinecone_index = get_pinecone_index()

            if _shared_openai_client is None:
                try:
                    from ..shared import get_openai_client

                    _shared_openai_client = get_openai_client()
                except Exception as e:
                    logger.warning("Failed to get shared OpenAI client: %s", e)
                    # Fallback: try direct initialization
                    try:
                        import os

                        import openai

                        openai_key = os.getenv("OPENAI_API_KEY")
                        if openai_key:
                            _shared_openai_client = openai.OpenAI(api_key=openai_key)
                        else:
                            logger.warning("OpenAI API key not found")
                            _shared_openai_client = None
                    except ImportError:
                        logger.warning(
                            "OpenAI package is not installed. Install it with: pip install openai"
                        )
                        _shared_openai_client = None
                    except Exception as fallback_error:
                        logger.warning(
                            "Failed to initialize OpenAI client directly: %s",
                            fallback_error,
                        )
                        _shared_openai_client = None

    return _shared_supabase, _shared_pinecone_index, _shared_openai_client


def get_chat_utilities() -> ChatUtilities:
    """Return the shared ChatUtilities instance (thread-safe)."""
    global _utilities_cache
    if _utilities_cache is None:
        with _cache_lock:
            # Double-checked locking pattern
            if _utilities_cache is None:
                _utilities_cache = ChatUtilities()
                logger.debug("Created shared ChatUtilities instance")
    return _utilities_cache


def get_conversation_manager(org_id: str) -> ConversationManager:
    """Return a cached ConversationManager for the organization (thread-safe)."""
    manager = _conversation_manager_cache.get(org_id)
    if manager is None:
        with _cache_lock:
            # Double-checked locking pattern
            if org_id not in _conversation_manager_cache:
                supabase, _, _ = _get_shared_clients()
                manager = ConversationManager(org_id=org_id, supabase_client=supabase)
                _conversation_manager_cache[org_id] = manager
                logger.debug("Created ConversationManager for org %s", org_id)
            else:
                manager = _conversation_manager_cache[org_id]
    return manager


def get_document_retrieval_service(
    org_id: str, context_config=None
) -> DocumentRetrievalService:
    """
    Build a new DocumentRetrievalService for this request.
    Fresh instances prevent context_config leakage across requests.
    """
    _, pinecone_index, openai_client = _get_shared_clients()

    # Validate that openai_client is not None before use
    if openai_client is None:
        raise ValueError(
            "OpenAI client is not available. Please check OPENAI_API_KEY environment variable."
        )

    logger.debug("Creating DocumentRetrievalService for org %s", org_id)
    return DocumentRetrievalService(
        org_id=org_id,
        openai_client=openai_client,
        pinecone_index=pinecone_index,
        context_config=context_config,
    )


def get_response_generation_service(
    org_id: str, chatbot_config: dict, context_config=None
) -> ResponseGenerationService:
    """
    Build a new ResponseGenerationService for this request.
    Chatbot/context settings are isolated per instance.
    """
    _, _, openai_client = _get_shared_clients()

    # Validate that openai_client is not None before use
    if openai_client is None:
        raise ValueError(
            "OpenAI client is not available. Please check OPENAI_API_KEY environment variable."
        )

    logger.debug("Creating ResponseGenerationService for org %s", org_id)
    return ResponseGenerationService(
        org_id=org_id,
        openai_client=openai_client,
        context_config=context_config,
        chatbot_config=chatbot_config,
    )


def get_analytics_service(org_id: str, context_config=None) -> AnalyticsService:
    """Build a new AnalyticsService for this request."""
    supabase, _, _ = _get_shared_clients()
    logger.debug("Creating AnalyticsService for org %s", org_id)
    return AnalyticsService(
        org_id=org_id,
        supabase_client=supabase,
        context_config=context_config,
    )


def get_error_handling_service(org_id: str) -> ErrorHandlingService:
    """Return (or create) the shared ErrorHandlingService for the org (thread-safe)."""
    from app.utils.error_monitoring import error_monitor

    handler = _error_handler_cache.get(org_id)
    if handler is None:
        with _cache_lock:
            # Double-checked locking pattern
            if org_id not in _error_handler_cache:
                handler = ErrorHandlingService(
                    org_id=org_id, error_monitor=error_monitor
                )
                _error_handler_cache[org_id] = handler
                logger.debug("Created ErrorHandlingService for org %s", org_id)
            else:
                handler = _error_handler_cache[org_id]
    return handler
