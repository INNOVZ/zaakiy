"""
Singleton client manager for shared OpenAI, Pinecone, and Supabase connections
"""
import logging
import os
from threading import Lock
from typing import Optional

import openai
from supabase import Client

logger = logging.getLogger(__name__)


class ClientManager:
    """Singleton manager for shared API clients"""

    _instance: Optional["ClientManager"] = None
    _lock = Lock()

    def __new__(cls) -> "ClientManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._openai_client: Optional[openai.OpenAI] = None
        self._pinecone_index = None
        self._supabase_client: Optional[Client] = None
        self._initialized = True

        # Initialize clients
        self._init_clients()

    def _init_clients(self):
        """Initialize all API clients with error handling"""
        try:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")

            self._openai_client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0,  # Add timeout
                max_retries=3,  # Add retry logic
            )
            logger.info("✅ OpenAI client initialized")

        except Exception as e:
            logger.error("❌ Failed to initialize OpenAI client: %s", str(e))
            raise

        try:
            # Use centralized Pinecone client
            from ..storage.pinecone_client import get_pinecone_index

            self._pinecone_index = get_pinecone_index()
            logger.info("✅ Pinecone client initialized")

        except Exception as e:
            logger.error("❌ Failed to initialize Pinecone client: %s", str(e))
            raise

        try:
            # Use centralized Supabase client
            from ..storage.supabase_client import get_supabase_client

            self._supabase_client = get_supabase_client()
            logger.info("✅ Supabase client initialized")

        except Exception as e:
            logger.error("❌ Failed to initialize Supabase client: %s", str(e))
            raise

    @property
    def openai(self) -> openai.OpenAI:
        """Get OpenAI client instance"""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not initialized")
        return self._openai_client

    @property
    def pinecone_index(self):
        """Get Pinecone index instance"""
        if self._pinecone_index is None:
            raise RuntimeError("Pinecone index not initialized")
        return self._pinecone_index

    @property
    def supabase(self) -> Client:
        """Get Supabase client instance"""
        if self._supabase_client is None:
            raise RuntimeError("Supabase client not initialized")
        return self._supabase_client

    def health_check(self) -> dict:
        """Check health of all clients"""
        health = {"openai": False, "pinecone": False, "supabase": False}

        try:
            # Test OpenAI
            self._openai_client.models.list()
            health["openai"] = True
        except Exception as e:
            logger.warning("OpenAI health check failed: %s", str(e))

        try:
            # Test Pinecone
            from ..storage.pinecone_client import get_pinecone_client

            pinecone_client = get_pinecone_client()
            pinecone_client.list_indexes()
            health["pinecone"] = True
        except Exception as e:
            logger.warning("Pinecone health check failed: %s", str(e))

        try:
            # Test Supabase
            self._supabase_client.table("organizations").select("id").limit(1).execute()
            health["supabase"] = True
        except Exception as e:
            logger.warning("Supabase health check failed: %s", str(e))

        return health


# Global singleton instance
client_manager = ClientManager()
