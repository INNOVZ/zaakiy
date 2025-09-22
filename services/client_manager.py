"""
Singleton client manager for shared OpenAI, Pinecone, and Supabase connections
"""
import os
import logging
from typing import Optional
from threading import Lock
import openai
from pinecone import Pinecone
from supabase import create_client, Client
from supabase.client import ClientOptions  # Import ClientOptions

logger = logging.getLogger(__name__)


class ClientManager:
    """Singleton manager for shared API clients"""

    _instance: Optional['ClientManager'] = None
    _lock = Lock()

    def __new__(cls) -> 'ClientManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._openai_client: Optional[openai.OpenAI] = None
        self._pinecone_client: Optional[Pinecone] = None
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
                max_retries=3   # Add retry logic
            )
            logger.info("✅ OpenAI client initialized")

        except Exception as e:
            logger.error("❌ Failed to initialize OpenAI client: %s", str(e))
            raise

        try:
            # Initialize Pinecone client
            pinecone_key = os.getenv("PINECONE_API_KEY")
            pinecone_index = os.getenv("PINECONE_INDEX")

            if not pinecone_key or not pinecone_index:
                raise ValueError("Pinecone credentials not found")

            self._pinecone_client = Pinecone(api_key=pinecone_key)
            self._pinecone_index = self._pinecone_client.Index(pinecone_index)
            logger.info("✅ Pinecone client initialized")

        except Exception as e:
            logger.error("❌ Failed to initialize Pinecone client: %s", str(e))
            raise

        try:
            # Initialize Supabase client - FIXED
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                raise ValueError("Supabase credentials not found")

            # Create proper ClientOptions object
            options = ClientOptions(
                auto_refresh_token=True,
                persist_session=False
            )

            self._supabase_client = create_client(
                supabase_url,
                supabase_key,
                options=options  # Use proper ClientOptions object
            )
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
        health = {
            "openai": False,
            "pinecone": False,
            "supabase": False
        }

        try:
            # Test OpenAI
            self._openai_client.models.list()
            health["openai"] = True
        except Exception as e:
            logger.warning("OpenAI health check failed: %s", str(e))

        try:
            # Test Pinecone
            self._pinecone_client.list_indexes()
            health["pinecone"] = True
        except Exception as e:
            logger.warning("Pinecone health check failed: %s", str(e))

        try:
            # Test Supabase
            self._supabase_client.table(
                "organizations").select("id").limit(1).execute()
            health["supabase"] = True
        except Exception as e:
            logger.warning("Supabase health check failed: %s", str(e))

        return health


# Global singleton instance
client_manager = ClientManager()
