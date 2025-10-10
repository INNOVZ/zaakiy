import httpx
import logging
from supabase import create_client, Client
from ...config.settings import get_database_config

logger = logging.getLogger(__name__)

# Get configuration
db_config = get_database_config()

# Validate required configuration
if not db_config.supabase_url:
    raise ValueError("SUPABASE_URL environment variable is required")
if not db_config.supabase_service_key:
    raise ValueError(
        "SUPABASE_SERVICE_ROLE_KEY environment variable is required")

logger.info("Initializing Supabase clients with centralized configuration")

# Create headers for HTTP client
headers = {
    "apikey": db_config.supabase_service_key,
    "Authorization": f"Bearer {db_config.supabase_service_key}",
    "Content-Type": "application/json",
}

# HTTP client for REST API calls
client = httpx.AsyncClient(
    base_url=f"{db_config.supabase_url}/rest/v1",
    headers=headers,
    timeout=30.0
)

# Supabase client for ORM-style operations
supabase: Client = create_client(
    db_config.supabase_url, db_config.supabase_service_key)

logger.info("Supabase clients initialized successfully")
