import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar

import httpx
from starlette.concurrency import run_in_threadpool
from supabase import Client, create_client

from ...config.settings import get_database_config
from ...utils.env_loader import is_test_environment

logger = logging.getLogger(__name__)

# Global variables for lazy initialization
_client: Optional[httpx.AsyncClient] = None
_supabase: Optional[Client] = None
_config_error: Optional[str] = None
T = TypeVar("T")


@dataclass
class _StubResult:
    data: list


class _StubSupabaseTable:
    """Minimal stub that mimics Supabase table chaining for tests."""

    def __init__(self, error: str):
        self._error = error

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def insert(self, *args, **kwargs):
        return self

    def upsert(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def delete(self, *args, **kwargs):
        return self

    def execute(self):
        return _StubResult(data=[])


class _StubSupabaseClient:
    """Stub Supabase client used when credentials are unavailable in tests."""

    def __init__(self, error: str):
        self._error = error
        self.auth = _StubAuthClient(error)

    def table(self, *args, **kwargs):
        return _StubSupabaseTable(self._error)

    def __getattr__(self, item):  # pragma: no cover - defensive
        raise RuntimeError(self._error)


class _StubAuthClient:
    """Stub authentication interface that mirrors Supabase client methods."""

    def __init__(self, error: str):
        self._error = error

    def sign_up(self, *args, **kwargs):  # pragma: no cover - simple stub
        raise RuntimeError(self._error)


def _build_configuration() -> Tuple[Optional[dict], Optional[str]]:
    """Load Supabase configuration and report missing keys."""

    config = get_database_config()
    missing = []

    if not config.supabase_url:
        missing.append("SUPABASE_URL")
    if not config.supabase_service_key:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if missing:
        message = (
            "Supabase configuration is incomplete. Missing environment variables: "
            + ", ".join(missing)
        )

        if is_test_environment():
            logger.warning(
                "%s. The Supabase client will remain disabled in this environment.",
                message,
            )
            return None, message

        raise ValueError(message)

    headers = {
        "apikey": config.supabase_service_key,
        "Authorization": f"Bearer {config.supabase_service_key}",
        "Content-Type": "application/json",
    }

    return {
        "config": config,
        "headers": headers,
    }, None


def get_supabase_client() -> Client:
    """Get the Supabase client instance with lazy initialization."""

    global _client, _supabase, _config_error

    if _supabase is not None:
        return _supabase

    logger.info("Initializing Supabase clients...")

    config_bundle, error = _build_configuration()
    _config_error = error

    if error:
        if is_test_environment():
            logger.warning("Using stub Supabase client for tests: %s", error)
            _supabase = _StubSupabaseClient(error)
            _client = None
            return _supabase

        raise RuntimeError(error)

    config = config_bundle["config"]
    headers = config_bundle["headers"]

    _client = httpx.AsyncClient(
        base_url=f"{config.supabase_url}/rest/v1", headers=headers, timeout=30.0
    )
    _supabase = create_client(config.supabase_url, config.supabase_service_key)

    logger.info("Supabase clients initialized successfully")

    return _supabase


def get_supabase_http_client():
    """Get the HTTP client instance for REST API calls"""
    # Initialize clients if not already done
    get_supabase_client()  # This will initialize both clients
    return _client


def get_connection_stats() -> dict:
    """Get connection statistics for monitoring"""
    # For now, return basic stats since we don't have detailed connection pooling
    if _config_error:
        return {
            "pool_size": 0,
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 1,
            "client_initialized": False,
            "error": _config_error,
        }

    current_supabase = get_supabase_client() if _supabase is None else _supabase
    return {
        "pool_size": 1,  # Single client instance
        "total_connections": 1,
        "active_connections": 1 if current_supabase else 0,
        "failed_connections": 0,
        "client_initialized": current_supabase is not None,
    }


async def run_supabase(operation: Callable[[], T]) -> T:
    """
    Execute a blocking Supabase operation inside a threadpool.

    This keeps async endpoints responsive while still using the
    synchronous Supabase Python client.
    """
    return await run_in_threadpool(operation)
