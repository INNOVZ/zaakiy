import sys
import types

import pytest

# Stub external dependencies that aren't available in the test environment
if "pinecone" not in sys.modules:
    pinecone_stub = types.ModuleType("pinecone")

    class _StubPinecone:  # pragma: no cover - simple dependency shim
        def __init__(self, *args, **kwargs):
            pass

        def Index(self, *args, **kwargs):
            class _Index:
                def describe_index_stats(self):
                    return {}

            return _Index()

    pinecone_stub.Pinecone = _StubPinecone
    sys.modules["pinecone"] = pinecone_stub


if "app.services.storage.vector_management" not in sys.modules:
    vm_stub = types.ModuleType("app.services.storage.vector_management")

    class _QueryBatchDeletion:  # pragma: no cover - simple dependency shim
        pass

    class _VectorDeletionStrategy:
        pass

    class _VectorManagementService:
        def __init__(self, *args, **kwargs):
            pass

    vm_stub.QueryBatchDeletion = _QueryBatchDeletion
    vm_stub.VectorDeletionStrategy = _VectorDeletionStrategy
    vm_stub.VectorManagementService = _VectorManagementService
    vm_stub.vector_management_service = _VectorManagementService()
    sys.modules["app.services.storage.vector_management"] = vm_stub

from app.services.chat.response_generation_service import ResponseGenerationService


def _build_service():
    """Create a minimal ResponseGenerationService for direct unit tests."""
    chatbot_config = {
        "name": "Zaakiy AI",
        "tone": "friendly",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 300,
    }

    return ResponseGenerationService(
        org_id="test-org",
        openai_client=None,
        context_config=None,
        chatbot_config=chatbot_config,
    )


class _DummyCache:
    """Minimal async cache stub for unit tests."""

    def __init__(self):
        self.storage = {}
        self.ttl = {}

    async def get(self, key):
        return self.storage.get(key)

    async def set(self, key, value, ttl_seconds):
        """Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to store (should be dict-like for response data)
            ttl_seconds: Time to live in seconds
        """
        # Validate that value is dict-like (has keys/items) for response data
        # If it's already a dict, make a shallow copy to avoid mutation issues
        if isinstance(value, dict):
            self.storage[key] = dict(value)
        elif hasattr(value, "items") and callable(getattr(value, "items")):
            # Handle dict-like objects (e.g., OrderedDict, defaultdict)
            self.storage[key] = dict(value.items())
        else:
            # For test flexibility, store non-dict values as-is
            # In production, cache service should only store dict-like response data
            self.storage[key] = value
        self.ttl[key] = ttl_seconds


def test_forbidden_phrase_removed_with_curly_apostrophe():
    """Ensure smart apostrophes don't bypass the forbidden phrase filter."""
    service = _build_service()
    raw_response = "I don’t have information about an office in Spain."
    context_data = {
        "demo_links": ["https://example.com/connect"],
        "contact_info": {},
    }
    cleaned = service._remove_forbidden_phrases(
        raw_response, context_data, "Do you have office in Spain?"
    )

    assert "don't have information" not in cleaned.lower()
    assert "don't have information about" not in cleaned.lower()
    # New Keplero-style response should contain constructive alternatives
    assert (
        "consultation" in cleaned.lower()
        or "team" in cleaned.lower()
        or "help" in cleaned.lower()
        or "collaborate" in cleaned.lower()
    )


def test_forbidden_phrase_straight_apostrophe():
    """Test detection with straight apostrophe."""
    service = _build_service()
    raw_response = "I don't have information about an office in Spain."
    context_data = {
        "demo_links": [],
        "contact_info": {},
    }
    cleaned = service._remove_forbidden_phrases(
        raw_response, context_data, "Do you have office in Spain?"
    )

    assert "don't have information" not in cleaned.lower()
    assert "don't have information about" not in cleaned.lower()


def test_forbidden_phrase_variations():
    """Test various forbidden phrase variations."""
    service = _build_service()

    test_cases = [
        "I don't have that information available.",
        "I don't have information about that.",
        "I don't have that information.",
        "I don't know about that.",
        "That information is not available.",
        "I can't help with that.",
    ]

    context_data = {
        "demo_links": ["https://example.com/connect"],
        "contact_info": {},
    }

    for raw_response in test_cases:
        cleaned = service._remove_forbidden_phrases(
            raw_response, context_data, "Test query"
        )

        # Verify forbidden phrases are removed
        assert "don't have that information available" not in cleaned.lower()
        assert "don't have information about" not in cleaned.lower()
        assert "don't have that information" not in cleaned.lower()
        assert "don't know" not in cleaned.lower()
        assert "information is not available" not in cleaned.lower()
        assert "can't help with that" not in cleaned.lower()

        # Verify replacement is constructive
        assert len(cleaned) > 0
        assert "team" in cleaned.lower() or "help" in cleaned.lower()


def test_location_query_detection():
    """Test that location queries trigger proper rewriting."""
    service = _build_service()
    raw_response = "I don't have information about an office in Spain."
    context_data = {
        "demo_links": ["https://example.com/connect"],
        "contact_info": {},
    }
    cleaned = service._remove_forbidden_phrases(
        raw_response, context_data, "Do you have office in Spain?"
    )

    # Should detect location query and rewrite accordingly
    assert "don't have information" not in cleaned.lower()
    assert "office locations" in cleaned.lower() or "locations" in cleaned.lower()
    assert "team" in cleaned.lower()


def test_demo_query_detection():
    """Test that demo queries trigger proper rewriting."""
    service = _build_service()
    raw_response = "I don't have that information available."
    context_data = {
        "demo_links": ["https://example.com/demo"],
        "contact_info": {},
    }
    cleaned = service._remove_forbidden_phrases(
        raw_response, context_data, "Is free demo available?"
    )

    # Should detect demo query and rewrite accordingly
    assert "don't have that information available" not in cleaned.lower()
    assert "demo" in cleaned.lower() or "available" in cleaned.lower()
    assert "team" in cleaned.lower()


def test_safe_response_not_modified():
    """Test that safe responses without forbidden phrases are not modified."""
    service = _build_service()
    safe_response = (
        "Yes, we have an office in Spain. You can contact us at info@example.com."
    )
    context_data = {
        "demo_links": [],
        "contact_info": {},
    }
    cleaned = service._remove_forbidden_phrases(
        safe_response, context_data, "Do you have office in Spain?"
    )

    # Safe response should pass through unchanged (or with minimal changes)
    assert "office in Spain" in cleaned
    assert "don't have information" not in cleaned.lower()


@pytest.mark.asyncio
async def test_cached_response_is_cleaned_and_refreshed(monkeypatch):
    """Old cached responses should be rewritten and stored back in cache."""
    service = _build_service()
    dummy_cache = _DummyCache()

    monkeypatch.setattr(
        "app.services.chat.response_generation_service.cache_service", dummy_cache
    )

    query = "Do you have office in Spain?"
    cache_key = service._generate_response_cache_key(query, [])
    dummy_cache.storage[cache_key] = {
        "response": "I don't have information about an office in Spain.",
        "demo_links": ["https://example.com/book"],
        "contact_info": {},
    }

    cached_response = await service._get_cached_response(query, [])

    assert cached_response is not None
    assert "don't have information" not in cached_response["response"].lower()
    assert (
        dummy_cache.storage[cache_key]["response"] == cached_response["response"]
    ), "sanitized response should be written back to cache"
