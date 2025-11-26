"""
Tests for Robust Cache Implementation
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.robust_cache import (
    CacheKeyGenerator,
    RobustCacheService,
    SmartTTLManager,
    cached,
)


class TestCacheKeyGenerator:
    """Test cache key generation"""

    def test_normalize_text(self):
        """Test text normalization"""
        # Case sensitivity
        assert CacheKeyGenerator.normalize_text("Hello World") == "hello world"
        assert CacheKeyGenerator.normalize_text("HELLO WORLD") == "hello world"

        # Extra whitespace
        assert CacheKeyGenerator.normalize_text("hello  world") == "hello world"
        assert CacheKeyGenerator.normalize_text("  hello world  ") == "hello world"

        # Punctuation
        assert CacheKeyGenerator.normalize_text("hello world?") == "hello world"
        assert CacheKeyGenerator.normalize_text("hello world!") == "hello world"
        assert CacheKeyGenerator.normalize_text("hello world.") == "hello world"

        # Filler words
        assert CacheKeyGenerator.normalize_text("please tell me") == "tell me"
        assert CacheKeyGenerator.normalize_text("can you help") == "help"

    def test_generate_key_consistency(self):
        """Test that same input generates same key"""
        text1 = "What are your products?"
        text2 = "what are your products"  # Different case, punctuation

        key1 = CacheKeyGenerator.generate_key("test", text1, normalize=True)
        key2 = CacheKeyGenerator.generate_key("test", text2, normalize=True)

        # Should be same after normalization
        assert key1 == key2

    def test_generate_key_different_inputs(self):
        """Test that different inputs generate different keys"""
        key1 = CacheKeyGenerator.generate_key("test", "hello")
        key2 = CacheKeyGenerator.generate_key("test", "world")

        assert key1 != key2

    def test_generate_embedding_key(self):
        """Test embedding key generation"""
        query = "What are your products?"

        key1 = CacheKeyGenerator.generate_embedding_key(query)
        key2 = CacheKeyGenerator.generate_embedding_key(query)

        # Should be consistent
        assert key1 == key2
        assert key1.startswith("embedding:")

    def test_generate_embedding_key_normalization(self):
        """Test embedding key normalization"""
        key1 = CacheKeyGenerator.generate_embedding_key("What are your products?")
        key2 = CacheKeyGenerator.generate_embedding_key("what are your products")

        # Should be same after normalization
        assert key1 == key2


class TestSmartTTLManager:
    """Test smart TTL management"""

    def test_default_ttls(self):
        """Test default TTL values"""
        manager = SmartTTLManager()

        assert manager.get_ttl("embedding") == 3600  # 1 hour
        assert manager.get_ttl("conversation") == 1800  # 30 minutes
        assert manager.get_ttl("chatbot_config") == 7200  # 2 hours

    def test_access_based_ttl(self):
        """Test TTL adjustment based on access count"""
        manager = SmartTTLManager()

        # Low access
        ttl_low = manager.get_ttl("embedding", access_count=5)

        # Medium access
        ttl_medium = manager.get_ttl("embedding", access_count=25)

        # High access
        ttl_high = manager.get_ttl("embedding", access_count=60)

        # Very high access
        ttl_very_high = manager.get_ttl("embedding", access_count=150)

        # Should increase with access count
        assert ttl_low < ttl_medium < ttl_high < ttl_very_high

    def test_ttl_cap(self):
        """Test TTL is capped at 24 hours"""
        manager = SmartTTLManager()

        # Even with very high access, should not exceed 24 hours
        ttl = manager.get_ttl("embedding", access_count=1000, base_ttl=50000)
        assert ttl <= 86400  # 24 hours

    def test_record_access(self):
        """Test access recording"""
        manager = SmartTTLManager()

        key = "test_key"

        # Record multiple accesses
        for _ in range(5):
            manager.record_access(key)

        assert manager.get_access_count(key) == 5


class TestRobustCacheService:
    """Test robust cache service"""

    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service"""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        return cache

    @pytest.fixture
    def robust_cache(self, mock_cache_service):
        """Create robust cache instance"""
        return RobustCacheService(mock_cache_service)

    @pytest.mark.asyncio
    async def test_get_hit(self, robust_cache, mock_cache_service):
        """Test cache hit"""
        mock_cache_service.get.return_value = "cached_value"

        value = await robust_cache.get("test_key")

        assert value == "cached_value"
        assert robust_cache.hits == 1
        assert robust_cache.misses == 0

    @pytest.mark.asyncio
    async def test_get_miss(self, robust_cache, mock_cache_service):
        """Test cache miss"""
        mock_cache_service.get.return_value = None

        value = await robust_cache.get("test_key")

        assert value is None
        assert robust_cache.hits == 0
        assert robust_cache.misses == 1

    @pytest.mark.asyncio
    async def test_get_error_handling(self, robust_cache, mock_cache_service):
        """Test error handling on get"""
        mock_cache_service.get.side_effect = Exception("Cache error")

        value = await robust_cache.get("test_key")

        # Should return None on error, not raise
        assert value is None
        assert robust_cache.errors == 1

    @pytest.mark.asyncio
    async def test_set_with_smart_ttl(self, robust_cache, mock_cache_service):
        """Test set with smart TTL"""
        await robust_cache.set("test_key", "value", data_type="embedding")

        # Should call underlying cache with smart TTL
        mock_cache_service.set.assert_called_once()
        call_args = mock_cache_service.set.call_args

        assert call_args[0][0] == "test_key"
        assert call_args[0][1] == "value"
        assert call_args[1]["ttl"] == 3600  # Default embedding TTL

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, robust_cache, mock_cache_service):
        """Test set with custom TTL"""
        await robust_cache.set("test_key", "value", ttl=7200)

        call_args = mock_cache_service.set.call_args
        assert call_args[1]["ttl"] == 7200

    @pytest.mark.asyncio
    async def test_get_or_set_cache_hit(self, robust_cache, mock_cache_service):
        """Test get_or_set with cache hit"""
        mock_cache_service.get.return_value = "cached_value"

        compute_fn = AsyncMock(return_value="computed_value")

        value = await robust_cache.get_or_set("test_key", compute_fn)

        # Should return cached value without computing
        assert value == "cached_value"
        compute_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_set_cache_miss(self, robust_cache, mock_cache_service):
        """Test get_or_set with cache miss"""
        mock_cache_service.get.return_value = None

        compute_fn = AsyncMock(return_value="computed_value")

        value = await robust_cache.get_or_set("test_key", compute_fn)

        # Should compute and cache value
        assert value == "computed_value"
        compute_fn.assert_called_once()
        mock_cache_service.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics(self, robust_cache, mock_cache_service):
        """Test metrics tracking"""
        # Simulate some cache operations
        mock_cache_service.get.side_effect = ["value1", None, "value2", None]

        await robust_cache.get("key1")  # Hit
        await robust_cache.get("key2")  # Miss
        await robust_cache.get("key3")  # Hit
        await robust_cache.get("key4")  # Miss

        metrics = robust_cache.get_metrics()

        assert metrics["hits"] == 2
        assert metrics["misses"] == 2
        assert metrics["hit_rate"] == 50.0
        assert metrics["total_requests"] == 4


class TestCachedDecorator:
    """Test cached decorator"""

    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test cached decorator functionality"""
        call_count = 0

        @cached(data_type="test", ttl=3600)
        async def expensive_function(arg1: str):
            nonlocal call_count
            call_count += 1
            return f"result_{arg1}"

        with patch("app.services.shared.cache_service") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            # First call - should execute function
            result1 = await expensive_function("test")
            assert result1 == "result_test"
            assert call_count == 1

            # Second call with cache hit - should not execute function
            mock_cache.get.return_value = "cached_result"
            result2 = await expensive_function("test")
            assert result2 == "cached_result"
            assert call_count == 1  # Not incremented


class TestConcurrentAccess:
    """Test concurrent cache access"""

    @pytest.mark.asyncio
    async def test_concurrent_get(self):
        """Test concurrent cache gets"""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="value")
        mock_cache.set = AsyncMock()

        robust_cache = RobustCacheService(mock_cache)

        # 100 concurrent gets
        tasks = [robust_cache.get("test_key") for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r == "value" for r in results)
        assert robust_cache.hits == 100

    @pytest.mark.asyncio
    async def test_concurrent_set(self):
        """Test concurrent cache sets"""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        robust_cache = RobustCacheService(mock_cache)

        # 50 concurrent sets
        tasks = [robust_cache.set(f"key_{i}", f"value_{i}") for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r is True for r in results)
        assert robust_cache.sets == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
