"""
Unit tests for WhatsApp Configuration Cache

Tests the multi-layer caching system to ensure:
- Cache hits/misses are tracked correctly
- TTL expiration works as expected
- Invalidation clears all cache layers
- LRU eviction works when cache is full
"""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.services.whatsapp.whatsapp_config_cache import WhatsAppConfigCache


class TestWhatsAppConfigCache:
    """Test suite for WhatsApp configuration caching"""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing"""
        redis_mock = MagicMock()
        redis_mock.get.return_value = None
        redis_mock.setex.return_value = True
        redis_mock.delete.return_value = 1
        return redis_mock

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache instance with mocked Redis"""
        return WhatsAppConfigCache(
            redis_client=mock_redis,
            ttl_seconds=2,  # Short TTL for testing
            max_memory_cache_size=3,  # Small size for testing eviction
        )

    def test_cache_miss_on_first_access(self, cache):
        """Test that first access results in cache miss"""
        result = cache.get("org_123")
        assert result is None

        stats = cache.get_stats()
        assert stats["cache_misses"] == 1
        assert stats["memory_hits"] == 0

    def test_cache_hit_after_set(self, cache):
        """Test that cache returns stored value"""
        config = {"twilio_account_sid": "AC123", "twilio_phone_number": "+1234567890"}

        cache.set("org_123", config)
        result = cache.get("org_123")

        assert result == config

        stats = cache.get_stats()
        assert stats["memory_hits"] == 1
        assert stats["cache_misses"] == 0

    def test_cache_expiration(self, cache):
        """Test that cache entries expire after TTL"""
        config = {"twilio_account_sid": "AC123"}

        cache.set("org_123", config)

        # Should hit cache immediately
        result = cache.get("org_123")
        assert result == config

        # Wait for TTL to expire
        time.sleep(2.5)

        # Should miss cache after expiration
        result = cache.get("org_123")
        assert result is None

        stats = cache.get_stats()
        assert stats["cache_misses"] == 1  # Expired entry counts as miss

    def test_lru_eviction(self, cache):
        """Test that LRU eviction works when cache is full"""
        # Fill cache to max size (3)
        cache.set("org_1", {"id": 1})
        cache.set("org_2", {"id": 2})
        cache.set("org_3", {"id": 3})

        # Add one more - should evict oldest (org_1)
        cache.set("org_4", {"id": 4})

        # org_1 should be evicted
        assert cache.get("org_1") is None

        # Others should still be cached
        assert cache.get("org_2") == {"id": 2}
        assert cache.get("org_3") == {"id": 3}
        assert cache.get("org_4") == {"id": 4}

    def test_invalidation_clears_cache(self, cache):
        """Test that invalidation removes entry from cache"""
        config = {"twilio_account_sid": "AC123"}

        cache.set("org_123", config)
        assert cache.get("org_123") == config

        # Invalidate
        cache.invalidate("org_123")

        # Should miss after invalidation
        assert cache.get("org_123") is None

        stats = cache.get_stats()
        assert stats["invalidations"] == 1

    def test_redis_fallback_on_memory_miss(self, cache, mock_redis):
        """Test that Redis is checked when memory cache misses"""
        config = {"twilio_account_sid": "AC123"}

        # Simulate Redis having the data
        mock_redis.get.return_value = json.dumps(config)

        result = cache.get("org_123")

        # Should get from Redis
        assert result == config

        # Should have called Redis
        mock_redis.get.assert_called_once()

        stats = cache.get_stats()
        assert stats["redis_hits"] == 1

    def test_redis_stores_on_set(self, cache, mock_redis):
        """Test that set() stores in both memory and Redis"""
        config = {"twilio_account_sid": "AC123"}

        cache.set("org_123", config)

        # Should have called Redis setex
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args

        # Verify Redis key format
        assert call_args[0][0] == "whatsapp:config:org_123"
        # Verify TTL
        assert call_args[0][1] == 2
        # Verify data is JSON
        assert json.loads(call_args[0][2]) == config

    def test_graceful_redis_failure(self, cache, mock_redis):
        """Test that cache works even if Redis fails"""
        # Make Redis raise an exception
        mock_redis.get.side_effect = Exception("Redis connection failed")
        mock_redis.setex.side_effect = Exception("Redis connection failed")

        config = {"twilio_account_sid": "AC123"}

        # Should still work with memory cache only
        cache.set("org_123", config)
        result = cache.get("org_123")

        assert result == config

    def test_cache_stats_accuracy(self, cache):
        """Test that cache statistics are tracked correctly"""
        # Initial stats
        stats = cache.get_stats()
        assert stats["total_requests"] == 0
        assert stats["hit_rate_percent"] == 0.0

        # Cache miss
        cache.get("org_1")

        # Cache hit
        cache.set("org_1", {"id": 1})
        cache.get("org_1")

        # Another cache hit
        cache.get("org_1")

        stats = cache.get_stats()
        assert stats["total_requests"] == 3
        assert stats["cache_misses"] == 1
        assert stats["memory_hits"] == 2
        assert stats["hit_rate_percent"] == pytest.approx(66.67, rel=0.1)

    def test_clear_all(self, cache):
        """Test that clear_all removes all cached entries"""
        cache.set("org_1", {"id": 1})
        cache.set("org_2", {"id": 2})
        cache.set("org_3", {"id": 3})

        cache.clear_all()

        # All should be cleared
        assert cache.get("org_1") is None
        assert cache.get("org_2") is None
        assert cache.get("org_3") is None

        stats = cache.get_stats()
        assert stats["memory_cache_size"] == 0

    def test_concurrent_access_same_org(self, cache):
        """Test that multiple accesses to same org work correctly"""
        config = {"twilio_account_sid": "AC123"}

        cache.set("org_123", config)

        # Multiple reads should all hit cache
        for _ in range(10):
            result = cache.get("org_123")
            assert result == config

        stats = cache.get_stats()
        assert stats["memory_hits"] == 10
        assert stats["cache_misses"] == 0


class TestWhatsAppConfigCacheIntegration:
    """Integration tests for cache with WhatsAppService"""

    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client"""
        supabase_mock = MagicMock()

        # Mock successful config fetch
        response_mock = MagicMock()
        response_mock.data = [
            {
                "id": "config_123",
                "org_id": "org_123",
                "twilio_account_sid": "AC123",
                "twilio_auth_token": "token123",
                "twilio_phone_number": "+1234567890",
                "is_active": True,
            }
        ]

        supabase_mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = (
            response_mock
        )

        return supabase_mock

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    def test_whatsapp_service_uses_cache(self, mock_get_supabase, mock_supabase):
        """Test that WhatsAppService properly uses the cache"""
        from app.services.whatsapp.whatsapp_service import WhatsAppService

        mock_get_supabase.return_value = mock_supabase

        service = WhatsAppService(org_id="org_123")

        # First call - should hit database
        config1 = service._get_whatsapp_config()

        # Second call - should hit cache (no DB query)
        config2 = service._get_whatsapp_config()

        # Should be same config
        assert config1 == config2

        # Database should only be called once (second call uses instance cache)
        assert mock_supabase.table.call_count == 1

        # The second call uses instance cache (self._config), not shared cache
        # So we verify the instance cache is working by checking DB call count
        # For shared cache testing, we need multiple service instances

        # Create a new service instance (different instance, same org)
        service2 = WhatsAppService(org_id="org_123")
        config3 = service2._get_whatsapp_config()

        # This should hit the shared cache (memory or Redis)
        # Database should still only be called once
        assert mock_supabase.table.call_count == 1
        assert config3 == config1

        # Now check shared cache stats - should have at least one hit
        stats = service2.get_cache_stats()
        # The second service instance should have gotten a cache hit
        assert stats["memory_hits"] >= 1 or stats["redis_hits"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
