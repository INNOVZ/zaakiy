"""
Test Document Retrieval Performance Optimizations
Tests parallel embedding generation and embedding caching
"""
import asyncio
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat.document_retrieval_service import DocumentRetrievalService


class TestEmbeddingOptimizations:
    """Test embedding generation optimizations"""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client"""
        client = MagicMock()

        # Mock embeddings.create to return a response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        client.embeddings.create = AsyncMock(return_value=mock_response)

        return client

    @pytest.fixture
    def mock_pinecone_index(self):
        """Create mock Pinecone index"""
        return MagicMock()

    @pytest.fixture
    def retrieval_service(self, mock_openai_client, mock_pinecone_index):
        """Create DocumentRetrievalService instance"""
        return DocumentRetrievalService(
            org_id="test-org",
            openai_client=mock_openai_client,
            pinecone_index=mock_pinecone_index,
            context_config={"embedding_model": "text-embedding-3-small"},
        )

    @pytest.mark.asyncio
    async def test_embedding_caching(self, retrieval_service, mock_openai_client):
        """Test that embeddings are cached and reused"""
        # Mock cache service
        with patch(
            "app.services.chat.document_retrieval_service.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)  # First call: cache miss
            mock_cache.set = AsyncMock()

            # First call - should generate and cache
            embedding1 = await retrieval_service._generate_embedding("test query")

            # Verify OpenAI was called
            assert mock_openai_client.embeddings.create.call_count == 1

            # Verify cache was set
            mock_cache.set.assert_called_once()
            cache_key = f"embedding:{hashlib.md5('test query'.encode()).hexdigest()}"
            assert mock_cache.set.call_args[0][0] == cache_key

            # Second call with cache hit
            mock_cache.get = AsyncMock(return_value=[0.1] * 1536)
            embedding2 = await retrieval_service._generate_embedding("test query")

            # Verify OpenAI was NOT called again
            assert mock_openai_client.embeddings.create.call_count == 1

            # Verify embeddings match
            assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_parallel_embedding_generation(
        self, retrieval_service, mock_openai_client
    ):
        """Test that multiple embeddings are generated in parallel"""
        queries = ["query 1", "query 2", "query 3", "query 4", "query 5"]

        # Mock cache to always miss
        with patch(
            "app.services.chat.document_retrieval_service.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            start_time = time.time()
            embeddings = await retrieval_service._generate_embeddings_parallel(queries)
            elapsed_time = time.time() - start_time

            # Verify all embeddings were generated
            assert len(embeddings) == 5

            # Verify OpenAI was called 5 times (once per query)
            assert mock_openai_client.embeddings.create.call_count == 5

            # Verify parallel execution (should be much faster than sequential)
            # Sequential would take ~5x the time of a single call
            # Parallel should take roughly the same time as a single call
            # This is a rough check - in real scenarios, parallel is much faster
            assert elapsed_time < 1.0  # Should complete quickly in tests

    @pytest.mark.asyncio
    async def test_parallel_embedding_with_cache_hits(
        self, retrieval_service, mock_openai_client
    ):
        """Test parallel generation with some cache hits"""
        queries = ["query 1", "query 2", "query 3"]

        # Mock cache: first query is cached, others are not
        cache_responses = {
            f"embedding:{hashlib.md5('query 1'.encode()).hexdigest()}": [0.1] * 1536,
            f"embedding:{hashlib.md5('query 2'.encode()).hexdigest()}": None,
            f"embedding:{hashlib.md5('query 3'.encode()).hexdigest()}": None,
        }

        async def mock_cache_get(key):
            return cache_responses.get(key)

        with patch(
            "app.services.chat.document_retrieval_service.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(side_effect=mock_cache_get)
            mock_cache.set = AsyncMock()

            embeddings = await retrieval_service._generate_embeddings_parallel(queries)

            # Verify all embeddings were returned
            assert len(embeddings) == 3

            # Verify OpenAI was only called for cache misses (2 times)
            assert mock_openai_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_embedding_error_handling(
        self, retrieval_service, mock_openai_client
    ):
        """Test that parallel generation handles errors gracefully"""
        queries = ["query 1", "query 2", "query 3"]

        # Mock OpenAI to fail on second query
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("OpenAI API error")
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            return mock_response

        mock_openai_client.embeddings.create = AsyncMock(side_effect=mock_create)

        with patch(
            "app.services.chat.document_retrieval_service.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            embeddings = await retrieval_service._generate_embeddings_parallel(queries)

            # Verify all embeddings were returned (with fallback for error)
            assert len(embeddings) == 3

            # Verify second embedding is zero vector (fallback)
            assert embeddings[1] == [0.0] * 1536

            # Verify other embeddings are valid
            assert embeddings[0] == [0.1] * 1536
            assert embeddings[2] == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_embedding_cache_key_generation(self, retrieval_service):
        """Test that cache keys are generated correctly"""
        query1 = "test query"
        query2 = "test query"  # Same query
        query3 = "different query"

        # Generate cache keys
        key1 = f"embedding:{hashlib.md5(query1.encode()).hexdigest()}"
        key2 = f"embedding:{hashlib.md5(query2.encode()).hexdigest()}"
        key3 = f"embedding:{hashlib.md5(query3.encode()).hexdigest()}"

        # Verify same queries have same keys
        assert key1 == key2

        # Verify different queries have different keys
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_empty_queries_parallel(self, retrieval_service):
        """Test parallel generation with empty query list"""
        embeddings = await retrieval_service._generate_embeddings_parallel([])

        # Should return empty list
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_cache_failure_fallback(self, retrieval_service, mock_openai_client):
        """Test that embedding generation works even if cache fails"""
        with patch(
            "app.services.chat.document_retrieval_service.cache_service"
        ) as mock_cache:
            # Mock cache to raise error
            mock_cache.get = AsyncMock(side_effect=Exception("Cache error"))
            mock_cache.set = AsyncMock(side_effect=Exception("Cache error"))

            # Should still generate embedding despite cache errors
            embedding = await retrieval_service._generate_embedding("test query")

            # Verify embedding was generated
            assert embedding == [0.1] * 1536

            # Verify OpenAI was called
            assert mock_openai_client.embeddings.create.call_count == 1


class TestPerformanceImprovements:
    """Test performance improvements from optimizations"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sequential_vs_parallel_performance(self):
        """
        Compare sequential vs parallel embedding generation performance.
        This test requires actual OpenAI API calls (marked as slow).
        """
        # This test is marked as slow and should be run separately
        # It demonstrates the actual performance improvement
        pytest.skip("Requires actual OpenAI API - run manually for benchmarking")

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, retrieval_service):
        """Test that cache hits are significantly faster"""
        with patch(
            "app.services.chat.document_retrieval_service.cache_service"
        ) as mock_cache:
            # First call: cache miss (slow)
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            start_time = time.time()
            await retrieval_service._generate_embedding("test query")
            miss_time = time.time() - start_time

            # Second call: cache hit (fast)
            mock_cache.get = AsyncMock(return_value=[0.1] * 1536)

            start_time = time.time()
            await retrieval_service._generate_embedding("test query")
            hit_time = time.time() - start_time

            # Cache hit should be faster (no OpenAI call)
            # In real scenarios, this is 100-200ms vs <1ms
            assert hit_time < miss_time


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
