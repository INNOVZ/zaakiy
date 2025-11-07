"""
Optimized Vector Search Service
High-performance wrapper around Pinecone with intelligent caching and query optimization
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import openai

from ..storage.pinecone_client import get_pinecone_index
from .vector_search_cache import vector_search_cache

logger = logging.getLogger(__name__)


class OptimizedVectorSearch:
    """High-performance vector search with intelligent caching and optimization"""

    def __init__(
        self,
        openai_client: openai.OpenAI,
        org_id: str,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.openai_client = openai_client
        self.org_id = org_id
        self.embedding_model = embedding_model
        self.pinecone_index = get_pinecone_index()
        self.embedding_cache = {}  # Local embedding cache

    async def search_with_caching(
        self,
        query: str,
        top_k: int = 10,
        namespace: str = "default",
        search_params: Optional[Dict[str, Any]] = None,
        context_config: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform vector search with intelligent caching
        Returns optimized results with performance metrics
        """
        search_start = datetime.now(timezone.utc)
        search_params = search_params or {}
        context_config = context_config or {}

        # Performance tracking
        performance_metrics = {
            "query": query[:100],
            "cache_hit": False,
            "embedding_cached": False,
            "total_time_ms": 0,
            "embedding_time_ms": 0,
            "search_time_ms": 0,
            "cache_time_ms": 0,
            "similarity_score": 0.0,
        }

        try:
            # Step 1: Check for cached results
            cache_start = datetime.now(timezone.utc)

            # Try to get cached embedding first
            cached_embedding = await vector_search_cache.get_cached_embedding(
                query, self.org_id
            )

            if cached_embedding:
                performance_metrics["embedding_cached"] = True
                logger.debug("Using cached embedding for query: %s", query[:50])
                query_embedding = cached_embedding
            else:
                # Generate new embedding
                embedding_start = datetime.now(timezone.utc)
                query_embedding = await self._generate_embedding(query)
                performance_metrics["embedding_time_ms"] = (
                    datetime.now(timezone.utc) - embedding_start
                ).total_seconds() * 1000

            # Check for cached vector search results
            cached_results = await vector_search_cache.get_cached_vector_results(
                query=query,
                embedding=query_embedding,
                org_id=self.org_id,
                search_params=search_params,
                context_config=context_config,
            )

            performance_metrics["cache_time_ms"] = (
                datetime.now(timezone.utc) - cache_start
            ).total_seconds() * 1000

            if cached_results:
                results, similarity_score = cached_results
                performance_metrics["cache_hit"] = True
                performance_metrics["similarity_score"] = similarity_score
                performance_metrics["total_time_ms"] = (
                    datetime.now(timezone.utc) - search_start
                ).total_seconds() * 1000

                logger.info(
                    "Vector search cache HIT: %s (similarity: %.3f, time: %.2fms)",
                    query[:50],
                    similarity_score,
                    performance_metrics["total_time_ms"],
                )

                return {
                    "matches": results[:top_k],
                    "performance": performance_metrics,
                    "total_results": len(results),
                    "namespace": namespace,
                }

            # Step 2: Perform actual vector search
            search_start_time = datetime.now(timezone.utc)

            if not self.pinecone_index:
                raise Exception("Pinecone index not available")

            # Execute vector search
            search_results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
                **search_params
            )

            performance_metrics["search_time_ms"] = (
                datetime.now(timezone.utc) - search_start_time
            ).total_seconds() * 1000

            # Step 3: Process and format results
            formatted_results = []
            for match in search_results.matches:
                formatted_results.append(
                    {
                        "id": match.id,
                        "score": float(match.score),
                        "metadata": match.metadata if include_metadata else {},
                        "text": match.metadata.get("text", "")
                        if match.metadata
                        else "",
                    }
                )

            # Step 4: Cache results asynchronously
            total_search_time = (
                datetime.now(timezone.utc) - search_start_time
            ).total_seconds() * 1000
            asyncio.create_task(
                vector_search_cache.cache_vector_results(
                    query=query,
                    embedding=query_embedding,
                    results=formatted_results,
                    org_id=self.org_id,
                    search_params=search_params,
                    context_config=context_config,
                    search_time_ms=total_search_time,
                )
            )

            performance_metrics["total_time_ms"] = (
                datetime.now(timezone.utc) - search_start
            ).total_seconds() * 1000

            logger.info(
                "Vector search completed: %s (time: %.2fms, results: %d)",
                query[:50],
                performance_metrics["total_time_ms"],
                len(formatted_results),
            )

            return {
                "matches": formatted_results,
                "performance": performance_metrics,
                "total_results": len(formatted_results),
                "namespace": namespace,
            }

        except Exception as e:
            error_time = (
                datetime.now(timezone.utc) - search_start
            ).total_seconds() * 1000
            logger.error("Vector search failed: %s (time: %.2fms)", e, error_time)

            performance_metrics["total_time_ms"] = error_time
            performance_metrics["error"] = str(e)

            return {
                "matches": [],
                "performance": performance_metrics,
                "total_results": 0,
                "namespace": namespace,
                "error": str(e),
            }

    async def batch_search_with_caching(
        self,
        queries: List[str],
        top_k: int = 10,
        namespace: str = "default",
        search_params: Optional[Dict[str, Any]] = None,
        context_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform batch vector searches with intelligent caching"""

        batch_start = datetime.now(timezone.utc)
        results = []
        cache_hits = 0

        # Process queries concurrently but limit concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent searches

        async def search_single(query: str) -> Dict[str, Any]:
            async with semaphore:
                result = await self.search_with_caching(
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    search_params=search_params,
                    context_config=context_config,
                )
                return result

        # Execute all searches
        search_tasks = [search_single(query) for query in queries]
        batch_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error("Batch search failed for query %d: %s", i, result)
                results.append(
                    {
                        "matches": [],
                        "performance": {"error": str(result)},
                        "total_results": 0,
                        "query_index": i,
                    }
                )
            else:
                if result.get("performance", {}).get("cache_hit", False):
                    cache_hits += 1
                result["query_index"] = i
                results.append(result)

        batch_time = (datetime.now(timezone.utc) - batch_start).total_seconds() * 1000

        logger.info(
            "Batch vector search completed: %d queries, %d cache hits, %.2fms total",
            len(queries),
            cache_hits,
            batch_time,
        )

        return results

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with local caching"""
        try:
            # Check local cache first
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]

            # Generate embedding
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )

            embedding = response.data[0].embedding

            # Cache locally (limited size)
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[text_hash] = embedding

            return embedding

        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise

    async def warm_cache_for_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Warm cache with specific queries"""
        try:
            warm_start = datetime.now(timezone.utc)
            warmed_count = 0

            for query in queries:
                try:
                    # Check if already cached
                    cached = await vector_search_cache.get_cached_vector_results(
                        query=query,
                        embedding=[],  # Empty embedding for check
                        org_id=self.org_id,
                        search_params={},
                        context_config={},
                    )

                    if not cached:
                        # Perform search to warm cache
                        await self.search_with_caching(query)
                        warmed_count += 1

                except Exception as e:
                    logger.warning("Failed to warm cache for query '%s': %s", query, e)
                    continue

            warm_time = (datetime.now(timezone.utc) - warm_start).total_seconds() * 1000

            return {
                "warmed_queries": warmed_count,
                "total_queries": len(queries),
                "warm_time_ms": warm_time,
            }

        except Exception as e:
            logger.error("Cache warming failed: %s", e)
            return {"error": str(e), "warmed_queries": 0}

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return vector_search_cache.get_vector_cache_stats()

    async def invalidate_cache(self) -> int:
        """Invalidate all cached results for this organization"""
        return await vector_search_cache.invalidate_org_vectors(self.org_id)
