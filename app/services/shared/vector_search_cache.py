"""
Optimized Vector Search Cache Service
Addresses performance bottlenecks in vector similarity searches through intelligent caching
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cache_service import CacheMetrics, cache_service

logger = logging.getLogger(__name__)


class VectorSearchCache:
    """High-performance vector search caching with similarity-aware optimization"""

    def __init__(self):
        self.enabled = True
        self.metrics = CacheMetrics()
        self.similarity_threshold = 0.95  # Cache hit threshold for similar queries
        self.embedding_cache_ttl = 3600  # 1 hour for embeddings
        self.results_cache_ttl = 1800  # 30 minutes for search results
        self.popular_queries_cache = {}  # In-memory popular queries

    async def get_cached_vector_results(
        self,
        query: str,
        embedding: List[float],
        org_id: str,
        search_params: Dict[str, Any],
        context_config: Dict[str, Any],
    ) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        """
        Get cached vector search results with similarity matching
        Returns: (results, similarity_score) or None
        """
        if not self.enabled:
            return None

        start_time = datetime.now(timezone.utc)

        try:
            # First try exact match
            exact_key = self._generate_exact_cache_key(
                query, search_params, context_config, org_id
            )
            cached_exact = await cache_service.get(exact_key)

            if cached_exact:
                response_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                self.metrics.update_hit(response_time)
                logger.debug("Vector cache EXACT HIT for query: %s", query[:50])
                return cached_exact["results"], 1.0

            # Try similarity-based matching for expensive queries
            if len(embedding) > 0:
                similar_result = await self._find_similar_cached_query(
                    embedding, org_id, search_params, context_config
                )

                if similar_result:
                    response_time = (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds() * 1000
                    self.metrics.update_hit(response_time)
                    logger.debug(
                        "Vector cache SIMILARITY HIT for query: %s (similarity: %.3f)",
                        query[:50],
                        similar_result[1],
                    )
                    return similar_result

            # Cache miss
            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            self.metrics.update_miss(response_time)
            logger.debug("Vector cache MISS for query: %s", query[:50])
            return None

        except Exception as e:
            logger.error("Vector cache retrieval error: %s", e)
            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            self.metrics.update_error()
            self.metrics.update_miss(response_time)
            return None

    async def cache_vector_results(
        self,
        query: str,
        embedding: List[float],
        results: List[Dict[str, Any]],
        org_id: str,
        search_params: Dict[str, Any],
        context_config: Dict[str, Any],
        search_time_ms: float,
    ) -> bool:
        """Cache vector search results with performance-based TTL"""
        if not self.enabled or not results:
            return False

        try:
            # Generate cache keys
            exact_key = self._generate_exact_cache_key(
                query, search_params, context_config, org_id
            )
            embedding_key = self._generate_embedding_cache_key(
                embedding, org_id, search_params
            )

            # Determine TTL based on query performance and result quality
            ttl = self._calculate_adaptive_ttl(results, search_time_ms)

            # Cache exact match
            cache_data = {
                "results": results,
                "query": query,
                "embedding": embedding,
                "search_time_ms": search_time_ms,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "result_count": len(results),
                "top_score": results[0].get("score", 0) if results else 0,
            }

            # Cache with both exact and embedding-based keys
            cache_tasks = [
                cache_service.set(exact_key, cache_data, ttl),
                cache_service.set(embedding_key, cache_data, ttl),
            ]
            await asyncio.gather(*cache_tasks)

            # Update popular queries tracking
            await self._update_popular_queries(query, org_id)

            # Cache embeddings separately for reuse
            embedding_only_key = (
                f"embedding:v1:{org_id}:{hashlib.md5(query.encode()).hexdigest()}"
            )
            await cache_service.set(
                embedding_only_key, embedding, self.embedding_cache_ttl
            )

            logger.debug(
                "Cached vector results for query: %s (TTL: %ds, Results: %d)",
                query[:50],
                ttl,
                len(results),
            )
            return True

        except Exception as e:
            logger.error("Vector cache storage error: %s", e)
            return False

    async def get_cached_embedding(
        self, query: str, org_id: str
    ) -> Optional[List[float]]:
        """Get cached embedding for a query"""
        try:
            embedding_key = (
                f"embedding:v1:{org_id}:{hashlib.md5(query.encode()).hexdigest()}"
            )
            cached_embedding = await cache_service.get(embedding_key)

            if cached_embedding:
                logger.debug("Embedding cache HIT for query: %s", query[:50])
                return cached_embedding

            return None

        except Exception as e:
            logger.error("Embedding cache retrieval error: %s", e)
            return None

    def _generate_exact_cache_key(
        self,
        query: str,
        search_params: Dict[str, Any],
        context_config: Dict[str, Any],
        org_id: str,
    ) -> str:
        """Generate exact match cache key"""
        composite = {
            "query": query,
            "search_params": search_params,
            "context_config": context_config,
            "org_id": org_id,
            "version": "v2",
        }

        composite_str = orjson.dumps(composite, sort_keys=True).decode("utf-8")
        cache_hash = hashlib.sha256(composite_str.encode()).hexdigest()[:16]

        return f"vector_exact:v2:{org_id}:{cache_hash}"

    def _generate_embedding_cache_key(
        self, embedding: List[float], org_id: str, search_params: Dict[str, Any]
    ) -> str:
        """Generate embedding-based cache key"""
        # Create hash from embedding vector
        embedding_str = ",".join(
            [f"{x:.6f}" for x in embedding[:50]]
        )  # First 50 dimensions
        params_str = orjson.dumps(search_params, sort_keys=True).decode("utf-8")

        composite = f"{org_id}:{embedding_str}:{params_str}"
        cache_hash = hashlib.sha256(composite.encode()).hexdigest()[:16]

        return f"vector_embed:v2:{org_id}:{cache_hash}"

    async def _find_similar_cached_query(
        self,
        query_embedding: List[float],
        org_id: str,
        search_params: Dict[str, Any],
        context_config: Dict[str, Any],
    ) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        """Find similar cached queries using embedding similarity"""
        try:
            # Get recent cached embeddings for this org
            pattern = f"vector_embed:v2:{org_id}:*"
            cached_keys = (
                await cache_service.redis_client.keys(pattern)
                if cache_service.enabled
                else []
            )

            # Limit similarity search
            if not cached_keys or len(cached_keys) > 50:
                return None

            best_similarity = 0
            best_result = None

            for key in cached_keys[:20]:  # Check top 20 recent queries
                try:
                    cached_data = await cache_service.get(key)
                    if not cached_data or "embedding" not in cached_data:
                        continue

                    cached_embedding = cached_data["embedding"]

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(
                        query_embedding, cached_embedding
                    )

                    if (
                        similarity > self.similarity_threshold
                        and similarity > best_similarity
                    ):
                        best_similarity = similarity
                        best_result = (cached_data["results"], similarity)

                except Exception:
                    continue

            return best_result

        except Exception as e:
            logger.warning("Similarity search error: %s", e)
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vec1) != len(vec2):
                return 0.0

            # Convert to numpy arrays for efficient computation
            a = np.array(vec1)
            b = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)

        except Exception:
            return 0.0

    def _calculate_adaptive_ttl(
        self, results: List[Dict[str, Any]], search_time_ms: float
    ) -> int:
        """Calculate adaptive TTL based on result quality and search performance"""
        base_ttl = self.results_cache_ttl

        # Longer TTL for high-quality results
        if results:
            top_score = results[0].get("score", 0)
            if top_score > 0.9:
                base_ttl *= 2  # High confidence results
            elif top_score < 0.5:
                base_ttl //= 2  # Low confidence results

        # Longer TTL for expensive searches
        if search_time_ms > 1000:  # > 1 second
            base_ttl *= 1.5
        elif search_time_ms > 2000:  # > 2 seconds
            base_ttl *= 2

        return min(int(base_ttl), 7200)  # Max 2 hours

    async def _update_popular_queries(self, query: str, org_id: str):
        """Track popular queries for cache warming"""
        try:
            key = f"popular_queries:v1:{org_id}"
            popular_queries = await cache_service.get(key, {})

            if not isinstance(popular_queries, dict):
                popular_queries = {}

            # Update query frequency
            query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
            popular_queries[query_hash] = popular_queries.get(query_hash, 0) + 1

            # Keep only top 100 queries
            if len(popular_queries) > 100:
                sorted_queries = sorted(
                    popular_queries.items(), key=lambda x: x[1], reverse=True
                )
                popular_queries = dict(sorted_queries[:100])

            # Cache for 24 hours
            await cache_service.set(key, popular_queries, 86400)

        except Exception as e:
            logger.warning("Failed to update popular queries: %s", e)

    async def warm_popular_vectors(
        self, org_id: str, limit: int = 20
    ) -> Dict[str, Any]:
        """Warm cache with popular vector queries for an organization"""
        try:
            key = f"popular_queries:v1:{org_id}"
            popular_queries = await cache_service.get(key, {})

            if not isinstance(popular_queries, dict):
                return {"warmed": 0, "error": "No popular queries found"}

            # Sort by frequency and get top queries
            sorted_queries = sorted(
                popular_queries.items(), key=lambda x: x[1], reverse=True
            )
            top_queries = sorted_queries[:limit]

            warmed_count = 0
            for query_hash, frequency in top_queries:
                try:
                    # Check if already cached
                    pattern = f"vector_exact:v2:{org_id}:*{query_hash}*"
                    existing = (
                        await cache_service.redis_client.keys(pattern)
                        if cache_service.enabled
                        else []
                    )

                    if not existing:
                        # Would trigger actual vector search here in production
                        # For now, just mark as needing warming
                        warm_key = f"needs_warming:v1:{org_id}:{query_hash}"
                        await cache_service.set(
                            warm_key, {"frequency": frequency, "hash": query_hash}, 3600
                        )
                        warmed_count += 1

                except Exception:
                    continue

            return {
                "warmed": warmed_count,
                "total_popular": len(popular_queries),
                "processed": len(top_queries),
            }

        except Exception as e:
            logger.error("Vector cache warming error: %s", e)
            return {"warmed": 0, "error": str(e)}

    async def invalidate_org_vectors(self, org_id: str) -> int:
        """Invalidate all vector caches for an organization"""
        try:
            patterns = [
                f"vector_exact:v2:{org_id}:*",
                f"vector_embed:v2:{org_id}:*",
                f"embedding:v1:{org_id}:*",
                f"popular_queries:v1:{org_id}",
            ]

            total_cleared = 0
            for pattern in patterns:
                cleared = await cache_service.clear_pattern(pattern)
                total_cleared += cleared
                logger.debug("Cleared %d vector cache entries: %s", cleared, pattern)

            return total_cleared

        except Exception as e:
            logger.error("Vector cache invalidation error: %s", e)
            return 0

    def get_vector_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector cache statistics"""
        try:
            base_stats = self.metrics.get_performance_summary()

            # Add vector-specific stats
            vector_stats = {
                **base_stats,
                "similarity_threshold": self.similarity_threshold,
                "embedding_cache_ttl": self.embedding_cache_ttl,
                "results_cache_ttl": self.results_cache_ttl,
                "cache_strategy": "similarity_aware_multilevel",
            }

            return vector_stats

        except Exception as e:
            logger.error("Failed to get vector cache stats: %s", e)
            return {"error": str(e)}


# Global vector search cache instance
vector_search_cache = VectorSearchCache()
