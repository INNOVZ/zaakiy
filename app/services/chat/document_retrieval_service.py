"""
Document Retrieval Service
Handles all vector search and document retrieval operations
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.chat.contact_extractor import contact_extractor
from app.services.shared import cache_service
from app.services.shared.optimized_vector_search import OptimizedVectorSearch

logger = logging.getLogger(__name__)


class DocumentRetrievalError(Exception):
    """Exception for document retrieval errors"""


class DocumentRetrievalService:
    """Handles document retrieval using various strategies"""

    def __init__(self, org_id: str, openai_client, pinecone_index, context_config):
        self.org_id = org_id
        self.namespace = f"org-{org_id}"
        self.openai_client = openai_client
        self.pinecone_index = pinecone_index
        self.context_config = context_config

        # Initialize optimized vector search
        if self.openai_client:
            self.optimized_vector_search = OptimizedVectorSearch(
                openai_client=self.openai_client,
                org_id=org_id,
                embedding_model="text-embedding-3-small",
            )
        else:
            self.optimized_vector_search = None

        # Optimized retrieval config for better response quality
        # Increased from emergency mode (2) to production mode (5-8) for richer context
        self.retrieval_config = {"initial": 10, "rerank": 8, "final": 5}

    async def retrieve_documents(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents with intelligent caching (Cache-Aside pattern)"""
        # Check cache first for the entire query set
        cache_key = self._generate_retrieval_cache_key(queries)

        # Try to get cached results
        cached_results = await self._get_cached_retrieval_results(cache_key)
        if cached_results:
            logger.info("Vector cache HIT for %d queries", len(queries))
            return cached_results

        # Cache miss - perform retrieval
        logger.info(
            "Vector cache MISS - performing retrieval for %d queries", len(queries)
        )
        all_docs = {}

        # Get retrieval strategy from context config
        if self.context_config and hasattr(self.context_config, "retrieval_strategy"):
            strategy = self.context_config.retrieval_strategy
            semantic_weight = self.context_config.semantic_weight
            keyword_weight = self.context_config.keyword_weight
        elif self.context_config and isinstance(self.context_config, dict):
            # Handle dict-based config
            strategy = self.context_config.get("retrieval_strategy", "vector_search")
            semantic_weight = self.context_config.get("semantic_weight", 0.8)
            keyword_weight = self.context_config.get("keyword_weight", 0.2)
        else:
            # Handle None or invalid config - use defaults
            strategy = "vector_search"
            semantic_weight = 0.8
            keyword_weight = 0.2

        logger.info(
            "Using retrieval strategy: %s (semantic: %s, keyword: %s)",
            strategy,
            semantic_weight,
            keyword_weight,
        )

        # OPTIMIZATION: Process all queries in parallel instead of sequentially
        async def retrieve_for_query(query: str) -> List[Dict[str, Any]]:
            """Retrieve documents for a single query with error handling"""
            try:
                # Strategy-based retrieval
                if strategy == "semantic_only":
                    return await self._semantic_retrieval(query)
                elif strategy == "hybrid":
                    return await self._hybrid_retrieval(
                        query, semantic_weight, keyword_weight
                    )
                elif strategy == "keyword_boost":
                    return await self._keyword_boost_retrieval(
                        query, semantic_weight, keyword_weight
                    )
                elif strategy == "domain_specific":
                    return await self._domain_specific_retrieval(query)
                else:
                    # Fallback to semantic only
                    return await self._semantic_retrieval(query)

            except Exception as e:
                logger.warning(
                    "Retrieval failed for query '%s' with strategy %s: %s",
                    query,
                    strategy,
                    e,
                )
                # Fallback to basic semantic search
                try:
                    return await self._semantic_retrieval(query)
                except Exception as fallback_e:
                    logger.error("Fallback retrieval also failed: %s", fallback_e)
                    raise DocumentRetrievalError(
                        f"Document retrieval failed: {e}"
                    ) from e

        # Execute all query retrievals in parallel with timeout
        try:
            # Add timeout for document retrieval - increased to 20 seconds for better reliability
            query_results = await asyncio.wait_for(
                asyncio.gather(
                    *[retrieve_for_query(query) for query in queries],
                    return_exceptions=True,
                ),
                timeout=20.0,  # 20 second timeout for all vector searches (increased from 10s)
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Document retrieval timed out after 20 seconds - returning empty results for fallback"
            )
            # Return empty list instead of raising error - allows chatbot to respond without context
            return []
        except Exception as e:
            logger.error("Parallel retrieval failed: %s", e)
            raise DocumentRetrievalError(f"Document retrieval failed: {e}") from e

        # Merge results from all queries, keeping highest scores
        for result in query_results:
            if isinstance(result, Exception):
                logger.error("Query retrieval error: %s", result)
                continue

            if not result:
                continue

            for doc in result:
                doc_id = doc["id"]
                if doc_id not in all_docs or doc["score"] > all_docs[doc_id]["score"]:
                    all_docs[doc_id] = doc

        # Return top documents sorted by score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)

        # For contact queries, return MORE documents to ensure we get phone/email info
        # Check if any query contains contact keywords
        is_contact_query = any(
            keyword in query.lower()
            for query in queries
            for keyword in [
                "phone",
                "number",
                "email",
                "contact",
                "call",
                "reach",
                "address",
                "demo",
                "booking",
                "book",
                "schedule",
            ]
        )

        if is_contact_query:
            # Return up to 15 documents for contact queries (increased from 10)
            # This ensures we capture all contact information
            final_count = min(15, len(sorted_docs))

            # Re-score and re-sort documents based on contact information content
            # This prioritizes chunks that actually contain contact info
            contact_scored_docs = []
            for doc in sorted_docs:
                chunk = doc.get("chunk", "")
                contact_score = contact_extractor.score_chunk_for_contact_query(chunk)

                # Boost score if chunk contains contact info
                if contact_score > 0:
                    # Combine original similarity score with contact score
                    # Contact score is normalized to 0-1 range and added as a boost
                    boosted_score = doc["score"] + (contact_score / 100.0)
                    doc["contact_boosted_score"] = boosted_score
                    doc["contact_info"] = contact_extractor.extract_contact_info(chunk)
                else:
                    doc["contact_boosted_score"] = doc["score"]
                    doc["contact_info"] = {"has_contact_info": False}

                contact_scored_docs.append(doc)

            # Re-sort by contact-boosted score
            contact_scored_docs.sort(
                key=lambda x: x["contact_boosted_score"], reverse=True
            )

            # Log contact information found
            contact_found = sum(
                1
                for doc in contact_scored_docs
                if doc["contact_info"].get("has_contact_info")
            )
            if contact_found > 0:
                logger.info(
                    "🔍 Contact query - found contact info in %d/%d documents",
                    contact_found,
                    len(contact_scored_docs[:final_count]),
                )
                # Log what contact info was found
                for doc in contact_scored_docs[:5]:  # Log top 5
                    info = doc["contact_info"]
                    if info.get("has_contact_info"):
                        logger.info(
                            "📞 Contact info in doc %s: phones=%d, emails=%d, demo_links=%d",
                            doc["id"][:20],
                            len(info.get("phones", [])),
                            len(info.get("emails", [])),
                            len(info.get("demo_links", [])),
                        )
            else:
                logger.warning(
                    "⚠️ Contact query detected but NO contact info found in any retrieved documents!"
                )
                # Log chunk previews for debugging
                for i, doc in enumerate(contact_scored_docs[:5]):
                    chunk_preview = doc.get("chunk", "")[:200]
                    logger.debug("Doc %d chunk preview: %s", i + 1, chunk_preview)

            final_docs = contact_scored_docs[:final_count]
            logger.info(
                "🔍 Contact query detected - returning %d documents (boosted by contact info)",
                final_count,
            )
        else:
            final_count = self.retrieval_config["final"]
            final_docs = sorted_docs[:final_count]

        # DEBUG: Log what we're returning
        logger.info(
            "📤 Returning %d documents (scores: %s)",
            len(final_docs),
            [f"{doc['score']:.3f}" for doc in final_docs[:5]],
        )

        # Cache results asynchronously to avoid blocking
        # Save task to prevent premature garbage collection
        cache_task = asyncio.create_task(
            self._cache_retrieval_results(cache_key, final_docs)
        )

        return final_docs

    async def _semantic_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Optimized semantic similarity search with intelligent caching"""
        try:
            if self.optimized_vector_search:
                # Use optimized vector search with caching
                search_params = {"filter": {}}  # Add any metadata filters here

                context_config = {
                    "retrieval_strategy": getattr(
                        self.context_config, "retrieval_strategy", "semantic_only"
                    ),
                    "semantic_weight": getattr(
                        self.context_config, "semantic_weight", 0.7
                    ),
                    "keyword_weight": getattr(
                        self.context_config, "keyword_weight", 0.3
                    ),
                }

                result = await self.optimized_vector_search.search_with_caching(
                    query=query,
                    top_k=self.retrieval_config["initial"],
                    namespace=self.namespace,
                    search_params=search_params,
                    context_config=context_config,
                    include_metadata=True,
                )

                # Convert to expected format
                docs = []
                for match in result.get("matches", []):
                    # Extract chunk with better handling
                    metadata = match.get("metadata", {})
                    chunk = metadata.get("chunk", match.get("text", ""))

                    # If chunk is empty or very short, try other metadata fields
                    if not chunk or len(chunk.strip()) < 10:
                        # Try text field
                        chunk = metadata.get("text", "")
                        # Try content field
                        if not chunk:
                            chunk = metadata.get("content", "")

                    # Log chunk extraction for debugging
                    if not chunk or len(chunk.strip()) < 10:
                        logger.warning(
                            "⚠️ Retrieved document %s has empty or very short chunk (%d chars)",
                            match["id"],
                            len(chunk) if chunk else 0,
                        )
                        logger.debug("Metadata keys: %s", list(metadata.keys())[:10])

                    docs.append(
                        {
                            "id": match["id"],
                            "score": match["score"],
                            "chunk": chunk,
                            "source": metadata.get("source", ""),
                            "metadata": metadata,
                            "query_variant": query,
                            "retrieval_method": "semantic_optimized",
                            "cache_hit": result.get("performance", {}).get(
                                "cache_hit", False
                            ),
                            "search_time_ms": result.get("performance", {}).get(
                                "total_time_ms", 0
                            ),
                        }
                    )

                # Log performance metrics
                perf = result.get("performance", {})
                if perf.get("cache_hit"):
                    logger.info(
                        "Vector search cache HIT: %s (%.2fms)",
                        query[:50],
                        perf.get("total_time_ms", 0),
                    )
                else:
                    logger.info(
                        "Vector search cache MISS: %s (%.2fms)",
                        query[:50],
                        perf.get("total_time_ms", 0),
                    )

                return docs
            else:
                # Fallback to original implementation
                return await self._fallback_semantic_retrieval(query)

        except Exception as e:
            logger.error("Optimized semantic retrieval failed: %s", e)
            # Try fallback
            try:
                return await self._fallback_semantic_retrieval(query)
            except Exception as fallback_e:
                logger.error("Fallback semantic retrieval also failed: %s", fallback_e)
                raise DocumentRetrievalError(f"Semantic retrieval failed: {e}") from e

    async def _fallback_semantic_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Fallback semantic search without optimization"""
        try:
            embedding = await self._generate_embedding(query)
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=self.retrieval_config["initial"],
                namespace=self.namespace,
                include_metadata=True,
            )

            docs = []
            for match in results.matches:
                # Extract chunk with better handling
                metadata = match.metadata or {}
                chunk = metadata.get("chunk", "")

                # If chunk is empty, try other fields
                if not chunk or len(chunk.strip()) < 10:
                    chunk = metadata.get("text", "") or metadata.get("content", "")

                docs.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "chunk": chunk,
                        "source": metadata.get("source", ""),
                        "metadata": metadata,
                        "query_variant": query,
                        "retrieval_method": "semantic_fallback",
                    }
                )
            return docs

        except Exception as e:
            logger.error("Fallback semantic retrieval failed: %s", e)
            raise DocumentRetrievalError(
                f"Fallback semantic retrieval failed: {e}"
            ) from e

    async def _hybrid_retrieval(
        self, query: str, semantic_weight: float, keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic and keyword search"""
        try:
            # Get semantic results
            semantic_docs = await self._semantic_retrieval(query)

            # Get keyword results
            keyword_docs = await self._keyword_matching(query)

            # Combine and reweight scores
            combined_docs = {}

            # Add semantic docs with weight
            for doc in semantic_docs:
                doc_id = doc["id"]
                doc["score"] = doc["score"] * semantic_weight
                doc["retrieval_method"] = "hybrid_semantic"
                combined_docs[doc_id] = doc

            # Add keyword docs with weight (and combine if already exists)
            for doc in keyword_docs:
                doc_id = doc["id"]
                weighted_score = doc["score"] * keyword_weight

                if doc_id in combined_docs:
                    # Combine scores
                    combined_docs[doc_id]["score"] += weighted_score
                    combined_docs[doc_id]["retrieval_method"] = "hybrid_combined"
                else:
                    doc["score"] = weighted_score
                    doc["retrieval_method"] = "hybrid_keyword"
                    combined_docs[doc_id] = doc

            # Sort by combined score
            sorted_docs = sorted(
                combined_docs.values(), key=lambda x: x["score"], reverse=True
            )
            return sorted_docs[: self.retrieval_config["initial"]]

        except Exception as e:
            logger.error("Hybrid retrieval failed: %s", e)
            # Fallback to semantic only
            return await self._semantic_retrieval(query)

    async def _keyword_boost_retrieval(
        self, query: str, semantic_weight: float, keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Semantic search with keyword boosting"""
        try:
            # Start with semantic results
            semantic_docs = await self._semantic_retrieval(query)

            # Extract keywords from query
            keywords = self._extract_keywords(query)

            # Boost scores for docs containing keywords
            for doc in semantic_docs:
                chunk_text = doc.get("chunk", "").lower()
                keyword_matches = sum(
                    1 for keyword in keywords if keyword.lower() in chunk_text
                )

                if keyword_matches > 0:
                    # Apply keyword boost
                    keyword_boost = (keyword_matches / len(keywords)) * keyword_weight
                    doc["score"] = (doc["score"] * semantic_weight) + keyword_boost
                    doc["retrieval_method"] = "keyword_boosted"
                    doc["keyword_matches"] = keyword_matches
                else:
                    doc["score"] = doc["score"] * semantic_weight
                    doc["retrieval_method"] = "semantic_only"

            # Re-sort by boosted scores
            sorted_docs = sorted(semantic_docs, key=lambda x: x["score"], reverse=True)
            return sorted_docs

        except Exception as e:
            logger.error("Keyword boost retrieval failed: %s", e)
            return await self._semantic_retrieval(query)

    async def _domain_specific_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Domain-specific retrieval with specialized filters"""
        try:
            # This would implement domain-specific logic
            # For now, fallback to semantic retrieval
            return await self._semantic_retrieval(query)

        except Exception as e:
            logger.error("Domain-specific retrieval failed: %s", e)
            return await self._semantic_retrieval(query)

    async def _keyword_matching(self, query: str) -> List[Dict[str, Any]]:
        """Basic keyword matching using metadata filters"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)

            if not keywords:
                return []

            # This is a simplified implementation
            # In a real system, you'd use full-text search or metadata filtering

            # For now, return empty list as we don't have keyword search implemented
            return []

        except Exception as e:
            logger.error("Keyword matching failed: %s", e)
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        try:
            # Simple keyword extraction (could be enhanced with NLP)
            import re

            # Remove common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "how",
                "what",
                "where",
                "when",
                "why",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "cannot",
            }

            # Extract words (alphanumeric only)
            words = re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())

            # Filter out stop words and short words
            keywords = [
                word for word in words if word not in stop_words and len(word) > 2
            ]

            return keywords[:10]  # Limit to top 10 keywords

        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with error handling"""
        try:
            # Get embedding model from config
            if hasattr(self.context_config, "embedding_model"):
                model = self.context_config.embedding_model
            else:
                model = self.context_config.get(
                    "embedding_model", "text-embedding-3-small"
                )

            response = self.openai_client.embeddings.create(model=model, input=text)
            return response.data[0].embedding

        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise DocumentRetrievalError(f"Embedding generation failed: {e}") from e

    def _generate_retrieval_cache_key(self, queries: List[str]) -> str:
        """Generate cache key for retrieval results"""
        import hashlib
        import json

        # Create composite string for hashing using json for compatibility
        queries_str = json.dumps(sorted(queries), sort_keys=True)

        # Handle both dict and object with dict() method - exclude datetime fields
        if hasattr(self.context_config, "dict"):
            config_dict = self.context_config.dict()
            # Remove datetime fields that can't be JSON serialized
            json_safe_config = {
                k: v
                for k, v in config_dict.items()
                if not str(type(v)).startswith("<class 'datetime")
            }
            config_str = json.dumps(json_safe_config, sort_keys=True)
        else:
            # For plain dict, filter out datetime objects
            json_safe_config = {
                k: v
                for k, v in self.context_config.items()
                if not str(type(v)).startswith("<class 'datetime")
            }
            config_str = json.dumps(json_safe_config, sort_keys=True)
        params_str = json.dumps(self.retrieval_config, sort_keys=True)

        composite = f"{self.org_id}:{queries_str}:{config_str}:{params_str}"
        # SECURITY NOTE: MD5 is used here for cache key generation only (non-cryptographic purpose)
        # This is acceptable as it's not used for security, passwords, or authentication
        # For cache keys, MD5 provides fast hashing with good distribution
        cache_hash = hashlib.md5(composite.encode("utf-8")).hexdigest()

        return f"vector_retrieval:v1:{self.org_id}:{cache_hash}"

    async def _get_cached_retrieval_results(
        self, cache_key: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval results"""
        if not cache_service:
            return None

        try:
            cached_data = await cache_service.get(cache_key)
            if cached_data:
                logger.info("Retrieved cached vector search results")
                return cached_data
            return None
        except Exception as e:
            logger.warning("Failed to get cached retrieval results: %s", e)
            return None

    async def _cache_retrieval_results(
        self, cache_key: str, results: List[Dict[str, Any]]
    ):
        """Cache retrieval results asynchronously"""
        if not cache_service:
            return

        try:
            # Cache for 30 minutes (1800 seconds)
            await cache_service.set(cache_key, results, 1800)
            logger.debug("Cached vector search results with key: %s", cache_key[:20])
        except Exception as e:
            logger.warning("Failed to cache retrieval results: %s", e)

    async def get_vector_search_stats(self) -> Dict[str, Any]:
        """Get vector search performance statistics"""
        try:
            if self.optimized_vector_search:
                return self.optimized_vector_search.get_cache_statistics()
            else:
                return {"error": "Optimized vector search not available"}
        except Exception as e:
            logger.error("Failed to get vector search stats: %s", e)
            return {"error": str(e)}

    async def warm_vector_cache(self, queries: List[str]) -> Dict[str, Any]:
        """Warm vector search cache with specific queries"""
        try:
            if self.optimized_vector_search:
                return await self.optimized_vector_search.warm_cache_for_queries(
                    queries
                )
            else:
                return {"error": "Optimized vector search not available"}
        except Exception as e:
            logger.error("Failed to warm vector cache: %s", e)
            return {"error": str(e)}
