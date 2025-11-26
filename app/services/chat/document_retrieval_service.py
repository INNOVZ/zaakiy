"""
Document Retrieval Service
Handles all vector search and document retrieval operations
"""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from app.services.chat.contact_extractor import contact_extractor
from app.services.chat.shared.keyword_extractor import get_keyword_extractor
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

        # Use shared keyword extractor to avoid duplication
        self.keyword_extractor = get_keyword_extractor()

        # Optimized retrieval config for better response quality
        # Adaptive k-values: contact=6, product=6, default=4 (boost only when needed)
        self.retrieval_config = {"initial": 8, "rerank": 6, "final": 4}

        # Optimization flags
        self.enable_reranking = True
        self.enable_metadata_filtering = True

    async def retrieve_documents(
        self,
        queries: List[str],
        intent_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents with intelligent caching (Cache-Aside pattern)"""
        original_retrieval_config = self.retrieval_config.copy()
        original_rerank_flag = self.enable_reranking

        intent_overrides = intent_config or {}
        try:
            if "k_values" in intent_overrides and intent_overrides["k_values"]:
                self.retrieval_config = intent_overrides["k_values"]
            if "rerank_enabled" in intent_overrides:
                self.enable_reranking = bool(intent_overrides["rerank_enabled"])

            all_docs = {}

            # Get retrieval strategy from context config
            if self.context_config and hasattr(
                self.context_config, "retrieval_strategy"
            ):
                strategy = self.context_config.retrieval_strategy
                semantic_weight = self.context_config.semantic_weight
                keyword_weight = self.context_config.keyword_weight
            elif self.context_config and isinstance(self.context_config, dict):
                # Handle dict-based config
                strategy = self.context_config.get(
                    "retrieval_strategy", "vector_search"
                )
                semantic_weight = self.context_config.get("semantic_weight", 0.8)
                keyword_weight = self.context_config.get("keyword_weight", 0.2)
            else:
                # Handle None or invalid config - use defaults
                strategy = "vector_search"
                semantic_weight = 0.8
                keyword_weight = 0.2

            strategy = intent_overrides.get("retrieval_strategy", strategy)

            logger.info(
                "Using retrieval strategy: %s (semantic: %s, keyword: %s)",
                strategy,
                semantic_weight,
                keyword_weight,
            )

            # Build metadata filters once for all queries (if enabled)
            combined_query = " ".join(queries)

            # Detect contact query early to adjust retrieval (using shared utility)
            from app.services.chat.chat_utilities import ChatUtilities

            is_contact_query_early = ChatUtilities.is_contact_query(combined_query)

            if not intent_overrides.get("k_values"):
                adaptive_initial_k = self._determine_initial_fetch_size(combined_query)

                # CRITICAL FIX: For contact queries, retrieve MORE documents initially
                # This ensures contact chunks are in the candidate pool
                if is_contact_query_early:
                    original_k = adaptive_initial_k
                    adaptive_initial_k = max(
                        adaptive_initial_k, 30
                    )  # At least 30 for contact queries
                    logger.info(
                        f"🔍 Contact query detected - increasing initial retrieval from {original_k} to {adaptive_initial_k}"
                    )

                self.retrieval_config["initial"] = adaptive_initial_k
                logger.debug(
                    "Adaptive initial top_k set to %d for query '%s'",
                    adaptive_initial_k,
                    combined_query[:80],
                )

            metadata_filters = (
                self._build_metadata_filters(combined_query)
                if self.enable_metadata_filtering
                else {}
            )

            # CRITICAL FIX: For contact queries, we DON'T use strict metadata filters
            # because they might return 0 results. Instead, we:
            # 1. Retrieve MORE documents (30+) to ensure contact chunks are in the pool
            # 2. Use contact scoring boost to prioritize contact chunks
            # This is more reliable than filtering which might miss contact chunks
            if is_contact_query_early:
                logger.info(
                    "🔍 Contact query detected - using increased retrieval + contact scoring boost"
                )

            intent_filters = intent_overrides.get("metadata_filters") or {}
            metadata_filters = self._combine_metadata_filters(
                metadata_filters, intent_filters
            )

            # Check cache now that filters/strategy are known
            cache_key = self._generate_retrieval_cache_key(
                queries,
                metadata_filters=metadata_filters,
                retrieval_strategy=strategy,
            )

            cached_results = await self._get_cached_retrieval_results(cache_key)
            if cached_results:
                logger.info("Vector cache HIT for %d queries", len(queries))
                return cached_results

            logger.info(
                "Vector cache MISS - performing retrieval for %d queries", len(queries)
            )

            # OPTIMIZATION: Process all queries in parallel instead of sequentially
            async def retrieve_for_query(query: str) -> List[Dict[str, Any]]:
                """Retrieve documents for a single query with error handling"""
                try:
                    # Strategy-based retrieval with metadata filters
                    if strategy == "semantic_only":
                        return await self._semantic_retrieval(query, metadata_filters)
                    elif strategy == "hybrid":
                        return await self._hybrid_retrieval(
                            query, semantic_weight, keyword_weight, metadata_filters
                        )
                    elif strategy == "keyword_boost":
                        return await self._keyword_boost_retrieval(
                            query, semantic_weight, keyword_weight, metadata_filters
                        )
                    elif strategy == "domain_specific":
                        return await self._domain_specific_retrieval(query)
                    else:
                        # Fallback to semantic only
                        return await self._semantic_retrieval(query, metadata_filters)

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
                    if (
                        doc_id not in all_docs
                        or doc["score"] > all_docs[doc_id]["score"]
                    ):
                        all_docs[doc_id] = doc

            # Return top documents sorted by score
            sorted_docs = sorted(
                all_docs.values(), key=lambda x: x["score"], reverse=True
            )

            # Determine query type for adaptive optimization
            # Use shared contact query detection utility (consolidated logic)
            from app.services.chat.chat_utilities import ChatUtilities

            is_contact_query = ChatUtilities.is_contact_query(combined_query)

            is_product_query = any(
                keyword in query.lower()
                for query in queries
                for keyword in [
                    "product",
                    "price",
                    "cost",
                    "buy",
                    "purchase",
                    "perfume",
                    "item",
                    "catalog",
                    "shop",
                    "store",
                    "available",
                    "offer",
                    "list",
                ]
            )

            # TOPIC-BASED BOOSTING (TENANT-AGNOSTIC)
            # Extract topics from query and match against metadata topics stored during ingestion
            # This is much more accurate than URL string matching

            combined_query_lower = combined_query.lower()
            query_terms = set(combined_query_lower.split())

            # Extract significant terms (filter out common words AND generic business terms)
            stop_words = {
                "a",
                "an",
                "the",
                "is",
                "are",
                "what",
                "how",
                "do",
                "does",
                "you",
                "your",
                "our",
                "we",
                "i",
                "me",
                "my",
                "can",
                "could",
                "would",
                "should",
                "about",
                "tell",
                "me",
                "us",
                "have",
                "has",
                "get",
                "any",
                # Generic business terms
                "services",
                "service",
                "products",
                "product",
                "company",
                "business",
                "offer",
                "offers",
                "offering",
            }

            # Extract query topics (longer, more specific terms)
            query_topics = [
                term for term in query_terms if len(term) > 2 and term not in stop_words
            ]

            # Also check for multi-word topics (e.g., "email marketing" = "email" + "marketing")
            for i in range(len(query_topics) - 1):
                multi_word = f"{query_topics[i]} {query_topics[i+1]}"
                if len(multi_word) > 6:  # Only meaningful combinations
                    query_topics.append(multi_word)

            # Sort by specificity (longer terms first)
            query_topics = sorted(set(query_topics), key=len, reverse=True)

            logger.info(
                f"🎯 Query: '{combined_query}' | Extracted topics: {query_topics[:5]}"
            )

            if query_topics:
                # Topic-based boosting: match query topics against document metadata topics
                topic_matched_count = 0
                topic_match_details = []

                for doc in sorted_docs:
                    doc_topics = doc.get("metadata", {}).get("topics", [])
                    if not doc_topics:
                        continue

                    # Calculate topic overlap
                    matched_topics = []
                    for q_topic in query_topics:
                        for d_topic in doc_topics:
                            # Match exact or partial (for hyphenated variations)
                            if q_topic in d_topic or d_topic in q_topic:
                                matched_topics.append((q_topic, d_topic))

                    if matched_topics:
                        original_score = doc["score"]
                        # Boost based on number of matching topics
                        # More matches = stronger boost
                        boost_amount = 0.15 * len(
                            matched_topics
                        )  # 0.15 per matching topic
                        boost_amount = min(boost_amount, 0.50)  # Cap at 0.50

                        doc["score"] = doc["score"] + boost_amount
                        doc["topic_boosted"] = True
                        doc["matched_topics"] = matched_topics
                        doc["topic_boost_amount"] = boost_amount
                        topic_matched_count += 1

                        topic_match_details.append(
                            {
                                "source": doc.get("source", "")[:60],
                                "matched": [f"{q}→{d}" for q, d in matched_topics],
                                "boost": boost_amount,
                                "new_score": doc["score"],
                            }
                        )

                        logger.debug(
                            f"   📈 Topic boost: {doc.get('source', '')[:60]}... | "
                            f"Matched: {matched_topics[:2]} | {original_score:.4f} → {doc['score']:.4f}"
                        )

                logger.info(f"🎯 Applied topic boost to {topic_matched_count} documents")

                # Log top matches
                if topic_match_details:
                    for detail in sorted(
                        topic_match_details, key=lambda x: x["new_score"], reverse=True
                    )[:3]:
                        logger.info(
                            f"   - {detail['source']} | Matched: {detail['matched']} | Boost: +{detail['boost']:.2f}"
                        )

                # Re-sort after boosting
                sorted_docs = sorted(
                    sorted_docs, key=lambda x: x["score"], reverse=True
                )

                # Log top 5 after boosting
                logger.info("📊 Top 5 documents after topic boosting:")
                for i, doc in enumerate(sorted_docs[:5], 1):
                    source = doc.get("source", "Unknown")
                    score = doc.get("score", 0)
                    boosted = "🎯" if doc.get("topic_boosted") else "  "
                    topics = doc.get("matched_topics", [])
                    logger.info(
                        f"   {i}. {boosted} {score:.4f} - {source[:50]}... {topics[:2] if topics else ''}"
                    )

            # Calculate optimal k-value based on query type
            optimal_k = self._calculate_optimal_k(
                combined_query,
                is_contact_query=is_contact_query,
                is_product_query=is_product_query,
                available_docs=len(sorted_docs),
            )

            if is_contact_query:
                # Use adaptive k-value (typically 6 for contact queries)
                final_count = min(optimal_k, len(sorted_docs))

                # CRITICAL FIX: Process MORE candidates for contact queries
                # Contact chunks might not be in top 10, so check top 20-30
                # This ensures we find contact info even if it's not top-ranked by semantic similarity
                # For contact queries, we want to check ALL retrieved documents
                candidates_to_score = len(
                    sorted_docs
                )  # Check ALL documents, not just top 30
                candidates = sorted_docs

                logger.info(
                    f"🔍 Contact query: Processing ALL {candidates_to_score} candidates to find contact info"
                )

                # Re-score and re-sort documents based on contact information content
                # This prioritizes chunks that actually contain contact info
                contact_scored_docs = []
                for doc in candidates:
                    chunk = doc.get("chunk", "")
                    contact_score = contact_extractor.score_chunk_for_contact_query(
                        chunk
                    )

                    # Boost score if chunk contains contact info
                    if contact_score > 0:
                        # CRITICAL FIX: Much stronger boost for contact info
                        # Original: contact_score / 100.0 (too weak)
                        # New: contact_score / 20.0 (5x stronger boost)
                        # This ensures contact chunks rank at the top
                        contact_boost = (
                            contact_score / 20.0
                        )  # Strong boost: 30 points = +1.5 score
                        boosted_score = doc["score"] + contact_boost
                        doc["contact_boosted_score"] = boosted_score
                        doc["contact_info"] = contact_extractor.extract_contact_info(
                            chunk
                        )
                        doc["contact_boost"] = contact_boost
                        logger.debug(
                            f"   📞 Contact boost: doc {doc['id'][:20]}... | "
                            f"score: {doc['score']:.3f} + {contact_boost:.3f} = {boosted_score:.3f} | "
                            f"phones: {len(doc['contact_info'].get('phones', []))}, "
                            f"emails: {len(doc['contact_info'].get('emails', []))}"
                        )
                    else:
                        doc["contact_boosted_score"] = doc["score"]
                        doc["contact_info"] = {"has_contact_info": False}
                        doc["contact_boost"] = 0.0

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
                        "⚠️ Contact query detected but NO contact info found in top %d candidates!",
                        candidates_to_score,
                    )
                    # FALLBACK: Search ALL documents for contact info if not found in top candidates
                    logger.info("🔍 Searching ALL documents for contact info...")
                    fallback_found = 0
                    for doc in sorted_docs[candidates_to_score:]:
                        chunk = doc.get("chunk", "")
                        if not chunk:
                            continue
                        contact_info = contact_extractor.extract_contact_info(chunk)
                        if contact_info.get("has_contact_info"):
                            # Found contact info! Add it to results with strong boost
                            contact_score = (
                                contact_extractor.score_chunk_for_contact_query(chunk)
                            )
                            contact_boost = contact_score / 20.0  # Same strong boost
                            doc["contact_boosted_score"] = doc["score"] + contact_boost
                            doc["contact_info"] = contact_info
                            doc["contact_boost"] = contact_boost
                            contact_scored_docs.append(doc)
                            fallback_found += 1
                            logger.info(
                                f"✅ Found contact info in doc {doc['id'][:20]}... "
                                f"(was ranked #{sorted_docs.index(doc) + 1}, now boosted) | "
                                f"phones: {len(contact_info.get('phones', []))}, "
                                f"emails: {len(contact_info.get('emails', []))}"
                            )
                            # Only add first few contact chunks found
                            if fallback_found >= 3:
                                break

                    if fallback_found > 0:
                        # Re-sort after adding fallback chunks
                        contact_scored_docs.sort(
                            key=lambda x: x["contact_boosted_score"], reverse=True
                        )
                        logger.info(
                            f"✅ Fallback search found {fallback_found} contact chunks and added them to results"
                        )

                    # LAST RESORT: Direct Pinecone query for contact chunks if still no contact info found
                    if not any(
                        d.get("contact_info", {}).get("has_contact_info")
                        for d in contact_scored_docs
                    ):
                        logger.warning(
                            "❌ NO contact info found in %d retrieved documents! Trying direct Pinecone query...",
                            len(sorted_docs),
                        )
                        try:
                            # Query Pinecone directly with metadata filter for contact chunks
                            contact_filter = {"has_contact_info": {"$eq": True}}
                            direct_contact_results = await self._semantic_retrieval(
                                combined_query, metadata_filters=contact_filter
                            )

                            if direct_contact_results:
                                logger.info(
                                    f"✅ Direct query found {len(direct_contact_results)} contact chunks!"
                                )
                                # Add contact chunks with high boost
                                for doc in direct_contact_results[:3]:  # Add top 3
                                    chunk = doc.get("chunk", "")
                                    contact_info = (
                                        contact_extractor.extract_contact_info(chunk)
                                    )
                                    if contact_info.get("has_contact_info"):
                                        contact_score = contact_extractor.score_chunk_for_contact_query(
                                            chunk
                                        )
                                        contact_boost = contact_score / 20.0
                                        doc["contact_boosted_score"] = (
                                            doc["score"] + contact_boost + 1.0
                                        )  # Extra boost
                                        doc["contact_info"] = contact_info
                                        doc["contact_boost"] = contact_boost
                                        contact_scored_docs.append(doc)
                                        logger.info(
                                            f"✅ Added contact chunk from direct query: "
                                            f"phones={len(contact_info.get('phones', []))}, "
                                            f"emails={len(contact_info.get('emails', []))}"
                                        )

                                # Re-sort after adding direct query results
                                contact_scored_docs.sort(
                                    key=lambda x: x["contact_boosted_score"],
                                    reverse=True,
                                )
                            else:
                                logger.error(
                                    "❌ Direct Pinecone query also returned no contact chunks"
                                )
                        except Exception as e:
                            logger.warning(f"Direct contact query failed: {e}")

                        # Log chunk previews for debugging
                        for i, doc in enumerate(contact_scored_docs[:5]):
                            chunk_preview = doc.get("chunk", "")[:200]
                            logger.debug(
                                "Doc %d chunk preview: %s", i + 1, chunk_preview
                            )

                final_docs = contact_scored_docs[:final_count]

                # Re-rank documents for better quality (if enabled)
                if self.enable_reranking and len(final_docs) > 3:
                    final_docs = self._rerank_documents(
                        combined_query, final_docs, top_k=final_count
                    )
                    logger.debug(f"Re-ranked {len(final_docs)} contact documents")

                logger.info(
                    "🔍 Contact query detected - returning %d documents (k=%d, boosted by contact info)",
                    final_count,
                    optimal_k,
                )
            elif is_product_query:
                # Use adaptive k-value (typically 6 for product queries)
                final_count = min(optimal_k, len(sorted_docs))
                final_docs = sorted_docs[:final_count]

                # Re-rank documents for better quality (if enabled)
                if self.enable_reranking and len(final_docs) > 3:
                    final_docs = self._rerank_documents(
                        combined_query, final_docs, top_k=final_count
                    )
                    logger.debug(f"Re-ranked {len(final_docs)} product documents")

                logger.info(
                    "🛍️ Product query detected - returning %d documents (k=%d, optimized)",
                    final_count,
                    optimal_k,
                )
            else:
                # Use adaptive k-value for general queries
                final_count = min(optimal_k, len(sorted_docs))
                final_docs = sorted_docs[:final_count]

                # Re-rank for complex queries (if enabled)
                if self.enable_reranking and len(final_docs) > 4:
                    final_docs = self._rerank_documents(
                        combined_query, final_docs, top_k=final_count
                    )
                    logger.debug(f"Re-ranked {len(final_docs)} general documents")

                logger.debug(f"Returning {final_count} documents (k={optimal_k})")

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
        finally:
            self.retrieval_config = original_retrieval_config
            self.enable_reranking = original_rerank_flag

    async def _semantic_retrieval(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        allow_filter_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """Optimized semantic similarity search with intelligent caching and metadata filtering"""
        try:
            if self.optimized_vector_search:
                # Use optimized vector search with caching and metadata filters
                search_params = (
                    {"filter": metadata_filters} if metadata_filters else {"filter": {}}
                )

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

                if metadata_filters and not docs and allow_filter_fallback:
                    logger.warning(
                        "Metadata filters yielded no documents for query '%s'. Retrying without filters.",
                        query[:80],
                    )
                    return await self._semantic_retrieval(
                        query, None, allow_filter_fallback=False
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
        self,
        query: str,
        semantic_weight: float,
        keyword_weight: float,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic and keyword search"""
        try:
            # Get semantic results
            semantic_docs = await self._semantic_retrieval(
                query, metadata_filters=metadata_filters
            )

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
            return await self._semantic_retrieval(
                query, metadata_filters=metadata_filters
            )

    async def _keyword_boost_retrieval(
        self,
        query: str,
        semantic_weight: float,
        keyword_weight: float,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search with keyword boosting"""
        try:
            # Start with semantic results (with metadata filters if provided)
            semantic_docs = await self._semantic_retrieval(
                query, metadata_filters=metadata_filters
            )

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
            # Use shared keyword extractor to avoid duplication
            # Limit to top 10 keywords for retrieval
            return self.keyword_extractor.extract_keywords(
                query, min_length=3, max_keywords=10
            )
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding with robust caching for high performance

        OPTIMIZED: Uses robust caching with:
        - Query normalization for consistent keys
        - Smart TTL based on access patterns
        - Automatic cache warming
        - Error handling with fallback
        """
        try:
            # Import robust cache utilities
            from app.utils.robust_cache import CacheKeyGenerator, get_robust_cache

            robust_cache = get_robust_cache()

            # Generate consistent cache key with normalization
            cache_key = CacheKeyGenerator.generate_embedding_key(
                text, model=self._get_embedding_model()
            )

            # Try robust cache first
            if robust_cache:
                cached_embedding = await robust_cache.get(cache_key)
                if cached_embedding:
                    logger.debug("Embedding cache HIT for query: %s", text[:50])
                    return cached_embedding

            # Cache miss - generate embedding
            logger.debug("Embedding cache MISS for query: %s", text[:50])

            # Get embedding model
            model = self._get_embedding_model()

            # Generate embedding
            response = self.openai_client.embeddings.create(model=model, input=text)
            embedding = response.data[0].embedding

            # Cache with smart TTL
            if robust_cache:
                await robust_cache.set(
                    cache_key,
                    embedding,
                    data_type="embedding",  # Uses smart TTL for embeddings
                )
                logger.debug("Cached embedding for query: %s", text[:50])

            return embedding

        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise DocumentRetrievalError(f"Embedding generation failed: {e}") from e

    def _get_embedding_model(self) -> str:
        """Get embedding model from config"""
        if hasattr(self.context_config, "embedding_model"):
            return self.context_config.embedding_model
        elif isinstance(self.context_config, dict):
            return self.context_config.get("embedding_model", "text-embedding-3-small")
        else:
            return "text-embedding-3-small"

    async def _generate_embeddings_parallel(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel for better performance.

        OPTIMIZATION: This method generates embeddings concurrently instead of sequentially,
        reducing total time from O(n) to O(1) for n queries.

        Expected performance improvement: ~100-200ms for 3-5 queries
        """
        try:
            if not texts:
                return []

            # OPTIMIZATION: Generate all embeddings in parallel
            embedding_tasks = [self._generate_embedding(text) for text in texts]
            embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)

            # Handle any errors in parallel execution
            valid_embeddings = []
            for i, result in enumerate(embeddings):
                if isinstance(result, Exception):
                    logger.error(
                        "Embedding generation failed for text %d: %s", i, result
                    )
                    # Use a zero vector as fallback (will have low similarity)
                    # This allows the query to continue even if one embedding fails
                    valid_embeddings.append(
                        [0.0] * 1536
                    )  # text-embedding-3-small dimension
                else:
                    valid_embeddings.append(result)

            logger.info(
                "Generated %d embeddings in parallel (cache hits: %d)",
                len(texts),
                sum(1 for e in embeddings if not isinstance(e, Exception)),
            )

            return valid_embeddings

        except Exception as e:
            logger.error("Parallel embedding generation failed: %s", e)
            raise DocumentRetrievalError(
                f"Parallel embedding generation failed: {e}"
            ) from e

    def _generate_retrieval_cache_key(
        self,
        queries: List[str],
        metadata_filters: Optional[Dict[str, Any]] = None,
        retrieval_strategy: Optional[str] = None,
    ) -> str:
        """Generate cache key for retrieval results"""

        # Create composite string for hashing using json for compatibility
        queries_str = json.dumps(sorted(queries), sort_keys=True)
        filters_str = json.dumps(metadata_filters or {}, sort_keys=True)
        strategy_str = retrieval_strategy or "default"

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

        composite = f"{self.org_id}:{queries_str}:{config_str}:{params_str}:{filters_str}:{strategy_str}"
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

    def _calculate_optimal_k(
        self,
        query: str,
        is_contact_query: bool = False,
        is_product_query: bool = False,
        available_docs: int = 0,
    ) -> int:
        """
        Intelligent top-k selection for optimal quality/performance balance.

        Strategy:
        - top-k = 3: Simple, focused queries (contact, hours, location)
        - top-k = 5: Standard queries (DEFAULT - best balance)
        - top-k = 10: Complex queries requiring comprehensive context

        Complex query indicators:
        - Multi-part questions (multiple "and", "or")
        - Technical documentation queries
        - Enterprise/detailed support
        - Product comparisons
        - Detailed explanations
        """
        query_lower = query.lower()
        query_words = query.split()
        word_count = len(query_words)

        # === SIMPLE QUERIES (top-k = 3) ===
        # Single-answer questions that need focused retrieval
        simple_query_patterns = [
            "what is your",
            "what are your",
            "where is",
            "where are",
            "when do you",
            "when are you",
            "do you have",
            "are you",
        ]

        simple_query_keywords = [
            "hours",
            "phone",
            "email",
            "address",
            "location",
            "open",
            "closed",
            "contact",
            "number",
        ]

        # Check if it's a simple query
        is_simple_pattern = any(
            pattern in query_lower for pattern in simple_query_patterns
        )
        is_simple_keyword = any(
            keyword in query_lower for keyword in simple_query_keywords
        )
        is_very_short = word_count <= 5

        if (
            is_simple_pattern or (is_simple_keyword and is_very_short)
        ) and word_count <= 8:
            logger.info(f"🎯 Simple query detected - using top-k=3 for speed")
            return 3

        # === COMPLEX QUERIES (top-k = 10) ===
        # Queries requiring comprehensive context

        # 1. Multi-part questions (multiple topics)
        multi_part_indicators = [
            " and ",
            " or ",
            " also ",
            " plus ",
            "as well as",
            "in addition",
            "along with",
        ]
        has_multiple_parts = (
            sum(1 for ind in multi_part_indicators if ind in query_lower) >= 2
        )

        # 2. Comparison queries
        comparison_keywords = [
            "compare",
            "comparison",
            "difference",
            "differences",
            "versus",
            "vs",
            "better",
            "best",
            "which",
            "between",
            "contrast",
            "pros and cons",
        ]
        is_comparison = any(keyword in query_lower for keyword in comparison_keywords)

        # 3. Technical/detailed queries
        technical_keywords = [
            "technical",
            "specification",
            "specifications",
            "details",
            "documentation",
            "api",
            "integration",
            "configure",
            "setup",
            "install",
            "implement",
            "architecture",
            "how does",
            "how do",
            "explain",
            "describe",
        ]
        is_technical = any(keyword in query_lower for keyword in technical_keywords)

        # 4. Enterprise/comprehensive support queries
        enterprise_keywords = [
            "enterprise",
            "business",
            "organization",
            "company",
            "comprehensive",
            "complete",
            "full",
            "all",
            "everything about",
            "tell me about",
            "overview",
        ]
        is_enterprise = any(keyword in query_lower for keyword in enterprise_keywords)

        # 5. Detailed product/feature queries
        detailed_keywords = [
            "features",
            "capabilities",
            "functionality",
            "what can",
            "what does",
            "how to",
            "guide",
            "tutorial",
            "walkthrough",
            "step by step",
        ]
        is_detailed = any(keyword in query_lower for keyword in detailed_keywords)

        # 6. Long, complex questions
        is_long_query = word_count > 15

        # Determine if complex
        is_complex = (
            has_multiple_parts
            or is_comparison
            or is_technical
            or is_enterprise
            or (is_detailed and word_count > 8)
            or is_long_query
        )

        if is_complex:
            logger.info(
                f"🔍 Complex query detected - using top-k=10 for comprehensive context "
                f"(multi_part={has_multiple_parts}, comparison={is_comparison}, "
                f"technical={is_technical}, enterprise={is_enterprise})"
            )
            return 10

        # === STANDARD QUERIES (top-k = 5) - DEFAULT ===
        # Most queries fall here - best balance of quality and speed
        logger.info(f"⚡ Standard query - using top-k=5 (optimal balance)")
        k = 5

        # Adjust based on specific query types
        if is_contact_query:
            k = 5  # Contact queries work well with 5
        elif is_product_query:
            k = 5  # Product queries work well with 5

        # Adjust based on available documents
        if available_docs > 0:
            k = min(k, available_docs)

        # Ensure minimum k
        k = max(k, 3)

        return k

    def _determine_initial_fetch_size(self, query: str) -> int:
        """Set a smaller default top_k and only boost when keywords/intent warrant it."""
        if not query:
            return max(self.retrieval_config.get("final", 4), 6)

        lowered = query.lower()
        tokens = lowered.split()
        baseline_k = 6
        boosted_k = 10

        contact_terms = [
            "phone",
            "email",
            "contact",
            "address",
            "office",
            "booking",
            "demo",
            "schedule",
            "how can i contact",
            "how to contact",
            "get in touch",
            "how can i reach",
            "how to reach",
            "what's your phone",
            "what is your email",
            "contact you",
            "reach you",
        ]
        pricing_terms = ["price", "pricing", "cost", "quote", "plan", "tier"]
        product_terms = [
            "product",
            "catalog",
            "collections",
            "catalogue",
            "items",
            "shop",
        ]

        complexity_triggers = ["compare", "difference", "versus", "steps", "process"]

        needs_boost = any(
            term in lowered for term in contact_terms + pricing_terms + product_terms
        )
        needs_boost = (
            needs_boost
            or len(tokens) > 12
            or any(trigger in lowered for trigger in complexity_triggers)
        )

        selected_k = boosted_k if needs_boost else baseline_k
        min_allowed = max(3, self.retrieval_config.get("final", 4))
        return max(selected_k, min_allowed)

    def _build_metadata_filters(self, query: str) -> Dict[str, Any]:
        """
        Build metadata filters based on query type and content.

        This narrows down the search space by filtering documents
        based on metadata (e.g., has_products, document type).
        """
        filters: Dict[str, Any] = {}
        or_conditions: List[Dict[str, Any]] = []
        lowered_query = query.lower()

        product_keywords = ["product", "buy", "purchase", "catalog", "catalogue"]
        pricing_keywords = ["price", "pricing", "cost", "quote", "$", "plan"]

        # Product query filters
        if any(keyword in lowered_query for keyword in product_keywords):
            or_conditions.append({"has_products": {"$eq": True}})

        # Pricing-specific filters
        if any(keyword in lowered_query for keyword in pricing_keywords):
            or_conditions.append({"has_pricing": {"$eq": True}})

        if or_conditions:
            if len(or_conditions) == 1:
                filters = or_conditions[0]
            else:
                filters["$or"] = or_conditions

        # Contact query filters - don't filter to ensure we get results
        # Document type filters could be added here
        # filters["type"] = {"$in": ["page", "article"]}

        return filters

    def _combine_metadata_filters(
        self,
        base_filters: Optional[Dict[str, Any]],
        intent_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Combine heuristic and intent-driven filters without losing OR clauses.
        """
        base_filters = base_filters or {}
        intent_filters = intent_filters or {}

        if base_filters and intent_filters:
            return {"$and": [base_filters, intent_filters]}
        return intent_filters or base_filters

    def _rerank_documents(
        self, query: str, documents: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using keyword matching and relevance scoring.
        Improves retrieval quality by scoring documents more accurately.
        """
        if not documents or len(documents) <= top_k:
            return documents

        try:
            query_keywords = set(self.keyword_extractor.extract_keywords(query.lower()))

            if not query_keywords:
                return documents[:top_k]

            # Score each document
            scored_docs = []
            for doc in documents:
                chunk = doc.get("chunk", "").lower()
                original_score = doc.get("score", 0.0)

                # Count keyword matches
                chunk_keywords = set(self.keyword_extractor.extract_keywords(chunk))
                keyword_matches = len(query_keywords & chunk_keywords)
                keyword_score = (
                    keyword_matches / len(query_keywords) if query_keywords else 0
                )

                # Combine original similarity score with keyword score
                # Weight: 70% original similarity, 30% keyword matching
                rerank_score = (original_score * 0.7) + (keyword_score * 0.3)

                scored_docs.append(
                    {
                        **doc,
                        "rerank_score": rerank_score,
                        "original_score": original_score,
                        "keyword_score": keyword_score,
                    }
                )

            # Sort by rerank score
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

            return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"Re-ranking failed, returning original order: {e}")
            return documents[:top_k]
