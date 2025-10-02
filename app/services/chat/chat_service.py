import os
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
# Import singleton
from pinecone import Pinecone
from supabase import create_client, Client
from ..analytics.context_config import context_config_manager
from ..shared import get_client_manager, cache_service
from ..analytics.context_analytics import context_analytics, ContextMetrics
from ...utils.error_handlers import ErrorHandler, retry_with_backoff, CircuitBreaker
from ...utils.error_context import ErrorContextManager, ErrorCategory, ErrorSeverity
from ...utils.error_monitoring import error_monitor

# Configure logger
logger = logging.getLogger(__name__)


class ChatServiceError(Exception):
    """Base exception for chat service errors"""
    pass


class RetrievalError(ChatServiceError):
    """Exception for retrieval-related errors"""
    pass


class ContextError(ChatServiceError):
    """Exception for context engineering errors"""
    pass


class ResponseGenerationError(ChatServiceError):
    """Exception for response generation errors"""
    pass


class ChatService:
    """Unified chat service with conversation management and context engineering"""

    def __init__(self, org_id: str, chatbot_config: dict):
        self.org_id = org_id
        self.namespace = f"org-{org_id}"
        self.chatbot_config = chatbot_config

        # Initialize clients with error handling
        try:
            client_manager = get_client_manager()
            self.openai_client = client_manager.openai
            self.index = client_manager.pinecone_index
            self.supabase = client_manager.supabase

            logger.info(
                "✅ ChatService initialized with shared clients for org %s", org_id)

        except Exception as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="ChatServiceInitializationError",
                severity="critical",
                service="chat_service",
                category="initialization"
            )
            
            # Use structured error handling
            ErrorHandler.log_and_raise(
                ChatServiceError,
                f"Service initialization failed: {e}",
                context="ChatService.__init__",
                original_exception=e,
                service="chat_service"
            )

        # Context engineering config will be loaded per request
        self.context_config = None

        # Initialize retrieval config with defaults (will be updated per request)
        self.retrieval_config = {
            "initial": 10,
            "rerank": 5,
            "final": 3
        }
        self.max_context_length = 4000

    # ==========================================
    # MAIN CHAT INTERFACE
    # ==========================================

    async def chat(
        self,
        message: str,
        session_id: str,
        chatbot_id: Optional[str] = None,
        channel: str = 'web'
    ) -> Dict[str, Any]:
        """Main chat interface - handles everything with improved error handling"""
        start_time = datetime.utcnow()

        try:
            # 0. Load context engineering configuration
            self.context_config = await context_config_manager.get_config(self.org_id)

            # Update retrieval config based on organization settings
            self.retrieval_config = {
                "initial": self.context_config.initial_retrieval_count,
                "rerank": self.context_config.semantic_rerank_count,
                "final": self.context_config.final_context_chunks
            }
            self.max_context_length = self.context_config.max_context_length

            # 1. Get or create conversation
            conversation = await self._get_or_create_conversation(
                session_id=session_id,
                chatbot_id=chatbot_id,
                channel=channel
            )

            # 2. Add user message
            user_message = await self._add_message(
                conversation_id=conversation["id"],
                role="user",
                content=message
            )

            # 3. Get conversation history for context
            history = await self._get_conversation_history(
                conversation_id=conversation["id"],
                limit=self.context_config.conversation_context_turns
            )

            # 4. Generate response with context engineering
            response_data = await self._generate_enhanced_response(
                message=message,
                conversation_history=history
            )

            # 5. Calculate processing time
            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000)

            # 6. Add assistant message
            assistant_message = await self._add_message(
                conversation_id=conversation["id"],
                role="assistant",
                content=response_data["response"],
                metadata={
                    "sources": response_data.get("sources", []),
                    "context_quality": response_data.get("context_quality", {}),
                    "retrieval_stats": response_data.get("retrieval_stats", {}),
                    "context_config_used": self.context_config.config_name
                },
                processing_time_ms=processing_time
            )

            # 7. Log context engineering analytics
            await self._log_analytics(
                conversation_id=conversation["id"],
                message_id=assistant_message["id"],
                query_original=message,
                response_data=response_data,
                processing_time=processing_time
            )

            return {
                "response": response_data["response"],
                "sources": response_data.get("sources", []),
                "conversation_id": conversation["id"],
                "message_id": assistant_message["id"],
                "processing_time_ms": processing_time,
                "context_quality": response_data.get("context_quality", {}),
                "config_used": self.context_config.config_name
            }

        except openai.OpenAIError as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="OpenAIError",
                severity="high",
                service="chat_service",
                category="external_service"
            )
            logging.error(f"OpenAI API error: {e}")
            return await self._fallback_response(message, session_id, "AI service temporarily unavailable")

        except ConnectionError as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="ConnectionError",
                severity="high",
                service="chat_service",
                category="database"
            )
            logging.error(f"Database connection error: {e}")
            return await self._fallback_response(message, session_id, "Database connection issue")

        except RetrievalError as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="RetrievalError",
                severity="medium",
                service="chat_service",
                category="retrieval"
            )
            logging.error(f"Retrieval error: {e}")
            return await self._fallback_response(message, session_id, "Knowledge retrieval issue")

        except ContextError as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="ContextError",
                severity="medium",
                service="chat_service",
                category="context_engineering"
            )
            logging.error(f"Context engineering error: {e}")
            return await self._fallback_response(message, session_id, "Context processing issue")

        except ResponseGenerationError as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="ResponseGenerationError",
                severity="medium",
                service="chat_service",
                category="response_generation"
            )
            logging.error(f"Response generation error: {e}")
            return await self._fallback_response(message, session_id, "Response generation issue")

        except Exception as e:
            # Record error in monitoring system
            error_monitor.record_error(
                error_type="UnexpectedError",
                severity="high",
                service="chat_service",
                category="unknown"
            )
            logging.error(f"Unexpected chat error: {e}")
            return await self._fallback_response(message, session_id, "Unexpected error occurred")

    # ==========================================
    # CONVERSATION MANAGEMENT - MISSING METHODS
    # ==========================================

    async def _get_or_create_conversation(
        self,
        session_id: str,
        chatbot_id: Optional[str] = None,
        channel: str = 'web'
    ) -> Dict[str, Any]:
        """Get existing conversation or create new one"""
        try:
            # Check for existing active conversation
            response = self.supabase.table("conversations").select("*").eq(
                "org_id", self.org_id
            ).eq(
                "session_id", session_id
            ).eq(
                "status", "active"
            ).order("updated_at", desc=True).limit(1).execute()

            if response.data and len(response.data) > 0:
                conversation = response.data[0]
                logging.info(
                    f"Found existing conversation: {conversation['id']}")
                return conversation

            # Create new conversation
            conversation_data = {
                "id": str(uuid.uuid4()),
                "org_id": self.org_id,
                "chatbot_id": chatbot_id or "default",
                "session_id": session_id,
                "channel": channel,
                "status": "active",
                "title": f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                "user_identifier": session_id,
                "metadata": {
                    "context_config": self.context_config.config_name if self.context_config else "default"
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            new_conv = self.supabase.table("conversations").insert(
                conversation_data).execute()

            if new_conv.data and len(new_conv.data) > 0:
                logging.info(
                    f"Created new conversation: {new_conv.data[0]['id']}")
                return new_conv.data[0]
            else:
                raise ConnectionError(
                    "Failed to create conversation in database")

        except Exception as e:
            logging.error(f"Error managing conversation: {e}")
            # Return a fallback conversation object
            fallback_conversation = {
                "id": f"fallback-{session_id}",
                "org_id": self.org_id,
                "session_id": session_id,
                "status": "active"
            }
            return fallback_conversation

    async def _add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: int = 0
    ) -> Dict[str, Any]:
        """Add message to conversation"""
        try:
            message_data = {
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "processing_time_ms": processing_time_ms,
                "token_count": len(content) // 4,  # Rough estimation
                "created_at": datetime.utcnow().isoformat()
            }

            response = self.supabase.table(
                "messages").insert(message_data).execute()

            if response.data and len(response.data) > 0:
                logging.info(
                    f"Added {role} message to conversation {conversation_id}")
                
                # Invalidate conversation history cache
                cache_service.clear_pattern(f"conversation_history:{conversation_id}:*")
                
                return response.data[0]
            else:
                # Return the message data even if database insert failed
                logging.warning(
                    f"Database insert failed for message, returning fallback")
                return message_data

        except Exception as e:
            logging.error(f"Error adding message: {e}")
            # Return fallback message object
            fallback_message = {
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "created_at": datetime.utcnow().isoformat()
            }
            return fallback_message

    async def _get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history with Redis caching"""
        try:
            # Check cache first
            cache_key = f"conversation_history:{conversation_id}:{limit}"
            cached_history = cache_service.get(cache_key)
            
            if cached_history:
                logging.info(f"Cache hit for conversation history: {conversation_id}")
                return cached_history

            # Get from database
            response = self.supabase.table("messages").select("*").eq(
                "conversation_id", conversation_id
            ).neq(
                "role", "system"
            ).order("created_at", desc=False).limit(limit).execute()

            history = response.data or []
            
            # Cache the history for 5 minutes
            cache_service.set(cache_key, history, 300)
            logging.info(
                f"Retrieved {len(history)} messages from conversation history and cached")
            return history

        except Exception as e:
            logging.error(f"Error getting conversation history: {e}")
            return []

    # ==========================================
    # CONTEXT ENGINEERING & RAG - MISSING METHODS
    # ==========================================

    async def _generate_enhanced_response(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate response with enhanced context engineering"""
        retrieval_start = datetime.utcnow()

        try:
            # Stage 1: Query Enhancement (if enabled)
            enhanced_queries = [message]  # Always include original
            if self.context_config.enable_query_rewriting:
                enhanced_queries = await self._enhance_query(message, conversation_history)

            # Stage 2: Multi-vector Retrieval
            candidates = await self._retrieve_documents(enhanced_queries)

            retrieval_time = int(
                (datetime.utcnow() - retrieval_start).total_seconds() * 1000)

            # Stage 3: Context Assembly
            context = await self._assemble_context(candidates, message)

            # Stage 4: Response Generation
            response = await self._generate_response_with_context(
                message=message,
                context=context,
                conversation_history=conversation_history
            )

            # Stage 5: Extract sources and product links
            sources = [doc.get("source", "") for doc in candidates[:3]]
            product_links = self._extract_product_links_from_documents(candidates)

            return {
                "response": response,
                "sources": list(filter(None, sources)),
                "product_links": product_links,
                "context_quality": self._assess_context_quality(context, message),
                "retrieval_stats": {
                    "candidates_found": len(candidates),
                    "context_length": len(context),
                    "retrieval_time_ms": retrieval_time,
                    "sources_used": len(sources)
                },
                "enhanced_queries": enhanced_queries,
                "context_used": context,
                "retrieved_documents": [
                    {
                        "source": doc.get("source", ""),
                        "score": doc.get("score", 0),
                        "chunk_preview": doc.get("chunk", "")[:200]
                    } for doc in candidates[:5]
                ]
            }

        except Exception as e:
            logging.error(f"Enhanced response generation error: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "sources": [],
                "context_quality": {"error": True},
                "retrieval_stats": {"error": True}
            }

    async def _enhance_query(self, query: str, history: List[Dict[str, Any]]) -> List[str]:
        """Enhance query with conversation context"""
        enhanced = [query]  # Always include original

        try:
            # Add conversation context if available and enabled
            if history and len(history) > 1:
                recent_context = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in history[-self.context_config.conversation_context_turns:]
                    if msg['role'] in ['user', 'assistant']
                ])

                context_prompt = f"""
Based on this conversation context, enhance the current query to be more specific and searchable:

Recent conversation:
{recent_context}

Current query: {query}

Enhanced query (maintain intent, add relevant context, make it more specific):
"""

                response = self.openai_client.chat.completions.create(
                    model=self.context_config.rerank_model,
                    messages=[{"role": "user", "content": context_prompt}],
                    max_tokens=100,
                    temperature=0.3
                )

                enhanced_query = response.choices[0].message.content.strip()
                if enhanced_query and enhanced_query != query and len(enhanced_query) > 5:
                    enhanced.append(enhanced_query)

            # Limit to max query variants
            return enhanced[:self.context_config.max_query_variants]

        except Exception as e:
            logging.warning(f"Query enhancement failed: {e}")
            return enhanced

    async def _retrieve_documents(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using configured retrieval strategy"""
        all_docs = {}
        
        # Get retrieval strategy from context config
        strategy = self.context_config.retrieval_strategy
        semantic_weight = self.context_config.semantic_weight
        keyword_weight = self.context_config.keyword_weight
        
        logger.info(f"Using retrieval strategy: {strategy} (semantic: {semantic_weight}, keyword: {keyword_weight})")

        for query in queries:
            try:
                # Strategy-based retrieval
                if strategy == "semantic_only":
                    docs = await self._semantic_retrieval(query)
                elif strategy == "hybrid":
                    docs = await self._hybrid_retrieval(query, semantic_weight, keyword_weight)
                elif strategy == "keyword_boost":
                    docs = await self._keyword_boost_retrieval(query, semantic_weight, keyword_weight)
                elif strategy == "domain_specific":
                    docs = await self._domain_specific_retrieval(query)
                else:
                    # Fallback to semantic only
                    docs = await self._semantic_retrieval(query)

                # Merge results, keeping highest scores
                for doc in docs:
                    doc_id = doc["id"]
                    if doc_id not in all_docs or doc["score"] > all_docs[doc_id]["score"]:
                        all_docs[doc_id] = doc

            except Exception as e:
                logging.warning(f"Retrieval failed for query '{query}' with strategy {strategy}: {e}")
                # Fallback to basic semantic search
                try:
                    docs = await self._semantic_retrieval(query)
                    for doc in docs:
                        doc_id = doc["id"]
                        if doc_id not in all_docs or doc["score"] > all_docs[doc_id]["score"]:
                            all_docs[doc_id] = doc
                except Exception as fallback_e:
                    logging.error(f"Fallback retrieval also failed: {fallback_e}")
                    raise RetrievalError(f"Document retrieval failed: {e}") from e

        # Return top documents sorted by score
        sorted_docs = sorted(
            all_docs.values(), key=lambda x: x["score"], reverse=True)
        return sorted_docs[:self.retrieval_config["final"]]

    async def _semantic_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Pure semantic similarity search"""
        try:
            embedding = await self._generate_embedding(query)
            results = self.index.query(
                vector=embedding,
                top_k=self.retrieval_config["initial"],
                namespace=self.namespace,
                include_metadata=True
            )

            docs = []
            for match in results.matches:
                docs.append({
                    "id": match.id,
                    "score": match.score,
                    "chunk": match.metadata.get('chunk', ''),
                    "source": match.metadata.get('source', ''),
                    "metadata": match.metadata,
                    "query_variant": query,
                    "retrieval_method": "semantic"
                })
            return docs

        except Exception as e:
            logging.error(f"Semantic retrieval failed: {e}")
            raise RetrievalError(f"Semantic retrieval failed: {e}") from e

    async def _keyword_matching(self, query: str) -> List[Dict[str, Any]]:
        """Basic keyword matching using metadata filters"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            # Use Pinecone metadata filtering for keyword matching
            results = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only search
                top_k=self.retrieval_config["initial"],
                namespace=self.namespace,
                include_metadata=True,
                filter={
                    "$or": [
                        {"keywords": {"$in": keywords}},
                        {"title": {"$in": keywords}},
                        {"content_type": {"$in": keywords}}
                    ]
                }
            )
            
            docs = []
            for match in results.matches:
                # Calculate keyword relevance score
                keyword_score = self._calculate_keyword_score(
                    match.metadata.get('chunk', ''), keywords
                )
                
                docs.append({
                    "id": match.id,
                    "score": keyword_score,
                    "chunk": match.metadata.get('chunk', ''),
                    "source": match.metadata.get('source', ''),
                    "metadata": match.metadata,
                    "query_variant": query,
                    "retrieval_method": "keyword"
                })
            
            return docs

        except Exception as e:
            logging.warning(f"Keyword matching failed: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        import re
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
        
        # Extract words (alphanumeric only)
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword relevance score"""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # Normalize by text length and keyword count
        return min(matches / len(keywords), 1.0)

    async def _hybrid_retrieval(self, query: str, semantic_weight: float, keyword_weight: float) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic and keyword matching"""
        try:
            # Get semantic results
            semantic_docs = await self._semantic_retrieval(query)
            
            # Get keyword results
            keyword_docs = await self._keyword_matching(query)
            
            # Combine and score
            all_docs = {}
            
            # Add semantic results with weighted scores
            for doc in semantic_docs:
                doc_id = doc["id"]
                all_docs[doc_id] = {
                    **doc,
                    "semantic_score": doc["score"],
                    "keyword_score": 0.0,
                    "score": doc["score"] * semantic_weight,
                    "retrieval_method": "hybrid_semantic"
                }
            
            # Add keyword results with weighted scores
            for doc in keyword_docs:
                doc_id = doc["id"]
                if doc_id in all_docs:
                    # Combine scores for existing documents
                    all_docs[doc_id]["keyword_score"] = doc["score"]
                    all_docs[doc_id]["score"] = (
                        all_docs[doc_id]["semantic_score"] * semantic_weight + 
                        doc["score"] * keyword_weight
                    )
                    all_docs[doc_id]["retrieval_method"] = "hybrid_combined"
                else:
                    # Add new document
                    all_docs[doc_id] = {
                        **doc,
                        "semantic_score": 0.0,
                        "keyword_score": doc["score"],
                        "score": doc["score"] * keyword_weight,
                        "retrieval_method": "hybrid_keyword"
                    }
            
            return list(all_docs.values())

        except Exception as e:
            logging.error(f"Hybrid retrieval failed: {e}")
            # Fallback to semantic only
            return await self._semantic_retrieval(query)

    async def _keyword_boost_retrieval(self, query: str, semantic_weight: float, keyword_weight: float) -> List[Dict[str, Any]]:
        """Keyword-boosted retrieval with enhanced keyword matching"""
        try:
            # Get semantic results
            semantic_docs = await self._semantic_retrieval(query)
            
            # Get keyword results
            keyword_docs = await self._keyword_matching(query)
            
            # Combine with keyword boost
            all_docs = {}
            
            # Process semantic results
            for doc in semantic_docs:
                doc_id = doc["id"]
                all_docs[doc_id] = {
                    **doc,
                    "semantic_score": doc["score"],
                    "keyword_score": 0.0,
                    "score": doc["score"] * semantic_weight,
                    "retrieval_method": "keyword_boost_semantic"
                }
            
            # Process keyword results with boost
            for doc in keyword_docs:
                doc_id = doc["id"]
                boosted_score = doc["score"] * 1.2  # 20% boost for keyword matches
                
                if doc_id in all_docs:
                    # Combine with existing semantic score
                    all_docs[doc_id]["keyword_score"] = boosted_score
                    all_docs[doc_id]["score"] = (
                        all_docs[doc_id]["semantic_score"] * semantic_weight + 
                        boosted_score * keyword_weight
                    )
                    all_docs[doc_id]["retrieval_method"] = "keyword_boost_combined"
                else:
                    # Add new document with boost
                    all_docs[doc_id] = {
                        **doc,
                        "semantic_score": 0.0,
                        "keyword_score": boosted_score,
                        "score": boosted_score * keyword_weight,
                        "retrieval_method": "keyword_boost_keyword"
                    }
            
            return list(all_docs.values())

        except Exception as e:
            logging.error(f"Keyword boost retrieval failed: {e}")
            # Fallback to semantic only
            return await self._semantic_retrieval(query)

    async def _domain_specific_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Domain-specific retrieval using business context"""
        try:
            # Get semantic results
            semantic_docs = await self._semantic_retrieval(query)
            
            # Apply domain-specific scoring
            domain_context = getattr(self.context_config, 'business_context', '')
            domain_knowledge = getattr(self.context_config, 'domain_knowledge', '')
            
            for doc in semantic_docs:
                # Boost score based on domain relevance
                domain_boost = self._calculate_domain_relevance(
                    doc["chunk"], domain_context, domain_knowledge
                )
                doc["score"] = doc["score"] * (1.0 + domain_boost)
                doc["retrieval_method"] = "domain_specific"
                doc["domain_boost"] = domain_boost
            
            return semantic_docs

        except Exception as e:
            logging.error(f"Domain-specific retrieval failed: {e}")
            # Fallback to semantic only
            return await self._semantic_retrieval(query)

    def _calculate_domain_relevance(self, text: str, domain_context: str, domain_knowledge: str) -> float:
        """Calculate domain-specific relevance boost"""
        if not domain_context and not domain_knowledge:
            return 0.0
        
        text_lower = text.lower()
        domain_terms = []
        
        if domain_context:
            domain_terms.extend(domain_context.lower().split())
        if domain_knowledge:
            domain_terms.extend(domain_knowledge.lower().split())
        
        if not domain_terms:
            return 0.0
        
        # Count domain term matches
        matches = sum(1 for term in domain_terms if term in text_lower)
        
        # Return boost factor (0.0 to 0.5)
        return min(matches / len(domain_terms), 0.5)

    def _extract_product_links_from_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and format product links from retrieved documents"""
        product_links = []
        seen_links = set()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            
            # Check if document has product links
            if metadata.get("has_products", False):
                doc_product_links = metadata.get("product_links", [])
                
                for link in doc_product_links:
                    if link and link not in seen_links:
                        seen_links.add(link)
                        
                        # Extract product name from chunk content
                        chunk = doc.get("chunk", "")
                        product_name = self._extract_product_name_from_chunk(chunk, link)
                        
                        product_links.append({
                            "url": link,
                            "name": product_name,
                            "source": doc.get("source", ""),
                            "relevance_score": doc.get("score", 0),
                            "chunk_preview": chunk[:150] + "..." if len(chunk) > 150 else chunk
                        })
        
        # Sort by relevance score and limit to top 5
        product_links.sort(key=lambda x: x["relevance_score"], reverse=True)
        return product_links[:5]

    def _extract_product_name_from_chunk(self, chunk: str, link: str) -> str:
        """Extract product name from chunk content"""
        import re
        
        # Try to find product names near the link or in the chunk
        # Look for patterns like "Product Name - Price" or "Product Name (Price)"
        product_patterns = [
            r'([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*[-–]\s*[A-Za-z]+\s*[0-9]+',
            r'([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*\([A-Za-z]+\s*[0-9]+\)',
            r'([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*for\s*[A-Za-z]+\s*[0-9]+',
            r'([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*[A-Za-z]+\s*[0-9]+',
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            if matches:
                # Return the first match, cleaned up
                product_name = matches[0].strip()
                # Remove extra whitespace and clean up
                product_name = re.sub(r'\s+', ' ', product_name)
                return product_name
        
        # If no pattern matches, try to extract from URL
        if link:
            # Extract product name from URL path
            url_parts = link.split('/')
            for part in reversed(url_parts):
                if part and part not in ['product', 'item', 'catalog', 'shop', 'buy', 'p']:
                    # Clean up the part to make it a readable product name
                    product_name = part.replace('-', ' ').replace('_', ' ')
                    product_name = re.sub(r'[^a-zA-Z\s]', '', product_name)
                    product_name = re.sub(r'\s+', ' ', product_name).strip()
                    if len(product_name) > 3:  # Only return if it's a meaningful name
                        return product_name.title()
        
        # Fallback: return a generic name
        return "Product"

    async def _assemble_context(self, documents: List[Dict[str, Any]], query: str) -> str:
        """Assemble context from retrieved documents"""
        if not documents:
            return "No relevant information found in uploaded documents."

        context_parts = []
        total_length = 0
        used_sources = set()

        for doc in documents:
            chunk = doc["chunk"]
            source = doc["source"]
            score = doc.get("score", 0)

            # Avoid too much content from same source unless it's highly relevant
            if source in used_sources and score < 0.8:
                continue

            # Estimate tokens
            chunk_tokens = len(chunk) // 4
            if total_length + chunk_tokens > self.max_context_length:
                # Try to fit a truncated version if there's meaningful space
                remaining_tokens = self.max_context_length - total_length
                if remaining_tokens > 100:
                    chunk = chunk[:remaining_tokens * 4]
                else:
                    break

            context_part = f"Source: {source} (Relevance: {score:.3f})\nContent: {chunk}\n"
            context_parts.append(context_part)
            total_length += len(chunk) // 4
            used_sources.add(source)

        return "\n---\n".join(context_parts)

    async def _generate_response_with_context(
        self,
        message: str,
        context: str,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Generate final response using context and conversation history"""

        # Build conversation context
        conv_context = ""
        if conversation_history:
            recent_turns = conversation_history[-3:]
            for turn in recent_turns:
                role = turn.get("role", "user")
                # Truncate long messages
                content = turn.get("content", "")[:200]
                conv_context += f"{role}: {content}\n"

        # Get model configuration
        model_config = context_config_manager.get_model_config(
            self.context_config.model_tier)

        # Enhanced system prompt with context engineering
        system_prompt = f"""You are {self.chatbot_config.get('name', 'Assistant')}, a {self.chatbot_config.get('tone', 'helpful')} AI assistant for INNOVZ.

ORGANIZATION CONTEXT: {self.context_config.business_context}

PERSONALITY: {self.chatbot_config.get('behavior', 'Be helpful and informative')}

SPECIALIZED INSTRUCTIONS: {self.context_config.specialized_instructions}

RELEVANT CONTEXT FROM DOCUMENTS:
{context}

RECENT CONVERSATION:
{conv_context}

INSTRUCTIONS:
- Answer based on the provided context when relevant
- Be conversational and maintain {self.chatbot_config.get('tone', 'friendly')} tone and use relevant emojis to make the response more engaging
- If context doesn't contain relevant info, say so politely
- Reference specific information from context when applicable
- Keep responses focused and concise
- Avoid unnecessary repetition
- Maintain a logical flow of information
- Use clear and concise language
- Avoid unnecessary jargon
- Be mindful of the user's perspective
- Always use the language of the user's query
- Format lists in bullet points with clear item names and details (e.g., prices, links, etc.).
- You can use markdown to format your response and also use bold and italic to make your response more engaging
- Encourage user engagement and questions
- Answer questions clearly and concisely in strictly less than 100 words
- Never make your own assumptions or fabricate information
- NEVER reveal that you are an AI model or AI assistant
- Always refer the contact details from the data and provide 
- Always refer to yourself as {self.chatbot_config.get('name', 'Assistant')} and be act like a human
- When mentioning products, always include clickable links in markdown format: [Product Name](URL)
- If you find product information in the context, make sure to mention the product names and include their links
- For product listings, format them as: **Product Name** - Description - [View Product](URL)
- Confidence threshold for responses: {self.context_config.confidence_threshold}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"]
            )

            generated_response = response.choices[0].message.content

            # Post-process response if hallucination checking is enabled
            if self.context_config.enable_hallucination_check:
                generated_response = await self._check_response_quality(
                    generated_response, context
                )

            return generated_response

        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return self.chatbot_config.get(
                'fallback_message',
                "I apologize, but I'm having trouble generating a response right now."
            )

    async def _check_response_quality(self, response: str, context: str) -> str:
        """Check response quality and add disclaimers if needed"""
        try:
            if not context or len(context.strip()) < 100:
                return response

            verification_prompt = f"""
Review this AI response against the provided context. Check if the response is well-supported by the context.

Context: {context[:1000]}...

AI Response: {response}

Rate the response support level (1-5):
1 = Not supported by context
2 = Poorly supported  
3 = Moderately supported
4 = Well supported
5 = Excellently supported

Respond with just the number (1-5):
"""

            verification = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": verification_prompt}],
                max_tokens=10,
                temperature=0.1
            )

            score_text = verification.choices[0].message.content.strip()
            try:
                score = int(score_text)
                # Quality check completed but no disclaimer added
                # Score: {score}/5 - keeping response as is
            except ValueError:
                pass  # If we can't parse the score, just return original response

            return response

        except Exception as e:
            logging.warning(f"Response quality check failed: {e}")
            return response

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.context_config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            raise

    def _assess_context_quality(self, context: str, query: str) -> Dict[str, Any]:
        """Assess context quality with enhanced metrics"""
        has_context = len(context.strip()) > 100
        context_length = len(context)

        # Simple coverage score based on context length vs optimal length
        optimal_length = self.max_context_length * 0.7  # 70% of max is considered good
        coverage_score = min(1.0, context_length /
                             optimal_length) if has_context else 0.0

        return {
            "context_length": context_length,
            "has_context": has_context,
            "coverage_score": coverage_score,
            "quality_tier": "high" if coverage_score > 0.7 else "medium" if coverage_score > 0.3 else "low",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _fallback_response(self, message: str, session_id: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced fallback response with error context"""
        fallback_messages = {
            "AI service temporarily unavailable": "I'm experiencing some issues connecting to my AI service. Please try again in a moment.",
            "Database connection issue": "I'm having trouble accessing my memory right now. Please try again shortly.",
            "Knowledge retrieval issue": "I'm having difficulty finding relevant information. Please rephrase your question.",
            "Context processing issue": "I'm having trouble understanding the context. Could you be more specific?",
            "Response generation issue": "I'm having trouble formulating a response. Please try rephrasing your question."
        }

        response_text = fallback_messages.get(error_type,
                                              self.chatbot_config.get('fallback_message',
                                                                      "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment."
                                                                      )
                                              )

        return {
            "response": response_text,
            "sources": [],
            "conversation_id": session_id,
            "message_id": None,
            "processing_time_ms": 0,
            "context_quality": {"error": True, "error_type": error_type},
            "error_context": error_type
        }

    # ==========================================
    # ANALYTICS AND LOGGING
    # ==========================================

    async def _log_analytics(
        self,
        conversation_id: str,
        message_id: str,
        query_original: str,
        response_data: Dict[str, Any],
        processing_time: int
    ) -> None:
        """Log comprehensive analytics data"""
        try:
            # Create context metrics (commented out - requires env setup)
            # metrics = ContextMetrics(
            #     org_id=self.org_id,
            #     conversation_id=conversation_id,
            #     message_id=message_id,
            #     query_original=query_original,
            #     query_enhanced=response_data.get("enhanced_queries", []),
            #     documents_retrieved=response_data.get(
            #         "retrieved_documents", []),
            #     context_length=len(response_data.get("context_used", "")),
            #     context_quality_score=response_data.get(
            #         "context_quality", {}).get("coverage_score", 0.5),
            #     retrieval_time_ms=response_data.get(
            #         "retrieval_stats", {}).get("retrieval_time_ms", 0),
            #     response_time_ms=processing_time,
            #     model_used=str(self.context_config.model_tier),
            #     sources_count=len(response_data.get("sources", []))
            # )

            # Log to analytics system (commented out - requires env setup)
            # await context_analytics.log_context_metrics(metrics)

            # Also log to context_analytics table (legacy support)
            context_data = {
                "message_id": message_id,
                "org_id": self.org_id,
                "query_original": query_original,
                "query_enhanced": response_data.get("enhanced_queries", []),
                "documents_retrieved": response_data.get("sources", []),
                "context_used": response_data.get("context_used", ""),
                "retrieval_stats": response_data.get("retrieval_stats", {}),
                "context_quality": response_data.get("context_quality", {}),
                "model_used": str(self.context_config.model_tier)
            }

            self.supabase.table("context_analytics").insert(
                context_data).execute()

        except Exception as e:
            logging.warning(f"Analytics logging failed: {e}")

    # ==========================================
    # CONVERSATION MANAGEMENT UTILITIES
    # ==========================================

    async def get_recent_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversations for this organization"""
        try:
            response = self.supabase.table("conversations").select(
                "*, messages!inner(content)"
            ).eq(
                "org_id", self.org_id
            ).order("updated_at", desc=True).limit(limit).execute()

            return response.data or []

        except Exception as e:
            logging.error(f"Error getting conversations: {e}")
            return []

    async def add_feedback(
        self,
        message_id: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """Add user feedback and update analytics"""
        try:
            # Get message to find conversation_id
            msg_response = self.supabase.table("messages").select(
                "conversation_id"
            ).eq("id", message_id).execute()

            if not msg_response.data:
                return False

            conversation_id = msg_response.data[0]["conversation_id"]

            feedback_data = {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "org_id": self.org_id,
                "rating": rating,
                "feedback_text": feedback_text
            }

            response = self.supabase.table(
                "conversation_feedback").insert(feedback_data).execute()

            # Update analytics with user satisfaction
            if response.data:
                try:
                    satisfaction_score = 1.0 if rating > 0 else 0.0
                    self.supabase.table("context_analytics").update({
                        "user_satisfaction": satisfaction_score,
                        "feedback_text": feedback_text
                    }).eq("message_id", message_id).execute()
                except Exception as e:
                    logging.warning(
                        f"Failed to update analytics with feedback: {e}")

            return bool(response.data)

        except Exception as e:
            logging.error(f"Error adding feedback: {e}")
            return False

    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for this organization"""
        try:
            # dashboard = await context_analytics.get_performance_dashboard(self.org_id, days=7)  # Requires env setup
            # return dashboard
            return {"message": "Performance insights require environment setup"}
        except Exception as e:
            logging.error(f"Error getting performance insights: {e}")
            return {}

    async def update_context_config(self, updates: Dict[str, Any]) -> bool:
        """Update context engineering configuration"""
        try:
            return await context_config_manager.update_config(self.org_id, updates)
        except Exception as e:
            logging.error(f"Error updating context config: {e}")
            return False
