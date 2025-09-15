import os
# import uuid
import logging
from typing import Dict, List, Any
from datetime import datetime
import openai
from pinecone import Pinecone
from supabase import create_client, Client
from services.context_config import context_config_manager
from services.context_analytics import context_analytics, ContextMetrics


class ChatService:
    """Unified chat service with conversation management and context engineering"""

    def __init__(self, org_id: str, chatbot_config: dict):
        self.org_id = org_id
        self.namespace = f"org-{org_id}"
        self.chatbot_config = chatbot_config

        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX"))

        # Initialize Supabase for conversation management
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)

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
        chatbot_id: str = None,
        channel: str = 'web'
    ) -> Dict:
        """Main chat interface - handles everything"""
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

        except (openai.OpenAIError, ValueError, KeyError, ConnectionError) as e:
            logging.error("Chat error: %s", e)
            return await self._fallback_response(message, session_id)

    # ==========================================
    # CONVERSATION MANAGEMENT
    # ==========================================

    async def _get_or_create_conversation(
        self,
        session_id: str,
        chatbot_id: str = None,
        channel: str = 'web'
    ) -> Dict:
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
                return response.data[0]

            # Create new conversation
            conversation_data = {
                "org_id": self.org_id,
                "chatbot_id": chatbot_id or "default",
                "session_id": session_id,
                "channel": channel,
                "status": "active",
                "metadata": {
                    "context_config": self.context_config.config_name if self.context_config else "default"
                }
            }

            new_conv = self.supabase.table("conversations").insert(
                conversation_data).execute()
            return new_conv.data[0]

        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error managing conversation: %s", e)
            raise

    async def _add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict = None,
        processing_time_ms: int = 0
    ) -> Dict:
        """Add message to conversation"""
        try:
            message_data = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "processing_time_ms": processing_time_ms,
                "token_count": len(content) // 4  # Rough estimation
            }

            response = self.supabase.table(
                "messages").insert(message_data).execute()
            return response.data[0]

        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error adding message: %s", e)
            raise

    async def _get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent conversation history"""
        try:
            response = self.supabase.table("messages").select("*").eq(
                "conversation_id", conversation_id
            ).neq(
                "role", "system"
            ).order("created_at", desc=False).limit(limit).execute()

            return response.data or []

        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error getting conversation history: %s", e)
            return []

    # ==========================================
    # CONTEXT ENGINEERING & RAG
    # ==========================================

    async def _generate_enhanced_response(
        self,
        message: str,
        conversation_history: List[Dict]
    ) -> Dict:
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

            # Stage 5: Extract sources
            sources = [doc.get("source", "") for doc in candidates[:3]]

            return {
                "response": response,
                "sources": list(filter(None, sources)),
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

        except (openai.OpenAIError, ValueError, KeyError, ConnectionError) as e:
            logging.error("Enhanced response generation error: %s", e)
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "sources": [],
                "context_quality": {"error": True},
                "retrieval_stats": {"error": True}
            }

    async def _enhance_query(self, query: str, history: List[Dict]) -> List[str]:
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

        except (openai.OpenAIError, ValueError, KeyError) as e:
            logging.warning("Query enhancement failed: %s", e)
            return enhanced

    async def _retrieve_documents(self, queries: List[str]) -> List[Dict]:
        """Retrieve relevant documents using multiple queries"""
        all_docs = {}

        for query in queries:
            try:
                embedding = await self._generate_embedding(query)
                results = self.index.query(
                    vector=embedding,
                    top_k=self.retrieval_config["initial"],
                    namespace=self.namespace,
                    include_metadata=True
                )

                for match in results.matches:
                    doc_id = match.id
                    # Keep highest score for duplicate documents
                    if doc_id not in all_docs or match.score > all_docs[doc_id]["score"]:
                        all_docs[doc_id] = {
                            "id": doc_id,
                            "score": match.score,
                            "chunk": match.metadata.get('chunk', ''),
                            "source": match.metadata.get('source', ''),
                            "metadata": match.metadata,
                            "query_variant": query
                        }

            except (ValueError, KeyError, ConnectionError) as e:
                logging.warning("Retrieval failed for query '%s': %s", query, e)

        # Return top documents sorted by score
        sorted_docs = sorted(
            all_docs.values(), key=lambda x: x["score"], reverse=True)
        return sorted_docs[:self.retrieval_config["final"]]

    async def _assemble_context(self, documents: List[Dict], query: str) -> str:
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
        conversation_history: List[Dict]
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
- Be conversational and maintain {self.chatbot_config.get('tone', 'helpful')} tone
- If context doesn't contain relevant info, say so politely
- Reference specific information from context when applicable
- Keep responses focused and concise
- Avoid unnecessary repetition
- Maintain a logical flow of information
- Use clear and concise language
- Avoid unnecessary jargon
- Be mindful of the user's perspective
- Encourage user engagement and questions
- Answer questions clearly and concisely in less than 100 words
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

        except (openai.OpenAIError, ValueError, KeyError) as e:
            logging.error("Response generation failed: %s", e)
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
                if score < 3:  # Below moderate support
                    response += "\n\n*Please note: This response may require additional verification as it might not be fully supported by the available context.*"
            except ValueError:
                pass  # If we can't parse the score, just return original response

            return response

        except (openai.OpenAIError, ValueError, KeyError) as e:
            logging.warning("Response quality check failed: %s", e)
            return response

    # ==========================================
    # ANALYTICS AND LOGGING
    # ==========================================

    async def _log_analytics(
        self,
        conversation_id: str,
        message_id: str,
        query_original: str,
        response_data: Dict,
        processing_time: int
    ):
        """Log comprehensive analytics data"""
        try:
            # Create context metrics
            metrics = ContextMetrics(
                org_id=self.org_id,
                conversation_id=conversation_id,
                message_id=message_id,
                query_original=query_original,
                query_enhanced=response_data.get("enhanced_queries", []),
                documents_retrieved=response_data.get(
                    "retrieved_documents", []),
                context_length=len(response_data.get("context_used", "")),
                context_quality_score=response_data.get(
                    "context_quality", {}).get("coverage_score", 0.5),
                retrieval_time_ms=response_data.get(
                    "retrieval_stats", {}).get("retrieval_time_ms", 0),
                response_time_ms=processing_time,
                model_used=str(self.context_config.model_tier),
                sources_count=len(response_data.get("sources", []))
            )

            # Log to analytics system
            await context_analytics.log_context_metrics(metrics)

            # Also log to context_logs table (legacy support)
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

            self.supabase.table("context_analytics").insert(context_data).execute()

        except (ValueError, KeyError, ConnectionError) as e:
            logging.warning("Analytics logging failed: %s", e)

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
        except (openai.OpenAIError, ValueError, KeyError) as e:
            logging.error("Embedding generation failed: %s", e)
            raise

    def _assess_context_quality(self, context: str, query: str) -> Dict:
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

    async def _fallback_response(self, message: str, session_id: str) -> Dict:
        """Fallback response when main chat fails"""
        return {
            "response": self.chatbot_config.get(
                'fallback_message',
                "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment."
            ),
            "sources": [],
            "conversation_id": session_id,
            "message_id": None,
            "processing_time_ms": 0,
            "context_quality": {"error": True}
        }

    # ==========================================
    # CONVERSATION MANAGEMENT UTILITIES
    # ==========================================

    async def get_recent_conversations(self, limit: int = 20) -> List[Dict]:
        """Get recent conversations for this organization"""
        try:
            response = self.supabase.table("conversations").select(
                "*, messages!inner(content)"
            ).eq(
                "org_id", self.org_id
            ).order("updated_at", desc=True).limit(limit).execute()

            return response.data or []

        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error getting conversations: %s", e)
            return []

    async def add_feedback(
        self,
        message_id: str,
        rating: int,
        feedback_text: str = None
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
                except (ValueError, KeyError, ConnectionError) as e:
                    logging.warning(
                        "Failed to update analytics with feedback: %s", e)

            return bool(response.data)

        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error adding feedback: %s", e)
            return False

    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for this organization"""
        try:
            dashboard = await context_analytics.get_performance_dashboard(self.org_id, days=7)
            return dashboard
        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error getting performance insights: %s", e)
            return {}

    async def update_context_config(self, updates: Dict[str, Any]) -> bool:
        """Update context engineering configuration"""
        try:
            return await context_config_manager.update_config(self.org_id, updates)
        except (ValueError, KeyError, ConnectionError) as e:
            logging.error("Error updating context config: %s", e)
            return False
