"""
Response Generation Service
Handles AI response generation and context engineering
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseGenerationError(Exception):
    """Exception for response generation errors"""


class ResponseGenerationService:
    """Handles AI response generation with context engineering"""

    def __init__(self, org_id: str, openai_client, context_config, chatbot_config):
        self.org_id = org_id
        self.openai_client = openai_client
        self.context_config = context_config
        self.chatbot_config = chatbot_config
        self.max_context_length = 4000

    async def generate_enhanced_response(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        retrieved_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate response with enhanced context engineering"""
        try:
            # Build context from retrieved documents
            context_data = self._build_context(retrieved_documents)

            # Create system prompt with context
            system_prompt = self._create_system_prompt(context_data)

            # Build conversation messages
            messages = self._build_conversation_messages(
                system_prompt, conversation_history, message
            )

            # Generate response using OpenAI
            openai_response = await self._call_openai(messages)

            # Process and format the response
            formatted_response = self._format_response(
                openai_response["content"], retrieved_documents, context_data
            )

            # Add token usage information to the response
            formatted_response["tokens_used"] = openai_response["tokens_used"]
            formatted_response["prompt_tokens"] = openai_response["prompt_tokens"]
            formatted_response["completion_tokens"] = openai_response[
                "completion_tokens"
            ]

            return formatted_response

        except Exception as e:
            logger.error("Enhanced response generation failed: %s", e)
            raise ResponseGenerationError(f"Response generation failed: {e}") from e

    def _build_context(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context from retrieved documents"""
        try:
            if not documents:
                return {
                    "context_text": "",
                    "sources": [],
                    "context_quality": {"coverage_score": 0.0, "relevance_score": 0.0},
                }

            # Combine document chunks
            context_chunks = []
            sources = []
            total_score = 0

            for doc in documents:
                chunk_text = doc.get("chunk", "")
                source = doc.get("source", "")
                score = doc.get("score", 0)

                if chunk_text and len(chunk_text.strip()) > 10:
                    context_chunks.append(chunk_text)
                    if source and source not in sources:
                        sources.append(source)
                    total_score += score

            # Combine context with length limit
            context_text = self._combine_context_chunks(context_chunks)

            # Calculate quality metrics
            avg_score = total_score / len(documents) if documents else 0
            coverage_score = min(len(context_text) / self.max_context_length, 1.0)

            return {
                "context_text": context_text,
                "sources": sources,
                "context_quality": {
                    "coverage_score": coverage_score,
                    "relevance_score": avg_score,
                    "document_count": len(documents),
                },
            }

        except Exception as e:
            logger.error("Context building failed: %s", e)
            return {
                "context_text": "",
                "sources": [],
                "context_quality": {"coverage_score": 0.0, "relevance_score": 0.0},
            }

    def _combine_context_chunks(self, chunks: List[str]) -> str:
        """Combine context chunks with length limits"""
        if not chunks:
            return ""

        combined = ""
        current_length = 0

        for chunk in chunks:
            chunk_length = len(chunk)

            # Check if adding this chunk would exceed the limit
            if (
                current_length + chunk_length + 50 > self.max_context_length
            ):  # 50 char buffer
                break

            if combined:
                combined += "\n\n---\n\n"
                current_length += 7  # Length of separator

            combined += chunk.strip()
            current_length += chunk_length

        return combined

    def _create_system_prompt(self, context_data: Dict[str, Any]) -> str:
        """Create system prompt with context"""
        base_prompt = self.chatbot_config.get(
            "system_prompt",
            "You are a helpful AI assistant. Use the provided context to answer questions accurately.",
        )

        context_text = context_data.get("context_text", "")

        if context_text:
            context_section = f"""
CONTEXT INFORMATION:
{context_text}

INSTRUCTIONS:
- Use the context information above to answer the user's question
- If the context doesn't contain relevant information, say so clearly
- Cite sources when possible
- Be concise but comprehensive
- If you're unsure about something, acknowledge the uncertainty

"""
            return base_prompt + "\n\n" + context_section
        else:
            return (
                base_prompt
                + "\n\nNo specific context information is available for this query."
            )

    def _build_conversation_messages(
        self, system_prompt: str, history: List[Dict[str, Any]], current_message: str
    ) -> List[Dict[str, str]]:
        """Build conversation messages for OpenAI API"""
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last few messages)
        history_limit = 10  # Limit to last 10 messages
        recent_history = (
            history[-history_limit:] if len(history) > history_limit else history
        )

        for msg in recent_history:
            role = msg.get("role")
            content = msg.get("content", "")

            if role in ["user", "assistant"] and content.strip():
                messages.append({"role": role, "content": content})

        # Add current message
        messages.append({"role": "user", "content": current_message})

        return messages

    async def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API with error handling"""
        try:
            # Get parameters from chatbot config with defaults
            model = self.chatbot_config.get("model", "gpt-3.5-turbo")
            temperature = self.chatbot_config.get("temperature", 0.7)
            max_tokens = self.chatbot_config.get("max_tokens", 500)

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            # Extract actual token usage from OpenAI response
            actual_tokens = response.usage.total_tokens if response.usage else 0

            return {
                "content": response.choices[0].message.content,
                "tokens_used": actual_tokens,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else 0,
            }

        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise ResponseGenerationError(f"OpenAI API call failed: {e}") from e

    def _format_response(
        self,
        response_text: str,
        retrieved_documents: List[Dict[str, Any]],
        context_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format the final response with metadata"""
        return {
            "response": response_text,
            "sources": context_data.get("sources", []),
            "context_used": context_data.get("context_text", ""),
            "context_quality": context_data.get("context_quality", {}),
            "document_count": len(retrieved_documents),
            "retrieval_method": "enhanced_rag",
            "model_used": self.chatbot_config.get("model", "gpt-3.5-turbo"),
            "generation_metadata": {
                "temperature": self.chatbot_config.get("temperature", 0.7),
                "max_tokens": self.chatbot_config.get("max_tokens", 500),
                "context_length": len(context_data.get("context_text", "")),
                "message_count": 1,  # Current message
            },
        }

    async def generate_fallback_response(
        self, message: str, session_id: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate fallback response when main generation fails"""
        fallback_messages = {
            "AI service temporarily unavailable": "I'm experiencing some issues connecting to my AI service. Please try again in a moment.",
            "Database connection issue": "I'm having trouble accessing my memory right now. Please try again shortly.",
            "Knowledge retrieval issue": "I'm having difficulty finding relevant information. Please rephrase your question.",
            "Context processing issue": "I'm having trouble understanding the context. Could you be more specific?",
            "Response generation issue": "I'm having trouble formulating a response. Please try rephrasing your question.",
        }

        response_text = fallback_messages.get(
            error_type,
            self.chatbot_config.get(
                "fallback_message",
                "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment.",
            ),
        )

        return {
            "response": response_text,
            "sources": [],
            "conversation_id": session_id,
            "message_id": None,
            "processing_time_ms": 0,
            "context_quality": {"error": True, "error_type": error_type},
            "error_context": error_type,
            "is_fallback": True,
        }

    async def enhance_query(
        self, query: str, history: List[Dict[str, Any]]
    ) -> List[str]:
        """Enhance query with context from conversation history"""
        enhanced = [query]  # Always include original query

        if not self.context_config.enable_query_rewriting:
            return enhanced

        try:
            # Get recent context from conversation
            recent_messages = history[-3:] if history else []
            context_summary = self._summarize_recent_context(recent_messages)

            if not context_summary:
                return enhanced

            # Generate enhanced query using OpenAI
            enhancement_prompt = f"""
Given this conversation context and user query, generate 1-2 alternative ways to ask the same question that might help find more relevant information.

Conversation Context:
{context_summary}

Original Query: {query}

Alternative queries (one per line, no numbering):"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": enhancement_prompt}],
                temperature=0.3,
                max_tokens=150,
            )

            # Parse enhanced queries
            enhanced_queries = response.choices[0].message.content.strip().split("\n")

            for enhanced_query in enhanced_queries:
                enhanced_query = enhanced_query.strip()
                if (
                    enhanced_query
                    and enhanced_query != query
                    and len(enhanced_query) > 5
                ):
                    enhanced.append(enhanced_query)

            # Limit to max query variants
            return enhanced[: self.context_config.max_query_variants]

        except Exception as e:
            logger.warning("Query enhancement failed: %s", e)
            return enhanced

    def _summarize_recent_context(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize recent conversation context"""
        if not messages:
            return ""

        context_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                context_parts.append(f"User asked: {content[:100]}")
            elif role == "assistant":
                context_parts.append(f"I responded about: {content[:100]}")

        return " | ".join(context_parts)
