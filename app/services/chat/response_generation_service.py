"""
Response Generation Service
Handles AI response generation and context engineering
"""
import asyncio
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
- Please provide the answer in the same language as the user's question.
- Keep responses focused and concise
- Avoid unnecessary repetition
- Format lists in bullet points with clear item names and details (e.g., prices, links, etc.).
- You can use markdown to format your response and also use bold and italic to make your response more engaging
- Encourage user engagement and questions
- Answer questions clearly and concisely in strictly less than 100 words
- Never make your own assumptions or fabricate information
- Be conversational and maintain {self.chatbot_config.get('tone', 'friendly')} tone and use relevant emojis to make the response more engaging
- If context doesn't contain relevant info, say so politely
- Maintain a logical flow of information
- IMPORTANT: When users ask for contact information (phone, email, address), extract the EXACT details from the context above
- NEVER use placeholders like [insert phone number] or [insert email] - always provide the actual contact details found in the context
- If you see phone numbers in the context (like +91 75 94 94 94 06), provide them exactly as shown
- If you see email addresses in the context, provide them exactly as shown
- Always refer to yourself as {self.chatbot_config.get('name', 'Assistant')} and be act like a human
- When mentioning products, always include clickable links in markdown format: [Product Name](URL)
- If you find product information in the context, make sure to mention the product names and include their links
- For product listings, format them as: **Product Name** - Description - [View Product](URL)
- Confidence threshold for responses: {self.context_config.confidence_threshold}

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
            # Validate OpenAI client is available
            if self.openai_client is None:
                raise ResponseGenerationError(
                    "OpenAI client is not initialized. Please check API key configuration."
                )

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
        """Enhance query with context from conversation history - OPTIMIZED"""
        enhanced = [query]  # Always include original query

        # SPECIAL CASE: Always enhance contact-related queries regardless of settings
        contact_variants = self._get_contact_query_variants(query)
        if contact_variants:
            enhanced.extend(contact_variants)
            logger.info(
                "Added %d contact-related query variants", len(contact_variants)
            )
            return enhanced[:5]  # Limit to 5 total queries

        # OPTIMIZATION: Skip query enhancement if disabled or no OpenAI client
        if not hasattr(self.context_config, "enable_query_rewriting"):
            return enhanced

        if not self.context_config.enable_query_rewriting or not self.openai_client:
            return enhanced

        # OPTIMIZATION: Skip enhancement for very short queries or no history
        if len(query.strip()) < 10 or not history:
            return enhanced

        try:
            # Get recent context from conversation
            recent_messages = history[-3:] if history else []
            context_summary = self._summarize_recent_context(recent_messages)

            if not context_summary:
                return enhanced

            # OPTIMIZATION: Add timeout to query enhancement (max 1 second)
            # This prevents slow OpenAI calls from blocking the entire request
            try:
                # Generate enhanced query using OpenAI
                enhancement_prompt = f"""
Given this conversation context and user query, generate 1-2 alternative ways to ask the same question that might help find more relevant information.

Conversation Context:
{context_summary}

Original Query: {query}

Alternative queries (one per line, no numbering):"""

                # Wrap the OpenAI call with a timeout
                async def enhance_with_openai():
                    return self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": enhancement_prompt}],
                        temperature=0.3,
                        max_tokens=100,  # Reduced from 150 for faster response
                    )

                response = await asyncio.wait_for(enhance_with_openai(), timeout=1.0)

                # Parse enhanced queries
                enhanced_queries = (
                    response.choices[0].message.content.strip().split("\n")
                )

                for enhanced_query in enhanced_queries:
                    enhanced_query = enhanced_query.strip()
                    if (
                        enhanced_query
                        and enhanced_query != query
                        and len(enhanced_query) > 5
                    ):
                        enhanced.append(enhanced_query)

                # Limit to max query variants
                max_variants = getattr(self.context_config, "max_query_variants", 3)
                return enhanced[:max_variants]

            except asyncio.TimeoutError:
                logger.warning(
                    "Query enhancement timed out after 1s - using original query only"
                )
                return enhanced

        except Exception as e:
            logger.warning("Query enhancement failed: %s", e)
            return enhanced

    def _get_contact_query_variants(self, query: str) -> List[str]:
        """Generate query variants for contact-related queries"""
        query_lower = query.lower()

        # Keywords that indicate a contact information query
        contact_keywords = {
            "phone": ["phone number", "contact number", "call", "telephone", "mobile"],
            "email": ["email address", "email", "contact email", "mail"],
            "address": ["address", "location", "where located", "office address"],
            "contact": [
                "contact information",
                "contact details",
                "reach",
                "get in touch",
            ],
        }

        variants = []

        # Check if query contains contact-related keywords
        for category, keywords in contact_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Add specific variants based on the category
                if category == "phone":
                    variants.extend(
                        [
                            "phone number contact information",
                            "call telephone number",
                            "contact phone details",
                        ]
                    )
                elif category == "email":
                    variants.extend(
                        ["email address contact", "email contact information"]
                    )
                elif category == "address":
                    variants.extend(["office address location", "company address"])
                elif category == "contact":
                    variants.extend(
                        [
                            "contact information phone email",
                            "contact details phone number email address",
                            "how to reach contact",
                        ]
                    )

                # Found contact keywords, return variants
                return list(set(variants))[:4]  # Return unique variants, max 4

        return []

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
