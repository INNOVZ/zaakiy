"""
Response Generation Service
Handles AI response generation and context engineering
"""
import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.models.chatbot_config import ChatbotConfig
from app.services.shared import cache_service

from .context_leakage_detector import get_context_leakage_detector
from .prompt_sanitizer import PromptInjectionDetector

logger = logging.getLogger(__name__)


class ResponseGenerationError(Exception):
    """Exception for response generation errors"""


class ResponseGenerationService:
    """Handles AI response generation with context engineering"""

    def __init__(
        self,
        org_id: str,
        openai_client,
        context_config,
        chatbot_config: Union[Dict[str, Any], ChatbotConfig],
    ):
        self.org_id = org_id
        self.openai_client = openai_client
        self.context_config = context_config

        # TYPE SAFETY: Convert dict to Pydantic model for validation and type safety
        # This prevents configuration errors and provides IDE autocompletion
        if isinstance(chatbot_config, dict):
            self.chatbot_config = ChatbotConfig.from_dict(chatbot_config)
            logger.debug("Converted dict chatbot_config to Pydantic model")
        elif isinstance(chatbot_config, ChatbotConfig):
            self.chatbot_config = chatbot_config
        else:
            # Fallback to default config
            logger.warning(
                "Invalid chatbot_config type, using defaults: %s", type(chatbot_config)
            )
            self.chatbot_config = ChatbotConfig()

        # Optimized context length: balanced for quality and performance
        # Increased from 2000 to 4000 to support richer responses with more documents
        self.max_context_length = 4000

        # SECURITY: Initialize security detectors
        self.injection_detector = PromptInjectionDetector()
        self.leakage_detector = get_context_leakage_detector()

    async def generate_enhanced_response(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        retrieved_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate response with enhanced context engineering"""
        try:
            # SECURITY: Check user message for prompt injection attempts
            (
                is_injection,
                pattern,
                matched,
            ) = self.injection_detector.check_for_injection(message)

            if is_injection:
                logger.warning(
                    f"🚨 SECURITY: Blocked prompt injection attempt in user message",
                    extra={
                        "pattern": pattern,
                        "matched_text": matched,
                        "message_preview": message[:100],
                        "org_id": self.org_id,
                    },
                )
                return {
                    "response": "I'm sorry, but I can't process that request. Please rephrase your question in a different way.",
                    "sources": [],
                    "tokens_used": 0,
                    "security_blocked": True,
                    "block_reason": "prompt_injection_attempt",
                    "processing_time_ms": 0,
                }

            # SECURITY: Check for context extraction attempts
            (
                is_extraction,
                extraction_pattern,
                extraction_matched,
            ) = self.leakage_detector.is_context_extraction_attempt(message)

            if is_extraction:
                logger.warning(
                    f"🚨 SECURITY: Blocked context extraction attempt",
                    extra={
                        "pattern": extraction_pattern,
                        "matched_text": extraction_matched,
                        "message_preview": message[:100],
                        "org_id": self.org_id,
                    },
                )
                safe_message = self.leakage_detector.get_safe_response_message(
                    extraction_pattern
                )
                return {
                    "response": safe_message,
                    "sources": [],
                    "tokens_used": 0,
                    "security_blocked": True,
                    "block_reason": "context_extraction_attempt",
                    "processing_time_ms": 0,
                }

            # SECURITY: Check for iterative extraction across conversation
            if conversation_history:
                previous_messages = [
                    msg.get("content", "")
                    for msg in conversation_history
                    if msg.get("role") == "user"
                ]
                is_iterative = self.leakage_detector.check_iterative_extraction(
                    message, previous_messages
                )

                if is_iterative:
                    logger.warning(
                        f"🚨 SECURITY: Blocked iterative extraction attempt",
                        extra={
                            "message_preview": message[:100],
                            "conversation_length": len(previous_messages),
                            "org_id": self.org_id,
                        },
                    )
                    return {
                        "response": self.leakage_detector.get_safe_response_message(
                            "iterative_extraction"
                        ),
                        "sources": [],
                        "tokens_used": 0,
                        "security_blocked": True,
                        "block_reason": "iterative_extraction_attempt",
                        "processing_time_ms": 0,
                    }

            # PERFORMANCE: Check response cache for instant results (20-40% hit rate expected)
            cache_hit_response = await self._get_cached_response(
                message, retrieved_documents
            )
            if cache_hit_response:
                logger.info(
                    "💨 CACHE HIT: Instant response for query: '%s'", message[:50]
                )
                return cache_hit_response

            # DEBUG: Log retrieved documents
            logger.info(
                "📄 Retrieved %d documents for query: '%s'",
                len(retrieved_documents),
                message[:100],
            )

            # Build context from retrieved documents
            context_data = self._build_context(retrieved_documents)

            # DEBUG: Log context information
            context_text = context_data.get("context_text", "")
            logger.info(
                "📝 Context built",
                extra={
                    "org_id": self.org_id,
                    "context_length": len(context_text),
                    "documents_count": len(retrieved_documents),
                    "has_context": len(context_text) > 0,
                    "context_preview": context_text[:200] if context_text else "EMPTY",
                },
            )

            # CRITICAL: Log if context is empty but documents were retrieved
            if not context_text and retrieved_documents:
                logger.error(
                    "🚨 CRITICAL: Documents retrieved but context is EMPTY!",
                    extra={
                        "org_id": self.org_id,
                        "documents_retrieved": len(retrieved_documents),
                        "document_ids": [
                            doc.get("id", "unknown") for doc in retrieved_documents[:5]
                        ],
                    },
                )

            # CRITICAL: Log if no documents found for a specific query
            if not retrieved_documents:
                # Check if query seems specific (contains product/price/contact keywords)
                specific_keywords = [
                    "product",
                    "price",
                    "cost",
                    "plan",
                    "pricing",
                    "contact",
                    "phone",
                    "email",
                    "address",
                    "location",
                    "buy",
                    "purchase",
                    "service",
                    "feature",
                    "what",
                    "which",
                    "how much",
                ]
                query_lower = message.lower()
                is_specific_query = any(
                    keyword in query_lower for keyword in specific_keywords
                )

                if is_specific_query:
                    logger.warning(
                        "⚠️ WARNING: Specific query but NO documents retrieved",
                        extra={
                            "org_id": self.org_id,
                            "query": message,
                            "query_type": "specific",
                            "suggested_action": "Check if documents are indexed in namespace",
                        },
                    )

            # Detect if this is a contact information query - if so, use ZERO temperature
            is_contact_query = self._is_contact_information_query(message)

            # DEBUG: Check if phone numbers are in context

            phone_pattern = r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
            phones_in_context = re.findall(phone_pattern, context_text)

            if phones_in_context:
                logger.info(
                    "📞 Found %d phone numbers in context: %s",
                    len(phones_in_context),
                    phones_in_context,
                )
            elif is_contact_query:
                # Contact query but no phone in context - this is the problem!
                logger.error(
                    "🚨 PROBLEM: Contact query detected but NO phone numbers in retrieved context!"
                )
                logger.error(
                    "Retrieved documents: %d, Context length: %d chars",
                    len(retrieved_documents),
                    len(context_text),
                )
                logger.error("First 500 chars of context: %s", context_text[:500])

                # DEBUG: Log each retrieved document
                for idx, doc in enumerate(retrieved_documents):
                    chunk = doc.get("chunk", "")[:200]
                    score = doc.get("score", 0)
                    logger.error("Doc %d (score %.3f): %s...", idx + 1, score, chunk)

            # Create system prompt with context
            system_prompt = self._create_system_prompt(context_data)

            # Build conversation messages
            messages = self._build_conversation_messages(
                system_prompt, conversation_history, message
            )

            # Generate response using OpenAI with dynamic temperature
            openai_response = await self._call_openai(
                messages, force_factual=is_contact_query
            )

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

            # PERFORMANCE: Cache response for instant future retrieval (fire and forget)
            asyncio.create_task(
                self._cache_response(message, retrieved_documents, formatted_response)
            )

            return formatted_response

        except Exception as e:
            logger.error("Enhanced response generation failed: %s", e)
            raise ResponseGenerationError(f"Response generation failed: {e}") from e

    def _build_context(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context from retrieved documents and fallback org knowledge"""
        try:
            # Track whether we had to rely on context config fallback
            fallback_sections_used: List[str] = []

            if not documents:
                (
                    fallback_text,
                    fallback_sections_used,
                ) = self._build_context_from_config()
                if fallback_text:
                    return {
                        "context_text": fallback_text,
                        "sources": [
                            f"context_config.{section}"
                            for section in fallback_sections_used
                        ],
                        "context_quality": {
                            "coverage_score": 0.3,
                            "relevance_score": 0.5,
                            "fallback_context_used": True,
                            "fallback_sections": fallback_sections_used,
                        },
                    }
                return {
                    "context_text": "",
                    "sources": [],
                    "context_quality": {
                        "coverage_score": 0.0,
                        "relevance_score": 0.0,
                        "fallback_context_used": False,
                    },
                }

            # Combine document chunks
            context_chunks = []
            sources = []
            total_score = 0

            for idx, doc in enumerate(documents):
                chunk_text = doc.get("chunk", "")
                source = doc.get("source", "")
                score = doc.get("score", 0)

                # Log document structure for debugging
                if idx == 0:  # Log first document structure
                    logger.info(
                        "📄 Document structure check",
                        extra={
                            "org_id": self.org_id,
                            "has_chunk": "chunk" in doc,
                            "has_content": "content" in doc,
                            "has_text": "text" in doc,
                            "doc_keys": list(doc.keys()),
                            "chunk_length": len(chunk_text) if chunk_text else 0,
                        },
                    )

                # Try multiple possible field names for chunk content
                if not chunk_text:
                    chunk_text = (
                        doc.get("content", "")
                        or doc.get("text", "")
                        or doc.get("metadata", {}).get("text", "")
                    )

                if chunk_text and len(chunk_text.strip()) > 10:
                    context_chunks.append(chunk_text)
                    if source and source not in sources:
                        sources.append(source)
                    total_score += score

                    # DEBUG: Log each chunk being added
                    logger.debug(
                        "📄 Chunk %d (score: %.3f, length: %d): %s...",
                        idx + 1,
                        score,
                        len(chunk_text),
                        chunk_text[:100],
                    )
                else:
                    logger.warning(
                        "⚠️ Skipping document %d - no valid chunk content",
                        idx + 1,
                        extra={
                            "org_id": self.org_id,
                            "score": score,
                            "chunk_length": len(chunk_text) if chunk_text else 0,
                            "has_chunk": "chunk" in doc,
                            "has_content": "content" in doc,
                        },
                    )

            # Combine context with length limit
            context_text = self._combine_context_chunks(context_chunks)

            # Append fallback org knowledge if available (helps when docs are sparse)
            fallback_text, fallback_sections_used = self._build_context_from_config()
            if fallback_text:
                if context_text:
                    context_text = f"{context_text}\n\n---\n\n{fallback_text}"
                else:
                    context_text = fallback_text
                for section in fallback_sections_used:
                    source_key = f"context_config.{section}"
                    if source_key not in sources:
                        sources.append(source_key)

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
                    "fallback_context_used": bool(
                        fallback_sections_used and not documents
                    ),
                    "fallback_sections": fallback_sections_used,
                },
            }

        except Exception as e:
            logger.error("Context building failed: %s", e)
            return {
                "context_text": "",
                "sources": [],
                "context_quality": {
                    "coverage_score": 0.0,
                    "relevance_score": 0.0,
                    "fallback_context_used": False,
                },
            }

    def _build_context_from_config(self) -> Tuple[str, List[str]]:
        """Construct fallback context text from stored context configuration."""
        cfg_data = self._context_config_as_dict()
        if not cfg_data:
            return "", []

        sections = []
        sections_used = []
        field_mappings = [
            ("business_context", "BUSINESS OVERVIEW"),
            ("domain_knowledge", "DOMAIN KNOWLEDGE"),
            ("specialized_instructions", "SPECIAL GUIDANCE"),
        ]

        for field_name, heading in field_mappings:
            value = cfg_data.get(field_name)
            if isinstance(value, str):
                value = value.strip()
            if value:
                sections.append(f"{heading}:\n{value}")
                sections_used.append(field_name)

        if not sections:
            return "", []

        return "\n\n".join(sections), sections_used

    def _context_config_as_dict(self) -> Dict[str, Any]:
        """Return context_config as plain dict regardless of underlying type."""
        config = getattr(self, "context_config", None)
        if not config:
            return {}

        if isinstance(config, dict):
            return config

        if hasattr(config, "model_dump"):
            try:
                return config.model_dump()
            except Exception:
                pass

        if hasattr(config, "__dict__"):
            try:
                return dict(config.__dict__)
            except Exception:
                return {}

        return {}

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
        # TYPE SAFE: Using Pydantic model attribute access
        base_prompt = (
            self.chatbot_config.system_prompt
            or "You are a helpful AI assistant. Use the provided context to answer questions accurately."
        )

        context_text = context_data.get("context_text", "")

        if context_text:
            context_section = f"""
CONTEXT INFORMATION:
{context_text}

⚠️ CRITICAL ANTI-HALLUCINATION RULES ⚠️

1. USE information from the CONTEXT INFORMATION above to provide helpful, accurate answers
2. SYNTHESIZE and COMBINE information from multiple parts of the context to answer questions fully
3. If the context contains relevant information (even if not exact), USE IT to provide a helpful answer
4. NEVER make up product names, prices, descriptions, or any facts that aren't in the context
5. NEVER use placeholders like "XXX", "[insert X]", "around X", or "approximately" for specific details
6. If context has partial info (e.g., product names but no prices), share what you have and indicate what's missing
7. Only say "I don't have that information" if the context is COMPLETELY unrelated to the question
8. Be helpful and informative - use all available context to provide the best possible answer

CONTACT INFORMATION - ZERO TOLERANCE FOR ERRORS:
- Phone numbers, emails, addresses MUST be copied EXACTLY character-by-character from context
- If contact info is NOT in context, say "I don't have contact information available"

EXAMPLES OF CORRECT BEHAVIOR:

✅ CONTACT INFO - Context has: "Call us at +91 75 94 94 94 06, email: support@company.com"
   Response: "You can reach us:

📞 **Phone**: [+91 75 94 94 94 06](tel:+917594949406)
📧 **Email**: [support@company.com](mailto:support@company.com)"

❌ WRONG - Context has phone/email
   Response: "Phone: +91 75 94 94 94 06 Email: support@company.com" (Missing emojis and markdown!)

✅ PRODUCT INFO - Context has: "Solar Panel 500W costs ₹50,000"
   Response: "**[Solar Panel 500W](product-url)** - *High efficiency panel* - **Price**: ₹50,000"

❌ WRONG - Context has product
   Response: "Solar panels cost around ₹40,000-60,000" (Don't modify prices!)

GENERAL INSTRUCTIONS:
- Provide helpful, informative answers based on the context above
- Be conversational and helpful - synthesize information to answer questions fully
- Always use appropriate emojis while chatting with the user
- Give responses in the same language as the user's question
- For product questions: describe what's available based on context
- For general questions: provide relevant information from the context
- Only say "I don't have that information" if the context is COMPLETELY unrelated to the question
- Cite exact facts, numbers, prices, and contact details from context without modification
- Keep responses clear, friendly, and under 150 words
- Maintain {self.chatbot_config.tone} tone
- Refer to yourself as {self.chatbot_config.name}
- If you have partial information, share it and indicate what additional details you might not have

FORMATTING REQUIREMENTS (CRITICAL - MUST FOLLOW EXACTLY):

1. **CONTACT INFORMATION - EXACT FORMAT REQUIRED:**

   📞 **Phone**: [number](tel:number)
   📧 **Email**: [email](mailto:email)
   📍 **Location**: *address*

   ✅ CORRECT Example:
   You can reach us through:

   📞 **Phone**: [0503789198](tel:0503789198)
   📧 **Email**: [support@email.com](mailto:support@email.com)
   📍 **Location**: *Dubai Production City, IMPZ, Dubai, UAE*

   ❌ WRONG: Missing emojis, no markdown, or plain text

2. **PRODUCT INFORMATION - EXACT FORMAT:**
   **[Product Name](URL)** - *Description* - **Price**: ₹Amount

   Example: **[Solar Panel 500W](https://example.com/solar)** - *High efficiency monocrystalline panel* - **Price**: ₹25,000

3. **TEXT FORMATTING:**
   - Use **bold** for: labels (Phone, Email, Location), product names, prices, key terms
   - Use *italics* for: descriptions, locations, emphasis
   - Use bullet points: Start lists with "- " (dash and space)

4. **LINE BREAKS ARE MANDATORY:**
   - Each contact detail MUST be on a new line
   - Add blank line before contact section
   - NEVER run contact details together on one line

5. **EMOJIS:**
   - ✅ ALWAYS use emojis with contact info: 📞 for phone, 📧 for email, 📍 for location
   - ✅ Use other relevant emojis sparingly to enhance readability

6. **FORBIDDEN:**
   - ❌ NO raw URLs (always use markdown links)
   - ❌ NO running contact details together on one line
   - ❌ NO plain text contact info without formatting

"""
            return base_prompt + "\n\n" + context_section
        else:
            # CRITICAL: When no context is available, be explicit about limitations
            no_context_section = f"""
⚠️ IMPORTANT: NO CONTEXT INFORMATION AVAILABLE ⚠️

No specific documents or knowledge base information is available for this query.

RESPONSE GUIDELINES WHEN NO CONTEXT:
1. You may use your general knowledge to provide helpful information
2. ALWAYS indicate when you're providing general information vs. specific company information
3. For specific questions about products, prices, or company details:
   - If you don't have specific information, say: "I don't have specific details about [topic] in my knowledge base. Please check our website or contact our support team for accurate information."
4. For general questions, you can provide helpful general information but clarify it's general knowledge
5. NEVER make up specific product names, prices, or company details that aren't in your general knowledge
6. Be helpful but honest about limitations

EXAMPLES:

✅ GOOD - General question:
   User: "What is AI?"
   Response: "AI (Artificial Intelligence) is technology that enables machines to learn and make decisions. [General explanation]..."

✅ GOOD - Specific question without context:
   User: "What products do you have?"
   Response: "I don't have specific product details available in my knowledge base. Please check our website or contact our sales team for the most up-to-date product information. I'm here to help with general questions though!"

❌ BAD - Making up specific details:
   User: "What products do you have?"
   Response: "We offer Product A, Product B, and Product C..." (Don't make up product names!)

GENERAL INSTRUCTIONS:
- Be helpful and conversational
- Use appropriate emojis
- Maintain {self.chatbot_config.tone} tone
- Refer to yourself as {self.chatbot_config.name}
- When in doubt, direct users to the website or support team for specific information
"""
            return base_prompt + "\n\n" + no_context_section

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

    async def _call_openai_streaming(
        self, messages: List[Dict[str, str]], force_factual: bool = False
    ):
        """
        Call OpenAI API with STREAMING enabled for instant perceived response time.

        This yields tokens as they're generated, making responses feel 10x faster!
        Use this for future streaming endpoint implementation.

        Yields: Token strings as they're generated by OpenAI
        """
        try:
            # Validate OpenAI client is available
            if self.openai_client is None:
                raise ResponseGenerationError(
                    "OpenAI client is not initialized. Please check API key configuration."
                )

            # Get parameters from chatbot config (TYPE SAFE with Pydantic defaults)
            model = self.chatbot_config.model
            max_tokens = self.chatbot_config.max_tokens

            # Set temperature based on query type
            if force_factual:
                temperature = 0.0
            else:
                temperature = self.chatbot_config.temperature

            # STREAMING: Enable stream=True for instant token delivery
            def call_openai_streaming():
                return self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    timeout=20.0,
                    stream=True,  # ✅ STREAMING ENABLED - Tokens arrive instantly!
                )

            # Run streaming call in executor
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(None, call_openai_streaming)

            # Yield tokens as they arrive
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("OpenAI streaming API call failed: %s", e)
            raise ResponseGenerationError(f"OpenAI streaming failed: {e}") from e

    async def _call_openai(
        self, messages: List[Dict[str, str]], force_factual: bool = False
    ) -> str:
        """Call OpenAI API with error handling and dynamic temperature control"""
        try:
            # Validate OpenAI client is available
            if self.openai_client is None:
                raise ResponseGenerationError(
                    "OpenAI client is not initialized. Please check API key configuration."
                )

            # Get parameters from chatbot config (TYPE SAFE with Pydantic defaults)
            # PERFORMANCE: Using gpt-3.5-turbo for fastest responses (1-3s vs 5-10s for gpt-4)
            model = self.chatbot_config.model
            # PERFORMANCE: Reduced from 500 to 300 for faster responses (20-30% speedup)
            max_tokens = self.chatbot_config.max_tokens

            # CRITICAL: Use low temperature to prevent hallucinations
            # Business chatbots should prioritize ACCURACY over creativity
            if force_factual:
                temperature = 0.0  # Completely deterministic for contact info
                logger.info(
                    "Using temperature=0.0 for factual/contact information query"
                )
            else:
                # Even for general queries, use LOW temperature to minimize hallucinations
                # 0.1-0.2 provides slight variation while maintaining factual accuracy
                temperature = self.chatbot_config.temperature
                logger.debug("Using temperature=%.1f for general query", temperature)

            # Add timeout to prevent hanging
            # OpenAI client is sync, so we run it in executor with timeout
            def call_openai_api():
                return self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    timeout=20.0,  # 20 second timeout for OpenAI call
                )

            try:
                # Run sync OpenAI call in executor with timeout
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, call_openai_api),
                    timeout=25.0,  # 25 second overall timeout
                )
            except asyncio.TimeoutError:
                logger.error("OpenAI API call timed out after 25 seconds")
                raise ResponseGenerationError(
                    "Response generation timed out - please try again"
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
        """Format the final response with metadata and validate for hallucinations"""

        # Validate contact information to prevent hallucinations
        validated_response = self._validate_contact_info(
            response_text, context_data.get("context_text", "")
        )

        # SECURITY: Sanitize response for context leakage
        # Check if response contains too much raw context
        sanitized_response = self.leakage_detector.sanitize_response_for_leakage(
            validated_response,
            context_data.get("context_text", ""),
            threshold=0.8,  # 80% overlap threshold
        )

        # Post-process to ensure proper markdown formatting
        formatted_response = self._ensure_markdown_formatting(sanitized_response)

        return {
            "response": formatted_response,
            "sources": context_data.get("sources", []),
            "context_used": context_data.get("context_text", ""),
            "context_quality": context_data.get("context_quality", {}),
            "document_count": len(retrieved_documents),
            "retrieval_method": "enhanced_rag",
            "model_used": self.chatbot_config.model,  # TYPE SAFE attribute access
            "generation_metadata": {
                "temperature": self.chatbot_config.temperature,  # TYPE SAFE
                "max_tokens": self.chatbot_config.max_tokens,  # TYPE SAFE
                "context_length": len(context_data.get("context_text", "")),
                "message_count": 1,  # Current message
            },
        }

    def _ensure_markdown_formatting(self, response: str) -> str:
        """Post-process response to ensure proper markdown formatting"""

        # Fix phone number formatting
        # Pattern: Find "Phone: 1234567890" or "📞 Phone: 1234567890" and convert to markdown
        # More flexible pattern to catch variations
        phone_pattern = r"(?:📞\s*)?(?:Phone|phone|PHONE):\s*(\+?[\d\s\-\(\)]+?)(?=\s*(?:📧|Email|email|EMAIL|📍|Location|location|LOCATION|$|\n))"

        def format_phone(match):
            number = match.group(1).strip()
            # Clean number for tel: link (remove spaces and dashes)
            clean_number = re.sub(r"[\s\-\(\)]", "", number)
            return f"\n📞 **Phone**: [{number}](tel:{clean_number})"

        response = re.sub(phone_pattern, format_phone, response)

        # Fix email formatting
        # Pattern: Find "Email: email@example.com" or "📧 Email: email@example.com" and convert to markdown
        email_pattern = r"(?:📧\s*)?(?:Email|email|EMAIL):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?=\s*(?:📍|Location|location|LOCATION|$|\n))"

        def format_email(match):
            email = match.group(1).strip()
            return f"\n📧 **Email**: [{email}](mailto:{email})"

        response = re.sub(email_pattern, format_email, response)

        # Fix location formatting
        # Pattern: Find "Location: address" or "📍 Location: address" and convert to markdown
        # Capture until we hit a sentence ending or emoji
        location_pattern = r"(?:📍\s*)?(?:Location|location|LOCATION):\s*([^\.!?\n]+?)(?=(?:Feel|feel|Thank|thank|$|\n|\.))"

        def format_location(match):
            location = match.group(1).strip()
            return f"\n📍 **Location**: *{location}*\n"

        response = re.sub(location_pattern, format_location, response)

        # Add blank line after common intro phrases before contact details
        response = re.sub(
            r"((?:contact details?|reach (?:me|us)|get in touch):\s*)(\n(?:📞|📧|📍)\s*\*\*(?:Phone|Email|Location))",
            r"\1\n\2",
            response,
            flags=re.IGNORECASE,
        )

        # Clean up multiple consecutive newlines
        response = re.sub(r"\n{3,}", "\n\n", response)

        # Remove any duplicate emojis (but keep the formatted ones)
        response = re.sub(r"([📞📧📍])\s*\1+", r"\1", response)

        response = response.strip()

        return response

    def _validate_contact_info(self, response: str, context: str) -> str:
        """Validate that factual information in response exists in context - prevents hallucinations"""

        # 1. VALIDATE PHONE NUMBERS
        phone_pattern = (
            r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        )
        response_phones = re.findall(phone_pattern, response)
        context_phones = re.findall(phone_pattern, context)

        # Normalize phone numbers for comparison (remove spaces, dashes, etc.)
        def normalize_phone(phone):
            return re.sub(r"[^\d+]", "", phone)

        normalized_context_phones = set(normalize_phone(p) for p in context_phones)

        # Check each phone number in response
        for response_phone in response_phones:
            normalized_response_phone = normalize_phone(response_phone)

            # If phone number in response is not in context, it's hallucinated
            if normalized_response_phone not in normalized_context_phones:
                logger.warning(
                    "🚨 HALLUCINATION DETECTED: Phone number '%s' not in context. Context has: %s",
                    response_phone,
                    context_phones,
                )

                # Replace hallucinated number with actual or remove it
                if context_phones:
                    response = response.replace(response_phone, context_phones[0])
                    logger.info("✅ Auto-corrected phone to: %s", context_phones[0])
                else:
                    response = re.sub(
                        re.escape(response_phone),
                        "[Contact number not available]",
                        response,
                    )

        # 2. VALIDATE EMAILS
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        response_emails = re.findall(email_pattern, response)
        context_emails = set(re.findall(email_pattern, context))

        for response_email in response_emails:
            if response_email not in context_emails:
                logger.warning(
                    "🚨 HALLUCINATION DETECTED: Email '%s' not in context. Context has: %s",
                    response_email,
                    context_emails,
                )

                if context_emails:
                    actual_email = list(context_emails)[0]
                    response = response.replace(response_email, actual_email)
                    logger.info("✅ Auto-corrected email to: %s", actual_email)

        # 3. VALIDATE PRICES (₹, Rs, INR, $, etc.)
        # Look for price patterns in both response and context
        price_pattern = r"(?:₹|Rs\.?|INR|\$|USD|EUR|£)\s*[\d,]+(?:\.\d{2})?"
        response_prices = re.findall(price_pattern, response, re.IGNORECASE)
        context_prices = set(re.findall(price_pattern, context, re.IGNORECASE))

        for response_price in response_prices:
            # Normalize for comparison (remove spaces, commas)
            normalized_response_price = re.sub(r"[\s,]", "", response_price.lower())
            normalized_context_prices = {
                re.sub(r"[\s,]", "", p.lower()) for p in context_prices
            }

            if normalized_response_price not in normalized_context_prices:
                logger.warning(
                    "🚨 POTENTIAL PRICE HALLUCINATION: '%s' not found in context. Context prices: %s",
                    response_price,
                    list(context_prices)[:3],  # Show first 3 prices
                )

        # 4. DETECT VAGUE PHRASES THAT INDICATE HALLUCINATION
        hallucination_phrases = [
            r"around \d+",  # "around 50000"
            r"approximately \d+",
            r"roughly \d+",
            r"about \d+",
            r"\d+-\d+ range",  # "40000-60000 range"
        ]

        for pattern in hallucination_phrases:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(
                    "⚠️ VAGUE/ESTIMATED LANGUAGE DETECTED: AI may be hallucinating. Pattern: %s",
                    pattern,
                )

        return response

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

        # TYPE SAFE: Use Pydantic model attribute for fallback message
        response_text = fallback_messages.get(
            error_type,
            self.chatbot_config.fallback_message,
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
        """Enhance query with context from conversation history - IMPROVED"""
        enhanced = [query]  # Always include original query

        # SPECIAL CASE: Always enhance contact-related queries regardless of settings
        contact_variants = self._get_contact_query_variants(query)
        if contact_variants:
            enhanced.extend(contact_variants)
            logger.info(
                "🔍 Contact query detected! Generated %d query variants: %s",
                len(enhanced),
                enhanced,
            )
            return enhanced[:5]  # Limit to 5 total queries

        # IMPROVED: Generate query variations for better retrieval
        # This helps when the exact query doesn't match indexed documents
        query_lower = query.lower().strip()

        # Generate semantic variations for common query types
        variations = []

        # Product/Service queries
        if any(
            word in query_lower
            for word in ["product", "service", "offer", "have", "sell"]
        ):
            variations.extend(
                [
                    query,  # Original
                    query + " details",  # Add "details"
                    query + " information",  # Add "information"
                    query.replace("what", "list").replace(
                        "which", "list"
                    ),  # Change question word
                ]
            )

        # Pricing queries
        elif any(
            word in query_lower
            for word in ["price", "cost", "pricing", "plan", "plans"]
        ):
            variations.extend(
                [
                    query,
                    query + " plans",
                    query.replace("what's", "what are").replace("what is", "what are"),
                    "pricing information " + query,
                ]
            )

        # General queries - add context words
        else:
            variations = [query]
            # Add variations with common business terms
            if len(query.split()) <= 5:  # Only for short queries
                variations.append(query + " company")
                variations.append(query + " business")

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_lower = v.lower().strip()
            if v_lower not in seen and len(v.strip()) > 0:
                seen.add(v_lower)
                unique_variations.append(v)

        # Limit to 3-4 variations to balance speed and coverage
        final_queries = unique_variations[:4]

        logger.info(
            "🔍 Query enhancement: Generated %d variations",
            len(final_queries),
            extra={
                "org_id": self.org_id,
                "original": query,
                "variations": final_queries,
            },
        )

        return final_queries

    def _is_contact_information_query(self, query: str) -> bool:
        """Detect if the query is asking for contact information"""
        query_lower = query.lower()

        # Keywords that indicate a contact information query
        contact_keywords = [
            "phone",
            "number",
            "call",
            "telephone",
            "mobile",
            "contact",
            "email",
            "mail",
            "address",
            "location",
            "reach",
            "get in touch",
            "whatsapp",
            "how to contact",
            "contact details",
            "contact info",
        ]

        return any(keyword in query_lower for keyword in contact_keywords)

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

    async def _get_cached_response(
        self, message: str, retrieved_documents: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for common queries (Cache-Aside pattern)
        Returns cached response if available, None otherwise
        """
        if not cache_service:
            return None

        try:
            # Generate cache key from message and context
            cache_key = self._generate_response_cache_key(message, retrieved_documents)

            # Try to get cached response
            cached_response = await cache_service.get(cache_key)

            if cached_response:
                # Add cache hit metadata
                cached_response["cache_hit"] = True
                cached_response["cached_at"] = cached_response.get(
                    "cached_at", "unknown"
                )
                return cached_response

            return None

        except Exception as e:
            logger.warning("Failed to get cached response: %s", e)
            return None

    async def _cache_response(
        self,
        message: str,
        retrieved_documents: List[Dict[str, Any]],
        response: Dict[str, Any],
    ):
        """
        Cache response for future instant retrieval
        TTL: 1 hour for common questions
        """
        if not cache_service:
            return

        try:
            # Generate cache key
            cache_key = self._generate_response_cache_key(message, retrieved_documents)

            # Prepare response for caching
            cache_data = {
                **response,
                "cached_at": asyncio.get_event_loop().time(),
            }

            # Cache for 1 hour (3600 seconds)
            # Common questions will be served instantly
            await cache_service.set(cache_key, cache_data, 3600)

            logger.debug("Cached response for query: %s", message[:50])

        except Exception as e:
            logger.warning("Failed to cache response: %s", e)

    def _generate_response_cache_key(
        self, message: str, retrieved_documents: List[Dict[str, Any]]
    ) -> str:
        """Generate cache key for response"""
        # Create hash from message and context
        # This ensures same question with same context gets same cached response
        doc_ids = sorted([doc.get("id", "") for doc in retrieved_documents[:3]])
        composite = f"{self.org_id}:{message.lower().strip()}:{':'.join(doc_ids)}"

        # SECURITY NOTE: SHA-256 used for cache key generation (non-cryptographic purpose)
        cache_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

        return f"response_cache:v1:{self.org_id}:{cache_hash}"
