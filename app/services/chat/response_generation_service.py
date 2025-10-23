"""
Response Generation Service
Handles AI response generation and context engineering
"""
import asyncio
import logging
import re
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
            logger.info("📝 Context length: %d characters", len(context_text))

            # Detect if this is a contact information query - if so, use ZERO temperature
            is_contact_query = self._is_contact_information_query(message)

            # DEBUG: Check if phone numbers are in context
            import re

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

            for idx, doc in enumerate(documents):
                chunk_text = doc.get("chunk", "")
                source = doc.get("source", "")
                score = doc.get("score", 0)

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

⚠️ CRITICAL ANTI-HALLUCINATION RULES ⚠️

1. ONLY use information from the CONTEXT INFORMATION above
2. If information is NOT in the context, say "I don't have that information in my knowledge base"
3. NEVER generate, assume, or fabricate any information
4. NEVER use placeholders like [insert X] or make up examples

CONTACT INFORMATION - ZERO TOLERANCE FOR ERRORS:
- Phone numbers, emails, addresses MUST be copied EXACTLY character-by-character
- If contact info is NOT in context, say "I don't have contact information available"

EXAMPLES OF CORRECT BEHAVIOR:
✅ Context has: "Call us at +91 75 94 94 94 06"
   Response: "You can reach us at +91 75 94 94 94 06 📞"

❌ WRONG - Context has: "Call us at +91 75 94 94 94 06"
   Response: "Call us at +91 9876543210" (NEVER make up numbers!)

✅ Context has: "Solar panels cost ₹50,000"
   Response: "Solar panels cost ₹50,000"

❌ WRONG - Context has: "Solar panels cost ₹50,000"
   Response: "Solar panels cost around ₹40,000-60,000" (Don't modify prices!)

GENERAL INSTRUCTIONS:
- ONLY provide information that exists in the context above
- Use emojis sparingly to enhance readability and engagement and only when appropriate
- If information is NOT in context, respond: "I don't have that specific information in my knowledge base"
- Be precise and factual - accuracy is MORE important than sounding friendly
- Cite exact facts, numbers, prices, and details from the context without modification
- Use the same language as the user's question
- Keep responses under 100 words and focused
- Format lists with bullet points when listing multiple items
- Use markdown, bold, and italic for readability
- Maintain {self.chatbot_config.get('tone', 'friendly')} tone but prioritize accuracy over friendliness
- Refer to yourself as {self.chatbot_config.get('name', 'Assistant')}
- When mentioning products, include clickable links: [Product Name](URL)
- For product listings: **Product Name** - Description - [View Product](URL)
- If unsure or information seems incomplete, acknowledge the uncertainty rather than guessing

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

            # Get parameters from chatbot config with defaults
            model = self.chatbot_config.get("model", "gpt-3.5-turbo")
            max_tokens = self.chatbot_config.get("max_tokens", 500)

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
                temperature = self.chatbot_config.get(
                    "temperature", 0.1
                )  # Changed from 0.7 to 0.1
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

        return {
            "response": validated_response,
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
                "🔍 Contact query detected! Generated %d query variants: %s",
                len(enhanced),
                enhanced,
            )
            return enhanced[:5]  # Limit to 5 total queries

        # EMERGENCY MODE: Always skip query enhancement for speed
        # Saves 500-1000ms per request!
        logger.info("⚡ EMERGENCY MODE: Skipping query enhancement for speed")
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
