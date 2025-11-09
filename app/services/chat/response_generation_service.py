"""
Response Generation Service
Handles AI response generation and context engineering
"""
import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Union

from app.models.chatbot_config import ChatbotConfig
from app.services.shared import cache_service

from .chat_utilities import ChatUtilities
from .contact_extractor import contact_extractor
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

        # UTILITY: Initialize chat utilities for product extraction
        self.chat_utilities = ChatUtilities()

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
            logger.info("📝 Context length: %d characters", len(context_text))

            # Detect if this is a contact information query - if so, use ZERO temperature
            is_contact_query = self._is_contact_information_query(message)

            # DEBUG: Check contact information in context
            contact_info = context_data.get("contact_info", {})
            phones_in_context = contact_info.get("phones", [])
            emails_in_context = contact_info.get("emails", [])
            demo_links_in_context = contact_info.get("demo_links", [])

            if phones_in_context:
                logger.info(
                    "📞 Found %d phone numbers in context: %s",
                    len(phones_in_context),
                    phones_in_context[:3],  # Log first 3
                )
            if emails_in_context:
                logger.info(
                    "📧 Found %d email addresses in context: %s",
                    len(emails_in_context),
                    emails_in_context[:3],  # Log first 3
                )
            if demo_links_in_context:
                logger.info(
                    "🔗 Found %d demo/booking links in context: %s",
                    len(demo_links_in_context),
                    demo_links_in_context,
                )

            if is_contact_query:
                # Check if we have any contact info
                has_contact = bool(
                    phones_in_context
                    or emails_in_context
                    or contact_info.get("addresses")
                )

                if not has_contact:
                    # Contact query but no contact info - this is the problem!
                    logger.error(
                        "🚨 PROBLEM: Contact query detected but NO contact information in retrieved context!"
                    )
                logger.error(
                    "Retrieved documents: %d, Context length: %d chars",
                    len(retrieved_documents),
                    len(context_text),
                )
                logger.error("First 500 chars of context: %s", context_text[:500])

                # DEBUG: Log each retrieved document with contact extraction
                for idx, doc in enumerate(retrieved_documents):
                    chunk = doc.get("chunk", "")[:200]
                    score = doc.get("score", 0)
                    doc_contact_info = contact_extractor.extract_contact_info(chunk)
                    logger.error(
                        "Doc %d (score %.3f, has_contact: %s): %s...",
                        idx + 1,
                        score,
                        doc_contact_info.get("has_contact_info", False),
                        chunk,
                    )

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
                openai_response["content"],
                retrieved_documents,
                context_data,
                is_contact_query,
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
        """Build context from retrieved documents with contact info prioritization"""
        try:
            if not documents:
                return {
                    "context_text": "",
                    "sources": [],
                    "contact_info": {},
                    "demo_links": [],
                    "context_quality": {"coverage_score": 0.0, "relevance_score": 0.0},
                }

            # Extract contact information from all documents
            all_phones = set()
            all_emails = set()
            all_addresses = []
            all_demo_links = set()
            all_links = set()

            # Combine document chunks with contact info prioritization
            context_chunks = []
            contact_chunks = []  # Chunks with contact info (prioritized)
            sources = []
            total_score = 0

            for idx, doc in enumerate(documents):
                chunk_text = doc.get("chunk", "")
                source = doc.get("source", "")
                score = doc.get("score", 0)

                if chunk_text and len(chunk_text.strip()) > 10:
                    # Extract contact info from this chunk
                    contact_info = contact_extractor.extract_contact_info(chunk_text)

                    # Collect contact information
                    if contact_info.get("phones"):
                        all_phones.update(contact_info["phones"])
                    if contact_info.get("emails"):
                        all_emails.update(contact_info["emails"])
                    if contact_info.get("addresses"):
                        all_addresses.extend(contact_info["addresses"])
                    if contact_info.get("demo_links"):
                        all_demo_links.update(contact_info["demo_links"])
                    if contact_info.get("links"):
                        all_links.update(contact_info["links"])

                    # Prioritize chunks with contact info
                    if contact_info.get("has_contact_info"):
                        contact_chunks.append(chunk_text)
                        logger.debug(
                            "📞 Contact chunk %d (score: %.3f): phones=%d, emails=%d, demo_links=%d",
                            idx + 1,
                            score,
                            len(contact_info.get("phones", [])),
                            len(contact_info.get("emails", [])),
                            len(contact_info.get("demo_links", [])),
                        )
                    else:
                        context_chunks.append(chunk_text)

                    if source and source not in sources:
                        sources.append(source)
                    total_score += score

                    # DEBUG: Log each chunk being added
                    logger.debug(
                        "📄 Chunk %d (score: %.3f, length: %d, has_contact: %s): %s...",
                        idx + 1,
                        score,
                        len(chunk_text),
                        contact_info.get("has_contact_info", False),
                        chunk_text[:100],
                    )

            # Prioritize contact chunks - add them first
            prioritized_chunks = contact_chunks + context_chunks

            # OPTIMIZATION: Compress long chunks to reduce token usage
            # Truncate chunks longer than 800 chars to fit within context limits
            compressed_chunks = []
            for chunk in prioritized_chunks:
                if len(chunk) > 800:  # Compress long chunks
                    # Extract first 400 and last 400 chars (key info usually at ends)
                    compressed = chunk[:400] + "... [compressed] ..." + chunk[-400:]
                    compressed_chunks.append(compressed)
                else:
                    compressed_chunks.append(chunk)
            prioritized_chunks = compressed_chunks

            # Combine context with length limit
            context_text = self._combine_context_chunks(prioritized_chunks)

            # Extract product links from documents
            product_links = self.chat_utilities.extract_product_links_from_documents(
                documents
            )
            logger.info(
                "🛍️ Extracted %d product links from documents", len(product_links)
            )

            # Build product section for context if we have products
            product_section = ""
            if product_links:
                product_section = self._build_product_section(product_links, documents)
                logger.debug("📦 Product section: %s", product_section[:200])

            # Create structured contact info section if we have contact info
            contact_info_text = ""
            if all_phones or all_emails or all_addresses or all_demo_links:
                contact_info_parts = []

                if all_phones:
                    contact_info_parts.append(
                        f"Phone Numbers: {', '.join(sorted(all_phones))}"
                    )

                if all_emails:
                    contact_info_parts.append(
                        f"Email Addresses: {', '.join(sorted(all_emails))}"
                    )

                if all_addresses:
                    # Take first 3 addresses (most relevant)
                    contact_info_parts.append(
                        f"Addresses: {' | '.join(all_addresses[:3])}"
                    )

                if all_demo_links:
                    contact_info_parts.append(
                        f"Demo/Booking Links: {', '.join(sorted(all_demo_links))}"
                    )

                if contact_info_parts:
                    contact_info_text = "\n\nCONTACT INFORMATION:\n" + "\n".join(
                        contact_info_parts
                    )

            # Build final context: Product section → Contact section → Regular context
            # This ensures products and contact info are prominent
            final_context_parts = []
            if product_section:
                final_context_parts.append(product_section)
            if contact_info_text:
                final_context_parts.append(contact_info_text)
            final_context_parts.append(context_text)

            context_text = "\n".join(final_context_parts)

            # Log extracted contact info
            if all_phones or all_emails or all_demo_links:
                logger.info(
                    "✅ Extracted contact info: phones=%d, emails=%d, demo_links=%d, addresses=%d",
                    len(all_phones),
                    len(all_emails),
                    len(all_demo_links),
                    len(all_addresses),
                )

            # Calculate quality metrics
            avg_score = total_score / len(documents) if documents else 0
            coverage_score = min(len(context_text) / self.max_context_length, 1.0)

            return {
                "context_text": context_text,
                "sources": sources,
                "contact_info": {
                    "phones": list(all_phones),
                    "emails": list(all_emails),
                    "addresses": all_addresses[:5],  # Limit to 5
                    "demo_links": list(all_demo_links),
                    "all_links": list(all_links),
                },
                "product_links": product_links,  # Include extracted product links
                "demo_links": list(all_demo_links),  # For backward compatibility
                "context_quality": {
                    "coverage_score": coverage_score,
                    "relevance_score": avg_score,
                    "document_count": len(documents),
                    "contact_chunks_count": len(contact_chunks),
                    "product_links_count": len(product_links),
                },
            }

        except Exception as e:
            logger.error("Context building failed: %s", e)
            return {
                "context_text": "",
                "sources": [],
                "contact_info": {},
                "product_links": [],  # Empty product links on error
                "demo_links": [],
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

    def _build_product_section(
        self, product_links: List[Dict[str, Any]], documents: List[Dict[str, Any]]
    ) -> str:
        """Build a structured product section from extracted product links

        Args:
            product_links: List of product dicts from ChatUtilities.extract_product_links_from_documents
            documents: Original retrieved documents with chunk content

        Returns:
            Formatted product catalog string
        """
        if not product_links:
            return ""

        lines = ["PRODUCT CATALOG:"]

        for idx, product in enumerate(product_links, start=1):
            url = product.get("url", "")
            name = product.get("name", "Product")
            chunk_preview = product.get("chunk_preview", "")

            # Extract price and description from chunk if available
            price = self._extract_price_from_chunk(chunk_preview)
            description = self._extract_description_from_chunk(chunk_preview, name)

            # Format: 1. **[Product Name](URL)** - *Description* - **Price**: Amount
            line = f"{idx}. **[{name}]({url})**"
            if description:
                line += f" - *{description}*"
            if price:
                line += f" - **Price**: {price}"

            lines.append(line)

        return "\n".join(lines)

    def _extract_price_from_chunk(self, chunk: str) -> str:
        """Extract price from chunk text"""
        if not chunk:
            return ""

        # Look for common price patterns
        price_patterns = [
            r"(?:price|cost|₹|Rs\.?|AED|Dhs\.?|\$)\s*:?\s*([\d,]+(?:\.\d{2})?)",
            r"(Dhs\.?\s*[\d,]+(?:\.\d{2})?)",
            r"(₹\s*[\d,]+(?:\.\d{2})?)",
            r"(AED\s*[\d,]+(?:\.\d{2})?)",
        ]

        for pattern in price_patterns:
            match = re.search(pattern, chunk, re.IGNORECASE)
            if match:
                return (
                    match.group(1).strip()
                    if len(match.groups()) >= 1
                    else match.group(0).strip()
                )

        return ""

    def _extract_description_from_chunk(self, chunk: str, product_name: str) -> str:
        """Extract product description from chunk text"""
        if not chunk:
            return ""

        # Look for description after product name or in surrounding text
        # Limit to first 50 words for conciseness
        lines = chunk.strip().split("\n")
        for line in lines:
            # Find lines that don't contain price indicators
            if not re.search(r"price|cost|₹|Rs\.|AED|Dhs\.|\$\d", line, re.IGNORECASE):
                # Extract meaningful description text
                clean_line = line.strip()
                # Remove product name from description to avoid redundancy
                clean_line = re.sub(
                    re.escape(product_name), "", clean_line, flags=re.IGNORECASE
                ).strip()
                # Remove leading/trailing punctuation and whitespace
                clean_line = clean_line.strip(" .-–—:")

                if len(clean_line) > 20:  # Meaningful description length
                    # Limit to ~50 words
                    words = clean_line.split()
                    if len(words) > 15:
                        clean_line = " ".join(words[:15]) + "..."
                    return clean_line

        return ""

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

YOUR ROLE:
You are {self.chatbot_config.name}, a {self.chatbot_config.tone} AI assistant. Your goal is to be helpful, constructive, and conversational while providing accurate information from the context above.

CORE PRINCIPLES:

1. **Be Constructive, Not Just Informative**
   - When you can't provide exactly what the user asks for, offer helpful alternatives
   - Always provide clear next steps or ways the user can get what they need
   - Turn "no" answers into opportunities to help in other ways

2. **Accuracy First**
   - ONLY use information that exists in the CONTEXT INFORMATION above
   - Use exact facts, numbers, prices, and contact details from context
   - If information isn't in context, acknowledge it honestly but offer alternatives
   - NEVER make up product names, prices, descriptions, or facts

3. **Conversational and Natural**
   - Write in a natural, flowing style - like a helpful colleague
   - Use multi-paragraph responses when helpful for clarity
   - Be warm and professional, not robotic or overly formal
   - Respond in the same language as the user's question

RESPONSE STRUCTURE FOR DIFFERENT QUERY TYPES:

**When Answering "Yes" or Providing Information:**
- Start with a direct, clear answer
- Provide relevant details from context
- Include any helpful links, prices, or next steps
- Format information clearly with proper markdown

**When Answering "No" or Information Not Available:**
- NEVER start with "I don't have that information" or "I don't know" - these are robotic and unhelpful
- Instead, provide context about what IS available or what CAN be done
- Immediately offer constructive alternatives or solutions
- Provide clear next steps (schedule consultation, contact sales, etc.)
- Include relevant links (demo/booking links) when available
- Make the user feel helped, not blocked

**CRITICAL: Forbidden Phrases - NEVER Use These:**
- ❌ "I don't have that information available"
- ❌ "I don't have that information"
- ❌ "I don't know"
- ❌ "That information is not available"
- ❌ "I can't help with that"
- ❌ "I'm not able to provide that information"

**Instead, Use Constructive Alternatives:**
- ✅ "Based on the information available to me..."
- ✅ "While I don't see [specific detail] available..."
- ✅ "At the moment, we [do/don't] offer [specific thing]..."
- ✅ "Currently, [company] [status/action]..."
- ✅ Always follow with: "However, I can help you by..."

**Example for "No" Answers (Office Location):**
User: "Do you have an office in Germany?"
❌ BAD: "I don't have that information available. If you have any other questions, feel free to ask!"
✅ GOOD: "{self.chatbot_config.name} is currently based in [location from context if available, or 'our headquarters']. While we don't have an office in Germany at this time, we work with clients globally and all consultations can be managed online.

If you're interested in our solutions or want to discuss how we can help from Germany, you can schedule a free consultation with our team. They can provide detailed information about our services and how we work with international clients.

Would you like to **[book a consultation here](demo-link-from-context)**?"

**Example for Missing Information (Free Demo):**
User: "Is there a free demo available?"
❌ BAD: "I don't have that information available. Feel free to explore more about {self.chatbot_config.name} on our website."
✅ GOOD: "At the moment, we do not offer a free demo or trial version of {self.chatbot_config.name}.

However, you can schedule a free consultation call with our team to see how {self.chatbot_config.name} works, discuss your specific needs, and get a personalized demonstration based on your requirements. This allows us to tailor the demo to your use case and answer any questions you might have.

If you're interested, you can **[book your free consultation here](demo-link-from-context)**."

**When Context Doesn't Have Specific Information:**
- Even if the exact answer isn't in context, you can still be helpful
- Mention what the context DOES contain (if anything relevant)
- Always pivot to offering ways to get the information (consultation, contact sales, etc.)
- Never leave the user with just "I don't have that information"

CONTACT INFORMATION HANDLING:
- ONLY provide contact information (phone, email, address) when the user EXPLICITLY asks for it
- If user asks about products, prices, or other topics, DO NOT include contact information unless asked
- Phone numbers, emails, addresses MUST be copied EXACTLY from context
- Format contact info clearly with emojis and markdown:
  - 📞 **Phone**: [number](tel:number)
  - 📧 **Email**: [email](mailto:email)
  - 📍 **Location**: *address*
- If contact info is NOT in context AND user asks for it, acknowledge and offer to connect them with the team

DEMO/BOOKING LINKS:
- If context has demo/booking links, use them when offering consultations or demos
- Use the EXACT URL from the "Demo/Booking Links" section in context
- Format as: **[Book a Consultation](exact-url-from-context)** or **[Schedule a Demo](exact-url-from-context)**
- Include these links when offering alternatives to direct requests

PRODUCT INFORMATION:
- If context includes a PRODUCT CATALOG section, use those exact product links, names, and prices
- Format: **[Product Name](URL)** - *Description* - **Price**: Amount (if available)
- If price is missing from context, list product without price - DO NOT insert placeholders
- When listing products, focus on product names, descriptions, prices, and links

FORMATTING GUIDELINES:
- Use **bold** for: labels (Phone, Email, Location), product names, prices, key terms
- Use *italics* for: descriptions, locations, emphasis
- Use bullet points with "- " for lists
- Each contact detail on a new line with blank line before contact section
- Always use markdown links, never raw URLs
- Use emojis appropriately: 📞 (phone), 📧 (email), 📍 (location), and others sparingly for readability

RESPONSE LENGTH:
- Aim for 2-4 paragraphs for comprehensive answers
- Keep responses informative but concise (typically 100-200 words)
- Don't be artificially brief - provide complete, helpful answers

TONE AND LANGUAGE:
- Maintain a {self.chatbot_config.tone} tone
- Be warm, helpful, and professional
- Avoid being robotic or overly technical
- Make users feel understood and supported

REMEMBER:
- Accuracy is important, but being helpful and constructive is equally important
- When you can't give a direct answer, always offer a constructive alternative
- Provide clear calls to action with links when relevant
- Make every response feel like it adds value, even when the answer is "no"
- NEVER use phrases like "I don't have that information" - always be constructive
- If context has demo/booking links, ALWAYS include them when offering consultations or demos
- Every response should make the user feel helped and supported, never blocked or dismissed
- Turn every "no" into an opportunity to help in another way
"""
            return base_prompt + "\n\n" + context_section
        else:
            # NO CONTEXT AVAILABLE - Provide helpful fallback instructions
            no_context_section = f"""
NO KNOWLEDGE BASE CONTEXT AVAILABLE

YOUR ROLE:
You are {self.chatbot_config.name}, a {self.chatbot_config.tone} AI assistant. While you don't have specific information in your knowledge base right now, your goal is to be helpful, constructive, and guide users toward getting the information they need.

CORE APPROACH:

1. **Be Honest and Constructive**
   - Acknowledge that you don't have specific details in your knowledge base
   - Immediately offer helpful alternatives and next steps
   - Never make up information, but always provide value through guidance

2. **Offer Clear Next Steps**
   - Connect them with the right people (sales team, support, etc.)
   - Suggest scheduling consultations or demos when relevant
   - Provide multiple ways to get the information they need

3. **Be Conversational and Helpful**
   - Write naturally, like a helpful colleague
   - Use multi-paragraph responses for clarity
   - Maintain a warm, professional {self.chatbot_config.tone} tone
   - Use emojis appropriately to keep responses engaging

RESPONSE PATTERNS:

**For Product Questions:**
"I'd love to help you with information about our products! 🎯

While I don't have specific product details in my knowledge base right now, I can help you in a few ways:

1. **Connect you with our team** - They can provide detailed product information and pricing
2. **Schedule a demo** - See our products in action
3. **Answer general questions** - About what we do and how we can help

What specific information are you looking for? I'll make sure you get the right details! 😊"

**For Pricing Questions:**
"Great question about pricing! 💰

While I don't have specific pricing details in my current knowledge base, here's how I can help:

• **Get a Custom Quote** - Our pricing varies based on your needs
• **Schedule a Consultation** - Discuss your requirements and get accurate pricing
• **Compare Plans** - I can connect you with someone who can explain our different options

Would you like me to help you get in touch with our sales team for detailed pricing information?"

**For Service/Plan Questions:**
"I'd be happy to help you understand our services and plans!

While I don't have the specific details in my knowledge base at the moment, here are some ways I can assist:

• **Schedule a Consultation** - Our team can explain all available options and help you find the best fit
• **Get More Information** - I can connect you with someone who can provide detailed information about features and plans
• **Answer General Questions** - I can help with general questions about what we do

What specific services or features are you most interested in?"

**For Contact/Location Questions:**
"I'd be happy to help you get in touch with us! 📞

While I don't have contact information in my knowledge base right now, here's how you can reach us:

• **Contact Support** - Our support team can provide contact details and answer your questions
• **Schedule a Call** - Book a consultation to speak directly with our team
• **Visit Our Website** - You can find our contact information on our main website

Is there something specific you'd like to discuss? I can help connect you with the right person!"

GENERAL GUIDELINES:
- Always acknowledge what you don't know, but immediately offer alternatives
- Provide 2-3 clear next steps in every response
- Use natural, conversational language
- Keep responses to 2-4 paragraphs (100-200 words)
- Make every response feel helpful and valuable
- Never say "I don't know" without offering what you CAN do
- Use emojis sparingly but effectively to enhance readability
- Maintain a {self.chatbot_config.tone} tone throughout

**CRITICAL: Forbidden Phrases - NEVER Use These:**
- ❌ "I don't have that information available"
- ❌ "I don't have that information"
- ❌ "I don't know"
- ❌ "That information is not available"
- ❌ "I can't help with that"

**Instead, Always:**
- ✅ Start with what you CAN do or offer
- ✅ Provide specific next steps (schedule consultation, contact sales, etc.)
- ✅ Include links when available (demo/booking links)
- ✅ Make the user feel helped, not blocked

REMEMBER:
- Being helpful is more important than having all the answers
- Turn limitations into opportunities to guide users
- Every response should add value and provide next steps
- Make users feel supported, not frustrated
- NEVER use robotic phrases - always be constructive and action-oriented
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
        is_contact_query: bool = False,
    ) -> Dict[str, Any]:
        """Format the final response with metadata and validate for hallucinations"""

        # Only validate contact information if user actually asked for it
        # This prevents false positives when LLM mentions numbers that aren't contact info
        validated_response = self._validate_contact_info(
            response_text, context_data.get("context_text", ""), is_contact_query
        )

        # Ensure demo links from context are included if missing
        demo_links = context_data.get("demo_links", [])
        contact_info = context_data.get("contact_info", {})
        if contact_info:
            demo_links = contact_info.get("demo_links", demo_links)

        # PROACTIVE: Always add demo links when offering consultations, demos, or when response seems incomplete
        # Check if response mentions consultation/demo keywords or seems to be offering help
        consultation_keywords = [
            "consultation",
            "demo",
            "book",
            "booking",
            "schedule",
            "call",
            "talk",
            "discuss",
            "connect",
            "contact sales",
            "reach out",
            "get in touch",
            "speak with",
        ]

        response_lower = validated_response.lower()
        mentions_consultation = any(
            keyword in response_lower for keyword in consultation_keywords
        )

        # Also check if response seems to be offering alternatives (common patterns)
        offers_alternatives = any(
            phrase in response_lower
            for phrase in [
                "however",
                "you can",
                "we can",
                "i can help",
                "let me",
                "schedule",
                "contact",
                "reach",
                "connect",
            ]
        )

        # If we have demo links and response is offering help/consultation, ensure link is included
        if demo_links and (mentions_consultation or offers_alternatives):
            # Check if any demo link from context is already in the response
            has_demo_link = any(link in validated_response for link in demo_links)

            if not has_demo_link:
                # Add the demo link proactively
                logger.info("Adding demo link to response: %s", demo_links[0])
                demo_link_text = f"**[Book a Consultation]({demo_links[0]})**"

                # Try to find a good insertion point (after consultation/demo mentions)
                # Search in the original response (case-insensitive)
                demo_pattern = r"(consultation|demo|call|talk|discuss|schedule|book|booking)[^\n\.!?]*(?:\.|!|\?|$)"
                match = re.search(demo_pattern, response_lower, re.IGNORECASE)

                if match:
                    # Insert link after the match position
                    pos = match.end()
                    # Add link with proper formatting
                    if pos < len(validated_response):
                        # Check if there's already punctuation or whitespace
                        next_char = (
                            validated_response[pos : pos + 1]
                            if pos < len(validated_response)
                            else ""
                        )
                        if next_char.strip() and next_char not in [".", "!", "?"]:
                            validated_response = (
                                validated_response[:pos]
                                + f". You can {demo_link_text}."
                                + validated_response[pos:]
                            )
                        else:
                            validated_response = (
                                validated_response[:pos]
                                + f" You can {demo_link_text}."
                                + validated_response[pos:].lstrip()
                            )
                    else:
                        # Append at end
                        validated_response += f"\n\nYou can {demo_link_text}."
                else:
                    # Check if response ends with punctuation, add link naturally
                    if validated_response.strip().endswith((".", "!", "?")):
                        validated_response += f" You can {demo_link_text}."
                    else:
                        validated_response += f"\n\nYou can {demo_link_text}."

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
            "contact_info": contact_info,  # Include extracted contact info
            "demo_links": demo_links,  # Include demo links
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

    def _validate_contact_info(
        self, response: str, context: str, is_contact_query: bool = False
    ) -> str:
        """Validate that factual information in response exists in context - prevents hallucinations

        Only validates contact info when user actually asked for it to avoid false positives.
        This prevents expensive regex operations on every response.
        """

        # OPTIMIZATION: Only validate contact information if this is actually a contact query
        # This prevents false positives where prices or other numbers are mistaken for phone numbers
        # and avoids expensive regex operations on every non-contact response
        if not is_contact_query:
            # For non-contact queries, do a quick check: if response mentions contact keywords,
            # we might want to do light validation. Otherwise, skip entirely.
            contact_keywords = ["phone", "contact", "call", "email", "reach", "number"]
            has_contact_keywords = any(
                keyword in response.lower() for keyword in contact_keywords
            )

            if not has_contact_keywords:
                # No contact keywords at all, skip validation entirely to save performance
                logger.debug(
                    "Skipping contact validation - not a contact query and no contact keywords"
                )
                return response

            # If response has contact keywords but it's not a contact query, the LLM might
            # have mentioned contact info incidentally. We'll do a minimal check but won't
            # run expensive validation. Just return as-is to avoid false positives.
            logger.debug(
                "Skipping expensive contact validation - not a contact query (keywords found but query wasn't about contact)"
            )
            return response

        # PERFORMANCE: Only reach here if is_contact_query is True
        # This means we only run expensive regex operations for actual contact queries

        # 1. VALIDATE PHONE NUMBERS
        # Improved pattern: more specific to avoid matching prices
        # Phone numbers typically have country codes, area codes, or specific formatting
        phone_pattern = r"(?:(?:\+|00)[1-9]\d{0,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        response_phones = re.findall(phone_pattern, response)
        context_phones = re.findall(phone_pattern, context)

        # Filter out false positives: prices, years, IDs that match phone pattern
        def is_likely_phone(text: str) -> bool:
            """Check if a matched pattern is likely a phone number vs price/ID"""
            # Remove common phone formatting
            cleaned = re.sub(r"[^\d+]", "", text)

            # Too short or too long to be a phone
            if len(cleaned) < 8 or len(cleaned) > 15:
                return False

            # If it's next to price indicators (₹, $, Dhs, etc.), it's probably a price
            if re.search(r"[₹$€£Dhs]|price|cost", text, re.IGNORECASE):
                return False

            # If it's clearly in a price context (Dhs. 70.00), skip it
            if re.search(r"dhs\.?\s*\d+|price.*\d+|\d+.*price", text, re.IGNORECASE):
                return False

            return True

        # Filter to only likely phone numbers
        response_phones = [p for p in response_phones if is_likely_phone(p)]
        context_phones = [p for p in context_phones if is_likely_phone(p)]

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

                # Only replace if this is a contact query - otherwise, just log it
                if is_contact_query:
                    # Replace hallucinated number with actual or remove it
                    if context_phones:
                        response = response.replace(response_phone, context_phones[0])
                        logger.info("✅ Auto-corrected phone to: %s", context_phones[0])
                    else:
                        # Only insert placeholder if user asked for contact info
                        response = re.sub(
                            re.escape(response_phone),
                            "[Contact number not available]",
                            response,
                        )
                else:
                    # Not a contact query - just remove the hallucinated phone to avoid confusion
                    logger.debug(
                        "Removing hallucinated phone from non-contact response: %s",
                        response_phone,
                    )
                    response = response.replace(response_phone, "").strip()

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

        # SPECIAL CASE: Always enhance product/pricing queries regardless of settings
        product_variants = self._get_product_query_variants(query)
        if product_variants:
            enhanced.extend(product_variants)
            logger.info(
                "🛍️ Product/pricing query detected! Generated %d query variants: %s",
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

    def _get_product_query_variants(self, query: str) -> List[str]:
        """Generate query variants for product/pricing queries"""
        query_lower = query.lower()

        # Keywords that indicate a product/pricing query
        product_keywords = {
            "product": [
                "product",
                "products",
                "item",
                "items",
                "offering",
                "offerings",
                "sell",
                "selling",
                "available",
            ],
            "pricing": [
                "price",
                "pricing",
                "cost",
                "costs",
                "how much",
                "pricing plan",
                "subscription",
                "fee",
                "fees",
            ],
            "plans": [
                "plan",
                "plans",
                "package",
                "packages",
                "tier",
                "tiers",
                "subscription",
                "subscriptions",
            ],
            "features": [
                "feature",
                "features",
                "what do you have",
                "what you have",
                "what's included",
                "what is included",
            ],
        }

        variants = []

        # Check if query contains product-related keywords
        for category, keywords in product_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Add specific variants based on the category
                if category == "product":
                    variants.extend(
                        [
                            "products services offerings catalog",
                            "what products available",
                            "product catalog list",
                        ]
                    )
                elif category == "pricing":
                    variants.extend(
                        [
                            "pricing cost price plans",
                            "how much does it cost pricing",
                            "price list pricing information",
                        ]
                    )
                elif category == "plans":
                    variants.extend(
                        [
                            "subscription plans packages tiers",
                            "pricing plans available options",
                            "plan features comparison",
                        ]
                    )
                elif category == "features":
                    variants.extend(
                        [
                            "features capabilities offerings",
                            "what's included in product",
                        ]
                    )
                break  # Only use first matching category

        return variants[:3]  # Limit to 3 variants

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
