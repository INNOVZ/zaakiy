"""
Response Generation Service
Handles AI response generation and context engineering
"""
import asyncio
import hashlib
import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from app.models.chatbot_config import ChatbotConfig
from app.services.shared import cache_service

from .chat_utilities import ChatUtilities
from .contact_extractor import contact_extractor
from .context_builder import ContextBuilder
from .context_leakage_detector import get_context_leakage_detector
from .intent_detection_service import IntentResult, IntentType
from .prompt_sanitizer import PromptInjectionDetector
from .response_post_processor import ResponsePostProcessor

logger = logging.getLogger(__name__)


class ResponseGenerationError(Exception):
    """Exception for response generation errors"""


class ResponseGenerationService:
    """Handles AI response generation with context engineering"""

    CACHE_TTL_SECONDS = 3600

    def __init__(
        self,
        org_id: str,
        openai_client,
        context_config,
        chatbot_config: Union[Dict[str, Any], ChatbotConfig],
    ):
        self.org_id = org_id
        self.openai_client = openai_client
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

        # SECURITY: Initialize security detectors
        self.injection_detector = PromptInjectionDetector()
        self.leakage_detector = get_context_leakage_detector()

        # UTILITY: Initialize chat utilities for product extraction
        self.chat_utilities = ChatUtilities()
        self.context_builder = ContextBuilder(self.chat_utilities)
        self.response_post_processor = ResponsePostProcessor(
            self.leakage_detector, self.chatbot_config
        )

        self._context_config = None
        self.max_context_length = 4000
        self.context_config = context_config

    @property
    def context_config(self):
        return self._context_config

    @context_config.setter
    def context_config(self, value):
        self._context_config = value
        self.max_context_length = self._determine_max_context_length(value)
        logger.debug("Using max_context_length: %d", self.max_context_length)

    def _has_pricing_context(
        self, documents: List[Dict[str, Any]], context_text: str
    ) -> bool:
        """Check if current context contains pricing details."""
        try:
            for doc in documents:
                metadata = doc.get("metadata", {})
                if metadata.get("has_pricing") or metadata.get("pricing_info"):
                    return True

            lowered = (context_text or "").lower()
            pricing_terms = [
                "$",
                "€",
                "£",
                "usd",
                "eur",
                "price",
                "pricing",
                "per month",
                "per year",
                "plan",
                "package",
                "tier",
                "/month",
                "/year",
            ]
            return any(term in lowered for term in pricing_terms)
        except Exception as e:
            logger.debug("Pricing context detection failed: %s", e)
            return False

    def _build_pricing_fallback_response(
        self, context_data: Dict[str, Any], intent_result: Optional[IntentResult]
    ) -> Dict[str, Any]:
        """Return a constructive fallback response when pricing data isn't in context."""
        contact_info = context_data.get("contact_info", {}) or {}
        demo_links = contact_info.get("demo_links") or context_data.get(
            "demo_links", []
        )
        demo_link = demo_links[0] if demo_links else None

        response_lines = [
            "Great question about pricing! 💰",
            "I don’t see ready-made plan details in this knowledge base right now, but here’s how we can help:",
            "- **Get a Custom Quote** – We’ll tailor pricing based on your goals.",
            "- **Schedule a Consultation** – Talk through your requirements and get accurate numbers.",
            "- **Compare Options** – We can explain different tiers and what’s included.",
        ]

        if demo_link:
            response_lines.append(
                f"\nWould you like me to help you **[connect with our team]({demo_link})** so they can share the latest plans?"
            )
        else:
            response_lines.append(
                "\nWould you like me to connect you with our team so they can share the latest plans?"
            )

        response_text = "\n".join(response_lines)
        return {
            "response": response_text,
            "sources": context_data.get("sources", []),
            "context_used": context_data.get("context_text", ""),
            "contact_info": contact_info,
            "demo_links": demo_links,
            "context_quality": context_data.get("context_quality", {}),
            "intent": intent_result.to_dict() if intent_result else None,
        }

    async def enhance_query(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        intent_result: Optional[IntentResult] = None,
    ) -> List[str]:
        """
        Generate enhanced query variants for better document retrieval.

        Uses light-weight heuristics keyed off detected intent and message keywords.
        """
        try:
            if not message:
                return [message]

            normalized_message = message.strip()
            if not normalized_message:
                return [message]

            primary_intent = intent_result.primary_intent if intent_result else None

            # Greetings or chit-chat don't benefit from query expansion
            if primary_intent == IntentType.GREETING:
                return [normalized_message]

            max_variants = 3
            if self.context_config and hasattr(
                self.context_config, "max_query_variants"
            ):
                max_variants = self.context_config.max_query_variants or max_variants
            elif isinstance(self.context_config, dict):
                max_variants = self.context_config.get(
                    "max_query_variants", max_variants
                )

            variants: List[str] = []
            seen: set[str] = set()

            def add_variant(text: Optional[str]):
                if not text:
                    return
                cleaned = text.strip()
                if not cleaned:
                    return
                lowered = cleaned.lower()
                if lowered not in seen:
                    seen.add(lowered)
                    variants.append(cleaned)

            add_variant(normalized_message)

            # Add intent-specific variants
            def extend_with(items: List[str]):
                for item in items:
                    add_variant(item)

            if primary_intent in {IntentType.CONTACT, IntentType.BOOKING} or (
                not primary_intent
                and self._is_contact_information_query(normalized_message)
            ):
                extend_with(self._get_contact_query_variants(normalized_message))

            if primary_intent in {
                IntentType.PRODUCT,
                IntentType.PRICING,
                IntentType.COMPARISON,
                IntentType.RECOMMENDATION,
            }:
                extend_with(self._get_product_query_variants(normalized_message))
            else:
                product_variants = self._get_product_query_variants(normalized_message)
                extend_with(product_variants)

            # Incorporate recent conversation context for follow-up questions
            if conversation_history:
                history_summary = self._summarize_recent_context(conversation_history)
                if history_summary:
                    add_variant(f"{history_summary} {normalized_message}")

            # Ensure we always return at least the original query
            if not variants:
                variants = [normalized_message]

            return variants[: max(1, max_variants)]

        except Exception as e:
            logger.warning("Query enhancement failed: %s", e)
            return [message]

    async def generate_enhanced_response(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        retrieved_documents: List[Dict[str, Any]],
        intent_result: Optional[IntentResult] = None,
        intent_response_config: Optional[Dict[str, Any]] = None,
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
            context_result = self.context_builder.build(
                documents=retrieved_documents,
                max_context_length=self.max_context_length,
                context_config=self.context_config,
            )
            context_data = asdict(context_result)

            # DEBUG: Log context information
            context_text = context_data.get("context_text", "")
            logger.info("📝 Context length: %d characters", len(context_text))

            # Detect if this is a contact information query - if so, use ZERO temperature
            intent_primary = intent_result.primary_intent if intent_result else None
            is_contact_query = self._is_contact_information_query(message)
            if intent_primary in {IntentType.CONTACT, IntentType.BOOKING}:
                is_contact_query = True
            intent_prompt_instruction = ""
            override_temperature = None
            override_max_tokens = None
            if intent_response_config:
                intent_prompt_instruction = intent_response_config.get(
                    "intent_specific_prompt", ""
                )
                override_temperature = intent_response_config.get("temperature")
                override_max_tokens = intent_response_config.get("max_tokens")

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

            if intent_primary == IntentType.PRICING:
                has_pricing_context = self._has_pricing_context(
                    retrieved_documents, context_text
                )
                logger.info(
                    "Pricing intent detected (has_pricing_context=%s) - returning consultation fallback",
                    has_pricing_context,
                )
                fallback_response = self._build_pricing_fallback_response(
                    context_data, intent_result
                )
                fallback_response.update(
                    {
                        "tokens_used": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "document_count": len(retrieved_documents),
                        "retrieval_method": "enhanced_rag",
                        "model_used": self.chatbot_config.model,
                        "generation_metadata": {
                            "temperature": intent_response_config.get("temperature")
                            if intent_response_config
                            else self.chatbot_config.temperature,
                            "max_tokens": intent_response_config.get("max_tokens")
                            if intent_response_config
                            else self.chatbot_config.max_tokens,
                            "context_length": len(context_text),
                            "message_count": 1,
                        },
                    }
                )
                return fallback_response

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
            system_prompt = self._create_system_prompt(
                context_data, intent_prompt_instruction
            )

            # Build conversation messages
            messages = self._build_conversation_messages(
                system_prompt, conversation_history, message
            )

            # Generate response using OpenAI with dynamic temperature
            openai_response = await self._call_openai(
                messages,
                force_factual=is_contact_query,
                override_temperature=override_temperature,
                override_max_tokens=override_max_tokens,
            )

            # Process and format the response
            formatted_response = self.response_post_processor.format_response(
                openai_response["content"],
                retrieved_documents,
                context_data,
                is_contact_query,
                message,
            )
            formatted_response["intent"] = (
                intent_result.to_dict() if intent_result else None
            )
            if intent_response_config:
                formatted_response["intent_response_config"] = intent_response_config

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
        """Backward-compatible wrapper around ContextBuilder."""
        context_result = self.context_builder.build(
            documents=documents,
            max_context_length=self.max_context_length,
            context_config=self.context_config,
        )
        return asdict(context_result)

    def _create_system_prompt(
        self, context_data: Dict[str, Any], intent_instruction: str = ""
    ) -> str:
        """Create system prompt with context"""
        # CRITICAL RULES - These must ALWAYS come first, before any other instructions
        # IMPORTANT: This is a MULTITENANT SaaS platform - responses must be GENERAL and work for ANY business
        critical_rules = """🚨 CRITICAL RESPONSE RULES - HIGHEST PRIORITY - MUST FOLLOW 🚨

IMPORTANT CONTEXT: You are part of a MULTITENANT SaaS platform. Different businesses use this chatbot, each with:
- Different products/services
- Different demo processes
- Different office locations
- Different business models
- Different contact methods

Your responses must be GENERAL and work for ANY business type. Do NOT make specific assumptions about what "we" offer or how "we" operate.

ABSOLUTELY FORBIDDEN PHRASES - NEVER USE THESE (EVEN IF OTHER INSTRUCTIONS SAY TO):
- "I don't have that information available"
- "I don't have information about..."
- "I don't have that information"
- "I don't know"
- "That information is not available"
- "I can't help with that"
- "I'm not able to provide that information"

IF YOU START TO WRITE ANY OF THESE PHRASES, STOP IMMEDIATELY AND REWRITE YOUR RESPONSE.

MANDATORY RESPONSE PATTERN (MULTITENANT - MUST BE GENERAL):
When you cannot provide the exact information requested, you MUST:
1. Start with a positive, constructive statement about what IS possible
2. Acknowledge the request naturally (without using forbidden phrases)
3. Offer general, helpful alternatives that work for any business type
4. Include actionable next steps with links when available in context
5. Make the user feel helped and supported
6. NEVER assume specific business models, offerings, or processes
7. Use general language like "the team", "our services", "contact us" rather than specific claims

THESE RULES OVERRIDE ALL OTHER INSTRUCTIONS. FOLLOW THEM ABSOLUTELY.

🚨 CRITICAL REMINDER: Even if the base_prompt or other instructions below tell you to "acknowledge honestly"
or use phrases like "I don't have that information", YOU MUST IGNORE THOSE INSTRUCTIONS and follow these
CRITICAL RULES instead. Always be constructive and helpful, never robotic.

"""

        # TYPE SAFE: Using Pydantic model attribute access
        # Get base prompt from chatbot config, but sanitize it to remove conflicting instructions
        raw_base_prompt = (
            self.chatbot_config.system_prompt
            or "You are a helpful AI assistant. Use the provided context to answer questions accurately."
        )

        # CONTEXT ENGINEERING: Integrate business_context and specialized_instructions from context_config
        # These come from the frontend UI and should be added to the system prompt
        business_context_text = ""
        specialized_instructions_text = ""

        intent_instruction_text = (
            f"\n\nINTENT GUIDANCE:\n{intent_instruction}" if intent_instruction else ""
        )

        if self.context_config:
            if (
                hasattr(self.context_config, "business_context")
                and self.context_config.business_context
            ):
                business_context_text = (
                    f"\n\nBUSINESS CONTEXT:\n{self.context_config.business_context}"
                )
            elif isinstance(self.context_config, dict) and self.context_config.get(
                "business_context"
            ):
                business_context_text = (
                    f"\n\nBUSINESS CONTEXT:\n{self.context_config['business_context']}"
                )

            if (
                hasattr(self.context_config, "specialized_instructions")
                and self.context_config.specialized_instructions
            ):
                specialized_instructions_text = f"\n\nSPECIALIZED INSTRUCTIONS:\n{self.context_config.specialized_instructions}"
            elif isinstance(self.context_config, dict) and self.context_config.get(
                "specialized_instructions"
            ):
                specialized_instructions_text = f"\n\nSPECIALIZED INSTRUCTIONS:\n{self.context_config['specialized_instructions']}"

        # Combine base prompt with context engineering settings
        enhanced_base_prompt = (
            raw_base_prompt
            + business_context_text
            + specialized_instructions_text
            + intent_instruction_text
        )

        # CRITICAL: Sanitize base_prompt to remove any instructions that conflict with our rules
        # Remove phrases that might lead to forbidden responses
        base_prompt = self._sanitize_base_prompt(enhanced_base_prompt)

        context_text = context_data.get("context_text", "")

        if context_text:
            context_section = f"""
CONTEXT INFORMATION:
{context_text}

🚨 CRITICAL RESPONSE RULES - MUST FOLLOW 🚨

ABSOLUTELY FORBIDDEN PHRASES - NEVER USE THESE:
- "I don't have that information available"
- "I don't have information about..."
- "I don't have that information"
- "I don't know"
- "That information is not available"
- "I can't help with that"
- "I'm not able to provide that information"

IF YOU START TO WRITE ANY OF THESE PHRASES, STOP IMMEDIATELY AND REWRITE YOUR RESPONSE.

YOUR ROLE:
You are {self.chatbot_config.name}, a {self.chatbot_config.tone} AI assistant. Your goal is to be helpful, constructive, and conversational while providing accurate information from the context above.

MANDATORY RESPONSE PATTERN FOR "NO" OR MISSING INFORMATION (MULTITENANT - MUST BE GENERAL):

When you cannot provide the exact information requested, you MUST:
1. Start with a positive, constructive statement about what IS possible
2. Acknowledge the request naturally (without using forbidden phrases)
3. Offer GENERAL alternatives that work for any business type (don't assume specific processes)
4. Include actionable next steps with links when available in context
5. Make the user feel helped and supported]

6. NEVER make specific claims about what the business offers (demos, trials, locations, etc.)
7. Use general language: "our team", "contact us", "get more information" rather than specific processes

EXAMPLE RESPONSES (MULTITENANT - GENERAL PATTERNS THAT WORK FOR ANY BUSINESS):

Query: "Do you have an office in Spain?"
❌ WRONG: "I don't have information about an office in Spain. If you have any other questions, feel free to ask!"
✅ CORRECT: "Curently {self.chatbot_config.name} is based in [location from context if available]. Based on the information available to me, I don't see details about other office locations right now.

However, I can help you get in touch with our team who can provide you with accurate information about locations and how we can assist you. They can also discuss our services and answer any questions you might have.

Would you like me to help you **[connect with our team](demo-link-from-context-if-available)** or do you have other questions I can help with?"

Query: "Is free demo available?"
❌ WRONG: "I don't have that information available. Feel free to explore more about {self.chatbot_config.name} on our website."
✅ CORRECT (GENERAL): "I'd be happy to help you learn more about {self.chatbot_config.name} and what's available!

Based on the information in my knowledge base, I don't see specific details about demo availability right now. However, I can help you get in touch with our team who can provide you with accurate information about demos, trials, consultations, or other ways to experience our services.

Would you like me to help you **[connect with our team](demo-link-from-context-if-available)** to learn more about available options?"

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

**When Answering "No" or Information Not Available (MULTITENANT - BE CONSTRUCTIVE LIKE KEPLERO AI):**
- Be direct and honest about what you don't offer or know (e.g., "At the moment, we do not offer a free demo")
- IMMEDIATELY pivot to constructive alternatives with clear benefits
- Explain WHAT the user gets from the alternative (e.g., "to see how it works, discuss your needs, get personalized demonstration")
- Provide clear, actionable next steps with links when available
- Use natural, conversational flow - structure: [Direct answer] → [However/But] → [Constructive alternative with benefits] → [Clear call-to-action]
- When location info is available in context, mention it constructively (e.g., "We are currently based in [location]")
- When demo/consultation links are available, ALWAYS include them in the response
- Make the user feel helped and supported, not blocked

**When Context Doesn't Have Specific Information:**
- Even if the exact answer isn't in context, you can still be helpful
- Mention what the context DOES contain (if anything relevant)
- Always pivot to offering GENERAL ways to get the information (connect with team, etc.)
- Use the example patterns above - they show exactly how to respond constructively
- Remember: This is multitenant - your response must work for ANY business using this platform

CONTACT INFORMATION HANDLING:
- ONLY provide contact information (phone, email, address) when the user EXPLICITLY asks for it
- If user asks about products, prices, or other topics, DO NOT include contact information unless asked
- Phone numbers, emails, addresses MUST be copied EXACTLY from context
- Format contact info clearly with emojis and markdown:
  - 📞 **Phone**: [number](tel:number)
  - 📧 **Email**: [email](mailto:email)
  - 📍 **Location**: *address*
- If contact info is NOT in context AND user asks for it, acknowledge and offer to connect them with the team

    CONTACT/BOOKING LINKS:
    - If context has consultation or contact links, include them when offering next steps
    - Use the EXACT URL from the "Demo/Booking Links" (or equivalent) section in context
    - Format as: **[Connect with our team](exact-url-from-context)** or another neutral label that works for any tenant
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

REMEMBER (CONSTRUCTIVE FALLBACKS):
- Accuracy is important, but being helpful and constructive is equally important
- When you can't give a direct answer, be direct about it, then IMMEDIATELY offer a constructive alternative
- Structure responses like Keplero AI: [Honest answer] → [However/But] → [Alternative with benefits] → [Clear call-to-action]
- Always explain WHAT the user gets from the alternative (benefits, value proposition)
- Provide clear calls to action with links when relevant - make it easy for users to take next steps
- Make every response feel like it adds value, even when the answer is "no"
- NEVER use phrases like "I don't have that information" - be direct but constructive
- If context has demo/booking links, ALWAYS include them when offering consultations or demos
- Every response should make the user feel helped and supported, never blocked or dismissed
- Turn every "no" into an opportunity to help in another way - show value in alternatives
- Extract and use available context information (locations, services) to make responses more specific and helpful
"""
            # CRITICAL: Put our rules FIRST and make them VERY prominent
            # Use explicit override language to ensure LLM follows our rules
            final_prompt = f"""{critical_rules}

🚨 OVERRIDE INSTRUCTION: The rules above ABSOLUTELY OVERRIDE any conflicting instructions below.
If the base_prompt below says something different, IGNORE IT and follow the CRITICAL RULES above instead.

BASE PROMPT (Only follow if it doesn't conflict with CRITICAL RULES above):
{base_prompt}

{context_section}
"""
            return final_prompt
        else:
            # NO CONTEXT AVAILABLE - Provide helpful fallback instructions
            no_context_section = f"""
NO KNOWLEDGE BASE CONTEXT AVAILABLE

🚨 CRITICAL RESPONSE RULES - MUST FOLLOW 🚨

ABSOLUTELY FORBIDDEN PHRASES - NEVER USE THESE:
- "I don't have that information available"
- "I don't have information about..."
- "I don't have that information"
- "I don't know"
- "That information is not available"
- "I can't help with that"
- "I'm not able to provide that information"

IF YOU START TO WRITE ANY OF THESE PHRASES, STOP IMMEDIATELY AND REWRITE YOUR RESPONSE.

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

**For Office/Location Questions (e.g., "Do you have an office in Spain?") - MULTITENANT GENERAL:**
"{self.chatbot_config.name} is currently based in [location from context if available]. We do not have an office in [requested location] at this time.

However, all interactions and consultations can be managed online, and our team can assist you regardless of your location. If you're interested in our solutions or want to collaborate, you can schedule a call with our team to discuss your needs and see how we can help. **[Connect with our team here](demo-link-if-available)**"

**For Demo Questions (e.g., "Is free demo available?") - MULTITENANT GENERAL:**
"At the moment, we do not offer a free demo or trial version of {self.chatbot_config.name}.

However, you can schedule a free consultation call with our team to see how {self.chatbot_config.name} works, discuss your specific needs, and get a personalized demonstration based on your requirements. This allows us to tailor the demo to your use case and answer any questions you might have.

If you're interested, you can **[connect with our team here](demo-link-if-available)**"

**For Contact Questions:**
"I'd be happy to help you get in touch with us! 📞

Our team can provide contact details and answer your questions. Here's how you can reach us:

• **Schedule a Call** - Book a consultation to speak directly with our team
• **Contact Support** - Our support team can provide contact details
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
            # CRITICAL: Put our rules FIRST so they override any conflicting instructions in base_prompt
            return critical_rules + base_prompt + "\n\n" + no_context_section

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
        self,
        messages: List[Dict[str, str]],
        force_factual: bool = False,
        override_temperature: Optional[float] = None,
        override_max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
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

            # HUMAN-LIKE RESPONSES: Increase max_tokens to allow for constructive, helpful responses
            # Keplero AI style responses need more tokens for proper explanations and alternatives
            # Minimum 500 tokens for constructive responses, but use config value if higher
            config_max_tokens = self.chatbot_config.max_tokens
            max_tokens = max(
                config_max_tokens, 500
            )  # Ensure minimum 500 tokens for quality responses
            if config_max_tokens < 500:
                logger.info(
                    f"⚠️ max_tokens increased from {config_max_tokens} to {max_tokens} for human-like responses"
                )
            if override_max_tokens:
                max_tokens = override_max_tokens
                logger.debug(
                    "Using override max_tokens=%d from intent config", max_tokens
                )

            # HUMAN-LIKE RESPONSES: Slightly higher temperature for more natural, conversational responses
            # Balance between accuracy (low temp) and human-like quality (higher temp)
            # 0.3-0.5 provides natural variation while maintaining factual accuracy
            if force_factual:
                temperature = 0.0  # Completely deterministic for contact info
                logger.info(
                    "Using temperature=0.0 for factual/contact information query"
                )
            else:
                # Use config temperature, but ensure it's not too low for human-like responses
                config_temp = self.chatbot_config.temperature
                # Minimum 0.3 for human-like responses, but allow higher if configured
                temperature = (
                    max(config_temp, 0.3) if config_temp < 0.3 else config_temp
                )
                if config_temp < 0.3:
                    logger.info(
                        f"⚠️ temperature increased from {config_temp} to {temperature} for more human-like responses"
                    )
                logger.debug("Using temperature=%.1f for general query", temperature)
                if override_temperature is not None:
                    temperature = override_temperature
                    logger.debug(
                        "Using override temperature=%.1f from intent config",
                        temperature,
                    )

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
        user_message: str = "",
    ) -> Dict[str, Any]:
        """Wrapper for ResponsePostProcessor (kept for backwards compatibility)."""
        return self.response_post_processor.format_response(
            response_text,
            retrieved_documents,
            context_data,
            is_contact_query,
            user_message,
        )

    def _sanitize_base_prompt(self, base_prompt: str) -> str:
        """
        Sanitize base prompt to remove conflicting instructions that might lead to forbidden phrases.
        This ensures the chatbot's custom system_prompt doesn't override our improved response patterns.
        """
        if not base_prompt:
            return ""

        # Remove or replace conflicting instructions
        conflicting_patterns = [
            (r"acknowledge this honestly", "offer constructive alternatives"),
            (
                r"if you don't have.*information.*acknowledge",
                "offer constructive alternatives and next steps",
            ),
            (r"don't have.*information.*say so", "offer constructive alternatives"),
            (r"be honest.*don't.*know", "be direct but offer helpful alternatives"),
        ]

        sanitized = base_prompt
        for pattern, replacement in conflicting_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # Remove any mentions of forbidden phrases in instructions
        forbidden_in_instructions = [
            r"i don't have that information",
            r"i don't have information about",
            r"that information is not available",
        ]
        for pattern in forbidden_in_instructions:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    @staticmethod
    def _normalize_forbidden_phrase_text(text: str) -> str:
        """Delegate to ResponsePostProcessor for normalization."""
        return ResponsePostProcessor.normalize_forbidden_phrase_text(text)

    def _remove_forbidden_phrases(
        self, response: str, context_data: Dict[str, Any], user_message: str
    ) -> str:
        """Delegate to ResponsePostProcessor sanitizer."""
        return self.response_post_processor.remove_forbidden_phrases(
            response, context_data, user_message
        )

    def _ensure_markdown_formatting(self, response: str) -> str:
        """Delegate to ResponsePostProcessor markdown formatter."""
        return self.response_post_processor.ensure_markdown_formatting(response)

    def _validate_contact_info(
        self, response: str, context: str, is_contact_query: bool = False
    ) -> str:
        """Delegate to ResponsePostProcessor contact validator."""
        return self.response_post_processor.validate_contact_info(
            response, context, is_contact_query
        )

    def _is_contact_information_query(self, query: str) -> bool:
        """Detect if the query is asking for contact, booking, or demo information"""
        query_lower = query.lower()

        # Direct contact keywords (phones, emails, addresses, etc.)
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
            "talk to someone",
            "speak to someone",
        ]

        if any(keyword in query_lower for keyword in contact_keywords):
            return True

        # Demo / consultation booking intent
        booking_patterns = [
            r"book(?:ing)? (?:a )?(?:demo|consultation|call|meeting|appointment)",
            r"schedule(?: a)? (?:demo|consultation|call|meeting)",
            r"(?:request|arrange|set up|organize) (?:a )?(?:demo|consultation|call)",
            r"(?:demo|consultation) (?:request|booking|schedule)",
            r"(?:talk|speak|connect) (?:with|to) (?:sales|support|an expert|the team)",
            r"how (?:can|do) i (?:book|schedule|arrange) (?:a )?demo",
            r"(?:book|schedule) (?:a )?time with (?:the )?team",
        ]

        for pattern in booking_patterns:
            if re.search(pattern, query_lower):
                return True

        # Combined keyword detection for short queries like "Book demo"
        demo_terms = ["demo", "trial", "consultation", "meeting", "appointment"]
        action_terms = ["book", "schedule", "arrange", "request", "set up", "organize"]
        if any(term in query_lower for term in demo_terms) and any(
            action in query_lower for action in action_terms
        ):
            return True

        return False

    def _determine_max_context_length(self, context_config: Any) -> int:
        """Calculate max context length from config with safe fallbacks."""
        if context_config and hasattr(context_config, "max_context_length"):
            value = getattr(context_config, "max_context_length") or 0
            if isinstance(value, int) and value > 0:
                return value
        if isinstance(context_config, dict):
            value = context_config.get("max_context_length", 0)
            if isinstance(value, int) and value > 0:
                return value
        return 4000

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
                cached_response_text = cached_response.get("response", "")
                context_data_for_cache = {
                    "demo_links": cached_response.get("demo_links", []),
                    "contact_info": cached_response.get("contact_info", {}),
                }

                cleaned_response = (
                    self.response_post_processor.sanitize_cached_response(
                        cached_response_text, context_data_for_cache, message
                    )
                )

                if cleaned_response != cached_response_text:
                    cached_response["response"] = cleaned_response
                    cached_response["cached_at"] = asyncio.get_event_loop().time()
                    try:
                        await cache_service.set(
                            cache_key, cached_response, self.CACHE_TTL_SECONDS
                        )
                        logger.info(
                            "♻️  Refreshed cached response after cleaning forbidden phrases"
                        )
                    except Exception as cache_update_error:
                        logger.warning(
                            "Failed to refresh sanitized cache entry: %s",
                            cache_update_error,
                        )
                else:
                    cached_response["response"] = cleaned_response

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
            await cache_service.set(cache_key, cache_data, self.CACHE_TTL_SECONDS)

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
