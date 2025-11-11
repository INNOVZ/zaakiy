"""
Intent Detection Service
Detects user intent from messages using rule-based patterns and LLM classification
"""
import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.services.shared import cache_service

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Primary intent types"""

    GREETING = "greeting"
    CONTACT = "contact"
    BOOKING = "booking"
    PRODUCT = "product"
    PRICING = "pricing"
    SUPPORT = "support"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    INFORMATION = "information"
    FEEDBACK = "feedback"
    ACCOUNT = "account"
    ORDER = "order"
    SHIPPING = "shipping"
    RETURN = "return"
    PARTNERSHIP = "partnership"
    CAREER = "career"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Intent detection result"""

    primary_intent: IntentType
    confidence: float
    secondary_intents: List[IntentType]
    intent_metadata: Dict[str, Any]
    detection_method: str  # "rule_based" or "llm_based"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/analytics"""
        return {
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "secondary_intents": [intent.value for intent in self.secondary_intents],
            "intent_metadata": self.intent_metadata,
            "detection_method": self.detection_method,
        }


class IntentDetectionService:
    """Detects user intent from messages"""

    # Intent patterns for rule-based detection (fast, cached)
    INTENT_PATTERNS = {
        IntentType.GREETING: [
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)[\s!?]*$",
            r"^hi[\s!?]*$",
            r"^hello[\s!?]*$",
        ],
        IntentType.CONTACT: [
            r"\b(phone|number|call|telephone|mobile|contact|email|mail|address|location|reach|get in touch|whatsapp|contact details|contact info|talk to someone|speak to someone)\b",
            r"how (can|do) i contact",
            r"where (are you|is your office|are you located)",
        ],
        IntentType.BOOKING: [
            r"\b(book|booking|schedule|arrange|request|set up|organize)\s+(?:a\s+)?(?:demo|consultation|call|meeting|appointment)\b",
            r"(?:demo|consultation)\s+(?:request|booking|schedule)",
            r"how (can|do) i (book|schedule|arrange)\s+(?:a\s+)?demo",
            r"(?:talk|speak|connect)\s+(?:with|to)\s+(?:sales|support|an expert|the team)",
            r"book\s+(?:a\s+)?time\s+with\s+(?:the\s+)?team",
        ],
        IntentType.PRODUCT: [
            r"\b(product|products|item|items|offering|offerings|catalog|catalogue|shop|store|available|list|show me|what do you have|what you have)\b",
            r"what\s+(?:products|items|offerings)\s+(?:do you have|are available)",
            r"show\s+me\s+(?:your\s+)?(?:products|items|catalog)",
        ],
        IntentType.PRICING: [
            r"\b(price|pricing|cost|costs|how much|pricing plan|subscription|fee|fees|plan|plans|package|packages|tier|tiers)\b",
            r"how\s+much\s+(?:does|do|is|are)",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:price|pricing|cost)",
        ],
        IntentType.SUPPORT: [
            r"\b(help|support|troubleshoot|troubleshooting|problem|issue|error|bug|fix|how to|how do|faq|frequently asked)\b",
            r"how\s+(?:to|do|can)\s+i",
            r"i\s+(?:have|am having|am experiencing)\s+(?:a\s+)?(?:problem|issue|error)",
        ],
        IntentType.COMPARISON: [
            r"\b(compare|comparison|difference|differences|vs|versus|which is better|which one|better|best)\b",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:difference|differences)",
            r"which\s+(?:one|is better|is best)",
        ],
        IntentType.RECOMMENDATION: [
            r"\b(suggest|recommend|recommendation|what should i|what do you recommend|which would you|advice|advise)\b",
            r"what\s+(?:should|would)\s+i\s+(?:choose|use|buy|get)",
            r"can\s+you\s+(?:suggest|recommend)",
        ],
        IntentType.INFORMATION: [
            r"\b(what is|what are|tell me about|explain|describe|information about|tell me more|what can you tell me)\b",
            r"can\s+you\s+(?:explain|describe|tell me about)",
        ],
        IntentType.FEEDBACK: [
            r"\b(feedback|review|complaint|complaints|rating|rate|suggestions|suggest|improve|improvement)\b",
            r"i\s+(?:want|would like)\s+to\s+(?:give|provide|submit)\s+(?:feedback|a review)",
        ],
        IntentType.ACCOUNT: [
            r"\b(account|login|logout|sign in|sign out|profile|settings|preferences|my account)\b",
            r"how\s+(?:can|do)\s+i\s+(?:access|log into|sign into)\s+my\s+account",
        ],
        IntentType.ORDER: [
            r"\b(order|orders|purchase|buy|checkout|cart|shopping cart|add to cart|place an order)\b",
            r"how\s+(?:can|do)\s+i\s+(?:order|purchase|buy)",
        ],
        IntentType.SHIPPING: [
            r"\b(shipping|delivery|ship|deliver|tracking|track|when will it arrive|when will my order arrive)\b",
            r"how\s+(?:long|much)\s+(?:does|will)\s+(?:shipping|delivery)\s+(?:take|cost)",
        ],
        IntentType.RETURN: [
            r"\b(return|returns|refund|refunds|cancel|cancellation|exchange)\b",
            r"how\s+(?:can|do)\s+i\s+(?:return|refund|cancel)",
        ],
        IntentType.PARTNERSHIP: [
            r"\b(partner|partnership|collaborate|collaboration|business opportunity|become a partner)\b",
            r"how\s+(?:can|do)\s+i\s+(?:become|apply to be)\s+a\s+partner",
        ],
        IntentType.CAREER: [
            r"\b(jobs|job|career|careers|hiring|hire|employment|apply|application|openings|positions)\b",
            r"are\s+you\s+hiring",
            r"what\s+(?:jobs|positions|openings)\s+(?:are available|do you have)",
        ],
    }

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.5
    LOW_CONFIDENCE_THRESHOLD = 0.3

    # Cache TTL for intent detection (5 minutes)
    CACHE_TTL = 300

    def __init__(self, openai_client=None, org_id: str = None):
        self.openai_client = openai_client
        self.org_id = org_id or "default"
        self.cache_enabled = cache_service is not None

    async def detect_intent(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None,
        use_llm: bool = True,
    ) -> IntentResult:
        """
        Detect intent from message using rule-based and/or LLM-based methods.

        Args:
            message: User message
            conversation_history: Previous messages in conversation
            context: Additional context (org_id, channel, etc.)
            use_llm: Whether to use LLM for detection (if rule-based fails)

        Returns:
            IntentResult with primary intent, confidence, and metadata
        """
        # Normalize message
        message_lower = message.lower().strip()

        # Check cache first
        if self.cache_enabled:
            cached_result = await self._get_cached_intent(message_lower)
            if cached_result:
                logger.debug(f"Intent cache HIT for: {message[:50]}")
                return cached_result

        # Step 1: Rule-based detection (fast, cached)
        rule_result = await self._detect_intent_rule_based(
            message_lower, conversation_history
        )

        # If high confidence from rule-based, return immediately
        if rule_result.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Rule-based intent detected: {rule_result.primary_intent.value} (confidence: {rule_result.confidence:.2f})"
            )
            # Cache result
            if self.cache_enabled:
                await self._cache_intent(message_lower, rule_result)
            return rule_result

        # Step 2: LLM-based detection (if enabled and rule-based confidence is low)
        if (
            use_llm
            and self.openai_client
            and rule_result.confidence < self.MEDIUM_CONFIDENCE_THRESHOLD
        ):
            try:
                llm_result = await self._detect_intent_llm_based(
                    message, conversation_history, context
                )
                # Use LLM result if confidence is higher
                if llm_result.confidence > rule_result.confidence:
                    logger.debug(
                        f"LLM-based intent detected: {llm_result.primary_intent.value} (confidence: {llm_result.confidence:.2f})"
                    )
                    # Cache result
                    if self.cache_enabled:
                        await self._cache_intent(message_lower, llm_result)
                    return llm_result
            except Exception as e:
                logger.warning(
                    f"LLM-based intent detection failed: {e}, using rule-based result"
                )

        # Return rule-based result (even if confidence is low)
        logger.debug(
            f"Using rule-based intent: {rule_result.primary_intent.value} (confidence: {rule_result.confidence:.2f})"
        )
        # Cache result
        if self.cache_enabled:
            await self._cache_intent(message_lower, rule_result)
        return rule_result

    async def _detect_intent_rule_based(
        self, message_lower: str, conversation_history: List[Dict[str, Any]] = None
    ) -> IntentResult:
        """Rule-based intent detection using patterns"""
        intent_scores: Dict[IntentType, float] = {}
        matched_patterns: Dict[IntentType, List[str]] = {}

        # Score each intent based on pattern matches
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            matched = []

            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    # Exact match gets higher score
                    if re.match(pattern, message_lower, re.IGNORECASE):
                        score += 1.0
                        matched.append(pattern)
                    else:
                        # Partial match gets lower score
                        score += 0.5
                        matched.append(pattern)

            if score > 0:
                intent_scores[intent_type] = score
                matched_patterns[intent_type] = matched

        # Determine primary intent (highest score)
        if not intent_scores:
            # No matches - likely UNKNOWN or INFORMATION
            # Check if it's a question (information intent)
            if message_lower.endswith("?") or any(
                word in message_lower
                for word in ["what", "how", "why", "when", "where", "who"]
            ):
                return IntentResult(
                    primary_intent=IntentType.INFORMATION,
                    confidence=0.4,
                    secondary_intents=[],
                    intent_metadata={"reason": "question_detected"},
                    detection_method="rule_based",
                )
            else:
                return IntentResult(
                    primary_intent=IntentType.UNKNOWN,
                    confidence=0.2,
                    secondary_intents=[],
                    intent_metadata={"reason": "no_pattern_match"},
                    detection_method="rule_based",
                )

        # Sort by score (descending)
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        # Calculate confidence (normalize score)
        # Max score is typically 1.0-2.0 for single intent, so normalize to 0.0-1.0
        max_possible_score = 2.0  # Assuming max 2 patterns per intent
        confidence = min(primary_score / max_possible_score, 1.0)

        # Boost confidence for exact matches
        if any(
            re.match(pattern, message_lower, re.IGNORECASE)
            for pattern in self.INTENT_PATTERNS[primary_intent]
        ):
            confidence = min(confidence + 0.2, 1.0)

        # Get secondary intents (other intents with scores > 0)
        secondary_intents = [
            intent for intent, score in sorted_intents[1:] if score > 0
        ][
            :3
        ]  # Limit to top 3 secondary intents

        # Build metadata
        intent_metadata = {
            "matched_patterns": matched_patterns.get(primary_intent, []),
            "intent_scores": {
                intent.value: score for intent, score in sorted_intents[:5]
            },
            "message_length": len(message_lower),
            "is_question": message_lower.endswith("?"),
        }

        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            intent_metadata=intent_metadata,
            detection_method="rule_based",
        )

    async def _detect_intent_llm_based(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None,
    ) -> IntentResult:
        """LLM-based intent detection using OpenAI"""
        try:
            # Build prompt for intent classification
            intent_list = [
                intent.value for intent in IntentType if intent != IntentType.UNKNOWN
            ]
            intent_list_str = ", ".join(intent_list)

            system_prompt = f"""You are an intent classification system. Classify the user's message into one of these intent types:

{intent_list_str}

Return ONLY a JSON object with this structure:
{{
    "primary_intent": "intent_type",
    "confidence": 0.0-1.0,
    "secondary_intents": ["intent1", "intent2"],
    "reasoning": "brief explanation"
}}

Rules:
- Choose the MOST SPECIFIC intent that matches the message
- Confidence should reflect how certain you are (0.8-1.0 = high, 0.5-0.8 = medium, 0.3-0.5 = low)
- Secondary intents are optional, only include if the message clearly has multiple intents
- Be conservative with confidence scores"""

            user_prompt = f"Classify this message: {message}"

            # Add conversation history if available
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 messages
                history_text = "\n".join(
                    [
                        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                        for msg in recent_history
                    ]
                )
                user_prompt = f"Previous conversation:\n{history_text}\n\nCurrent message: {message}"

            # Call OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            # Parse response
            import json

            result = json.loads(response.choices[0].message.content)

            # Convert to IntentResult
            primary_intent_str = result.get("primary_intent", IntentType.UNKNOWN.value)
            try:
                primary_intent = IntentType(primary_intent_str)
            except ValueError:
                primary_intent = IntentType.UNKNOWN

            confidence = float(result.get("confidence", 0.5))
            valid_intent_values = {intent.value for intent in IntentType}
            secondary_intents = []
            for intent in result.get("secondary_intents", []):
                if intent in valid_intent_values:
                    try:
                        secondary_intents.append(IntentType(intent))
                    except ValueError:
                        continue

            intent_metadata = {
                "reasoning": result.get("reasoning", ""),
                "llm_model": "gpt-3.5-turbo",
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }

            return IntentResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary_intents,
                intent_metadata=intent_metadata,
                detection_method="llm_based",
            )

        except Exception as e:
            logger.error(f"LLM-based intent detection failed: {e}")
            # Fallback to rule-based
            return await self._detect_intent_rule_based(message.lower())

    def _build_cache_key(self, message_lower: str) -> str:
        """Generate deterministic cache key for message."""
        digest = hashlib.sha256(message_lower.encode("utf-8")).hexdigest()
        return f"intent:{self.org_id}:{digest}"

    async def _get_cached_intent(self, message_lower: str) -> Optional[IntentResult]:
        """Get cached intent detection result"""
        if not self.cache_enabled:
            return None

        try:
            cache_key = self._build_cache_key(message_lower)
            cached_data = await cache_service.get(cache_key)
            if cached_data:
                # Reconstruct IntentResult from cached data with error handling
                try:
                    primary_intent_str = cached_data.get(
                        "primary_intent", IntentType.UNKNOWN.value
                    )
                    try:
                        primary_intent = IntentType(primary_intent_str)
                    except ValueError:
                        logger.warning(
                            f"Invalid cached intent type: {primary_intent_str}, using UNKNOWN"
                        )
                        primary_intent = IntentType.UNKNOWN

                    confidence = float(cached_data.get("confidence", 0.5))

                    # Reconstruct secondary intents with error handling
                    secondary_intents = []
                    valid_intent_values = {intent.value for intent in IntentType}
                    for intent_str in cached_data.get("secondary_intents", []):
                        if intent_str in valid_intent_values:
                            try:
                                secondary_intents.append(IntentType(intent_str))
                            except ValueError:
                                continue

                    return IntentResult(
                        primary_intent=primary_intent,
                        confidence=confidence,
                        secondary_intents=secondary_intents,
                        intent_metadata=cached_data.get("intent_metadata", {}),
                        detection_method=cached_data.get(
                            "detection_method", "rule_based"
                        ),
                    )
                except Exception as reconstruction_error:
                    logger.warning(
                        f"Failed to reconstruct cached intent: {reconstruction_error}"
                    )
                    return None
        except Exception as e:
            logger.warning(f"Failed to get cached intent: {e}")

        return None

    async def _cache_intent(self, message_lower: str, intent_result: IntentResult):
        """Cache intent detection result"""
        if not self.cache_enabled:
            return

        try:
            cache_key = self._build_cache_key(message_lower)
            cache_data = intent_result.to_dict()
            await cache_service.set(cache_key, cache_data, self.CACHE_TTL)
        except Exception as e:
            logger.warning(f"Failed to cache intent: {e}")

    def get_intent_retrieval_config(
        self, intent_result: IntentResult
    ) -> Dict[str, Any]:
        """Get retrieval configuration based on intent"""
        intent = intent_result.primary_intent
        confidence = intent_result.confidence

        # Base k-values by intent
        k_values = {
            IntentType.CONTACT: {"initial": 8, "rerank": 6, "final": 6},
            IntentType.BOOKING: {"initial": 6, "rerank": 4, "final": 4},
            IntentType.PRODUCT: {"initial": 10, "rerank": 8, "final": 6},
            IntentType.PRICING: {"initial": 8, "rerank": 6, "final": 5},
            IntentType.SUPPORT: {"initial": 6, "rerank": 4, "final": 4},
            IntentType.COMPARISON: {"initial": 10, "rerank": 8, "final": 6},
            IntentType.RECOMMENDATION: {"initial": 8, "rerank": 6, "final": 5},
            IntentType.INFORMATION: {"initial": 6, "rerank": 4, "final": 4},
            IntentType.GREETING: {
                "initial": 0,
                "rerank": 0,
                "final": 0,
            },  # No retrieval needed
        }

        # Default k-values
        default_k = {"initial": 6, "rerank": 4, "final": 4}
        k_config = k_values.get(intent, default_k)

        # Adjust based on confidence
        if confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            # Low confidence - retrieve more documents
            k_config = {
                "initial": k_config["initial"] + 2,
                "rerank": k_config["rerank"] + 1,
                "final": k_config["final"] + 1,
            }

        # Metadata filters by intent
        metadata_filters = {}
        if intent == IntentType.PRODUCT:
            metadata_filters["has_products"] = {"$eq": True}
        elif intent == IntentType.BOOKING:
            metadata_filters["has_booking"] = {"$eq": True}

        # Do not force a `has_pricing` filter – most ingested docs don't set it yet.
        # Rely on the query itself so pricing answers still work.

        return {
            "k_values": k_config,
            "metadata_filters": metadata_filters,
            "retrieval_strategy": self._get_retrieval_strategy(intent),
            "rerank_enabled": confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD,
        }

    def _get_retrieval_strategy(self, intent: IntentType) -> str:
        """Get retrieval strategy for intent"""
        strategy_map = {
            IntentType.CONTACT: "semantic_only",  # Contact info is usually precise
            IntentType.PRODUCT: "hybrid",  # Products benefit from keyword matching
            IntentType.PRICING: "semantic_only",  # Pricing is usually in specific docs
            IntentType.SUPPORT: "keyword_boost",  # Support queries benefit from keyword matching
            IntentType.COMPARISON: "hybrid",  # Comparisons need both semantic and keyword
            IntentType.RECOMMENDATION: "hybrid",  # Recommendations need both
            IntentType.INFORMATION: "semantic_only",  # General information queries
        }
        return strategy_map.get(intent, "semantic_only")

    def get_intent_response_config(self, intent_result: IntentResult) -> Dict[str, Any]:
        """Get response generation configuration based on intent"""
        intent = intent_result.primary_intent

        # Temperature by intent (factual vs creative)
        temperature_map = {
            IntentType.CONTACT: 0.0,  # Completely factual
            IntentType.PRICING: 0.0,  # Completely factual
            IntentType.BOOKING: 0.1,  # Mostly factual
            IntentType.PRODUCT: 0.2,  # Some creativity in descriptions
            IntentType.SUPPORT: 0.1,  # Mostly factual
            IntentType.COMPARISON: 0.3,  # Some analysis
            IntentType.RECOMMENDATION: 0.4,  # More creative
            IntentType.INFORMATION: 0.2,  # Balanced
            IntentType.GREETING: 0.3,  # Friendly but consistent
        }

        # Max tokens by intent
        max_tokens_map = {
            IntentType.GREETING: 50,  # Short greetings
            IntentType.CONTACT: 200,  # Contact info is concise
            IntentType.BOOKING: 300,  # Booking instructions
            IntentType.PRODUCT: 400,  # Product descriptions
            IntentType.PRICING: 300,  # Pricing information
            IntentType.SUPPORT: 500,  # Support responses can be detailed
            IntentType.COMPARISON: 600,  # Comparisons need more tokens
            IntentType.RECOMMENDATION: 500,  # Recommendations need explanation
            IntentType.INFORMATION: 400,  # General information
        }

        temperature = temperature_map.get(intent, 0.2)
        max_tokens = max_tokens_map.get(intent, 300)

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "intent_specific_prompt": self._get_intent_prompt(intent),
        }

    def _get_intent_prompt(self, intent: IntentType) -> str:
        """Get intent-specific prompt additions"""
        prompt_additions = {
            IntentType.CONTACT: "Focus on providing accurate contact information. Include phone numbers, emails, and addresses exactly as they appear in the context.",
            IntentType.BOOKING: "Help the user book a demo or consultation. Provide clear next steps and include booking links if available.",
            IntentType.PRODUCT: "Provide detailed product information including names, descriptions, prices, and links. Format as a product catalog.",
            IntentType.PRICING: "Provide accurate pricing information. Include plans, features, and pricing details exactly as they appear in the context.",
            IntentType.SUPPORT: "Provide helpful troubleshooting steps and solutions. Be clear and concise.",
            IntentType.COMPARISON: "Compare the options clearly. Highlight differences and help the user make an informed decision.",
            IntentType.RECOMMENDATION: "Provide thoughtful recommendations based on the user's needs. Explain your reasoning.",
        }
        return prompt_additions.get(intent, "")
