"""
Response Post Processor
-----------------------
Validates and formats LLM responses before returning them to the caller.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResponsePostProcessor:
    """Central place for response validation, sanitation, and formatting."""

    def __init__(self, leakage_detector, chatbot_config):
        self.leakage_detector = leakage_detector
        self.chatbot_config = chatbot_config

    def format_response(
        self,
        response_text: str,
        retrieved_documents: List[Dict[str, Any]],
        context_data: Dict[str, Any],
        is_contact_query: bool,
        user_message: str,
    ) -> Dict[str, Any]:
        validated_response = self._validate_contact_info(
            response_text, context_data.get("context_text", ""), is_contact_query
        )

        enriched_response = self._ensure_contact_link_presence(
            validated_response, context_data
        )

        cleaned_response = self._sanitize_response(
            enriched_response, context_data, user_message
        )

        return {
            "response": cleaned_response,
            "sources": context_data.get("sources", []),
            "context_used": context_data.get("context_text", ""),
            "contact_info": context_data.get("contact_info", {}),
            "demo_links": context_data.get("demo_links", []),
            "context_quality": context_data.get("context_quality", {}),
            "document_count": len(retrieved_documents),
            "retrieval_method": "enhanced_rag",
            "model_used": self.chatbot_config.model,
            "generation_metadata": {
                "temperature": self.chatbot_config.temperature,
                "max_tokens": self.chatbot_config.max_tokens,
                "context_length": len(context_data.get("context_text", "")),
                "message_count": 1,
            },
        }

    def sanitize_cached_response(
        self, response_text: str, context_data: Dict[str, Any], user_message: str
    ) -> str:
        """Expose forbidden-phrase removal for cached responses."""
        return self._remove_forbidden_phrases(response_text, context_data, user_message)

    def validate_contact_info(
        self, response: str, context: str, is_contact_query: bool = False
    ) -> str:
        """Public wrapper for contact validation."""
        return self._validate_contact_info(response, context, is_contact_query)

    def ensure_markdown_formatting(self, response: str) -> str:
        """Public wrapper for markdown formatting."""
        return self._ensure_markdown_formatting(response)

    def remove_forbidden_phrases(
        self, response: str, context_data: Dict[str, Any], user_message: str
    ) -> str:
        """Public wrapper for forbidden phrase sanitization."""
        return self._remove_forbidden_phrases(response, context_data, user_message)

    @staticmethod
    def normalize_forbidden_phrase_text(text: str) -> str:
        """Public wrapper for normalization helper."""
        return ResponsePostProcessor._normalize_forbidden_phrase_text(text)

    def _ensure_contact_link_presence(
        self, response_text: str, context_data: Dict[str, Any]
    ) -> str:
        demo_links = context_data.get("demo_links", [])
        contact_info = context_data.get("contact_info", {}) or {}
        if contact_info.get("demo_links"):
            demo_links = contact_info["demo_links"]

        if not demo_links:
            return response_text

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

        response_lower = response_text.lower()
        mentions_consultation = any(
            keyword in response_lower for keyword in consultation_keywords
        )
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

        if not (mentions_consultation or offers_alternatives):
            return response_text

        has_demo_link = any(link in response_text for link in demo_links)
        if has_demo_link:
            return response_text

        logger.info("Adding demo/contact link to response: %s", demo_links[0])
        demo_link_text = f"**[Connect with our team]({demo_links[0]})**"
        response_lower = response_text.lower()
        demo_pattern = r"(consultation|demo|call|talk|discuss|schedule|book|booking)[^\n\.!?]*(?:\.|!|\?|$)"
        match = re.search(demo_pattern, response_lower, re.IGNORECASE)

        if match:
            pos = match.end()
            if pos < len(response_text):
                next_char = response_text[pos : pos + 1]
                if next_char.strip() and next_char not in [".", "!", "?"]:
                    return (
                        response_text[:pos]
                        + f". You can {demo_link_text}."
                        + response_text[pos:]
                    )
                return (
                    response_text[:pos]
                    + f" You can {demo_link_text}."
                    + response_text[pos:].lstrip()
                )
            return f"{response_text}\n\nYou can {demo_link_text}."

        if response_text.strip().endswith((".", "!", "?")):
            return f"{response_text} You can {demo_link_text}."

        return f"{response_text}\n\nYou can {demo_link_text}."

    def _sanitize_response(
        self, response_text: str, context_data: Dict[str, Any], user_message: str
    ) -> str:
        sanitized_response = self.leakage_detector.sanitize_response_for_leakage(
            response_text,
            context_data.get("context_text", ""),
            threshold=0.8,
        )

        formatted_response = self._ensure_markdown_formatting(sanitized_response)

        final_response = self._remove_forbidden_phrases(
            formatted_response, context_data, user_message
        )

        if final_response != formatted_response:
            logger.error(
                "🚨 FINAL CHECK: Forbidden phrase removed during post-processing. "
                "Original: %s... Rewritten to: %s...",
                formatted_response[:200],
                final_response[:200],
            )
        return final_response

    def _validate_contact_info(
        self, response: str, context: str, is_contact_query: bool = False
    ) -> str:
        if not is_contact_query:
            contact_keywords = ["phone", "contact", "call", "email", "reach", "number"]
            has_contact_keywords = any(
                keyword in response.lower() for keyword in contact_keywords
            )
            if not has_contact_keywords:
                logger.debug(
                    "Skipping contact validation - not a contact query and no contact keywords"
                )
                return response

            logger.debug(
                "Skipping expensive contact validation - not a contact query (keywords found but query wasn't about contact)"
            )
            return response

        phone_pattern = r"(?:(?:\+|00)[1-9]\d{0,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        response_phones = re.findall(phone_pattern, response)
        context_phones = re.findall(phone_pattern, context)

        def is_likely_phone(text: str) -> bool:
            cleaned = re.sub(r"[^\d+]", "", text)
            if len(cleaned) < 8 or len(cleaned) > 15:
                return False
            if re.search(r"[₹$€£Dhs]|price|cost", text, re.IGNORECASE):
                return False
            if re.search(r"dhs\.?\s*\d+|price.*\d+|\d+.*price", text, re.IGNORECASE):
                return False
            return True

        response_phones = [p for p in response_phones if is_likely_phone(p)]
        context_phones = [p for p in context_phones if is_likely_phone(p)]

        def normalize_phone(phone: str) -> str:
            return re.sub(r"[^\d+]", "", phone)

        normalized_context_phones = {normalize_phone(p) for p in context_phones}

        for response_phone in response_phones:
            normalized_response_phone = normalize_phone(response_phone)
            if normalized_response_phone not in normalized_context_phones:
                logger.warning(
                    "🚨 HALLUCINATION DETECTED: Phone number '%s' not in context. Context has: %s",
                    response_phone,
                    context_phones,
                )

                if context_phones:
                    response = response.replace(response_phone, context_phones[0])
                    logger.info("✅ Auto-corrected phone to: %s", context_phones[0])
                else:
                    response = re.sub(
                        re.escape(response_phone),
                        "",
                        response,
                        flags=re.IGNORECASE,
                    )

        email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
        response_emails = re.findall(email_pattern, response)
        context_emails = re.findall(email_pattern, context)

        normalized_context_emails = {email.lower() for email in context_emails}
        for email in response_emails:
            if email.lower() not in normalized_context_emails:
                logger.warning(
                    "🚨 HALLUCINATION DETECTED: Email '%s' not in context. Context has: %s",
                    email,
                    context_emails,
                )
                if context_emails:
                    response = response.replace(email, context_emails[0])
                else:
                    response = response.replace(email, "")

        return response

    def _ensure_markdown_formatting(self, response: str) -> str:
        phone_pattern = r"(?:📞\s*)?(?:Phone|phone|PHONE):\s*(\+?[\d\s\-\(\)]+?)(?=\s*(?:📧|Email|email|EMAIL|📍|Location|location|LOCATION|$|\n))"

        def format_phone(match):
            number = match.group(1).strip()
            clean_number = re.sub(r"[\s\-\(\)]", "", number)
            return f"\n📞 **Phone**: [{number}](tel:{clean_number})"

        response = re.sub(phone_pattern, format_phone, response)

        email_pattern = r"(?:📧\s*)?(?:Email|email|EMAIL):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?=\s*(?:📍|Location|location|LOCATION|$|\n))"

        def format_email(match):
            email = match.group(1).strip()
            return f"\n📧 **Email**: [{email}](mailto:{email})"

        response = re.sub(email_pattern, format_email, response)

        location_pattern = r"(?:📍\s*)?(?:Location|location|LOCATION):\s*([^\.!?\n]+?)(?=(?:Feel|feel|Thank|thank|$|\n|\.))"

        def format_location(match):
            location = match.group(1).strip()
            return f"\n📍 **Location**: *{location}*\n"

        response = re.sub(location_pattern, format_location, response)

        response = re.sub(
            r"((?:contact details?|reach (?:me|us)|get in touch):\s*)(\n(?:📞|📧|📍)\s*\*\*(?:Phone|Email|Location))",
            r"\1\n\2",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(r"\n{3,}", "\n\n", response)
        response = re.sub(r"([📞📧📍])\s*\1+", r"\1", response)
        return response.strip()

    def _remove_forbidden_phrases(
        self, response: str, context_data: Dict[str, Any], user_message: str
    ) -> str:
        if not response or not isinstance(response, str):
            logger.warning("⚠️ Empty or invalid response passed to sanitizer")
            return response or ""

        response_lower = response.lower()
        normalized_response = self._normalize_forbidden_phrase_text(response_lower)

        forbidden_strings = [
            "i don't have that information available",
            "i don't have information about",
            "i don't have that information",
            "i don't know",
            "that information is not available",
            "i can't help with that",
            "i'm not able to provide that information",
            "don't have information about",
            "don't have that information",
            "i do not have information about",
            "i do not have that information",
            "i cannot help with that",
            "i am not able to provide that information",
        ]

        forbidden_patterns = [
            r"i\s+don['’]?t\s+have\s+that\s+information\s+available",
            r"i\s+don['’]?t\s+have\s+information\s+about",
            r"i\s+don['’]?t\s+have\s+.*information\s+about",
            r"i\s+don['’]?t\s+have\s+that\s+information",
            r"i\s+don['’]?t\s+have\s+.*information.*available",
            r"i\s+don['’]?t\s+know",
            r"that\s+information\s+is\s+not\s+available",
            r"i\s+can['’]?t\s+help\s+with\s+that",
            r"i['’]?m\s+not\s+able\s+to\s+provide\s+that\s+information",
            r"don['’]?t\s+have\s+.*information",
            r"don['’]?t\s+have\s+information\s+about",
            r"i\s+do\s+not\s+have\s+.*information",
            r"i\s+am\s+not\s+able\s+to\s+provide\s+that\s+information",
            r"i\s+cannot\s+help\s+with\s+that",
        ]

        has_forbidden_phrase = False
        matched_pattern = None
        detection_method = None

        for forbidden_str in forbidden_strings:
            if forbidden_str in normalized_response:
                has_forbidden_phrase = True
                matched_pattern = forbidden_str
                detection_method = "string_match"
                break

        if not has_forbidden_phrase:
            for pattern in forbidden_patterns:
                if re.search(pattern, normalized_response, re.IGNORECASE):
                    has_forbidden_phrase = True
                    matched_pattern = pattern
                    detection_method = "regex"
                    break

        if not has_forbidden_phrase:
            return response

        logger.error(
            "🚨 Forbidden phrase detected (%s) via %s. Rewriting response.",
            matched_pattern,
            detection_method,
        )

        user_message_lower = (user_message or "").lower()
        demo_links = context_data.get("demo_links", [])

        if any(
            keyword in user_message_lower
            for keyword in ["location", "office", "branch", "city", "country"]
        ):
            addresses = (context_data.get("contact_info") or {}).get("addresses", [])
            base_location = addresses[0] if addresses else None
            if base_location:
                response = (
                    f"Our main operations are currently based in {base_location}. "
                )
            else:
                response = "Right now I only have confirmed details about our current locations, and nothing about additional offices yet. "
            response += (
                "However, all interactions and consultations can be managed online, and our team can assist you virtually regardless "
                "of your location.\n\n"
            )
            if demo_links:
                response += (
                    "If you're interested in our solutions or want to collaborate, you can schedule time with our team to "
                    f"discuss your needs and see how we can help. **[Connect with our team here]({demo_links[0]})**"
                )
            else:
                response += (
                    "If you're interested in our solutions or want to collaborate, I can help you connect with our team who "
                    "can provide accurate information about locations and discuss how we can assist you. Would you like me to help "
                    "you get in touch with them?"
                )
            return response

        pricing_keywords = [
            "price",
            "pricing",
            "cost",
            "plan",
            "plans",
            "tier",
            "subscription",
        ]
        if any(keyword in user_message_lower for keyword in pricing_keywords):
            pricing_lines = self._extract_pricing_highlights(
                context_data.get("context_text", "")
            )
            if pricing_lines:
                return self._build_pricing_response_from_context(
                    pricing_lines, demo_links
                )

        if any(
            keyword in user_message_lower
            for keyword in ["demo", "trial", "free demo", "test", "free trial"]
        ):
            chatbot_name = self.chatbot_config.name
            response = (
                f"The best way to experience {chatbot_name} is through a live consultation with our team. "
                "They'll tailor the walkthrough to your goals, show the exact features you care about, and share pricing options that fit. "
            )
            if demo_links:
                response += (
                    "You can pick a convenient time here to get started:\n\n"
                    f"**[Connect with our team]({demo_links[0]})**"
                )
            else:
                response += "Let me know and I can connect you with the right person to schedule that consultation."
            return response

        if demo_links:
            connection_text = f" You can **[Connect with our team here]({demo_links[0]})** to get the information you need."
        else:
            connection_text = (
                " You can connect with our team to get the information you need."
            )

        return (
            f"I'd be happy to help you with that!{connection_text}\n\n"
            "Our team can provide detailed information and answer any questions you might have. "
            "What specific information are you looking for? 😊"
        )

    @staticmethod
    def _extract_pricing_highlights(context_text: str, limit: int = 3) -> List[str]:
        if not context_text:
            return []

        highlights: List[str] = []
        seen: set[str] = set()
        price_pattern = re.compile(
            r"(price|pricing|cost|usd|aed|dhs|\$|₹|€|£|per\s+month)", re.IGNORECASE
        )

        for raw_line in context_text.splitlines():
            line = raw_line.strip().lstrip("-• ")
            if not line or len(line) < 4:
                continue
            if not re.search(r"\d", line):
                continue
            if not price_pattern.search(line):
                continue
            normalized = line.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            highlights.append(line)
            if len(highlights) >= limit:
                break

        return highlights

    def _build_pricing_response_from_context(
        self, pricing_lines: List[str], demo_links: List[str]
    ) -> str:
        response = [
            "Here are the plan details that are documented in this workspace:",
        ]
        response.extend(f"- {line}" for line in pricing_lines)

        if demo_links:
            response.append(
                f"\nNeed tailored numbers? You can **[connect with our team]({demo_links[0]})** for an in-depth walkthrough."
            )
        else:
            response.append(
                "\nNeed tailored numbers? Let me know and I can connect you with the right person for a detailed quote."
            )

        return "\n".join(response)

    @staticmethod
    def _normalize_forbidden_phrase_text(text: str) -> str:
        if not text:
            return ""

        normalized = text
        replacements = {"’": "'", "‘": "'", "“": '"', "”": '"'}
        for original, replacement in replacements.items():
            normalized = normalized.replace(original, replacement)
        return normalized
