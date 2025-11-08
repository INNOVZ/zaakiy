"""
Contact Information Validator
Validates contact info in responses against context to prevent hallucinations

Extracted from response_generation_service.py for better separation of concerns
"""
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ContactValidator:
    """Validates contact information in AI responses to prevent hallucinations"""

    def __init__(self):
        """Initialize validator with pre-compiled patterns"""

        # Pre-compile patterns for better performance
        self.phone_pattern = re.compile(
            r"(?:(?:\+|00)[1-9]\d{0,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        )

        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        self.price_pattern = re.compile(
            r"(?:₹|Rs\.?|INR|\$|USD|EUR|£|Dhs\.?|AED)\s*[\d,]+(?:\.\d{2})?",
            re.IGNORECASE,
        )

        # Hallucination phrase patterns
        self.hallucination_patterns = [
            re.compile(r"around \d+", re.IGNORECASE),
            re.compile(r"approximately \d+", re.IGNORECASE),
            re.compile(r"roughly \d+", re.IGNORECASE),
            re.compile(r"about \d+", re.IGNORECASE),
            re.compile(r"\d+-\d+ range", re.IGNORECASE),
        ]

        # Contact keywords for context detection
        self.contact_keywords = {
            "phone",
            "contact",
            "call",
            "email",
            "reach",
            "number",
            "telephone",
            "mobile",
            "whatsapp",
        }

    def validate_response(
        self, response: str, context: str, is_contact_query: bool = False
    ) -> str:
        """
        Validate and fix contact information in response

        Args:
            response: AI generated response
            context: Source context used for generation
            is_contact_query: Whether user explicitly asked for contact info

        Returns:
            Validated and corrected response
        """
        # Skip validation for non-contact queries without contact keywords
        if not is_contact_query and not self._has_contact_keywords(response):
            logger.debug("Skipping validation - not a contact query")
            return response

        # Validate in order of importance
        response = self._validate_phone_numbers(response, context, is_contact_query)
        response = self._validate_emails(response, context)
        response = self._validate_prices(response, context)
        self._detect_hallucination_phrases(response)

        return response

    def _has_contact_keywords(self, text: str) -> bool:
        """Check if text contains contact-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.contact_keywords)

    def _validate_phone_numbers(
        self, response: str, context: str, is_contact_query: bool
    ) -> str:
        """Validate phone numbers against context"""

        response_phones = self.phone_pattern.findall(response)
        context_phones = self.phone_pattern.findall(context)

        # Filter out false positives (prices, years, etc.)
        response_phones = [
            p for p in response_phones if self._is_likely_phone(p, response)
        ]
        context_phones = [
            p for p in context_phones if self._is_likely_phone(p, context)
        ]

        if not response_phones:
            return response

        # Normalize for comparison
        normalized_context = {self._normalize_phone(p) for p in context_phones}

        # Check each phone in response
        for response_phone in response_phones:
            normalized_response = self._normalize_phone(response_phone)

            if normalized_response not in normalized_context:
                logger.warning(
                    "🚨 HALLUCINATION: Phone '%s' not in context. Context has: %s",
                    response_phone,
                    context_phones[:3],
                )

                # Correct or remove based on query type
                if is_contact_query:
                    # Replace with actual phone if available
                    if context_phones:
                        response = response.replace(response_phone, context_phones[0])
                        logger.info("✅ Auto-corrected phone to: %s", context_phones[0])
                    else:
                        # Replace with placeholder if user asked for it
                        response = re.sub(
                            re.escape(response_phone),
                            "[Contact number not available]",
                            response,
                        )
                else:
                    # Remove hallucinated phone from non-contact response
                    response = response.replace(response_phone, "").strip()
                    logger.debug("Removed hallucinated phone: %s", response_phone)

        return response

    def _validate_emails(self, response: str, context: str) -> str:
        """Validate email addresses against context"""

        response_emails = self.email_pattern.findall(response)
        context_emails = set(self.email_pattern.findall(context))

        for response_email in response_emails:
            if response_email not in context_emails:
                logger.warning(
                    "🚨 HALLUCINATION: Email '%s' not in context. Context has: %s",
                    response_email,
                    list(context_emails)[:3],
                )

                # Replace with actual email if available
                if context_emails:
                    actual_email = list(context_emails)[0]
                    response = response.replace(response_email, actual_email)
                    logger.info("✅ Auto-corrected email to: %s", actual_email)

        return response

    def _validate_prices(self, response: str, context: str) -> None:
        """Validate prices against context (log only, don't modify)"""

        response_prices = self.price_pattern.findall(response)
        context_prices = set(self.price_pattern.findall(context))

        for response_price in response_prices:
            # Normalize for comparison
            normalized_response = re.sub(r"[\s,]", "", response_price.lower())
            normalized_context = {
                re.sub(r"[\s,]", "", p.lower()) for p in context_prices
            }

            if normalized_response not in normalized_context:
                logger.warning(
                    "🚨 PRICE MISMATCH: '%s' not in context. Context has: %s",
                    response_price,
                    list(context_prices)[:3],
                )

    def _detect_hallucination_phrases(self, response: str) -> None:
        """Detect vague/estimated language that indicates hallucination"""

        for pattern in self.hallucination_patterns:
            if pattern.search(response):
                logger.warning(
                    "⚠️ VAGUE LANGUAGE: Detected pattern '%s' - AI may be hallucinating",
                    pattern.pattern,
                )

    @staticmethod
    def _is_likely_phone(phone_str: str, context_text: str) -> bool:
        """
        Determine if a matched pattern is likely a phone number

        Filters out prices, years, IDs that match phone patterns
        """
        # Remove formatting to count digits
        cleaned = re.sub(r"[^\d+]", "", phone_str)

        # Length check
        if len(cleaned) < 8 or len(cleaned) > 15:
            return False

        # Check surrounding context for price indicators
        # Find position of this phone in text
        try:
            pos = context_text.index(phone_str)
            # Check 20 chars before and after
            surrounding = context_text[
                max(0, pos - 20) : min(len(context_text), pos + len(phone_str) + 20)
            ]

            # If surrounded by price indicators, it's probably a price
            if re.search(r"[₹$€£Dhs]|price|cost|aed", surrounding, re.IGNORECASE):
                return False
        except ValueError:
            # Phone not found in context, but still could be valid
            pass

        return True

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normalize phone number for comparison"""
        return re.sub(r"[^\d+]", "", phone)


# Global singleton instance
contact_validator = ContactValidator()
