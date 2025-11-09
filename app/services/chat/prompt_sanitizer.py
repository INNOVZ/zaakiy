"""
Prompt Injection Protection
Sanitizes user-controlled data before including in system prompts

WHEN TO USE THIS MODULE:
- Sanitizing chatbot configuration fields (name, tone, behavior, description, greeting, fallback)
- When user input will be included in AI system prompts
- Before storing chatbot configuration in database

WHEN NOT TO USE:
- For user chat messages → Use ChatSecurityService instead
- For general text processing → Use ChatUtilities.sanitize_text() instead
- For AI responses → Use ChatSecurityService.sanitize_response() instead

See SANITIZATION_GUIDE.md for detailed usage guidelines.
"""
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptInjectionDetector:
    """Detect and block prompt injection attempts"""

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        (
            r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?",
            "ignore_instruction",
        ),
        (r"disregard\s+(all\s+)?(previous|above|prior)", "disregard_instruction"),
        (r"new\s+instructions?:", "new_instruction"),
        (r"system\s*(message|prompt)?:", "system_override"),
        (r"---+.*---+", "delimiter_injection"),
        (r"forget\s+(everything|all|previous)", "forget_instruction"),
        (r"you\s+are\s+now", "identity_override"),
        (r"debug\s+mode", "debug_mode"),
        (r"reveal\s+(all|everything|context)", "context_extraction"),
        (r"repeat\s+(everything|all)", "repeat_attack"),
        (r"list\s+all", "list_attack"),
        (r"===\s*.*\s*===", "section_delimiter"),
        (r"\[SYSTEM\]", "system_tag"),
        (r"<system>", "system_xml_tag"),
        (r"\\n\\n", "double_newline_escape"),
        (r"override", "override_keyword"),
    ]

    def check_for_injection(
        self, text: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if text contains prompt injection attempts

        Returns:
            (is_injection, pattern_name, matched_text)
        """
        if not text:
            return False, None, None

        text_lower = text.lower()

        for pattern, pattern_name in self.INJECTION_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            if match:
                logger.warning(
                    f"Prompt injection detected: {pattern_name}",
                    extra={
                        "pattern": pattern_name,
                        "matched_text": match.group(0),
                        "input_preview": text[:100],
                    },
                )
                return True, pattern_name, match.group(0)

        return False, None, None


class PromptSanitizer:
    """Sanitize user inputs for safe inclusion in prompts"""

    # Allowed values for tone field
    ALLOWED_TONES = ["helpful", "professional", "friendly", "casual", "formal"]

    # Maximum lengths for fields
    MAX_NAME_LENGTH = 50
    MAX_TONE_LENGTH = 20
    MAX_BEHAVIOR_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 500
    MAX_GREETING_LENGTH = 200
    MAX_FALLBACK_LENGTH = 200

    def __init__(self):
        self.injection_detector = PromptInjectionDetector()

    def sanitize_chatbot_name(self, name: str) -> str:
        """Sanitize chatbot name for safe prompt inclusion"""
        if not name or not isinstance(name, str):
            return "Assistant"

        # Check for injection
        is_injection, pattern, matched = self.injection_detector.check_for_injection(
            name
        )
        if is_injection:
            logger.error(f"Blocked prompt injection in name: {pattern}")
            return "[BLOCKED]"

        # Remove newlines and control characters
        name = re.sub(r"[\n\r\t\v\f]", " ", name)

        # Remove multiple spaces
        name = re.sub(r"\s+", " ", name)

        # Remove special characters that could be used for injection
        name = re.sub(r"[<>{}[\]\\]", "", name)

        # Truncate to max length
        name = name[: self.MAX_NAME_LENGTH].strip()

        # Validate it's not empty after sanitization
        if not name or len(name) < 2:
            return "Assistant"

        return name

    def sanitize_tone(self, tone: str) -> str:
        """Sanitize tone with strict whitelist"""
        if not tone or not isinstance(tone, str):
            return "helpful"

        tone = tone.lower().strip()

        # Enforce whitelist
        if tone not in self.ALLOWED_TONES:
            logger.warning(f"Invalid tone '{tone}', using default")
            return "helpful"

        return tone

    def sanitize_behavior(self, behavior: str) -> str:
        """Sanitize behavior description"""
        if not behavior or not isinstance(behavior, str):
            return "Be helpful and informative"

        # Check for injection
        is_injection, pattern, matched = self.injection_detector.check_for_injection(
            behavior
        )
        if is_injection:
            logger.error(f"Blocked prompt injection in behavior: {pattern}")
            return "Be helpful and informative"

        # Remove newlines (replace with space)
        behavior = re.sub(r"[\n\r\t\v\f]", " ", behavior)

        # Remove multiple spaces
        behavior = re.sub(r"\s+", " ", behavior)

        # Remove delimiters and special markers
        behavior = re.sub(r"---+", "", behavior)
        behavior = re.sub(r"===+", "", behavior)

        # Remove special characters
        behavior = re.sub(r"[<>{}[\]\\]", "", behavior)

        # Truncate
        behavior = behavior[: self.MAX_BEHAVIOR_LENGTH].strip()

        if not behavior or len(behavior) < 5:
            return "Be helpful and informative"

        return behavior

    def sanitize_description(self, description: str) -> str:
        """Sanitize description field"""
        if not description or not isinstance(description, str):
            return ""

        # Check for injection
        is_injection, pattern, matched = self.injection_detector.check_for_injection(
            description
        )
        if is_injection:
            logger.error(f"Blocked prompt injection in description: {pattern}")
            return "[Content blocked]"

        # Remove newlines (replace with space)
        description = re.sub(r"[\n\r\t]", " ", description)

        # Remove multiple spaces
        description = re.sub(r"\s+", " ", description)

        # Truncate
        description = description[: self.MAX_DESCRIPTION_LENGTH].strip()

        return description

    def sanitize_greeting(self, greeting: str) -> str:
        """Sanitize greeting message"""
        if not greeting or not isinstance(greeting, str):
            return "Hello! How can I help you today?"

        # Check for injection
        is_injection, pattern, matched = self.injection_detector.check_for_injection(
            greeting
        )
        if is_injection:
            logger.error(f"Blocked prompt injection in greeting: {pattern}")
            return "Hello! How can I help you today?"

        # Remove control characters but keep newlines for formatting
        greeting = re.sub(r"[\t\v\f]", " ", greeting)

        # Limit newlines
        greeting = re.sub(r"\n{3,}", "\n\n", greeting)

        # Remove delimiters
        greeting = re.sub(r"---+", "", greeting)
        greeting = re.sub(r"===+", "", greeting)

        # Truncate
        greeting = greeting[: self.MAX_GREETING_LENGTH].strip()

        if not greeting:
            return "Hello! How can I help you today?"

        return greeting

    def sanitize_fallback(self, fallback: str) -> str:
        """Sanitize fallback message"""
        # Use a constructive fallback message that doesn't contain forbidden phrases
        safe_fallback = "I'd be happy to help you with that! Could you provide more details or rephrase your question so I can assist you better?"

        if not fallback or not isinstance(fallback, str):
            return safe_fallback

        # Check for injection
        is_injection, pattern, matched = self.injection_detector.check_for_injection(
            fallback
        )
        if is_injection:
            logger.error(f"Blocked prompt injection in fallback: {pattern}")
            return safe_fallback

        # Remove control characters but keep newlines
        fallback = re.sub(r"[\t\v\f]", " ", fallback)

        # Limit newlines
        fallback = re.sub(r"\n{3,}", "\n\n", fallback)

        # Remove delimiters
        fallback = re.sub(r"---+", "", fallback)
        fallback = re.sub(r"===+", "", fallback)

        # Truncate
        fallback = fallback[: self.MAX_FALLBACK_LENGTH].strip()

        if not fallback:
            return safe_fallback

        return fallback

    def sanitize_all_fields(self, chatbot_data: dict) -> dict:
        """Sanitize all user-controlled fields in chatbot configuration"""
        sanitized = {}

        # Sanitize each field
        if "name" in chatbot_data:
            sanitized["name"] = self.sanitize_chatbot_name(chatbot_data["name"])

        if "tone" in chatbot_data:
            sanitized["tone"] = self.sanitize_tone(chatbot_data["tone"])

        if "behavior" in chatbot_data:
            sanitized["behavior"] = self.sanitize_behavior(chatbot_data["behavior"])

        if "description" in chatbot_data:
            sanitized["description"] = self.sanitize_description(
                chatbot_data["description"]
            )

        if "greeting_message" in chatbot_data:
            sanitized["greeting_message"] = self.sanitize_greeting(
                chatbot_data["greeting_message"]
            )

        if "fallback_message" in chatbot_data:
            sanitized["fallback_message"] = self.sanitize_fallback(
                chatbot_data["fallback_message"]
            )

        # Log sanitization
        logger.info(
            "Chatbot configuration sanitized",
            extra={
                "original_name": chatbot_data.get("name", "")[:50],
                "sanitized_name": sanitized.get("name", "")[:50],
                "fields_sanitized": list(sanitized.keys()),
            },
        )

        return sanitized


# Singleton instance
_sanitizer = None


def get_prompt_sanitizer() -> PromptSanitizer:
    """Get singleton sanitizer instance"""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = PromptSanitizer()
    return _sanitizer
