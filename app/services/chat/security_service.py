"""
Production-Grade Security Service for Chat
Implements comprehensive security measures for public chat endpoints

WHEN TO USE THIS MODULE:
- Validating and sanitizing user messages in public chat endpoints
- Sanitizing AI responses before sending to clients
- Rate limiting and spam detection
- Session security management

WHEN NOT TO USE:
- For chatbot configuration → Use PromptSanitizer instead
- For internal text processing → Use ChatUtilities.sanitize_text() instead

See SANITIZATION_GUIDE.md for detailed usage guidelines.
"""
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ChatSecurityService:
    """
    Production-grade security service for chat operations.

    Features:
    - Input sanitization and validation
    - XSS/injection attack prevention
    - Rate limiting per session/IP
    - Content filtering
    - Suspicious pattern detection
    - Message length enforcement
    """

    # Security configuration
    MAX_MESSAGE_LENGTH = 2000
    MIN_MESSAGE_LENGTH = 1
    MAX_MESSAGES_PER_SESSION = 100  # Per day
    MAX_MESSAGES_PER_MINUTE = 10
    SUSPICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"on\w+\s*=",  # Event handlers
        r"eval\s*\(",  # eval() calls
        r"base64,",  # Base64 encoded content
        r"\bUNION\b.*\bSELECT\b",  # SQL injection
        r"--\s*$",  # SQL comments
        r"['\"];?\s*(DROP|DELETE|INSERT|UPDATE)\s+",  # SQL commands
    ]

    def __init__(self):
        self.session_message_counts: Dict[str, list] = {}
        self.suspicious_sessions: set = set()

    def validate_message(
        self, message: str, session_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive message validation.

        Returns:
            (is_valid, error_message)
        """
        # 1. Length validation
        if not message or len(message.strip()) < self.MIN_MESSAGE_LENGTH:
            return False, "Message is too short"

        if len(message) > self.MAX_MESSAGE_LENGTH:
            return (
                False,
                f"Message exceeds maximum length of {self.MAX_MESSAGE_LENGTH} characters",
            )

        # 2. Check for null bytes (potential injection attack)
        if "\x00" in message:
            logger.warning(
                "Null byte detected in message", extra={"session_id": session_id}
            )
            return False, "Invalid characters detected"

        # 3. Detect suspicious patterns (XSS, SQL injection, etc.)
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                logger.warning(
                    "Suspicious pattern detected",
                    extra={
                        "session_id": session_id,
                        "pattern": pattern,
                        "message_preview": message[:100],
                    },
                )
                self.suspicious_sessions.add(session_id)
                return False, "Message contains potentially harmful content"

        # 4. Check for excessive repetition (spam)
        if self._is_spam(message):
            return False, "Message appears to be spam"

        # 5. Rate limiting check
        if not self._check_rate_limit(session_id):
            logger.warning("Rate limit exceeded", extra={"session_id": session_id})
            return False, "Too many messages. Please wait a moment."

        return True, None

    def sanitize_message(self, message: str) -> str:
        """
        Sanitize user message to prevent XSS and injection attacks.
        """
        # Remove any HTML tags
        message = re.sub(r"<[^>]+>", "", message)

        # Remove any JavaScript-like content
        message = re.sub(r"javascript:", "", message, flags=re.IGNORECASE)

        # Remove event handlers
        message = re.sub(
            r"on\w+\s*=\s*[\"'][^\"']*[\"']", "", message, flags=re.IGNORECASE
        )

        # Normalize whitespace
        message = " ".join(message.split())

        # Trim to max length
        if len(message) > self.MAX_MESSAGE_LENGTH:
            message = message[: self.MAX_MESSAGE_LENGTH]

        return message.strip()

    def _is_spam(self, message: str) -> bool:
        """
        Detect spam patterns.
        """
        # Check for excessive character repetition
        if re.search(r"(.)\1{10,}", message):
            return True

        # Check for excessive word repetition
        words = message.lower().split()
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > 5:  # Same word repeated >5 times
                    return True

        # Check for all caps (excessive)
        if len(message) > 20 and message.isupper():
            return True

        return False

    def _check_rate_limit(self, session_id: str) -> bool:
        """
        Check if session has exceeded rate limits.
        """
        now = datetime.now()

        # Initialize session tracking if needed
        if session_id not in self.session_message_counts:
            self.session_message_counts[session_id] = []

        # Remove old timestamps (older than 1 minute)
        self.session_message_counts[session_id] = [
            ts
            for ts in self.session_message_counts[session_id]
            if now - ts < timedelta(minutes=1)
        ]

        # Check rate limit
        if len(self.session_message_counts[session_id]) >= self.MAX_MESSAGES_PER_MINUTE:
            return False

        # Add current timestamp
        self.session_message_counts[session_id].append(now)

        return True

    def is_session_suspicious(self, session_id: str) -> bool:
        """
        Check if session has been flagged as suspicious.
        """
        return session_id in self.suspicious_sessions

    def generate_secure_session_id(self, user_identifier: Optional[str] = None) -> str:
        """
        Generate a cryptographically secure session ID.
        """
        import secrets

        timestamp = str(datetime.now().timestamp())
        random_bytes = secrets.token_hex(16)
        user_part = (
            # SECURITY NOTE: SHA-256 for session key generation (appropriate cryptographic hash)
            hashlib.sha256(user_identifier.encode()).hexdigest()[:8]
            if user_identifier
            else ""
        )

        session_data = f"{timestamp}-{random_bytes}-{user_part}"
        # SECURITY NOTE: SHA-256 for session fingerprinting (appropriate cryptographic hash)
        return hashlib.sha256(session_data.encode()).hexdigest()

    def validate_chatbot_id(self, chatbot_id: str) -> bool:
        """
        Validate chatbot ID format (UUID).
        """
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        return bool(re.match(uuid_pattern, chatbot_id, re.IGNORECASE))

    def sanitize_response(self, response: str) -> str:
        """
        Sanitize AI response before sending to client.
        Ensures no leaked sensitive information or malicious content.
        """
        # Remove any accidentally leaked API keys or tokens
        response = re.sub(
            r"(api[_-]?key|token|secret|password)\s*[:=]\s*[a-zA-Z0-9_-]+",
            "[REDACTED]",
            response,
            flags=re.IGNORECASE,
        )

        # Remove email addresses if policy requires
        # response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', response)

        return response

    def log_security_event(
        self, event_type: str, session_id: str, details: Dict, severity: str = "INFO"
    ):
        """
        Log security-related events for monitoring and auditing.
        """
        logger.log(
            getattr(logger, severity.lower(), logger.info).__self__.level,
            f"Security Event: {event_type}",
            extra={
                "event_type": event_type,
                "session_id": session_id,
                "severity": severity,
                **details,
            },
        )

    def cleanup_old_sessions(self):
        """
        Cleanup old session data to prevent memory leaks.
        Should be called periodically.
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=24)

        # Clean up message counts
        sessions_to_remove = []
        for session_id, timestamps in self.session_message_counts.items():
            # Remove old timestamps
            self.session_message_counts[session_id] = [
                ts for ts in timestamps if now - ts < timedelta(hours=24)
            ]
            # Mark empty sessions for removal
            if not self.session_message_counts[session_id]:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.session_message_counts[session_id]

        logger.info(
            "Session cleanup completed",
            extra={"removed_sessions": len(sessions_to_remove)},
        )


# Singleton instance
_security_service = None


def get_security_service() -> ChatSecurityService:
    """Get the singleton security service instance."""
    global _security_service
    if _security_service is None:
        _security_service = ChatSecurityService()
    return _security_service
