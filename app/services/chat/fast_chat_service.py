"""
EMERGENCY FAST CHAT MODE
Bypasses slow vector search for immediate responses
"""
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FastChatMode:
    """Emergency fast mode that skips document retrieval for common queries"""

    # Only handle basic greetings in fast mode
    GREETING_PATTERNS = (
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    )

    GREETING_RESPONSE = "Hello! How may I help you today?"

    @classmethod
    def is_simple_query(cls, message: str) -> bool:
        """Check if query is a simple greeting that can be answered without context"""
        message_lower = message.lower().strip()

        # Very short messages (1-2 characters only)
        if len(message_lower) <= 2:
            return True

        # Only match if the ENTIRE message is a greeting (not just contains it)
        # This prevents "hello, can you tell me about products" from being treated as simple
        if (
            message_lower in cls.GREETING_PATTERNS
            or message_lower.rstrip("!?.") in cls.GREETING_PATTERNS
        ):
            return True

        return False

    @classmethod
    async def get_fast_response(
        cls, message: str, chatbot_config: dict
    ) -> Dict[str, Any]:
        """Get fast response for simple greetings without vector search"""
        is_simple = cls.is_simple_query(message)

        if not is_simple:
            return None

        # Use custom greeting from config or default
        response = chatbot_config.get("greeting_message", cls.GREETING_RESPONSE)

        logger.info("⚡ FAST MODE: Responding to greeting without vector search")

        return {
            "response": response,
            "sources": [],
            "conversation_id": f"fast-{int(time.time())}",
            "message_id": f"msg-{int(time.time())}",
            "processing_time_ms": 50,  # Very fast!
            "context_quality": {"fast_mode": True, "skip_retrieval": True},
            "config_used": "fast_mode",
        }
