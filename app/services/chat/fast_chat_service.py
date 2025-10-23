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

    SIMPLE_PATTERNS = [
        # Greetings
        ("hello", "hi", "hey", "greetings"),
        # About queries
        ("about you", "who are you", "what are you", "tell me about"),
        # Help
        ("help", "can you help", "assist"),
    ]

    SIMPLE_RESPONSES = {
        "greeting": "Hello! I'm your AI assistant. How can I help you today?",
        "about": "I'm an AI assistant powered by your knowledge base. I can help answer questions about your business, products, and services. What would you like to know?",
        "help": "I'm here to help! You can ask me questions about our services, products, contact information, or anything else you'd like to know.",
    }

    @classmethod
    def is_simple_query(cls, message: str) -> tuple[bool, str]:
        """Check if query is simple and can be answered without context"""
        message_lower = message.lower().strip()

        # Very short messages
        if len(message_lower) < 3:
            return True, "greeting"

        # Greetings
        if any(pattern in message_lower for pattern in cls.SIMPLE_PATTERNS[0]):
            return True, "greeting"

        # About queries
        if any(pattern in message_lower for pattern in cls.SIMPLE_PATTERNS[1]):
            return True, "about"

        # Help queries
        if any(pattern in message_lower for pattern in cls.SIMPLE_PATTERNS[2]):
            return True, "help"

        return False, ""

    @classmethod
    async def get_fast_response(
        cls, message: str, chatbot_config: dict
    ) -> Dict[str, Any]:
        """Get fast response without vector search"""
        is_simple, response_type = cls.is_simple_query(message)

        if not is_simple:
            return None

        # Get bot name from config
        bot_name = chatbot_config.get("name", "Assistant")

        # Customize response
        if response_type == "greeting":
            greeting = chatbot_config.get(
                "greeting_message", cls.SIMPLE_RESPONSES["greeting"]
            )
            response = greeting
        elif response_type == "about":
            response = f"I'm {bot_name}, " + cls.SIMPLE_RESPONSES["about"]
        else:
            response = cls.SIMPLE_RESPONSES.get(
                response_type, cls.SIMPLE_RESPONSES["help"]
            )

        logger.info(
            f"⚡ FAST MODE: Responding to {response_type} query without vector search"
        )

        return {
            "response": response,
            "sources": [],
            "conversation_id": f"fast-{int(time.time())}",
            "message_id": f"msg-{int(time.time())}",
            "processing_time_ms": 50,  # Very fast!
            "context_quality": {"fast_mode": True, "skip_retrieval": True},
            "config_used": "fast_mode",
        }
