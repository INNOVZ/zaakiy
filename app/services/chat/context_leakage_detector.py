"""
Context Leakage Detection Service
Prevents attackers from extracting RAG context through crafted queries
"""
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContextLeakageDetector:
    """Detect and prevent context extraction attempts"""

    # Suspicious query patterns that attempt to extract context
    SUSPICIOUS_QUERY_PATTERNS = [
        (r"repeat\s+(everything|all|your\s+context)", "repeat_attack"),
        (r"list\s+all\s+information", "list_attack"),
        (r"word\s+for\s+word", "verbatim_request"),
        (r"exactly\s+as\s+(written|appears|stored)", "exact_copy_request"),
        (r"dump\s+(your\s+)?(knowledge|context|database|memory)", "dump_request"),
        (r"first\s+sentence", "iterative_extraction"),
        (r"tell\s+me\s+everything", "everything_request"),
        (r"all\s+the\s+information", "all_info_request"),
        (r"raw\s+(data|context|text)", "raw_data_request"),
        (r"verbatim", "verbatim_keyword"),
        (r"copy\s+paste", "copy_paste_request"),
        (r"entire\s+(context|knowledge|database)", "entire_request"),
        (r"show\s+me\s+(everything|all)", "show_all_request"),
        (r"what\s+do\s+you\s+know\s+about\s+everything", "comprehensive_extraction"),
        (r"list\s+(every|each|all)", "enumeration_attack"),
        (r"give\s+me\s+(all|everything)", "give_all_request"),
        (r"output\s+(all|everything)", "output_all_request"),
        (r"print\s+(all|everything)", "print_all_request"),
        (r"what\s+is\s+in\s+your\s+(context|memory|knowledge)", "context_query"),
        (r"tell\s+me\s+what\s+you\s+have", "inventory_request"),
    ]

    # Patterns that indicate structured extraction attempts
    STRUCTURED_EXTRACTION_PATTERNS = [
        r"format:",
        r"template:",
        r"structure:",
        r"in\s+the\s+following\s+format",
        r"organize\s+as\s+follows",
        r"present\s+in\s+this\s+format",
        r"list\s+in\s+the\s+format",
    ]

    # Keywords that combined with short queries indicate extraction
    EXTRACTION_KEYWORDS = [
        "all",
        "everything",
        "list",
        "enumerate",
        "dump",
        "show",
        "display",
        "output",
        "print",
        "reveal",
        "tell",
    ]

    def is_context_extraction_attempt(
        self, query: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if query is attempting to extract context

        Returns:
            (is_extraction, pattern_name, matched_text)
        """
        if not query or not isinstance(query, str):
            return False, None, None

        query_lower = query.lower().strip()

        # Check for direct suspicious patterns
        for pattern, pattern_name in self.SUSPICIOUS_QUERY_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                logger.warning(
                    f"Context extraction attempt detected: {pattern_name}",
                    extra={
                        "pattern": pattern_name,
                        "matched_text": match.group(0),
                        "query_preview": query[:100],
                    },
                )
                return True, pattern_name, match.group(0)

        # Check for unusual patterns
        if self._is_unusual_extraction_pattern(query_lower):
            logger.warning(
                "Unusual extraction pattern detected",
                extra={"query_preview": query[:100]},
            )
            return True, "unusual_pattern", None

        return False, None, None

    def _is_unusual_extraction_pattern(self, query: str) -> bool:
        """Detect unusual query patterns that indicate extraction attempts"""

        # Pattern 1: Very short queries requesting "everything"
        word_count = len(query.split())
        if word_count < 10:
            extraction_keyword_count = sum(
                1 for keyword in self.EXTRACTION_KEYWORDS if keyword in query
            )
            if extraction_keyword_count >= 2:
                return True

        # Pattern 2: Queries with unusual formatting requests
        for pattern in self.STRUCTURED_EXTRACTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        # Pattern 3: Queries asking for numbered lists of everything
        if re.search(r"(list|give|show).*(1\.|2\.|3\.|\d+\.)", query, re.IGNORECASE):
            if any(keyword in query for keyword in ["all", "everything", "each"]):
                return True

        # Pattern 4: Queries with multiple extraction indicators
        extraction_indicators = [
            "complete",
            "full",
            "entire",
            "comprehensive",
            "detailed",
            "exhaustive",
            "total",
            "whole",
        ]
        indicator_count = sum(
            1 for indicator in extraction_indicators if indicator in query
        )
        if indicator_count >= 2:
            return True

        return False

    def detect_response_leakage(
        self, response: str, context_text: str, threshold: float = 0.8
    ) -> Tuple[bool, float]:
        """
        Detect if response contains too much raw context

        Args:
            response: AI-generated response
            context_text: Original context provided
            threshold: Maximum allowed overlap ratio (0.8 = 80%)

        Returns:
            (is_leaking, overlap_ratio)
        """
        if not response or not context_text:
            return False, 0.0

        # Normalize texts for comparison
        response_lower = response.lower()
        context_lower = context_text.lower()

        # Split into words
        response_words = set(response_lower.split())
        context_words = set(context_lower.split())

        # Remove common words
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
        }

        response_words = response_words - common_words
        context_words = context_words - common_words

        if not response_words:
            return False, 0.0

        # Calculate overlap
        overlap = response_words & context_words
        overlap_ratio = len(overlap) / len(response_words) if response_words else 0.0

        is_leaking = overlap_ratio > threshold

        if is_leaking:
            logger.warning(
                f"Response leakage detected: {overlap_ratio:.2%} overlap with context",
                extra={
                    "overlap_ratio": overlap_ratio,
                    "threshold": threshold,
                    "response_word_count": len(response_words),
                    "overlap_word_count": len(overlap),
                },
            )

        return is_leaking, overlap_ratio

    def sanitize_response_for_leakage(
        self, response: str, context_text: str, threshold: float = 0.8
    ) -> str:
        """
        Sanitize response to prevent context leakage

        Args:
            response: AI-generated response
            context_text: Original context provided
            threshold: Maximum allowed overlap ratio

        Returns:
            Sanitized response or replacement message
        """
        is_leaking, overlap_ratio = self.detect_response_leakage(
            response, context_text, threshold
        )

        if is_leaking:
            logger.warning(
                f"Blocking response due to context leakage: {overlap_ratio:.2%}",
                extra={"overlap_ratio": overlap_ratio},
            )
            return (
                "I found information related to your query. "
                "Could you ask a more specific question so I can provide "
                "a more focused answer?"
            )

        return response

    def check_iterative_extraction(
        self, query: str, conversation_history: List[str]
    ) -> bool:
        """
        Detect iterative extraction attempts across multiple queries

        Args:
            query: Current query
            conversation_history: Previous queries in conversation

        Returns:
            True if iterative extraction detected
        """
        if not conversation_history or len(conversation_history) < 2:
            return False

        # Pattern: Multiple queries asking for "more", "next", "continue", etc.
        continuation_patterns = [
            r"(what|tell).*(next|more|else|another)",
            r"continue",
            r"go\s+on",
            r"keep\s+going",
            r"what\s+else",
            r"anything\s+else",
            r"and\s+then",
            r"after\s+that",
        ]

        # Check if current query is a continuation
        query_lower = query.lower()
        is_continuation = any(
            re.search(pattern, query_lower, re.IGNORECASE)
            for pattern in continuation_patterns
        )

        if not is_continuation:
            return False

        # Check if previous queries were also continuations
        recent_history = conversation_history[-5:]  # Last 5 queries
        continuation_count = sum(
            1
            for prev_query in recent_history
            if any(
                re.search(pattern, prev_query.lower(), re.IGNORECASE)
                for pattern in continuation_patterns
            )
        )

        # If 3+ continuation queries in sequence, flag as iterative extraction
        if continuation_count >= 2:
            logger.warning(
                "Iterative extraction attempt detected",
                extra={
                    "continuation_count": continuation_count,
                    "recent_queries": len(recent_history),
                },
            )
            return True

        return False

    def get_safe_response_message(self, pattern_name: Optional[str] = None) -> str:
        """
        Get appropriate safe response message based on detection pattern

        Args:
            pattern_name: Name of detected pattern

        Returns:
            User-friendly response message
        """
        messages = {
            "repeat_attack": "I can answer specific questions about our services. What would you like to know?",
            "list_attack": "I'm designed to answer specific questions. What particular information can I help you with?",
            "verbatim_request": "I provide information in a conversational way. What specific question do you have?",
            "dump_request": "I can help with specific questions. What would you like to know about?",
            "iterative_extraction": "I notice you're asking for a lot of information. Could you ask about something specific?",
            "unusual_pattern": "Could you rephrase your question more specifically? I'm here to help with particular inquiries.",
        }

        return messages.get(
            pattern_name,
            "I'm here to answer specific questions. How can I help you with something in particular?",
        )


# Singleton instance
_leakage_detector = None


def get_context_leakage_detector() -> ContextLeakageDetector:
    """Get singleton context leakage detector instance"""
    global _leakage_detector
    if _leakage_detector is None:
        _leakage_detector = ContextLeakageDetector()
    return _leakage_detector
