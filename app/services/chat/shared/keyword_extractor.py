"""
Shared Keyword Extractor
Single implementation for keyword extraction used across all chat services
"""
import re
from typing import List

from .constants import STOP_WORDS


class KeywordExtractor:
    """
    Unified keyword extraction utility.

    This class provides a single implementation for keyword extraction
    to avoid duplication across services. All chat services should use
    this instead of implementing their own keyword extraction.
    """

    def __init__(self, stop_words: set = None):
        """
        Initialize keyword extractor.

        Args:
            stop_words: Custom stop words set. If None, uses shared STOP_WORDS.
        """
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS

    def extract_keywords(
        self, text: str, min_length: int = 3, max_keywords: int = None
    ) -> List[str]:
        """
        Extract meaningful keywords from text.

        Args:
            text: Input text to extract keywords from
            min_length: Minimum keyword length (default: 3)
            max_keywords: Maximum number of keywords to return (default: None = all)

        Returns:
            List of extracted keywords
        """
        if not text or not isinstance(text, str):
            return []

        # Extract words (alphanumeric only)
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out stop words and short words
        keywords = [
            word
            for word in words
            if len(word) >= min_length and word not in self.stop_words
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        # Limit to max_keywords if specified
        if max_keywords is not None:
            unique_keywords = unique_keywords[:max_keywords]

        return unique_keywords

    def calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword relevance score for text.

        Args:
            text: Text to score
            keywords: List of keywords to match

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not keywords or not text:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)

        # Normalize by keyword count
        return min(matches / len(keywords), 1.0) if keywords else 0.0


# Singleton instance for convenience
_keyword_extractor = None


def get_keyword_extractor() -> KeywordExtractor:
    """Get singleton keyword extractor instance"""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor()
    return _keyword_extractor
