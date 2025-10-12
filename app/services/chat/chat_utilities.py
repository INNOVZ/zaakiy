"""
Chat Utilities Service
Provides utility methods for message validation, formatting, and text processing
"""
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ChatUtilitiesError(Exception):
    """Exception for chat utilities errors"""


class ChatUtilities:
    """Utility methods for chat operations"""

    def __init__(self):
        # Common stop words for keyword extraction
        self.stop_words = {
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
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "shall",
        }

    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        if not query:
            return []

        # Extract words (alphanumeric only)
        words = re.findall(r"\b\w+\b", query.lower())

        # Filter out stop words and short words
        keywords = [
            word for word in words if len(word) > 2 and word not in self.stop_words
        ]

        return keywords

    def calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword relevance score"""
        if not keywords or not text:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)

        # Normalize by keyword count
        return min(matches / len(keywords), 1.0)

    def calculate_domain_relevance(
        self, text: str, domain_context: str, domain_knowledge: str
    ) -> float:
        """Calculate domain-specific relevance score"""
        if not text or not domain_context:
            return 0.0

        # Combine domain context and knowledge
        domain_terms = []
        if domain_context:
            domain_terms.extend(self.extract_keywords(domain_context))
        if domain_knowledge:
            domain_terms.extend(self.extract_keywords(domain_knowledge))

        if not domain_terms:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for term in domain_terms if term.lower() in text_lower)

        # Return boost factor (0.0 to 0.5)
        return min(matches / len(domain_terms), 0.5)

    def extract_product_links_from_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and format product links from retrieved documents"""
        product_links = []
        seen_links = set()

        for doc in documents:
            metadata = doc.get("metadata", {})

            # Check if document has product links
            if metadata.get("has_products", False):
                doc_product_links = metadata.get("product_links", [])

                for link in doc_product_links:
                    if link and link not in seen_links:
                        seen_links.add(link)

                        # Extract product name from chunk content
                        chunk = doc.get("chunk", "")
                        product_name = self.extract_product_name_from_chunk(chunk, link)

                        product_links.append(
                            {
                                "url": link,
                                "name": product_name,
                                "source": doc.get("source", ""),
                                "relevance_score": doc.get("score", 0),
                                "chunk_preview": chunk[:150] + "..."
                                if len(chunk) > 150
                                else chunk,
                            }
                        )

        # Sort by relevance score and limit to top 5
        product_links.sort(key=lambda x: x["relevance_score"], reverse=True)
        return product_links[:5]

    def extract_product_name_from_chunk(self, chunk: str, link: str) -> str:
        """Extract product name from chunk content"""
        if not chunk and not link:
            return "Product"

        # Try to find product names near the link or in the chunk
        # Look for patterns like "Product Name - Price" or "Product Name (Price)"
        product_patterns = [
            r"([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*[-–]\s*[A-Za-z]+\s*[0-9]+",
            r"([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*\([A-Za-z]+\s*[0-9]+\)",
            r"([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*for\s*[A-Za-z]+\s*[0-9]+",
            r"([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)\s*[A-Za-z]+\s*[0-9]+",
        ]

        for pattern in product_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            if matches:
                # Return the first match, cleaned up
                product_name = matches[0].strip()
                # Remove extra whitespace and clean up
                product_name = re.sub(r"\s+", " ", product_name)
                return product_name

        # If no pattern matches, try to extract from URL
        if link:
            # Extract product name from URL path
            url_parts = link.split("/")
            for part in reversed(url_parts):
                if part and part not in [
                    "product",
                    "item",
                    "catalog",
                    "shop",
                    "buy",
                    "p",
                ]:
                    # Clean up the part to make it a readable product name
                    product_name = part.replace("-", " ").replace("_", " ")
                    product_name = re.sub(r"[^a-zA-Z\s]", "", product_name)
                    product_name = re.sub(r"\s+", " ", product_name).strip()
                    if len(product_name) > 3:  # Only return if it's a meaningful name
                        return product_name.title()

        # Fallback: return a generic name
        return "Product"

    def validate_message_content(
        self, content: str, max_length: int = 4000
    ) -> Dict[str, Any]:
        """Validate message content"""
        validation_result = {"valid": True, "errors": [], "warnings": []}

        if not content:
            validation_result["valid"] = False
            validation_result["errors"].append("Message content cannot be empty")
            return validation_result

        if len(content) > max_length:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Message content exceeds maximum length of {max_length} characters"
            )

        # Check for potentially problematic content
        if len(content.strip()) < 3:
            validation_result["warnings"].append("Message content is very short")

        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)
            if repetition_ratio > 5:
                validation_result["warnings"].append(
                    "Message contains excessive repetition"
                )

        return validation_result

    def sanitize_text(self, text: str) -> str:
        """Sanitize text content for safe processing"""
        if not text:
            return ""

        # Remove control characters and normalize whitespace
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        return sanitized

    def format_context_from_documents(
        self, documents: List[Dict[str, Any]], max_context_length: int = 8000
    ) -> str:
        """Format context from retrieved documents"""
        if not documents:
            return "No relevant information found in uploaded documents."

        context_parts = []
        total_length = 0
        used_sources = set()

        for doc in documents:
            chunk = doc.get("chunk", "")
            source = doc.get("source", "Unknown")
            score = doc.get("score", 0)

            # Avoid too much content from same source unless it's highly relevant
            if source in used_sources and score < 0.8:
                continue

            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            chunk_tokens = len(chunk) // 4
            if total_length + chunk_tokens > max_context_length:
                # Try to fit a truncated version if there's meaningful space
                remaining_tokens = max_context_length - total_length
                if remaining_tokens > 100:
                    chunk = chunk[: remaining_tokens * 4]
                else:
                    break

            context_part = (
                f"Source: {source} (Relevance: {score:.3f})\nContent: {chunk}\n"
            )
            context_parts.append(context_part)
            total_length += len(chunk) // 4
            used_sources.add(source)

        return "\n---\n".join(context_parts)

    def format_conversation_history(
        self,
        history: List[Dict[str, Any]],
        max_turns: int = 3,
        max_length_per_turn: int = 200,
    ) -> str:
        """Format conversation history for context"""
        if not history:
            return ""

        conv_context = ""
        recent_turns = history[-max_turns:] if len(history) > max_turns else history

        for turn in recent_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            # Truncate long messages
            if len(content) > max_length_per_turn:
                content = content[:max_length_per_turn] + "..."

            conv_context += f"{role}: {content}\n"

        return conv_context.strip()

    def estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count for text"""
        if not text:
            return 0

        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately fit within token limit"""
        if not text:
            return text

        estimated_tokens = self.estimate_token_count(text)
        if estimated_tokens <= max_tokens:
            return text

        # Calculate approximate character limit
        char_limit = max_tokens * 4

        if len(text) <= char_limit:
            return text

        # Truncate and try to end at word boundary
        truncated = text[:char_limit]
        last_space = truncated.rfind(" ")

        if (
            last_space > char_limit * 0.8
        ):  # Only use word boundary if it's not too far back
            truncated = truncated[:last_space]

        return truncated + "..."

    def clean_json_content(self, content: str) -> str:
        """Clean content for safe JSON serialization"""
        if not content:
            return content

        # Remove or replace problematic characters
        cleaned = content.replace("\x00", "").replace("\\", "\\\\").replace('"', '\\"')

        # Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        return cleaned
