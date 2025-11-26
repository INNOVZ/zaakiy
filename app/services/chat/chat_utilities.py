"""
Chat Utilities Service
Provides utility methods for message validation, formatting, and text processing
"""
import logging
import re
from typing import Any, Dict, List

from .shared.keyword_extractor import get_keyword_extractor

logger = logging.getLogger(__name__)


class ChatUtilitiesError(Exception):
    """Exception for chat utilities errors"""


class ChatUtilities:
    """Utility methods for chat operations"""

    def __init__(self):
        # Use shared keyword extractor to avoid duplication
        self.keyword_extractor = get_keyword_extractor()

    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Delegate to shared keyword extractor
        return self.keyword_extractor.extract_keywords(query, min_length=3)

    def calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword relevance score"""
        # Delegate to shared keyword extractor
        return self.keyword_extractor.calculate_keyword_score(text, keywords)

    def calculate_domain_relevance(
        self, text: str, domain_context: str, domain_knowledge: str
    ) -> float:
        """Calculate domain-specific relevance score"""
        if not text or not domain_context:
            return 0.0

        # Combine domain context and knowledge using shared keyword extractor
        domain_terms = []
        if domain_context:
            domain_terms.extend(self.keyword_extractor.extract_keywords(domain_context))
        if domain_knowledge:
            domain_terms.extend(
                self.keyword_extractor.extract_keywords(domain_knowledge)
            )

        if not domain_terms:
            return 0.0

        # Use shared keyword extractor's scoring method
        score = self.keyword_extractor.calculate_keyword_score(text, domain_terms)

        # Return boost factor (0.0 to 0.5)
        return min(score, 0.5)

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
        """
        Sanitize text content for safe internal processing.

        NOTE: This is for basic text cleanup only, NOT for security validation.
        For user input validation, use ChatSecurityService.validate_message().
        For chatbot configuration, use PromptSanitizer.

        See SANITIZATION_GUIDE.md for when to use which sanitizer.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text with control characters removed and whitespace normalized
        """
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

    @staticmethod
    def is_contact_query(query: str) -> bool:
        """
        Detect if the query is asking for contact, booking, or demo information.

        This is a shared utility to avoid duplicate detection logic across services.
        Used by both document_retrieval_service and response_generation_service.

        Args:
            query: User query string

        Returns:
            True if query is about contact information, False otherwise
        """
        if not query:
            return False

        query_lower = query.lower().strip()

        # Direct contact keywords (phones, emails, addresses, etc.)
        contact_keywords = [
            "phone",
            "number",
            "call",
            "telephone",
            "mobile",
            "contact",
            "email",
            "mail",
            "address",
            "location",
            "reach",
            "get in touch",
            "whatsapp",
            "how to contact",
            "contact details",
            "contact info",
            "talk to someone",
            "speak to someone",
            "how can i contact",
            "how can i reach",
            "how to reach",
            "what's your phone",
            "what is your email",
            "contact you",
            "reach you",
        ]

        # Check for contact keywords
        if any(keyword in query_lower for keyword in contact_keywords):
            return True

        # Demo / consultation booking intent
        booking_patterns = [
            r"book(?:ing)? (?:a )?(?:demo|consultation|call|meeting|appointment)",
            r"schedule(?: a)? (?:demo|consultation|call|meeting)",
            r"(?:request|arrange|set up|organize) (?:a )?(?:demo|consultation|call)",
            r"(?:demo|consultation) (?:request|booking|schedule)",
            r"(?:talk|speak|connect) (?:with|to) (?:sales|support|an expert|the team)",
            r"how (?:can|do) i (?:book|schedule|arrange) (?:a )?demo",
            r"(?:book|schedule) (?:a )?time with (?:the )?team",
        ]

        for pattern in booking_patterns:
            if re.search(pattern, query_lower):
                return True

        # Combined keyword detection for short queries like "Book demo"
        demo_terms = ["demo", "trial", "consultation", "meeting", "appointment"]
        action_terms = ["book", "schedule", "arrange", "request", "set up", "organize"]

        has_demo_term = any(term in query_lower for term in demo_terms)
        has_action_term = any(term in query_lower for term in action_terms)

        if has_demo_term and has_action_term:
            return True

        return False
