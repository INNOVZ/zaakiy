"""
Contact Information Extractor
Extracts phone numbers, emails, addresses, and links from chunks
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContactExtractor:
    """Extract contact information from text chunks"""

    # Enhanced phone number patterns
    PHONE_PATTERNS = [
        r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
        r"\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}",  # UAE format
        r"\+971[-.\s]?\d{1,2}[-.\s]?\d{3}[-.\s]?\d{4}",  # UAE specific
        r"\d{2,3}[-.\s]?\d{3}[-.\s]?\d{4}",  # Local format
    ]

    # Email pattern
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # URL patterns (including demo/booking links)
    URL_PATTERNS = [
        r"https?://[^\s<>\"{}|\\^`\[\]]{1,2000}",  # Standard URLs
        r"www\.[^\s<>\"{}|\\^`\[\]]{1,2000}",  # www URLs
    ]

    # Demo/booking link keywords
    DEMO_KEYWORDS = [
        "demo",
        "booking",
        "book",
        "schedule",
        "appointment",
        "onboarding",
        "surveysparrow",
        "calendly",
        "zoom",
        "meet",
        "contact",
        "form",
    ]

    def extract_contact_info(self, chunk: str) -> Dict[str, Any]:
        """
        Extract all contact information from a chunk

        Returns:
            Dict with keys: phones, emails, addresses, links, demo_links
        """
        result = {
            "phones": [],
            "emails": [],
            "addresses": [],
            "links": [],
            "demo_links": [],
            "has_contact_info": False,
        }

        if not chunk or not isinstance(chunk, str):
            return result

        # Try to parse JSON if chunk appears to be JSON-like
        parsed_chunk = self._try_parse_json(chunk)
        if parsed_chunk != chunk:
            # If we parsed JSON, extract from the parsed structure
            chunk = self._extract_text_from_json(parsed_chunk)

        # Extract phone numbers
        result["phones"] = self._extract_phones(chunk)

        # Extract emails
        result["emails"] = self._extract_emails(chunk)

        # Extract addresses
        result["addresses"] = self._extract_addresses(chunk)

        # Extract all links
        result["links"] = self._extract_links(chunk)

        # Identify demo/booking links
        result["demo_links"] = self._identify_demo_links(result["links"])

        # Check if we found any contact info
        result["has_contact_info"] = bool(
            result["phones"]
            or result["emails"]
            or result["addresses"]
            or result["demo_links"]
        )

        return result

    def _try_parse_json(self, text: str) -> Any:
        """Try to parse JSON from text, return original if fails"""
        if not text or not isinstance(text, str):
            return text

        # Try parsing as JSON directly
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from escaped strings
        # Look for patterns like "key": "value" or { "key": "value" }
        json_match = re.search(
            r'\{[^{}]*"(?:phone|email|address|link|demo|contact)[^{}]*\}',
            text,
            re.IGNORECASE,
        )
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return text

    def _extract_text_from_json(self, parsed: Any) -> str:
        """Extract text content from parsed JSON structure"""
        if isinstance(parsed, str):
            return parsed
        elif isinstance(parsed, dict):
            # Extract values from dictionary
            texts = []
            for key, value in parsed.items():
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, (dict, list)):
                    texts.append(self._extract_text_from_json(value))
            return " ".join(texts)
        elif isinstance(parsed, list):
            texts = [self._extract_text_from_json(item) for item in parsed]
            return " ".join(texts)
        else:
            return str(parsed)

    def _extract_phones(self, text: str) -> List[str]:
        """Extract phone numbers from text"""
        phones = set()

        for pattern in self.PHONE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean and validate phone number
                cleaned = re.sub(r"[\s\-\(\)\.]", "", match.strip())
                # Must have at least 10 digits
                if len(re.sub(r"\D", "", cleaned)) >= 10:
                    # Preserve original format but normalize
                    normalized = match.strip()
                    phones.add(normalized)

        return list(phones)

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        emails = set(re.findall(self.EMAIL_PATTERN, text, re.IGNORECASE))

        # Filter out common noise emails
        noise_patterns = [r"example\.com", r"test\.com", r"noreply", r"no-reply"]
        filtered = []
        for email in emails:
            if not any(
                re.search(pattern, email, re.IGNORECASE) for pattern in noise_patterns
            ):
                filtered.append(email.lower())

        return filtered

    def _extract_addresses(self, text: str) -> List[str]:
        """Extract addresses from text"""
        addresses = []

        # Look for address-like patterns
        # Common patterns: "Building", "Street", "Road", "Avenue", "Dubai", "UAE", etc.
        address_indicators = [
            r"Building[^.!?]{10,100}",
            r"Street[^.!?]{10,100}",
            r"Road[^.!?]{10,100}",
            r"Avenue[^.!?]{10,100}",
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Building|Street|Road|Avenue)[^.!?]{10,100}",
            r"[^.!?]{20,150}(?:Dubai|Abu Dhabi|Sharjah|UAE|United Arab Emirates)[^.!?]{0,50}",
        ]

        for pattern in address_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the address
                cleaned = re.sub(r"\s+", " ", match.strip())
                if len(cleaned) > 15 and cleaned not in addresses:
                    addresses.append(cleaned)

        return addresses

    def _extract_links(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        links = set()

        for pattern in self.URL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean URL (remove trailing punctuation)
                cleaned = re.sub(r"[.,;!?]+$", "", match.strip())
                # Add protocol if missing
                if cleaned.startswith("www."):
                    cleaned = "https://" + cleaned
                if cleaned.startswith("http://") or cleaned.startswith("https://"):
                    links.add(cleaned)

        return list(links)

    def _identify_demo_links(self, links: List[str]) -> List[str]:
        """Identify demo/booking links from list of links"""
        demo_links = []

        for link in links:
            link_lower = link.lower()
            # Check if link contains demo/booking keywords
            if any(keyword in link_lower for keyword in self.DEMO_KEYWORDS):
                demo_links.append(link)
            # Check for common booking platforms
            elif any(
                platform in link_lower
                for platform in [
                    "calendly",
                    "surveysparrow",
                    "cal.com",
                    "acuity",
                    "appointlet",
                    "schedule",
                    "booking",
                ]
            ):
                demo_links.append(link)

        return demo_links

    def score_chunk_for_contact_query(self, chunk: str) -> float:
        """
        Score a chunk based on how likely it contains contact information
        Higher score = more likely to contain contact info
        """
        contact_info = self.extract_contact_info(chunk)
        score = 0.0

        # Phone numbers are very important
        score += len(contact_info["phones"]) * 10.0

        # Emails are important
        score += len(contact_info["emails"]) * 8.0

        # Demo/booking links are very important
        score += len(contact_info["demo_links"]) * 15.0

        # Addresses are moderately important
        score += len(contact_info["addresses"]) * 5.0

        # Regular links might be useful
        score += len(contact_info["links"]) * 2.0

        # Check for contact-related keywords in text
        contact_keywords = [
            "phone",
            "contact",
            "call",
            "email",
            "address",
            "location",
            "reach",
            "demo",
            "booking",
            "schedule",
            "appointment",
        ]
        chunk_lower = chunk.lower()
        keyword_matches = sum(
            1 for keyword in contact_keywords if keyword in chunk_lower
        )
        score += keyword_matches * 1.0

        return score


# Global instance
contact_extractor = ContactExtractor()
