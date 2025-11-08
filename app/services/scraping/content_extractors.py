"""
Shared content extraction utilities for web scraping.

This module provides reusable functions for extracting contact information,
text cleaning, and other content extraction tasks used across multiple scrapers.
"""

import re
from typing import Set

from bs4 import BeautifulSoup


class ContactExtractor:
    """Extract contact information (phone, email) from HTML content"""

    # Phone number patterns for various international formats
    PHONE_PATTERNS = [
        r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
        r"\+\d{1,3}\s?\d{1,14}",  # Simple international format
        r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}",  # US format with parentheses
        r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}",  # US format
        r"\d{4}[-.\s]?\d{6,7}",  # Some Asian formats
    ]

    # Email pattern
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # Contact-related keywords for finding contact sections
    CONTACT_KEYWORDS = ["contact", "phone", "email", "address", "call", "reach"]

    @classmethod
    def extract_phone_numbers(cls, text: str) -> Set[str]:
        """
        Extract phone numbers from text using various international formats.

        Args:
            text: Text content to search

        Returns:
            Set of unique phone numbers found
        """
        phone_numbers = set()

        for pattern in cls.PHONE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned_match = match.strip()
                # Filter out numbers that are likely dates or other non-phone numbers
                digit_count = len(re.sub(r"\D", "", cleaned_match))
                if 10 <= digit_count <= 15:
                    digits_only = re.sub(r"\D", "", cleaned_match)
                    # Filter out dates and repeated digits
                    if len(set(digits_only)) > 1 and not re.search(
                        r"19\d{2}|20\d{2}", digits_only
                    ):
                        phone_numbers.add(cleaned_match)

        return phone_numbers

    @classmethod
    def extract_emails(cls, text: str, filter_noise: bool = True) -> Set[str]:
        """
        Extract email addresses from text.

        Args:
            text: Text content to search
            filter_noise: Filter out example/test/noreply emails

        Returns:
            Set of unique email addresses found
        """
        emails = set(re.findall(cls.EMAIL_PATTERN, text))

        if filter_noise:
            # Filter noise emails
            emails = {
                e for e in emails if not re.search(r"example|test|noreply", e, re.I)
            }

        return emails

    @classmethod
    def extract_from_soup(
        cls, soup: BeautifulSoup, include_emoji: bool = False, max_items: int = None
    ) -> str:
        """
        Extract contact information from BeautifulSoup object.

        This extracts from the entire page including header/footer before they're removed.

        Args:
            soup: BeautifulSoup object to extract from
            include_emoji: Whether to include emoji prefixes (📞, 📧)
            max_items: Maximum number of items per category (None = unlimited)

        Returns:
            Formatted contact information string
        """
        contact_parts = []

        # Find all text in the page (including header, footer, etc.)
        all_text = soup.get_text()

        # Extract phone numbers
        phone_numbers = cls.extract_phone_numbers(all_text)
        if phone_numbers:
            phone_list = sorted(phone_numbers)
            if max_items:
                phone_list = phone_list[:max_items]
            prefix = "📞 " if include_emoji else ""
            label = "Contact Numbers: " if not include_emoji else ""
            contact_parts.append(f"{prefix}{label}{', '.join(phone_list)}")

        # Extract email addresses
        emails = cls.extract_emails(all_text, filter_noise=True)
        if emails:
            email_list = sorted(emails)
            if max_items:
                email_list = email_list[:max_items]
            prefix = "📧 " if include_emoji else ""
            label = "Email Addresses: " if include_emoji else "Email: "
            contact_parts.append(f"{prefix}{label}{', '.join(email_list)}")

        # Look for common contact-related elements
        for keyword in cls.CONTACT_KEYWORDS:
            elements = soup.find_all(
                ["div", "span", "p", "a"], class_=re.compile(keyword, re.IGNORECASE)
            )
            elements += soup.find_all(
                ["div", "span", "p", "a"], id=re.compile(keyword, re.IGNORECASE)
            )

            for elem in elements[:3]:  # Limit to avoid too much noise
                elem_text = elem.get_text(strip=True)
                if elem_text and len(elem_text) < 200:  # Not too long
                    # Check if it contains useful contact info we haven't already captured
                    if any(phone in elem_text for phone in phone_numbers) or any(
                        email in elem_text for email in emails
                    ):
                        continue  # Already captured

                    # Check if it looks like contact information
                    if re.search(r"\d{3,}", elem_text) or "@" in elem_text:
                        if elem_text not in str(contact_parts):  # Avoid duplicates
                            contact_parts.append(elem_text)

        # Format output
        if include_emoji:
            return "\n".join(contact_parts) if contact_parts else ""
        else:
            return " | ".join(contact_parts) if contact_parts else ""

    @classmethod
    def extract_from_text(
        cls, text: str, include_emoji: bool = False, max_items: int = None
    ) -> str:
        """
        Extract contact information from plain text.

        Args:
            text: Plain text content
            include_emoji: Whether to include emoji prefixes
            max_items: Maximum number of items per category

        Returns:
            Formatted contact information string
        """
        contact_parts = []

        # Extract phone numbers
        phone_numbers = cls.extract_phone_numbers(text)
        if phone_numbers:
            phone_list = sorted(phone_numbers)
            if max_items:
                phone_list = phone_list[:max_items]
            prefix = "📞 " if include_emoji else ""
            label = "Contact Numbers: " if not include_emoji else ""
            contact_parts.append(f"{prefix}{label}{', '.join(phone_list)}")

        # Extract email addresses
        emails = cls.extract_emails(text, filter_noise=True)
        if emails:
            email_list = sorted(emails)
            if max_items:
                email_list = email_list[:max_items]
            prefix = "📧 " if include_emoji else ""
            label = "Email Addresses: " if include_emoji else "Email: "
            contact_parts.append(f"{prefix}{label}{', '.join(email_list)}")

        # Format output
        if include_emoji:
            return "\n".join(contact_parts) if contact_parts else ""
        else:
            return " | ".join(contact_parts) if contact_parts else ""


# Convenience function for backward compatibility
def extract_contact_information(
    soup: BeautifulSoup = None, text: str = None, include_emoji: bool = False
) -> str:
    """
    Convenience function to extract contact information.

    Args:
        soup: BeautifulSoup object (preferred)
        text: Plain text (fallback if soup not provided)
        include_emoji: Whether to include emoji prefixes

    Returns:
        Formatted contact information string
    """
    if soup:
        return ContactExtractor.extract_from_soup(soup, include_emoji=include_emoji)
    elif text:
        return ContactExtractor.extract_from_text(text, include_emoji=include_emoji)
    else:
        return ""
