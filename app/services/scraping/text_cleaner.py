"""
Shared text cleaning utilities for web scraping.

This module provides reusable functions for cleaning extracted text,
removing UI noise, and normalizing content across multiple scrapers.
"""

import re
from typing import List


class TextCleaner:
    """Clean and normalize text extracted from web pages"""

    # Common e-commerce UI noise patterns
    ECOMMERCE_NOISE_PATTERNS = [
        # Navigation and UI
        r"(Sign In|Log in|Sign Up|Sign up|Create Account|My Account)\s*",
        r"(Add to cart|Add to bag|Add to wishlist|Quick view|Quick buy)\s*",
        r"(View Cart|Checkout|Continue shopping)\s*",
        r"(Sort by|Filter by|Showing \d+-\d+ of \d+)\s*",
        r"(Skip to content|Your cart is empty|Shopping cart Loading)\s*",
        r"(Have an account|Log in to check out faster)\s*",
        r"(Add note|Calculate shipping|Subtotal|Taxes and shipping calculated)\s*",
        r"(Update|Check out|View Cart)\s*",
        # Cookie and privacy
        r"(Cookie Policy|Privacy Policy|Terms of Service|Accept Cookies)\s*",
        # Newsletter/Marketing
        r"(Subscribe|Newsletter|Sign up for|Email signup)\s*",
        # Social media
        r"(Follow us|Share|Facebook|Twitter|Instagram|Pinterest)\s*",
        # Country selectors
        r"(Shipping Country|Select Country|Choose Region|Province|Zip/Postal Code)\s*",
        # Payment badges
        r"(Visa|Mastercard|PayPal|American Express|Discover)\s*",
        # Common footer text
        r"Copyright\s+©\s+\d{4}\s*",
        r"All rights reserved\s*",
    ]

    # Country name patterns (for aggressive removal)
    COUNTRY_PATTERNS = [
        r"Australia|Austria|Belgium|Canada|Czechia|Denmark|Finland|France|Germany|Hong Kong SAR|Ireland|Israel|Italy|Japan|Malaysia|Netherlands|New Zealand|Norway|Poland|Portugal|Singapore|South Korea|Spain|Sweden|Switzerland|United Arab Emirates|United Kingdom|United States",
    ]

    @classmethod
    def clean_whitespace(cls, text: str) -> str:
        """
        Clean and normalize whitespace in text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text with normalized whitespace
        """
        if not text:
            return ""

        # Remove extra whitespace and normalize line breaks
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned = " ".join(chunk for chunk in chunks if chunk)

        return cleaned.strip()

    @classmethod
    def remove_ecommerce_noise(cls, text: str, aggressive: bool = False) -> str:
        """
        Remove common e-commerce UI noise patterns.

        Args:
            text: Text to clean
            aggressive: Whether to use aggressive cleaning (removes more patterns)

        Returns:
            Cleaned text with noise removed
        """
        if not text:
            return ""

        # Remove common e-commerce UI patterns
        for pattern in cls.ECOMMERCE_NOISE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        if aggressive:
            # Remove country names (very aggressive)
            for pattern in cls.COUNTRY_PATTERNS:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)

            # Remove excessive country lists
            country_list = r"(?:Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia|Botswana|Brazil|Brunei|Bulgaria|Burkina|Burundi|Cambodia|Cameroon|Canada|Chad|Chile|China|Colombia|Congo|Costa Rica|Croatia|Cuba|Cyprus|Czechia|Denmark|Djibouti|Dominica|Ecuador|Egypt|El Salvador|Estonia|Ethiopia|Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Korea|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Mauritania|Mauritius|Mexico|Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nepal|Netherlands|New Zealand|Nicaragua|Niger|Nigeria|Norway|Oman|Pakistan|Panama|Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Samoa|San Marino|Saudi Arabia|Senegal|Serbia|Seychelles|Singapore|Slovakia|Slovenia|Somalia|South Africa|Spain|Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Togo|Trinidad|Tunisia|Turkey|Turkmenistan|Uganda|Ukraine|United Arab Emirates|United Kingdom|United States|Uruguay|Uzbekistan|Vanuatu|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe)"

            # Remove lines with 5+ countries
            lines = text.split("\n")
            cleaned_lines = []
            for line in lines:
                country_matches = re.findall(country_list, line, re.IGNORECASE)
                if len(country_matches) < 5:  # Keep line if less than 5 countries
                    cleaned_lines.append(line)

            text = "\n".join(cleaned_lines)

        # Clean up whitespace created by pattern removal
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @classmethod
    def clean_text(
        cls, text: str, remove_noise: bool = True, aggressive: bool = False
    ) -> str:
        """
        Comprehensive text cleaning pipeline.

        Args:
            text: Text to clean
            remove_noise: Whether to remove UI noise patterns
            aggressive: Whether to use aggressive cleaning

        Returns:
            Fully cleaned text
        """
        if not text:
            return ""

        # Step 1: Clean whitespace
        cleaned = cls.clean_whitespace(text)

        # Step 2: Remove noise if requested
        if remove_noise:
            cleaned = cls.remove_ecommerce_noise(cleaned, aggressive=aggressive)

        # Step 3: Final whitespace cleanup
        cleaned = cls.clean_whitespace(cleaned)

        return cleaned

    @classmethod
    def filter_noise_chunks(cls, chunks: List[str], min_length: int = 60) -> List[str]:
        """
        Filter out chunks that are clearly UI noise.

        Args:
            chunks: List of text chunks to filter
            min_length: Minimum chunk length to keep

        Returns:
            Filtered list of chunks
        """
        if not chunks:
            return []

        # Regexes target common e-commerce UI artifacts
        noise_patterns = [
            r"\b(Sign In|Sign Up|Log in|Create Account|Forgot your password)\b",
            r"\b(Add to (cart|bag|wishlist)|Quick (view|buy)|View Cart|Checkout)\b",
            r"\b(Sort by|Filter by|Showing \d+-\d+ of \d+)\b",
            r"\b(Your cart is empty|Shopping cart Loading)\b",
            r"\b(Subscribe|Newsletter|Email signup)\b",
            r"\b(Choose Region|Select Country|Province|Zip/Postal Code)\b",
            r"\b(Visa|Mastercard|PayPal|American Express|Discover)\b",
        ]

        compiled = [re.compile(pat, re.IGNORECASE) for pat in noise_patterns]

        def is_noise(text: str) -> bool:
            stripped = text.strip()
            if len(stripped) < min_length:
                return True

            # Large comma counts usually indicate country/region lists
            if stripped.count(",") >= 20:
                return True

            # Track how much of the chunk is actually noise. Some Shopify chunks
            # contain phrases like "Showing 1-8 of 20" next to real product data,
            # so we only drop the chunk if noise dominates the content.
            noise_chars = 0
            for pattern in compiled:
                for match in pattern.finditer(stripped):
                    noise_chars += len(match.group(0))
                    # Bail out early if noise clearly overwhelms the chunk
                    if noise_chars >= len(stripped) * 0.65:
                        return True

            # If we saw any noise but the chunk is still relatively short,
            # treat it as noise (pure nav/login text, cookie banners, etc.).
            if noise_chars and len(stripped) <= max(min_length * 2, 180):
                return True

            return False

        filtered = [c for c in chunks if not is_noise(c)]

        # Deduplicate identical chunks to reduce redundant vectors
        seen = set()
        unique_filtered = []
        for c in filtered:
            key = c.strip()
            if key not in seen:
                seen.add(key)
                unique_filtered.append(c)

        return unique_filtered


# Convenience functions for backward compatibility
def clean_text(text: str, remove_noise: bool = True) -> str:
    """Quick function to clean text"""
    return TextCleaner.clean_text(text, remove_noise=remove_noise)


def remove_ecommerce_noise(text: str, aggressive: bool = False) -> str:
    """Quick function to remove e-commerce noise"""
    return TextCleaner.remove_ecommerce_noise(text, aggressive=aggressive)
