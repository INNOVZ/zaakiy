"""
Playwright-based web scraper for JavaScript-rendered websites.

This scraper uses Playwright to render JavaScript content before extraction,
solving the issue where React/Vue/Angular apps don't work with traditional scrapers.
"""

import asyncio
import logging
import re
from typing import Dict, Optional

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class PlaywrightWebScraper:
    """Web scraper using Playwright for JavaScript-rendered content"""

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None

    async def __aenter__(self):
        """Context manager entry"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def scrape_url(self, url: str) -> Dict[str, str]:
        """
        Scrape a URL with JavaScript rendering

        Args:
            url: The URL to scrape

        Returns:
            Dict with 'text' (extracted content) and 'error' (if any)
        """
        if not self.browser:
            raise RuntimeError(
                "Browser not initialized. Use 'async with' context manager."
            )

        page: Optional[Page] = None
        try:
            logger.info(f"🌐 Scraping URL with Playwright: {url}")

            # Create new page
            page = await self.browser.new_page()

            # Navigate to URL and wait for network to be idle
            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            # Wait a bit for any lazy-loaded content
            await page.wait_for_timeout(2000)

            # HYBRID APPROACH: Get rendered HTML and pass to BeautifulSoup for structured parsing
            html_content = await page.content()

            # Use BeautifulSoup for intelligent parsing and extraction
            cleaned_text = self._extract_and_clean_text(html_content)

            logger.info(
                f"✅ Successfully scraped {len(cleaned_text)} characters from {url}"
            )

            return {"text": cleaned_text, "error": None}

        except Exception as e:
            logger.error(f"❌ Failed to scrape {url}: {str(e)}")
            return {"text": "", "error": str(e)}
        finally:
            if page:
                await page.close()

    def _extract_and_clean_text(self, html_content: str) -> str:
        """
        Extract and clean text from rendered HTML using BeautifulSoup.

        HYBRID APPROACH:
        1. Playwright renders JavaScript → Full HTML
        2. BeautifulSoup parses HTML → Structured extraction
        3. Remove unwanted elements (nav, footer, ads, scripts)
        4. Extract contact information intelligently
        5. Return clean, structured text
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # FIRST: Extract contact information from headers/footers before removing them
        contact_info = self._extract_contact_information(soup)

        # Remove unwanted elements (but we already saved contact info)
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Remove e-commerce UI clutter that adds noise
        ui_selectors_to_remove = [
            # Shopping cart and checkout
            {"class_": re.compile(r"cart|checkout|basket|bag|wishlist", re.I)},
            {"id": re.compile(r"cart|checkout|basket|bag|wishlist", re.I)},
            # Login/signup forms
            {"class_": re.compile(r"login|signup|sign-up|sign-in|auth|modal", re.I)},
            {"id": re.compile(r"login|signup|sign-up|sign-in|auth|modal", re.I)},
            # Navigation and filters
            {"class_": re.compile(r"filter|sidebar|breadcrumb|pagination|sort", re.I)},
            # Promotional banners
            {"class_": re.compile(r"banner|promo|sale-banner|announcement", re.I)},
            # Country/region selectors
            {"class_": re.compile(r"country|region|locale|currency-selector", re.I)},
        ]

        for selector in ui_selectors_to_remove:
            for element in soup.find_all(**selector):
                element.decompose()

        # Extract text
        text = soup.get_text()

        # Clean text - remove extra whitespace and normalize
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)

        # Additional cleaning: remove common e-commerce noise patterns
        noise_patterns = [
            r"(Sign In|Log in|Sign Up|Sign up|Create Account|Forgot Password)\??",
            r"(Add to (cart|basket|wishlist|bag))",
            r"(Quick (view|buy|shop))",
            r"(Shop now|Buy now|Sold out)",
            r"(Sort by|Filter by|Showing \d+-\d+ of \d+ Results)",
            r"(My Account|My Orders|Track Order)",
            r"(Cookie Policy|Privacy Policy|Terms of Service)",
            # Remove excessive repeating state/country names
            r"(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia){5,}",
        ]

        for pattern in noise_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

        # Remove extra spaces created by pattern removal
        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

        # Prepend contact information to ensure it's included in the indexed content
        if contact_info:
            cleaned_text = contact_info + "\n\n" + cleaned_text

        return cleaned_text.strip()

    def _extract_contact_information(self, soup: BeautifulSoup) -> str:
        """
        Extract contact information (phone, email, address) from BeautifulSoup object.

        This extracts from the entire page including header/footer before they're removed.
        """
        contact_parts = []

        # Find all text in the page (including header, footer, etc.)
        all_text = soup.get_text()

        # Extract phone numbers using various international formats
        phone_patterns = [
            r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
            r"\+\d{1,3}\s?\d{1,14}",  # Simple international format
            r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}",  # US format with parentheses
            r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}",  # US format
            r"\d{4}[-.\s]?\d{6,7}",  # Some Asian formats
        ]

        phone_numbers = set()
        for pattern in phone_patterns:
            matches = re.findall(pattern, all_text)
            for match in matches:
                # Clean up the match and validate it looks like a real phone number
                cleaned_match = match.strip()
                # Filter out numbers that are likely dates or other non-phone numbers
                if len(re.sub(r"\D", "", cleaned_match)) >= 10:  # At least 10 digits
                    phone_numbers.add(cleaned_match)

        if phone_numbers:
            contact_parts.append(
                "📞 Contact Numbers: " + ", ".join(sorted(phone_numbers))
            )

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = set(re.findall(email_pattern, all_text))

        if emails:
            contact_parts.append("📧 Email Addresses: " + ", ".join(sorted(emails)))

        # Look for common contact-related elements
        contact_keywords = ["contact", "phone", "email", "address", "call", "reach"]

        # Search for elements with contact-related classes or IDs
        for keyword in contact_keywords:
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

        return "\n".join(contact_parts) if contact_parts else ""


async def scrape_url_with_playwright(url: str) -> str:
    """
    Convenience function to scrape a single URL

    Args:
        url: The URL to scrape

    Returns:
        Extracted text content
    """
    async with PlaywrightWebScraper() as scraper:
        result = await scraper.scrape_url(url)
        if result["error"]:
            raise ValueError(f"Failed to scrape URL: {result['error']}")
        return result["text"]


# For backward compatibility with existing code
async def scrape_url_text(url: str) -> str:
    """
    Fallback function that tries Playwright first, then falls back to traditional scraping
    """
    try:
        # Try Playwright first
        return await scrape_url_with_playwright(url)
    except Exception as e:
        logger.warning(
            f"Playwright scraping failed: {e}, falling back to traditional scraper"
        )
        # Import and use the traditional scraper as fallback
        from .web_scraper import SecureWebScraper

        scraper = SecureWebScraper()
        return await scraper.scrape_url_text(url)
