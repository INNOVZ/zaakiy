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
from .content_extractors import ContactExtractor

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
            # Navigation and UI
            r"(Sign In|Log in|Sign Up|Sign up|Create Account|My Account)\s*",
            r"(Add to cart|Add to bag|Add to wishlist|Quick view|Quick buy)\s*",
            r"(View Cart|Checkout|Continue shopping)\s*",
            r"(Sort by|Filter by|Showing \d+-\d+ of \d+)\s*",
            # Cookie and privacy
            r"(Cookie Policy|Privacy Policy|Terms of Service|Accept Cookies)\s*",
            # Newsletter/Marketing
            r"(Subscribe|Newsletter|Sign up for|Email signup)\s*",
            # Social media
            r"(Follow us|Share|Facebook|Twitter|Instagram|Pinterest)\s*",
            # Country/Region selectors (aggressive removal)
            r"(United States|United Kingdom|Australia|Canada|France|Germany|Italy|Spain|Japan|China|India|Brazil)\s+"
            * 3,  # Multiple countries
            r"(Shipping Country|Select Country|Choose Region)\s*",
            # Payment badges
            r"(Visa|Mastercard|PayPal|American Express|Discover)\s*",
            # Common footer text
            r"Copyright\s+©\s+\d{4}\s*",
            r"All rights reserved\s*",
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
        Uses the shared ContactExtractor utility to avoid code duplication.
        """
        return ContactExtractor.extract_from_soup(soup, include_emoji=True)


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
