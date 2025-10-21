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

            # Extract all text content
            text_content = await page.evaluate("() => document.body.innerText")

            # Extract contact information (phone, email) before cleaning
            contact_info = await self._extract_contact_information(page)

            # Clean and combine
            cleaned_text = self._clean_text(text_content)

            # Prepend contact info if found
            if contact_info:
                cleaned_text = contact_info + "\n\n" + cleaned_text

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

    async def _extract_contact_information(self, page: Page) -> str:
        """Extract contact information (phone, email, address) from the page"""
        try:
            # Get the full HTML content
            html_content = await page.content()
            soup = BeautifulSoup(html_content, "html.parser")

            contact_parts = []
            all_text = soup.get_text()

            # Extract phone numbers
            phone_patterns = [
                r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
                r"\+\d{1,3}\s?\d{1,14}",
                r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}",
                r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}",
                r"\d{10,}",  # 10+ digits
            ]

            phone_numbers = set()
            for pattern in phone_patterns:
                matches = re.findall(pattern, all_text)
                for match in matches:
                    cleaned_match = match.strip()
                    # Only include if it has at least 10 digits
                    if len(re.sub(r"\D", "", cleaned_match)) >= 10:
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

            return "\n".join(contact_parts) if contact_parts else ""

        except Exception as e:
            logger.warning(f"Failed to extract contact information: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""

        # Split into lines and clean each
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines and very short lines
            if len(line) > 1:
                cleaned_lines.append(line)

        # Join with single newlines
        return "\n".join(cleaned_lines)


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
