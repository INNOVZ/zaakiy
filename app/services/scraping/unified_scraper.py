"""
Unified Scraping System - Robust scraper for all website types.

This module provides a comprehensive scraping solution that handles:
- Modern JavaScript frameworks (React, Next.js, Vue, Angular, etc.)
- E-commerce platforms (Shopify, WooCommerce, BigCommerce, etc.)
- Traditional HTML websites
- PDF documents
- JSON data files

The system uses intelligent fallbacks and multiple strategies to ensure
maximum success rate across different website types.
"""

import asyncio
from typing import Dict

from ...utils.logging_config import get_logger
from .url_utils import is_ecommerce_url, log_domain_safely

logger = get_logger(__name__)

# Import scrapers with fallback handling
try:
    from .ecommerce_scraper import EnhancedEcommerceProductScraper, scrape_ecommerce_url

    ECOMMERCE_AVAILABLE = True
except ImportError:
    ECOMMERCE_AVAILABLE = False
    logger.warning("E-commerce scraper not available")

try:
    from .playwright_scraper import PlaywrightWebScraper, scrape_url_with_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright scraper not available")

try:
    from .web_scraper import SecureWebScraper, scrape_url_text

    TRADITIONAL_AVAILABLE = True
except ImportError:
    TRADITIONAL_AVAILABLE = False
    logger.warning("Traditional scraper not available")


class UnifiedScraper:
    """
    Unified scraper that intelligently selects the best scraping strategy
    for any given URL, with comprehensive fallbacks.
    """

    def __init__(
        self,
        ecommerce_timeout: int = 90000,
        playwright_timeout: int = 30000,
        max_retries: int = 3,
    ):
        """
        Initialize the unified scraper.

        Args:
            ecommerce_timeout: Timeout for e-commerce scrapers (ms)
            playwright_timeout: Timeout for Playwright (ms)
            max_retries: Maximum retry attempts for failed scrapes
        """
        self.ecommerce_timeout = ecommerce_timeout
        self.playwright_timeout = playwright_timeout
        self.max_retries = max_retries

    async def scrape(self, url: str, extract_products: bool = True) -> Dict[str, any]:
        """
        Scrape a URL using the best available strategy with fallbacks.

        Args:
            url: The URL to scrape
            extract_products: Whether to extract structured product data for e-commerce sites

        Returns:
            Dict with:
                - 'text': Extracted text content
                - 'products': List of products (for e-commerce sites)
                - 'product_urls': List of product URLs (for e-commerce sites)
                - 'method': Scraping method used
                - 'success': Whether scraping succeeded
                - 'error': Error message if failed
        """
        result = {
            "text": "",
            "products": [],
            "product_urls": [],
            "method": "unknown",
            "success": False,
            "error": None,
        }

        # Track results from each strategy for error aggregation
        ecommerce_result = None
        playwright_result = None
        traditional_result = None

        # Strategy 1: E-commerce scraper (for Shopify, WooCommerce, etc.)
        if ECOMMERCE_AVAILABLE and is_ecommerce_url(url):
            ecommerce_result = await self._try_ecommerce_scraper(url, extract_products)
            if ecommerce_result["success"]:
                return ecommerce_result

        # Strategy 2: Playwright (for modern JS frameworks)
        if PLAYWRIGHT_AVAILABLE:
            playwright_result = await self._try_playwright_scraper(url)
            if playwright_result["success"]:
                return playwright_result

        # Strategy 3: Traditional scraper (fallback)
        if TRADITIONAL_AVAILABLE:
            traditional_result = await self._try_traditional_scraper(url)
            if traditional_result["success"]:
                return traditional_result

        # All strategies failed - aggregate error messages
        error_messages = []
        if ecommerce_result:
            error_messages.append(
                f"e-commerce: {ecommerce_result.get('error', 'unknown')}"
            )
        if playwright_result:
            error_messages.append(
                f"playwright: {playwright_result.get('error', 'unknown')}"
            )
        if traditional_result:
            error_messages.append(
                f"traditional: {traditional_result.get('error', 'unknown')}"
            )

        # If no strategies were available, note that
        if not error_messages:
            unavailable = []
            if not ECOMMERCE_AVAILABLE:
                unavailable.append("e-commerce")
            if not PLAYWRIGHT_AVAILABLE:
                unavailable.append("playwright")
            if not TRADITIONAL_AVAILABLE:
                unavailable.append("traditional")
            error_msg = f"All strategies unavailable: {', '.join(unavailable)}"
        else:
            error_msg = "; ".join(error_messages)

        result["error"] = f"All scraping strategies failed: {error_msg}"
        logger.error(
            f"All scraping strategies failed for {log_domain_safely(url)}: {error_msg}"
        )
        return result

    async def _try_ecommerce_scraper(
        self, url: str, extract_products: bool
    ) -> Dict[str, any]:
        """Try e-commerce scraper with retries."""
        result = {
            "text": "",
            "products": [],
            "product_urls": [],
            "method": "ecommerce",
            "success": False,
            "error": None,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"[E-commerce] Attempt {attempt + 1}/{self.max_retries} for {log_domain_safely(url)}"
                )

                async with EnhancedEcommerceProductScraper(
                    headless=True, timeout=self.ecommerce_timeout
                ) as scraper:
                    scrape_result = await scraper.scrape_product_collection(url)

                if scrape_result and scrape_result.get("text"):
                    text_content = scrape_result["text"].strip()

                    if len(text_content) > 0:
                        result["text"] = text_content
                        result["products"] = scrape_result.get("products", [])
                        result["product_urls"] = scrape_result.get("product_urls", [])
                        result["success"] = True

                        logger.info(
                            f"[E-commerce] ✅ Success: {len(text_content)} chars, "
                            f"{len(result['products'])} products, "
                            f"{len(result['product_urls'])} product URLs"
                        )
                        return result
                    else:
                        logger.warning(
                            f"[E-commerce] Attempt {attempt + 1}: Empty text content"
                        )
                else:
                    logger.warning(
                        f"[E-commerce] Attempt {attempt + 1}: No result returned"
                    )

            except Exception as e:
                logger.warning(f"[E-commerce] Attempt {attempt + 1} failed: {str(e)}")
                result["error"] = str(e)

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return result

    async def _try_playwright_scraper(self, url: str) -> Dict[str, any]:
        """Try Playwright scraper with retries."""
        result = {
            "text": "",
            "products": [],
            "product_urls": [],
            "method": "playwright",
            "success": False,
            "error": None,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"[Playwright] Attempt {attempt + 1}/{self.max_retries} for {log_domain_safely(url)}"
                )

                text = await scrape_url_with_playwright(url)

                if text and len(text.strip()) > 50:
                    # Check for JavaScript error messages
                    if "enable javascript" not in text.lower():
                        result["text"] = text.strip()
                        result["success"] = True

                        logger.info(
                            f"[Playwright] ✅ Success: {len(result['text'])} chars"
                        )
                        return result
                    else:
                        logger.warning(
                            f"[Playwright] Attempt {attempt + 1}: JavaScript not enabled message"
                        )
                else:
                    logger.warning(
                        f"[Playwright] Attempt {attempt + 1}: Insufficient content ({len(text.strip()) if text else 0} chars)"
                    )

            except Exception as e:
                logger.warning(f"[Playwright] Attempt {attempt + 1} failed: {str(e)}")
                result["error"] = str(e)

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return result

    async def _try_traditional_scraper(self, url: str) -> Dict[str, any]:
        """Try traditional scraper with retries."""
        result = {
            "text": "",
            "products": [],
            "product_urls": [],
            "method": "traditional",
            "success": False,
            "error": None,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"[Traditional] Attempt {attempt + 1}/{self.max_retries} for {log_domain_safely(url)}"
                )

                text = await scrape_url_text(url)

                if text and len(text.strip()) > 50:
                    result["text"] = text.strip()
                    result["success"] = True

                    logger.info(f"[Traditional] ✅ Success: {len(result['text'])} chars")
                    return result
                else:
                    logger.warning(
                        f"[Traditional] Attempt {attempt + 1}: Insufficient content"
                    )

            except Exception as e:
                logger.warning(f"[Traditional] Attempt {attempt + 1} failed: {str(e)}")
                result["error"] = str(e)

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return result

    async def scrape_with_products(self, url: str) -> Dict[str, any]:
        """
        Scrape URL and extract structured product information.

        For e-commerce sites, this returns detailed product data.
        For other sites, returns standard text extraction.

        Returns:
            Dict with text, products, product_urls, and metadata
        """
        result = await self.scrape(url, extract_products=True)

        # If e-commerce scraper didn't work but we got text, try to extract products from text
        if result["success"] and result["text"] and not result["products"]:
            if is_ecommerce_url(url):
                # Try to extract product links from text
                from .ingestion_worker import extract_product_links_from_chunk

                product_links = extract_product_links_from_chunk(result["text"], url)
                if product_links:
                    result["product_urls"] = product_links
                    logger.info(
                        f"Extracted {len(product_links)} product links from text"
                    )

        return result


# Convenience function
async def scrape_url_unified(url: str) -> str:
    """
    Unified scraping function that returns text content.

    This is a convenience wrapper around UnifiedScraper for simple use cases.

    Args:
        url: URL to scrape

    Returns:
        Extracted text content
    """
    scraper = UnifiedScraper()
    result = await scraper.scrape(url)

    if result["success"]:
        return result["text"]
    else:
        raise ValueError(
            f"Failed to scrape URL: {result.get('error', 'Unknown error')}"
        )


async def scrape_url_with_products(url: str) -> Dict[str, any]:
    """
    Scrape URL and return structured data including products.

    Args:
        url: URL to scrape

    Returns:
        Dict with text, products, and product_urls
    """
    scraper = UnifiedScraper()
    return await scraper.scrape_with_products(url)
