"""
Cached Document Processor - Adds caching to PDF and JSON extraction

This module provides caching wrappers for expensive document processing operations.
"""

import logging
import time
from typing import Optional

from .scraping_cache_service import scraping_cache_service
from .url_utils import log_domain_safely

logger = logging.getLogger(__name__)


class CachedDocumentProcessor:
    """
    Wrapper for document processing with intelligent caching.

    Provides caching for:
    - PDF text extraction (very expensive - 2-10 seconds)
    - JSON text extraction (moderate - 0.5-2 seconds)
    - URL scraping (already handled by CachedWebScraper)
    """

    @staticmethod
    async def extract_pdf_with_cache(
        extract_function,
        url: str,
        org_id: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> str:
        """
        Extract PDF with caching wrapper.

        Args:
            extract_function: The actual PDF extraction function
            url: PDF URL
            org_id: Organization ID for multi-tenant caching
            bypass_cache: Force fresh extraction

        Returns:
            Extracted text

        Performance:
        - Cache hit: ~10ms
        - Cache miss: ~2000-10000ms (PDF processing is SLOW!)
        """
        start_time = time.time()

        # Try cache first
        if not bypass_cache:
            try:
                cached_result = await scraping_cache_service.get_cached_content(
                    url=url, content_type="pdf", org_id=org_id
                )

                if cached_result:
                    cache_time_ms = (time.time() - start_time) * 1000

                    logger.info(
                        "PDF cache HIT - Retrieved in %.2fms (saved %.2fms)",
                        cache_time_ms,
                        cached_result.scrape_time_ms,
                        extra={
                            "url": log_domain_safely(url),
                            "cache_hit": True,
                            "cache_time_ms": cache_time_ms,
                            "extraction_time_saved_ms": cached_result.scrape_time_ms,
                            "content_size": cached_result.content_size,
                            "org_id": org_id,
                        },
                    )

                    return cached_result.content

            except Exception as cache_error:
                logger.warning(
                    "PDF cache retrieval failed, falling back to fresh extraction: %s",
                    cache_error,
                    extra={"url": log_domain_safely(url)},
                )

        # Cache miss - perform actual extraction
        logger.info(
            "PDF cache MISS - Performing extraction",
            extra={"url": log_domain_safely(url), "cache_hit": False, "org_id": org_id},
        )

        extraction_start = time.time()
        text = extract_function(url)  # Call the original function
        extraction_time_ms = (time.time() - extraction_start) * 1000

        # Cache the result
        if not bypass_cache:
            try:
                await scraping_cache_service.cache_content(
                    url=url,
                    content=text,
                    content_type="pdf",
                    scrape_time_ms=extraction_time_ms,
                    org_id=org_id,
                    metadata={"extraction_timestamp": time.time()},
                )

                logger.info(
                    "Cached PDF extraction result",
                    extra={
                        "url": log_domain_safely(url),
                        "extraction_time_ms": extraction_time_ms,
                        "content_size": len(text),
                        "org_id": org_id,
                    },
                )

            except Exception as cache_error:
                logger.warning(
                    "Failed to cache PDF extraction: %s",
                    cache_error,
                    extra={"url": log_domain_safely(url)},
                )

        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "PDF extraction completed in %.2fms",
            total_time_ms,
            extra={
                "url": log_domain_safely(url),
                "total_time_ms": total_time_ms,
                "extraction_time_ms": extraction_time_ms,
            },
        )

        return text

    @staticmethod
    async def extract_json_with_cache(
        extract_function,
        url: str,
        org_id: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> str:
        """
        Extract JSON with caching wrapper.

        Args:
            extract_function: The actual JSON extraction function
            url: JSON URL
            org_id: Organization ID for multi-tenant caching
            bypass_cache: Force fresh extraction

        Returns:
            Extracted text

        Performance:
        - Cache hit: ~10ms
        - Cache miss: ~500-2000ms
        """
        start_time = time.time()

        # Try cache first
        if not bypass_cache:
            try:
                cached_result = await scraping_cache_service.get_cached_content(
                    url=url, content_type="json", org_id=org_id
                )

                if cached_result:
                    cache_time_ms = (time.time() - start_time) * 1000

                    logger.info(
                        "JSON cache HIT - Retrieved in %.2fms (saved %.2fms)",
                        cache_time_ms,
                        cached_result.scrape_time_ms,
                        extra={
                            "url": log_domain_safely(url),
                            "cache_hit": True,
                            "cache_time_ms": cache_time_ms,
                            "extraction_time_saved_ms": cached_result.scrape_time_ms,
                            "content_size": cached_result.content_size,
                            "org_id": org_id,
                        },
                    )

                    return cached_result.content

            except Exception as cache_error:
                logger.warning(
                    "JSON cache retrieval failed, falling back to fresh extraction: %s",
                    cache_error,
                    extra={"url": log_domain_safely(url)},
                )

        # Cache miss - perform actual extraction
        logger.info(
            "JSON cache MISS - Performing extraction",
            extra={"url": log_domain_safely(url), "cache_hit": False, "org_id": org_id},
        )

        extraction_start = time.time()
        text = extract_function(url)  # Call the original function
        extraction_time_ms = (time.time() - extraction_start) * 1000

        # Cache the result
        if not bypass_cache:
            try:
                await scraping_cache_service.cache_content(
                    url=url,
                    content=text,
                    content_type="json",
                    scrape_time_ms=extraction_time_ms,
                    org_id=org_id,
                    metadata={"extraction_timestamp": time.time()},
                )

                logger.info(
                    "Cached JSON extraction result",
                    extra={
                        "url": log_domain_safely(url),
                        "extraction_time_ms": extraction_time_ms,
                        "content_size": len(text),
                        "org_id": org_id,
                    },
                )

            except Exception as cache_error:
                logger.warning(
                    "Failed to cache JSON extraction: %s",
                    cache_error,
                    extra={"url": log_domain_safely(url)},
                )

        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "JSON extraction completed in %.2fms",
            total_time_ms,
            extra={
                "url": log_domain_safely(url),
                "total_time_ms": total_time_ms,
                "extraction_time_ms": extraction_time_ms,
            },
        )

        return text


# Convenience decorator for easy integration
def with_caching(content_type: str):
    """
    Decorator to add caching to extraction functions.

    Usage:
        ```python
        @with_caching("pdf")
        def extract_pdf_text(url: str) -> str:
            # expensive PDF extraction
            return text
        ```
    """

    def decorator(func):
        async def wrapper(url: str, org_id: Optional[str] = None, **kwargs):
            bypass_cache = kwargs.pop("bypass_cache", False)

            if content_type == "pdf":
                return await CachedDocumentProcessor.extract_pdf_with_cache(
                    lambda u: func(u, **kwargs), url, org_id, bypass_cache
                )
            elif content_type == "json":
                return await CachedDocumentProcessor.extract_json_with_cache(
                    lambda u: func(u, **kwargs), url, org_id, bypass_cache
                )
            else:
                # No caching for unknown types
                return func(url, **kwargs)

        return wrapper

    return decorator
