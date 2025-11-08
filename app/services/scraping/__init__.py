"""
Web scraping services module

This module contains all web scraping and content ingestion services including
secure web scraping, adaptive concurrency, intelligent caching, and content processing.
"""

from .adaptive_scraper import AdaptiveConcurrencyManager, PriorityTaskQueue
from .cached_document_processor import CachedDocumentProcessor
from .cached_web_scraper import (
    CachedWebScraper,
    create_cached_scraper,
    get_default_cached_scraper,
    scrape_url_text_cached,
)
from .content_extractors import ContactExtractor, extract_contact_information
from .scraping_cache_service import (
    ScrapingCacheConfig,
    ScrapingCacheService,
    scraping_cache_service,
)
from .text_cleaner import TextCleaner, clean_text, remove_ecommerce_noise
from .url_utils import (
    URLSanitizer,
    create_safe_fetch_message,
    is_ecommerce_url,
    log_domain_safely,
)
from .web_scraper import RobotsTxtChecker, SecureWebScraper, URLSecurityValidator

# from .ingestion_worker import IngestionWorker  # No class, just functions

__all__ = [
    # Core scrapers
    "SecureWebScraper",
    "CachedWebScraper",
    "create_cached_scraper",
    "get_default_cached_scraper",
    # Security
    "URLSecurityValidator",
    "RobotsTxtChecker",
    # Adaptive/Advanced
    "AdaptiveConcurrencyManager",
    "PriorityTaskQueue",
    # Caching
    "ScrapingCacheService",
    "ScrapingCacheConfig",
    "scraping_cache_service",
    "CachedDocumentProcessor",
    "scrape_url_text_cached",
    # Content Extraction
    "ContactExtractor",
    "extract_contact_information",
    # Text Cleaning
    "TextCleaner",
    "clean_text",
    "remove_ecommerce_noise",
    # Utilities
    # "IngestionWorker",  # No class, just functions
    "URLSanitizer",
    "log_domain_safely",
    "create_safe_fetch_message",
    "is_ecommerce_url",
]
