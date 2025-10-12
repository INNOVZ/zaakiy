"""
Web scraping services module

This module contains all web scraping and content ingestion services including
secure web scraping, adaptive concurrency, and content processing.
"""

from .adaptive_scraper import AdaptiveConcurrencyManager, PriorityTaskQueue
# from .ingestion_worker import IngestionWorker  # No class, just functions
from .url_utils import URLSanitizer, create_safe_fetch_message, log_domain_safely
from .web_scraper import RobotsTxtChecker, SecureWebScraper, URLSecurityValidator

__all__ = [
    "SecureWebScraper",
    "URLSecurityValidator",
    "RobotsTxtChecker",
    "AdaptiveConcurrencyManager",
    "PriorityTaskQueue",
    # "IngestionWorker",  # No class, just functions
    "URLSanitizer",
    "log_domain_safely",
    "create_safe_fetch_message",
]
