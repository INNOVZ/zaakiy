"""
Topic Extraction Module
Extracts topic keywords from URLs for intent-based filtering
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def extract_topics_from_url(url: str) -> list:
    """
    Extract topic keywords from URL path for intent-based filtering.

    This is TENANT-AGNOSTIC - works for any URL structure.

    Examples:
        https://ohhzones.com/digital-marketing/seo/
        → ["digital-marketing", "seo", "digital marketing", "seo"]

        https://ohhzones.com/branding-services/brand-identity/
        → ["branding-services", "brand-identity", "branding services", "brand identity"]

        https://ambassadorscentworks.com/collections/essential-series
        → ["collections", "essential-series", "essential series"]
    """
    if not url:
        return []

    topics = []

    try:
        # Parse URL
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return []

        # Split path into segments
        segments = [s for s in path.split("/") if s]

        # Extract topics from each segment
        stop_words = {
            "the",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "a",
            "an",
            "www",
            "http",
            "https",
            "com",
            "net",
            "org",
            "io",
            "co",
        }

        for segment in segments:
            # Skip very short segments
            if len(segment) < 3:
                continue

            # Skip numbers
            if segment.isdigit():
                continue

            segment_lower = segment.lower()

            # Skip stop words
            if segment_lower in stop_words:
                continue

            # Add original segment (with hyphens)
            topics.append(segment_lower)

            # Also add space-separated version for matching
            if "-" in segment or "_" in segment:
                # Convert hyphens/underscores to spaces
                space_version = segment.replace("-", " ").replace("_", " ").lower()
                topics.append(space_version)

        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        logger.debug(f"Extracted topics from URL: {url} → {unique_topics}")
        return unique_topics

    except Exception as e:
        logger.warning(f"Failed to extract topics from URL {url}: {e}")
        return []
