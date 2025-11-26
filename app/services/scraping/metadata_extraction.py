"""
Metadata Extraction Module
Extracts metadata flags, product links, and other metadata from chunks
"""

import logging
import re
from urllib.parse import urljoin, urlparse

from ..chat.contact_extractor import contact_extractor

logger = logging.getLogger(__name__)

# Metadata detection patterns
PRICING_PATTERN = re.compile(
    r"(\$|€|£|aed|usd|price|pricing|cost|plans|tier|package|per\s+(?:month|year)|monthly|yearly)",
    re.IGNORECASE,
)
BOOKING_PATTERN = re.compile(
    r"(book|booking|schedule|consultation|demo|trial|appointment|talk to sales|contact sales)",
    re.IGNORECASE,
)


def extract_product_links_from_chunk(chunk: str, source_url: str = None) -> list:
    """Extract potential product links from text chunk"""
    product_links = []

    # Common patterns for product URLs
    product_patterns = [
        r"https?://[^\s]+/products?/[^\s]+",
        r"https?://[^\s]+/product/[^\s]+",
        r"https?://[^\s]+/p/[^\s]+",
        r"https?://[^\s]+/item/[^\s]+",
        r"https?://[^\s]+/shop/[^\s]+",
    ]

    # Extract URLs from text
    url_pattern = r"https?://[^\s<>\"']+"
    urls = re.findall(url_pattern, chunk)

    for url in urls:
        # Check if URL matches product patterns
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in product_patterns):
            product_links.append(url)
        # If source_url is provided, check for relative links that might be products
        elif source_url and not url.startswith("http"):
            try:
                absolute_url = urljoin(source_url, url)
                if any(
                    re.search(pattern, absolute_url, re.IGNORECASE)
                    for pattern in product_patterns
                ):
                    product_links.append(absolute_url)
            except:
                pass

    # Also check for product mentions without explicit URLs
    product_mentions = re.findall(
        r"(product|item|merchandise|goods)\s*:?\s*([A-Z][a-zA-Z\s]+)",
        chunk,
        re.IGNORECASE,
    )
    if product_mentions and source_url and not product_links:
        # If we found product mentions but no links, try to construct URLs
        domain = urlparse(source_url).netloc
        for mention_type, product_name in product_mentions[:3]:  # Limit to 3
            # Try common product URL patterns
            product_slug = product_name.lower().replace(" ", "-")
            potential_urls = [
                f"https://{domain}/products/{product_slug}",
                f"https://{domain}/product/{product_slug}",
                f"https://{domain}/shop/{product_slug}",
            ]
            product_links.extend(potential_urls[:1])  # Add first match only

    return list(set(product_links))  # Remove duplicates


def extract_metadata_flags(chunk: str) -> dict:
    """Detect key metadata flags (pricing, booking, contact info) for a chunk."""
    flags = {
        "has_pricing": False,
        "has_booking": False,
        "has_contact_info": False,
        "contact_has_phone": False,
        "contact_has_email": False,
        "contact_has_address": False,
    }

    if not chunk or len(chunk) < 5:
        return flags

    if PRICING_PATTERN.search(chunk):
        flags["has_pricing"] = True

    if BOOKING_PATTERN.search(chunk):
        flags["has_booking"] = True

    try:
        contact_info = contact_extractor.extract_contact_info(chunk)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Contact extraction failed for metadata flags: %s", exc)
        return flags

    has_phone = bool(contact_info.get("phones"))
    has_email = bool(contact_info.get("emails"))
    has_address = bool(contact_info.get("addresses"))

    if has_phone or has_email or has_address:
        flags["has_contact_info"] = True
        flags["contact_has_phone"] = has_phone
        flags["contact_has_email"] = has_email
        flags["contact_has_address"] = has_address

    return flags
