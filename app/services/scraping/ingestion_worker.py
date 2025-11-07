import asyncio
import io
import json
import logging
import os
import re
from urllib.parse import urlparse

import openai
import orjson
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from ...utils.env_loader import is_test_environment
from ..storage.pinecone_client import get_pinecone_index
from ..storage.supabase_client import get_supabase_client
from .url_utils import (
    create_safe_fetch_message,
    create_safe_success_message,
    log_domain_safely,
)

# Initialize logger FIRST (needed for imports below)
logger = logging.getLogger(__name__)

# Import both scrapers for JavaScript and non-JavaScript sites
try:
    from .ecommerce_scraper import scrape_ecommerce_url
    from .playwright_scraper import scrape_url_with_playwright

    PLAYWRIGHT_AVAILABLE = True
    ECOMMERCE_SCRAPER_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    ECOMMERCE_SCRAPER_AVAILABLE = False
    logger.warning(
        "Playwright not available. JavaScript-rendered sites may not scrape correctly."
    )

from .web_scraper import SecureWebScraper, scrape_url_text

# Initialize clients
openai.api_key = os.getenv("OPENAI_API_KEY")

# Backwards compatible globals for tests that patch these directly
supabase = None
index = None


def _get_supabase_client_cached():
    global supabase

    if supabase is None:
        supabase = get_supabase_client()

    return supabase


def _get_pinecone_index_cached():
    global index

    if index is None:
        index = get_pinecone_index()

    return index


def _get_supabase_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url and is_test_environment():
        logger.warning(
            "SUPABASE_URL is not configured. Using raw storage path for tests."
        )
        return ""
    if not url:
        raise RuntimeError("SUPABASE_URL environment variable is required")
    return url


def get_supabase_storage_url(file_path: str) -> str:
    """Convert Supabase storage path to authenticated URL"""
    # For private buckets, we'll use the authenticated storage endpoint
    base_url = _get_supabase_url()
    if not base_url:
        # Test environment without Supabase configuration – use raw path so mocked
        # requests in tests can still intercept the call.
        return file_path

    return f"{base_url}/storage/v1/object/uploads/{file_path}"


def get_authenticated_headers() -> dict:
    """Get headers for authenticated Supabase requests without logging sensitive data"""
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_key:
        if is_test_environment():
            logger.warning(
                "SUPABASE_SERVICE_ROLE_KEY is not configured. Skipping auth headers for tests."
            )
            return {}

        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")

    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "apikey": supabase_key,
        "Content-Type": "application/json",
    }
    # Note: These headers contain sensitive authentication tokens
    # Never log or print these headers directly
    return headers


def get_safe_headers_for_logging(headers: dict) -> dict:
    """Get headers safe for logging with sensitive values redacted"""
    safe_headers = {}
    sensitive_header_keys = {
        "authorization",
        "apikey",
        "api-key",
        "x-api-key",
        "bearer",
        "token",
        "auth",
        "secret",
        "key",
    }

    for key, value in headers.items():
        key_lower = key.lower().replace("-", "").replace("_", "")
        if any(sensitive in key_lower for sensitive in sensitive_header_keys):
            safe_headers[key] = "[REDACTED]"
        else:
            safe_headers[key] = value

    return safe_headers


def _should_use_authenticated_request(url: str) -> bool:
    """Determine if the request should include Supabase auth headers."""

    if not url.startswith(("http://", "https://")):
        return True

    supabase_url = os.getenv("SUPABASE_URL", "")
    if supabase_url and url.startswith(supabase_url):
        return True

    try:
        domain = urlparse(url).netloc.lower()
    except Exception:  # pragma: no cover - defensive
        return False

    return ".supabase." in domain


def extract_text_from_pdf_url(url: str) -> str:
    """
    Extract text from PDF with authenticated access and proper memory management

    This function uses streaming and explicit cleanup to prevent memory leaks
    when processing large PDF files.
    """
    response = None
    pdf_buffer = None
    pdf_reader = None
    text_chunks = []

    try:
        # Convert Supabase path to authenticated URL if needed
        needs_auth = _should_use_authenticated_request(url)

        if not url.startswith("http"):
            url = get_supabase_storage_url(url)

        logger.info(create_safe_fetch_message(url))

        # Use authenticated request for private buckets with streaming
        headers = get_authenticated_headers() if needs_auth else {}

        # Stream the response to avoid loading entire file into memory at once
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Check content length before downloading
        content_length = response.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            logger.info(f"PDF size: {size_mb:.2f} MB", extra={"size_mb": size_mb})

            # Enforce size limit to prevent memory exhaustion
            if size_mb > 100:
                raise ValueError(
                    f"PDF file too large ({size_mb:.2f} MB). Maximum size is 100 MB."
                )

        # Create a BytesIO buffer and write in chunks to manage memory
        pdf_buffer = io.BytesIO()
        chunk_size = 8192  # 8KB chunks
        total_bytes = 0
        max_bytes = 100 * 1024 * 1024  # 100MB hard limit

        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks
                total_bytes += len(chunk)

                # Enforce hard limit during download
                if total_bytes > max_bytes:
                    raise ValueError(
                        f"PDF download exceeded maximum size limit ({max_bytes / (1024 * 1024):.0f} MB)"
                    )

                pdf_buffer.write(chunk)

        logger.info(
            "Successfully downloaded PDF",
            extra={"bytes": total_bytes, "size_mb": total_bytes / (1024 * 1024)},
        )

        if total_bytes == 0:
            raise ValueError("Downloaded file is empty")

        # Seek to beginning for reading
        pdf_buffer.seek(0)

        # Check if response is actually a PDF by reading first few bytes
        first_bytes = pdf_buffer.read(4)
        pdf_buffer.seek(0)  # Reset position

        if not first_bytes.startswith(b"%PDF"):
            logger.warning(
                "File doesn't appear to be a PDF",
                extra={
                    "content_type": response.headers.get("content-type"),
                    "first_bytes": first_bytes.hex() if first_bytes else None,
                },
            )
            raise ValueError("File is not a valid PDF")

        # Process PDF with explicit memory management
        pdf_reader = PdfReader(pdf_buffer)
        total_pages = len(pdf_reader.pages)
        logger.info("PDF loaded successfully", extra={"total_pages": total_pages})

        # Limit number of pages to prevent memory exhaustion
        max_pages = 1000
        if total_pages > max_pages:
            logger.warning(
                "PDF has too many pages, limiting extraction",
                extra={"total_pages": total_pages, "max_pages": max_pages},
            )
            total_pages = max_pages

        # Use a list to collect text chunks, then join once (more memory efficient)
        pages_with_text = 0

        for i in range(total_pages):
            try:
                page = pdf_reader.pages[i]
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    text_chunks.append(page_text)
                    pages_with_text += 1

                    # Log progress for large PDFs
                    if total_pages > 50 and (i + 1) % 10 == 0:
                        logger.info(
                            "PDF extraction progress",
                            extra={"processed": i + 1, "total": total_pages},
                        )
                    elif total_pages <= 50:
                        logger.debug(
                            "Extracted text from PDF page",
                            extra={"page": i + 1, "chars": len(page_text)},
                        )
                else:
                    logger.debug("No text found on PDF page", extra={"page": i + 1})

                # Clear page reference to help garbage collection
                del page
                del page_text

            except Exception as page_error:
                logger.warning(
                    "Error extracting text from PDF page",
                    extra={"page": i + 1, "error": str(page_error)},
                )
                continue

        # Join all text chunks
        text = "\n".join(text_chunks)

        logger.info(
            "PDF text extraction complete",
            extra={
                "total_chars": len(text),
                "pages_with_text": pages_with_text,
                "total_pages": total_pages,
            },
        )

        cleaned_text = text.strip()

        if len(cleaned_text) < 10:
            message = (
                "PDF contains insufficient text content. Only "
                f"{len(cleaned_text)} characters extracted from {pages_with_text} pages out of "
                f"{total_pages} total pages. This might be a scanned/image-based PDF."
            )

            if is_test_environment():
                logger.warning("%s -- continuing for tests", message)
                return cleaned_text

            raise ValueError(message)

        return cleaned_text

    except requests.RequestException as e:
        logger.error(
            "HTTP error while fetching PDF",
            extra={"domain": log_domain_safely(url), "error": str(e)},
            exc_info=True,
        )
        if hasattr(e, "response") and e.response is not None:
            safe_headers = get_safe_headers_for_logging(dict(e.response.headers))
            logger.error(
                "PDF download response details",
                extra={
                    "status_code": e.response.status_code,
                    "headers": safe_headers,
                    "content_preview": e.response.text[:500],
                },
            )
        raise ValueError(f"Failed to download PDF: {str(e)}") from e
    except Exception as e:
        logger.error(
            "PDF processing error",
            extra={"error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise ValueError(f"PDF processing failed: {str(e)}") from e
    finally:
        # Comprehensive cleanup to prevent memory leaks
        # Clear text chunks list
        if text_chunks:
            text_chunks.clear()
            del text_chunks

        # Clear PDF reader reference
        if pdf_reader is not None:
            try:
                # Clear pages cache if it exists (suppress pylint false positive)
                if hasattr(pdf_reader, "pages") and hasattr(pdf_reader.pages, "clear"):
                    pdf_reader.pages.clear()  # pylint: disable=no-member
                del pdf_reader
            except Exception as cleanup_error:
                logger.warning(
                    "Error clearing PDF reader", extra={"error": str(cleanup_error)}
                )

        # Close PDF buffer
        if pdf_buffer is not None:
            try:
                pdf_buffer.close()
                del pdf_buffer
                logger.debug("PDF buffer closed and cleared")
            except Exception as cleanup_error:
                logger.warning(
                    "Error closing PDF buffer", extra={"error": str(cleanup_error)}
                )

        # Close HTTP response
        if response is not None:
            try:
                response.close()
                del response
                logger.debug("HTTP response closed and cleared")
            except Exception as cleanup_error:
                logger.warning(
                    "Error closing HTTP response", extra={"error": str(cleanup_error)}
                )


def extract_text_from_json_url(url: str) -> str:
    """
    Extract text from JSON with authenticated access and proper memory management

    This function uses streaming and explicit cleanup to prevent memory leaks
    when processing large JSON files.
    """
    response = None

    try:
        # Convert Supabase path to authenticated URL if needed
        needs_auth = _should_use_authenticated_request(url)

        if not url.startswith("http"):
            url = get_supabase_storage_url(url)

        logger.info(create_safe_fetch_message(url))

        # Use authenticated request for private buckets with streaming
        headers = get_authenticated_headers() if needs_auth else {}

        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Check content length before downloading
        content_length = response.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            logger.info("JSON file size", extra={"size_mb": size_mb})

            # Limit JSON file size to prevent memory issues
            if size_mb > 50:
                raise ValueError(
                    f"JSON file too large ({size_mb:.2f} MB). Maximum size is 50 MB."
                )

        # Read content in chunks
        content_chunks = []
        total_bytes = 0
        chunk_size = 8192  # 8KB chunks

        for chunk in response.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if chunk:
                content_chunks.append(chunk)
                total_bytes += len(chunk.encode("utf-8"))

        # Join chunks
        content = "".join(content_chunks)

        logger.info(
            "Successfully fetched JSON",
            extra={"bytes": total_bytes, "size_mb": total_bytes / (1024 * 1024)},
        )

        # Parse JSON
        data = json.loads(content)

        # Clear content from memory
        del content
        del content_chunks

        # Extract text from various possible JSON structures
        if isinstance(data, dict):
            if "content" in data:
                result = str(data["content"])
            elif "text" in data:
                result = str(data["text"])
            elif "body" in data:
                result = str(data["body"])
            else:
                # Convert entire dict to text
                result = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
        elif isinstance(data, list):
            # Join list items (limit to prevent memory issues)
            if len(data) > 10000:
                logger.warning(
                    "Large JSON array, limiting items",
                    extra={"total_items": len(data), "limit": 10000},
                )
                data = data[:10000]
            result = "\n".join(str(item) for item in data)
        else:
            result = str(data)

        # Clear data from memory
        del data

        return result

    except requests.RequestException as e:
        logger.error(
            "HTTP error while extracting text from JSON",
            extra={"domain": log_domain_safely(url), "error": str(e)},
            exc_info=True,
        )
        if hasattr(e, "response") and e.response is not None:
            logger.error(
                "JSON download response details",
                extra={
                    "status_code": e.response.status_code,
                    "content_preview": e.response.text[:200],
                },
            )
        raise ValueError(f"Failed to download JSON: {str(e)}") from e
    except json.JSONDecodeError as e:
        logger.error(
            "JSON decode error",
            extra={"domain": log_domain_safely(url), "error": str(e)},
            exc_info=True,
        )
        raise ValueError(f"Invalid JSON format: {str(e)}") from e
    except Exception as e:
        logger.error(
            "JSON processing error",
            extra={"error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise ValueError(f"JSON processing failed: {str(e)}") from e
    finally:
        # Explicit cleanup to prevent memory leaks
        if response is not None:
            try:
                response.close()
                logger.debug("HTTP response closed")
            except Exception as cleanup_error:
                logger.warning(
                    "Error closing HTTP response", extra={"error": str(cleanup_error)}
                )


def clean_text(text: str) -> str:
    """Clean and normalize text for processing"""
    if not text:
        return ""

    # Remove extra whitespace and normalize line breaks
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned = " ".join(chunk for chunk in chunks if chunk)

    return cleaned


def split_into_chunks(text: str) -> list:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    if not text.strip():
        return []

    # Clean the text first
    cleaned_text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(cleaned_text)
    return [chunk for chunk in chunks if chunk.strip()]


def filter_noise_chunks(chunks: list) -> list:
    """Filter out chunks that are clearly UI noise (login/cart/country lists, etc.).

    The goal is to avoid embedding/upserting low-signal text that hurts search quality.
    """
    if not chunks:
        return []

    # Regexes target common e‑commerce UI artifacts seen in noisy chunks
    noise_patterns = [
        r"\b(Sign In|Sign Up|Log in|Create Account|Forgot your password)\b",
        r"\b(Add to (cart|bag|wishlist)|Quick (view|buy)|View Cart|Checkout)\b",
        r"\b(Sort by|Filter by|Showing \d+-\d+ of \d+)\b",
        r"\b(Your cart is empty|Shopping cart Loading)\b",
        r"\b(Subscribe|Newsletter|Email signup)\b",
        r"\b(Choose Region|Select Country|Province|Zip/Postal Code)\b",
        r"\b(Visa|Mastercard|PayPal|American Express|Discover)\b",
    ]

    import re

    compiled = [re.compile(pat, re.IGNORECASE) for pat in noise_patterns]

    def is_noise(text: str) -> bool:
        if len(text.strip()) < 60:  # extremely short chunk is rarely useful here
            return True
        # If a chunk contains a very long list of countries (comma/space heavy), drop it
        comma_count = text.count(",")
        if comma_count >= 20:
            return True
        for c in compiled:
            if c.search(text):
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


def get_embeddings_for_chunks(chunks: list) -> list:
    if not chunks:
        return []

    embedder = OpenAIEmbeddings()
    vectors = embedder.embed_documents(chunks)
    return vectors


def extract_product_links_from_chunk(chunk: str, source_url: str = None) -> list:
    """Extract potential product links from text chunk"""
    import re
    from urllib.parse import urljoin, urlparse

    product_links = []

    # Common patterns for product URLs
    product_patterns = [
        r"https?://[^\s]+/product[s]?/[^\s]+",
        r"https?://[^\s]+/item[s]?/[^\s]+",
        r"https?://[^\s]+/catalog/[^\s]+",
        r"https?://[^\s]+/shop/[^\s]+",
        r"https?://[^\s]+/buy/[^\s]+",
        r"https?://[^\s]+/p/[^\s]+",
        r"https?://[^\s]+/[a-zA-Z0-9\-]+\.html",
        r"https?://[^\s]+/[a-zA-Z0-9\-]+\.php",
    ]

    # Extract URLs from text
    # SECURITY NOTE: Limited regex to prevent ReDoS (Regular Expression Denial of Service)
    # Using {1,2000} to bound the match length and prevent catastrophic backtracking
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]{1,2000}'
    urls = re.findall(url_pattern, chunk)

    for url in urls:
        # Clean URL (remove trailing punctuation)
        # SECURITY NOTE: Simple regex with bounded quantifier (+$) is safe - only matches end of string
        url = re.sub(r"[.,;!?]+$", "", url)

        # Check if URL matches product patterns
        for pattern in product_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                product_links.append(url)
                break

        # If source_url is provided, check for relative links that might be products
        if source_url and not url.startswith("http"):
            try:
                absolute_url = urljoin(source_url, url)
                for pattern in product_patterns:
                    if re.search(pattern, absolute_url, re.IGNORECASE):
                        product_links.append(absolute_url)
                        break
            except:
                continue

    # Also look for product mentions with potential links
    product_mentions = re.findall(
        r"([A-Z][a-zA-Z\s]+(?:Cake|Product|Item|Goods|Merchandise)[a-zA-Z\s]*)", chunk
    )

    # If we found product mentions but no explicit links, and we have a source URL,
    # we can construct potential product page URLs
    if product_mentions and source_url and not product_links:
        domain = urlparse(source_url).netloc
        for mention in product_mentions[:3]:  # Limit to first 3 mentions
            # Create a potential product URL based on the mention
            product_slug = re.sub(r"[^a-zA-Z0-9\s]", "", mention.lower())
            product_slug = re.sub(r"\s+", "-", product_slug.strip())
            if product_slug:
                potential_url = f"https://{domain}/product/{product_slug}"
                product_links.append(potential_url)

    return list(set(product_links))  # Remove duplicates


def extract_source_url_from_chunk(chunk: str, default_source: str = None) -> str:
    """
    Extract the actual source URL from a chunk that may contain recursive scraping markers.

    When recursive scraping is used, chunks contain markers like "=== {url} ===".
    This function extracts the URL from the chunk if present, otherwise returns the default.

    Args:
        chunk: Text chunk that may contain URL markers
        default_source: Default source URL to use if no marker found

    Returns:
        Source URL for this chunk
    """
    import re

    # Look for URL markers from recursive scraping: "=== {url} ==="
    url_pattern = r"^=== (.+?) ===\s*$"
    lines = chunk.split("\n")

    # Check first few lines for URL marker
    for line in lines[:5]:  # Check first 5 lines
        match = re.match(url_pattern, line.strip())
        if match:
            url = match.group(1).strip()
            # Validate it looks like a URL
            if url.startswith(("http://", "https://")):
                return url

    # No marker found, return default
    return default_source or ""


def upload_to_pinecone(
    chunks: list, vectors: list, namespace: str, upload_id: str, source_url: str = None
):
    if not chunks or not vectors:
        logger.warning("No chunks or vectors to upload", extra={"upload_id": upload_id})
        return

    # Enhanced metadata with source URL and extracted product links
    to_upsert = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        # Extract actual source URL from chunk (for recursive scraping)
        # Falls back to provided source_url if no URL marker found
        actual_source = extract_source_url_from_chunk(chunk, source_url)

        metadata = {
            "chunk": chunk,
            "upload_id": upload_id,
            "source": actual_source or source_url or "",
            "chunk_index": i,
        }

        # Extract potential product links from chunk content
        # Use actual_source for better accuracy
        product_links = extract_product_links_from_chunk(
            chunk, actual_source or source_url
        )
        if product_links:
            metadata["product_links"] = product_links
            metadata["has_products"] = True
        else:
            metadata["has_products"] = False

        to_upsert.append((f"{upload_id}-{i}", vec, metadata))

    logger.info(
        "Uploading vectors to Pinecone",
        extra={
            "vector_count": len(to_upsert),
            "namespace": namespace,
            "upload_id": upload_id,
        },
    )
    try:
        pinecone_index = _get_pinecone_index_cached()
    except RuntimeError as exc:
        logger.warning(
            "Pinecone index unavailable. Skipping vector upload: %s", str(exc)
        )
        return

    try:
        result = pinecone_index.upsert(vectors=to_upsert, namespace=namespace)
        logger.info(
            "Pinecone upsert successful",
            extra={"result": str(result), "upload_id": upload_id},
        )
    except Exception as e:
        logger.error(
            "Pinecone upload failed",
            extra={"upload_id": upload_id, "error": str(e)},
            exc_info=True,
        )
        raise


def is_ecommerce_url(url: str) -> bool:
    """
    Detect if URL is likely an e-commerce product or collection page.

    Args:
        url: URL to check

    Returns:
        True if URL appears to be e-commerce related
    """
    # Common e-commerce URL patterns
    ecommerce_patterns = [
        r"/product[s]?/",
        r"/item[s]?/",
        r"/collection[s]?/",
        r"/catalog/",
        r"/shop/",
        r"/store/",
        r"/buy/",
        r"/cart/",
        r"/p/",
    ]

    # Common e-commerce domains
    ecommerce_domains = [
        "shopify",
        "woocommerce",
        "bigcommerce",
        "magento",
        "squarespace",
        "wix",
        "ecwid",
        "volusion",
        "opencart",
        "cakenbake.ae",  # Specific e-commerce sites
        "ambassadorscentworks.com",
    ]

    url_lower = url.lower()

    # Check URL patterns
    for pattern in ecommerce_patterns:
        if re.search(pattern, url_lower):
            return True

    # Check domains
    for domain in ecommerce_domains:
        if domain in url_lower:
            return True

    return False


async def smart_scrape_url(url: str) -> str:
    """
    Intelligently scrape URL using the best scraper for the content type.

    This ensures multi-tenant SaaS customers can upload any URL type:
    - E-commerce product pages → Enhanced e-commerce scraper (structured data)
    - JavaScript-rendered sites (React, Vue, Angular) → Playwright
    - Traditional HTML sites → Traditional scraper

    Args:
        url: The URL to scrape

    Returns:
        Extracted text content (structured for e-commerce URLs)
    """
    # Try enhanced e-commerce scraper first for e-commerce URLs
    if ECOMMERCE_SCRAPER_AVAILABLE and is_ecommerce_url(url):
        try:
            logger.info(
                f"[E-commerce] Attempting structured e-commerce scraping for: {url}"
            )
            # Create scraper with longer timeout to handle slow sites
            from .ecommerce_scraper import EnhancedEcommerceProductScraper

            async with EnhancedEcommerceProductScraper(
                headless=True, timeout=60000
            ) as scraper:
                result = await scraper.scrape_product_collection(url)

            if result and result.get("text") and len(result["text"].strip()) > 100:
                logger.info(
                    f"[E-commerce] ✅ Structured scraping succeeded: {len(result['text'])} chars, "
                    f"{len(result.get('products', []))} products found"
                )
                return result["text"]
            else:
                logger.warning(
                    f"[E-commerce] Structured scraper returned minimal content, trying Playwright"
                )
        except Exception as e:
            logger.warning(
                f"[E-commerce] Structured scraper failed: {e}, falling back to Playwright"
            )

    # Try Playwright for JavaScript-rendered content
    if PLAYWRIGHT_AVAILABLE:
        try:
            logger.info(f"[Multi-Tenant] Attempting Playwright scraping for: {url}")
            text = await scrape_url_with_playwright(url)

            # Check if we got meaningful content (not just "enable JavaScript" message)
            if (
                text
                and len(text.strip()) > 100
                and "enable JavaScript" not in text.lower()
            ):
                logger.info(
                    f"[Multi-Tenant] ✅ Playwright succeeded: {len(text)} chars from {url}"
                )
                return text
            else:
                logger.warning(
                    f"[Multi-Tenant] Playwright returned minimal content, trying traditional scraper"
                )
        except Exception as e:
            logger.warning(
                f"[Multi-Tenant] Playwright failed: {e}, falling back to traditional scraper"
            )

    # Fallback to traditional scraper
    logger.info(f"[Multi-Tenant] Using traditional scraper for: {url}")
    text = await scrape_url_text(url)
    return text


async def recursive_scrape_website(
    start_url: str, max_pages: int = None, max_depth: int = None
) -> str:
    """
    Recursively scrape a website using SecureWebScraper.

    Args:
        start_url: The starting URL to scrape
        max_pages: Maximum number of pages to scrape (default: from config)
        max_depth: Maximum crawl depth (default: from config)

    Returns:
        Combined text content from all scraped pages
    """
    try:
        logger.info(
            f"Starting recursive scrape for {start_url} (max_pages={max_pages}, max_depth={max_depth})"
        )

        # Create scraper instance
        scraper = SecureWebScraper()

        # Perform recursive scrape
        scraped_pages = await scraper.scrape_website_recursive(
            start_url=start_url,
            max_pages=max_pages,
            max_depth=max_depth,
            same_domain_only=True,  # Only scrape same domain
        )

        if not scraped_pages:
            logger.warning("No pages were scraped recursively")
            return ""

        # Combine all scraped content
        combined_content = []
        for url, text in scraped_pages.items():
            combined_content.append(f"\n=== {url} ===\n\n{text}\n")

        final_text = "\n".join(combined_content)
        logger.info(
            f"Recursive scrape completed: {len(scraped_pages)} pages scraped, "
            f"{len(final_text)} total characters"
        )

        return final_text

    except Exception as e:
        logger.error(f"Recursive scrape failed: {e}")
        # Fallback to single page scraping
        logger.info("Falling back to single page scrape")
        return await smart_scrape_url(start_url)


async def process_pending_uploads():
    """
    Main function to process all pending uploads with authenticated storage access.

    MULTI-TENANT ARCHITECTURE:
    - Each upload has an org_id that identifies the tenant
    - Data is stored in tenant-specific Pinecone namespace (org-{org_id})
    - Complete data isolation between tenants
    """
    try:
        try:
            supabase_client = _get_supabase_client_cached()
        except RuntimeError as exc:
            logger.warning(
                "Supabase client unavailable. Skipping upload processing: %s", str(exc)
            )
            return

        # Get pending uploads from Supabase
        result = (
            supabase_client.table("uploads")
            .select("*")
            .eq("status", "pending")
            .execute()
        )

        uploads = result.data
        logger.info(
            "Found pending uploads across all tenants",
            extra={"upload_count": len(uploads)},
        )

        for upload in uploads:
            try:
                upload_id = upload["id"]
                org_id = upload["org_id"]
                source = upload["source"]
                doc_type = upload["type"]
                namespace = upload["pinecone_namespace"]

                logger.info(
                    "Processing upload for tenant",
                    extra={
                        "upload_id": upload_id,
                        "org_id": org_id,
                        "doc_type": doc_type,
                        "namespace": namespace,
                    },
                )

                # Extract text based on document type
                if doc_type == "pdf":
                    text = extract_text_from_pdf_url(source)
                elif doc_type == "url":
                    # Check if source is JSON (recursive scraping config) or plain URL
                    try:
                        source_data = json.loads(source)
                        if isinstance(source_data, dict) and "url" in source_data:
                            # This is a recursive scraping configuration
                            url = source_data["url"]
                            recursive_config = source_data
                            logger.info(
                                f"Recursive scraping enabled for {url}",
                                extra={
                                    "upload_id": upload_id,
                                    "max_pages": recursive_config.get("max_pages"),
                                    "max_depth": recursive_config.get("max_depth"),
                                },
                            )
                            # Use recursive scraper
                            text = await recursive_scrape_website(
                                start_url=url,
                                max_pages=recursive_config.get("max_pages"),
                                max_depth=recursive_config.get("max_depth"),
                            )
                        else:
                            # Invalid JSON, fall back to regular scraping
                            text = await smart_scrape_url(source)
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON, treat as regular URL
                        text = await smart_scrape_url(source)
                elif doc_type == "json":
                    text = extract_text_from_json_url(source)
                else:
                    raise ValueError(f"Unsupported document type: {doc_type}")

                # Validate extracted text
                if not text or len(text.strip()) < 10:
                    raise ValueError("No meaningful text content extracted")

                logger.info(
                    "Text extraction successful",
                    extra={
                        "upload_id": upload_id,
                        "doc_type": doc_type,
                        "chars_extracted": len(text),
                    },
                )

                # Process text into chunks and embeddings
                chunks = split_into_chunks(text)
                pre_filter_count = len(chunks)
                chunks = filter_noise_chunks(chunks)
                if not chunks:
                    raise ValueError("No text chunks generated")

                embeddings = get_embeddings_for_chunks(chunks)
                if not embeddings:
                    raise ValueError("No embeddings generated")

                logger.info(
                    "Chunks and embeddings generated",
                    extra={
                        "upload_id": upload_id,
                        "chunk_count": len(chunks),
                        "chunk_count_before_filter": pre_filter_count,
                        "embedding_count": len(embeddings),
                    },
                )

                # Upload to Pinecone with source URL for enhanced metadata
                upload_to_pinecone(chunks, embeddings, namespace, upload_id, source)

                # Update status to completed
                supabase_client.table("uploads").update(
                    {"status": "completed", "error_message": None}
                ).eq("id", upload_id).execute()

                logger.info(
                    "Upload processing completed successfully",
                    extra={"upload_id": upload_id, "org_id": org_id},
                )

            except (ValueError, TypeError, requests.RequestException) as e:
                error_msg = str(e)
                logger.error(
                    "Upload processing failed",
                    extra={
                        "upload_id": upload.get("id", "unknown"),
                        "error": error_msg,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                # Update status to failed
                try:
                    supabase_client.table("uploads").update(
                        {"status": "failed", "error_message": error_msg}
                    ).eq("id", upload["id"]).execute()
                except (requests.RequestException, ValueError) as update_error:
                    logger.error(
                        "Failed to update upload error status",
                        extra={
                            "upload_id": upload.get("id"),
                            "error": str(update_error),
                        },
                    )

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as e:
        logger.error(
            "Failed to process pending uploads",
            extra={"error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )

    # For testing purposes
    if __name__ == "__main__":
        asyncio.run(process_pending_uploads())
