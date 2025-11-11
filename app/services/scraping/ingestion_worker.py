import asyncio
import io
import json
import logging
import os
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import openai

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    orjson = None  # type: ignore
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from ...utils.env_loader import is_test_environment
from ..storage.pinecone_client import get_pinecone_index
from ..storage.supabase_client import get_supabase_client
from .text_cleaner import TextCleaner
from .url_utils import create_safe_fetch_message, is_ecommerce_url, log_domain_safely
from .web_scraper import SecureWebScraper

# Initialize logger FIRST (needed for imports below)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_UPLOADS = int(os.getenv("SCRAPER_MAX_CONCURRENCY", "3"))
MAX_UPLOAD_BATCH = int(os.getenv("SCRAPER_MAX_UPLOAD_BATCH", "25"))

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


def _safe_json_dumps(data: Any) -> str:
    """
    Serialize JSON content using orjson when available, falling back to the
    stdlib json module otherwise. This avoids mypy/pylint attr errors when the
    orjson stubs are missing.
    """
    if orjson is not None and hasattr(orjson, "dumps"):
        indent_option = getattr(orjson, "OPT_INDENT_2", 2)
        return orjson.dumps(data, option=indent_option).decode(
            "utf-8"
        )  # pylint: disable=no-member
    return json.dumps(data, indent=2)


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


async def extract_text_from_pdf_url(url: str) -> str:
    """
    Extract text from PDF with authenticated access and proper memory management.

    This function is async and uses asyncio.to_thread() for blocking I/O operations
    to prevent blocking the event loop during PDF processing (2-10 seconds).

    This function uses streaming and explicit cleanup to prevent memory leaks
    when processing large PDF files.
    """

    def _download_pdf() -> tuple:
        """Synchronous PDF download function to run in thread"""
        # Convert Supabase path to authenticated URL if needed
        needs_auth = _should_use_authenticated_request(url)
        download_url = url

        if not download_url.startswith("http"):
            download_url = get_supabase_storage_url(download_url)

        logger.info(create_safe_fetch_message(download_url))

        # Use authenticated request for private buckets with streaming
        headers = get_authenticated_headers() if needs_auth else {}

        # Stream the response to avoid loading entire file into memory at once
        response = requests.get(download_url, headers=headers, timeout=30, stream=True)
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

        return pdf_buffer, response

    def _process_pdf(pdf_buffer: io.BytesIO) -> str:
        """Synchronous PDF processing function to run in thread"""
        pdf_reader = None
        text_chunks = []

        try:
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
                    if hasattr(pdf_reader, "pages") and hasattr(
                        pdf_reader.pages, "clear"
                    ):
                        pdf_reader.pages.clear()  # pylint: disable=no-member
                    del pdf_reader
                except Exception as cleanup_error:
                    logger.warning(
                        "Error clearing PDF reader", extra={"error": str(cleanup_error)}
                    )

    # Run blocking I/O operations in thread pool to avoid blocking event loop
    try:
        # Download PDF in thread (blocking HTTP + file I/O)
        pdf_buffer, response = await asyncio.to_thread(_download_pdf)

        try:
            # Process PDF in thread (blocking CPU operations)
            result = await asyncio.to_thread(_process_pdf, pdf_buffer)
            return result
        finally:
            # Cleanup PDF buffer
            if pdf_buffer is not None:
                try:
                    pdf_buffer.close()
                    logger.debug("PDF buffer closed and cleared")
                except Exception as cleanup_error:
                    logger.warning(
                        "Error closing PDF buffer", extra={"error": str(cleanup_error)}
                    )

            # Close HTTP response
            if response is not None:
                try:
                    response.close()
                    logger.debug("HTTP response closed and cleared")
                except Exception as cleanup_error:
                    logger.warning(
                        "Error closing HTTP response",
                        extra={"error": str(cleanup_error)},
                    )

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


async def extract_text_from_json_url(url: str) -> str:
    """
    Extract text from JSON with authenticated access and proper memory management.

    This function is async and uses asyncio.to_thread() for blocking I/O operations
    to prevent blocking the event loop during JSON processing.

    This function uses streaming and explicit cleanup to prevent memory leaks
    when processing large JSON files.
    """

    def _download_and_process_json() -> str:
        """Synchronous JSON download and processing function to run in thread"""
        response = None

        try:
            # Convert Supabase path to authenticated URL if needed
            needs_auth = _should_use_authenticated_request(url)
            download_url = url

            if not download_url.startswith("http"):
                download_url = get_supabase_storage_url(download_url)

            logger.info(create_safe_fetch_message(download_url))

            # Use authenticated request for private buckets with streaming
            headers = get_authenticated_headers() if needs_auth else {}

            response = requests.get(
                download_url, headers=headers, timeout=30, stream=True
            )
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

            for chunk in response.iter_content(
                chunk_size=chunk_size, decode_unicode=True
            ):
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
                    result = _safe_json_dumps(data)
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
                        "Error closing HTTP response",
                        extra={"error": str(cleanup_error)},
                    )

    # Run blocking I/O operations in thread pool to avoid blocking event loop
    return await asyncio.to_thread(_download_and_process_json)


def split_into_chunks(text: str) -> list:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    if not text.strip():
        return []

    # Clean the text first using shared TextCleaner utility
    cleaned_text = TextCleaner.clean_text(text, remove_noise=False)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(cleaned_text)
    return [chunk for chunk in chunks if chunk.strip()]


def filter_noise_chunks(chunks: list) -> list:
    """Filter out chunks that are clearly UI noise (login/cart/country lists, etc.).

    The goal is to avoid embedding/upserting low-signal text that hurts search quality.

    This function now uses the shared TextCleaner utility for consistency.
    """
    return TextCleaner.filter_noise_chunks(chunks, min_length=60)


def get_embeddings_for_chunks(chunks: list) -> list:
    if not chunks:
        return []

    embedder = OpenAIEmbeddings()
    vectors = embedder.embed_documents(chunks)
    return vectors


def extract_product_links_from_chunk(chunk: str, source_url: str = None) -> list:
    """Extract potential product links from text chunk"""

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


def upload_to_pinecone(
    chunks: list, vectors: list, namespace: str, upload_id: str, source_url: str = None
):
    if not chunks or not vectors:
        logger.warning("No chunks or vectors to upload", extra={"upload_id": upload_id})
        return

    # Enhanced metadata with source URL and extracted product links
    to_upsert = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        metadata = {
            "chunk": chunk,
            "upload_id": upload_id,
            "source": source_url or "",
            "chunk_index": i,
        }

        # Extract potential product links from chunk content
        product_links = extract_product_links_from_chunk(chunk, source_url)
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


# is_ecommerce_url is now imported from url_utils to avoid duplication


async def smart_scrape_url(url: str) -> str:
    """
    Intelligently scrape URL using the best scraper for the content type.

    Uses UnifiedScraper which provides:
    - Automatic strategy selection (e-commerce → Playwright → traditional)
    - Retry logic with exponential backoff
    - Better error handling

    This ensures multi-tenant SaaS customers can upload any URL type:
    - E-commerce product pages → Enhanced e-commerce scraper (structured data)
    - JavaScript-rendered sites (React, Vue, Angular) → Playwright
    - Traditional HTML sites → Traditional scraper

    """
    # Use UnifiedScraper (consolidates all scraping logic with retries)
    try:
        from .unified_scraper import UnifiedScraper

        scraper = UnifiedScraper()
        result = await scraper.scrape(url, extract_products=True)

        if result["success"] and result["text"]:
            logger.info(
                f"[Unified] ✅ Scraping succeeded using {result['method']}: "
                f"{len(result['text'])} chars"
            )
            if result.get("products"):
                logger.info(
                    f"[Unified] Extracted {len(result['products'])} products, "
                    f"{len(result.get('product_urls', []))} product URLs"
                )
            # Return text regardless of whether products were found
            return result["text"]
        else:
            error_msg = result.get("error", "Unknown error")
            raise ValueError(f"Failed to scrape URL: {error_msg}")

    except ImportError:
        # Fallback if UnifiedScraper not available (shouldn't happen in production)
        logger.error("UnifiedScraper not available - this should not happen!")
        raise RuntimeError("Scraping system not properly configured")
    except Exception as e:
        logger.error(
            f"Scraping failed for {log_domain_safely(url)}: {str(e)}",
            exc_info=True,
        )
        raise


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


async def _process_upload_record(upload: dict, supabase_client):
    """
    Process a single upload record. This mirrors the previous sequential logic
    but allows the caller to schedule uploads concurrently with a semaphore.
    """
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
    # First, extract the actual URL for e-commerce detection
    actual_url = source
    if doc_type == "url":
        try:
            source_data = json.loads(source)
            if isinstance(source_data, dict) and "url" in source_data:
                actual_url = source_data["url"]
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON, use source as-is

    if doc_type == "pdf":
        text = await extract_text_from_pdf_url(source)
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
        text = await extract_text_from_json_url(source)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    # Validate extracted text
    if not text or len(text.strip()) < 10:
        # For e-commerce URLs, provide more specific error message
        if doc_type == "url" and is_ecommerce_url(actual_url):
            raise ValueError(
                f"Failed to extract meaningful content from Shopify/e-commerce site. "
                f"The site may require JavaScript rendering or have anti-scraping measures. "
                f"Extracted {len(text.strip()) if text else 0} characters."
            )
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

    # If we have very few chunks or very short text, log a warning
    if pre_filter_count == 0:
        logger.warning(
            "No chunks created from extracted text",
            extra={
                "upload_id": upload_id,
                "text_length": len(text),
                "text_preview": text[:500] if text else "No text",
            },
        )
    elif pre_filter_count < 3 and len(text) < 500:
        logger.warning(
            "Very few chunks created from short text",
            extra={
                "upload_id": upload_id,
                "chunk_count": pre_filter_count,
                "text_length": len(text),
            },
        )

    # Check if this is an e-commerce URL for less aggressive filtering
    # Use actual_url which handles JSON-encoded sources
    is_ecommerce = is_ecommerce_url(actual_url) if doc_type == "url" else False

    # Filter chunks - use less aggressive filtering for e-commerce sites
    if is_ecommerce:
        # For e-commerce sites, use a more lenient filter
        # First try standard filtering
        filtered_chunks = filter_noise_chunks(chunks)

        # If all chunks were filtered out, try progressively more lenient filtering
        if not filtered_chunks and pre_filter_count > 0:
            logger.warning(
                "All chunks filtered out for e-commerce site, trying lenient filtering",
                extra={
                    "upload_id": upload_id,
                    "pre_filter_count": pre_filter_count,
                    "source": log_domain_safely(actual_url),
                },
            )
            # Use TextCleaner directly with lower min_length
            from .text_cleaner import TextCleaner

            # Try progressively more lenient thresholds
            for min_len in [30, 20, 10]:
                filtered_chunks = TextCleaner.filter_noise_chunks(
                    chunks, min_length=min_len
                )
                if filtered_chunks:
                    logger.info(
                        f"Lenient filtering succeeded with min_length={min_len}",
                        extra={
                            "upload_id": upload_id,
                            "chunks_kept": len(filtered_chunks),
                            "min_length": min_len,
                        },
                    )
                    break

            # Last resort: if still no chunks, keep any chunk > 5 chars that doesn't match noise patterns
            if not filtered_chunks:
                logger.warning(
                    "Even lenient filtering removed all chunks, using last resort filter",
                    extra={
                        "upload_id": upload_id,
                        "pre_filter_count": pre_filter_count,
                        "text_length": len(text),
                    },
                )
                # Keep chunks that are at least 5 chars and don't match strict noise patterns
                import re

                strict_noise = [
                    r"^\s*(Sign In|Sign Up|Log in)\s*$",
                    r"^\s*(Add to cart|View Cart|Checkout)\s*$",
                    r"^\s*(Cookie Policy|Privacy Policy)\s*$",
                ]
                compiled_strict = [
                    re.compile(pat, re.IGNORECASE) for pat in strict_noise
                ]

                filtered_chunks = []
                for chunk in chunks:
                    chunk_stripped = chunk.strip()
                    if len(chunk_stripped) >= 5:
                        # Only filter if it matches strict noise patterns
                        is_strict_noise = any(
                            pat.search(chunk_stripped) for pat in compiled_strict
                        )
                        if not is_strict_noise:
                            filtered_chunks.append(chunk)

                if filtered_chunks:
                    logger.info(
                        f"Last resort filter kept {len(filtered_chunks)} chunks",
                        extra={
                            "upload_id": upload_id,
                            "chunks_kept": len(filtered_chunks),
                        },
                    )
                else:
                    # Absolute last resort: if we have ANY text at all, create a single chunk from it
                    # This handles cases where collection pages have very minimal but valid content
                    if text and len(text.strip()) > 10:
                        logger.warning(
                            "All chunks filtered, creating single chunk from full text as absolute last resort",
                            extra={
                                "upload_id": upload_id,
                                "text_length": len(text),
                                "text_preview": text[:200],
                            },
                        )
                        # Create one chunk from the full text (will be truncated by chunker if too long)
                        filtered_chunks = [text.strip()]

        chunks = filtered_chunks
    else:
        chunks = filter_noise_chunks(chunks)

    # CRITICAL: If still no chunks (including case where pre_filter_count == 0),
    # create a single chunk from the full text as absolute last resort
    if not chunks and text and len(text.strip()) > 10:
        logger.warning(
            "No chunks after all processing, creating single chunk from full text as absolute last resort",
            extra={
                "upload_id": upload_id,
                "text_length": len(text),
                "pre_filter_count": pre_filter_count,
                "text_preview": text[:200],
            },
        )
        chunks = [text.strip()]

    if not chunks:
        # Provide detailed error message with diagnostics
        error_details = {
            "pre_filter_count": pre_filter_count,
            "text_length": len(text),
            "text_preview": text[:200] if text else "No text",
            "is_ecommerce": is_ecommerce,
            "source": log_domain_safely(actual_url if doc_type == "url" else source),
        }
        logger.error(
            "No text chunks generated after filtering",
            extra={"upload_id": upload_id, **error_details},
        )
        raise ValueError(
            f"No text chunks generated after filtering. "
            f"Extracted {len(text)} characters, created {pre_filter_count} chunks before filtering. "
            f"This may indicate the content was mostly UI noise or the scraper needs adjustment."
        )

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

        if not uploads:
            logger.info("No pending uploads found")
            return

        if len(uploads) > MAX_UPLOAD_BATCH:
            logger.warning(
                "Large pending upload backlog detected, limiting batch size",
                extra={"requested": len(uploads), "limit": MAX_UPLOAD_BATCH},
            )
            uploads = uploads[:MAX_UPLOAD_BATCH]

        semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_UPLOADS))

        async def worker(upload):
            async with semaphore:
                try:
                    await _process_upload_record(upload, supabase_client)
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

        await asyncio.gather(*(worker(upload) for upload in uploads))

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
