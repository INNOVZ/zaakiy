"""
Pinecone Upload Module
Handles uploading chunks and vectors to Pinecone with metadata
"""

import logging
from typing import Dict, Tuple

from ..storage.pinecone_client import get_pinecone_index
from .metadata_extraction import (
    extract_metadata_flags,
    extract_product_links_from_chunk,
)
from .topic_extraction import extract_topics_from_url

logger = logging.getLogger(__name__)

# Cache for Pinecone index
_index_cache = None


def _get_pinecone_index_cached():
    """Get Pinecone index with caching"""
    global _index_cache
    if _index_cache is None:
        _index_cache = get_pinecone_index()
    return _index_cache


def upload_to_pinecone_with_url_mapping(
    chunks: list,
    vectors: list,
    namespace: str,
    upload_id: str,
    url_to_chunks_map: Dict[str, Tuple[int, int]],
    start_url: str = None,
):
    """
    Upload chunks to Pinecone with URL-specific topics for recursive scraping.

    Each URL's chunks get topics extracted from that specific URL, not just the start URL.
    This enables better topic-based retrieval for recursive scraping.

    Args:
        chunks: List of text chunks
        vectors: List of embedding vectors
        namespace: Pinecone namespace
        upload_id: Upload ID
        url_to_chunks_map: Dict mapping URLs to (start_idx, end_idx) tuple for their chunks
        start_url: Parent/start URL for reference
    """
    if not chunks or not vectors:
        logger.warning("No chunks or vectors to upload", extra={"upload_id": upload_id})
        return

    # Enhanced metadata with URL-specific topics
    to_upsert = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        # Find which URL this chunk belongs to
        chunk_url = start_url  # Default to start URL
        url_topics = []

        for url, (start_idx, end_idx) in url_to_chunks_map.items():
            if start_idx <= i < end_idx:
                chunk_url = url
                # Extract topics from this specific URL
                url_topics = extract_topics_from_url(url)
                break

        # If no URL found, use start URL
        if not url_topics and start_url:
            url_topics = extract_topics_from_url(start_url)
            chunk_url = start_url

        metadata = {
            "chunk": chunk,
            "upload_id": upload_id,
            "source": chunk_url or "",
            "chunk_index": i,
        }

        # Add URL-specific topics for intent-based filtering
        if url_topics:
            metadata["topics"] = url_topics
            logger.debug(
                f"Adding URL-specific topics to chunk {i} from {chunk_url[:50]}: {url_topics[:3]}",
                extra={"upload_id": upload_id},
            )

        # Extract potential product links from chunk content
        product_links = extract_product_links_from_chunk(chunk, chunk_url)
        if product_links:
            metadata["product_links"] = product_links
            metadata["has_products"] = True
        else:
            metadata["has_products"] = False

        metadata.update(extract_metadata_flags(chunk))

        to_upsert.append((f"{upload_id}-{i}", vec, metadata))

    logger.info(
        "Uploading vectors to Pinecone with URL-specific topics",
        extra={
            "vector_count": len(to_upsert),
            "namespace": namespace,
            "upload_id": upload_id,
            "url_count": len(url_to_chunks_map),
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
            "Pinecone upsert successful with URL-specific topics",
            extra={"result": str(result), "upload_id": upload_id},
        )
    except Exception as e:
        logger.error(
            "Pinecone upsert failed",
            extra={"error": str(e), "upload_id": upload_id},
            exc_info=True,
        )
        raise


def upload_to_pinecone(
    chunks: list, vectors: list, namespace: str, upload_id: str, source_url: str = None
):
    """
    Upload chunks to Pinecone with metadata.

    Args:
        chunks: List of text chunks
        vectors: List of embedding vectors
        namespace: Pinecone namespace
        upload_id: Upload ID
        source_url: Source URL for topic extraction
    """
    if not chunks or not vectors:
        logger.warning("No chunks or vectors to upload", extra={"upload_id": upload_id})
        return

    # Extract topics from URL ONCE (same for all chunks from this source)
    url_topics = extract_topics_from_url(source_url) if source_url else []

    # Enhanced metadata with source URL, topics, and extracted product links
    to_upsert = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        metadata = {
            "chunk": chunk,
            "upload_id": upload_id,
            "source": source_url or "",
            "chunk_index": i,
        }

        # Add URL-extracted topics for intent-based filtering
        if url_topics:
            metadata["topics"] = url_topics
            logger.debug(
                f"Adding topics to chunk {i}: {url_topics}",
                extra={"upload_id": upload_id},
            )

        # Extract potential product links from chunk content
        product_links = extract_product_links_from_chunk(chunk, source_url)
        if product_links:
            metadata["product_links"] = product_links
            metadata["has_products"] = True
        else:
            metadata["has_products"] = False

        metadata.update(extract_metadata_flags(chunk))

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
