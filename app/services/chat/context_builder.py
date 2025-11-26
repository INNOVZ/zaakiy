"""
Context Builder
---------------
Creates structured context payloads for response generation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .chat_utilities import ChatUtilities
from .contact_extractor import contact_extractor

logger = logging.getLogger(__name__)


@dataclass
class ContextBuildResult:
    """Structured representation of context-building output."""

    context_text: str
    sources: List[str]
    contact_info: Dict[str, Any]
    product_links: List[Dict[str, Any]]
    demo_links: List[str]
    context_quality: Dict[str, Any]


class ContextBuilder:
    """Builds prioritized context strings with contact/product enrichment."""

    def __init__(self, chat_utilities: Optional[ChatUtilities] = None):
        self.chat_utilities = chat_utilities or ChatUtilities()

    def build(
        self,
        documents: List[Dict[str, Any]],
        max_context_length: int,
        context_config: Optional[Any] = None,
    ) -> ContextBuildResult:
        if not documents:
            return ContextBuildResult(
                context_text="",
                sources=[],
                contact_info={},
                product_links=[],
                demo_links=[],
                context_quality={"coverage_score": 0.0, "relevance_score": 0.0},
            )

        all_phones: set[str] = set()
        all_emails: set[str] = set()
        all_addresses: List[str] = []
        all_demo_links: set[str] = set()
        all_links: set[str] = set()

        # IMPORTANT: Keep chunks in original order (they're already sorted by relevance)
        # Don't separate contact/non-contact chunks as that breaks relevance ordering
        all_chunks: List[str] = []
        sources: List[str] = []
        total_score = 0.0

        for idx, doc in enumerate(documents):
            chunk_text = doc.get("chunk", "")
            source = doc.get("source", "")
            score = doc.get("score", 0.0)

            if not chunk_text or len(chunk_text.strip()) <= 10:
                continue

            # OPTIMIZATION: Use already-extracted contact info if available (from retrieval service)
            # This avoids duplicate extraction and improves performance
            if "contact_info" in doc and doc["contact_info"].get("has_contact_info"):
                contact_info = doc["contact_info"]
            else:
                # Only extract if not already done
                contact_info = contact_extractor.extract_contact_info(chunk_text)

            # Extract contact info for the contact info section
            if contact_info.get("phones"):
                all_phones.update(contact_info["phones"])
            if contact_info.get("emails"):
                all_emails.update(contact_info["emails"])
            if contact_info.get("addresses"):
                all_addresses.extend(contact_info["addresses"])
            if contact_info.get("demo_links"):
                all_demo_links.update(contact_info["demo_links"])
            if contact_info.get("links"):
                all_links.update(contact_info["links"])

            # Keep ALL chunks in original order (preserves relevance ranking)
            all_chunks.append(chunk_text)

            logger.debug(
                "📄 Chunk %d (score: %.3f, length: %d, has_contact: %s): %s...",
                idx + 1,
                score,
                len(chunk_text),
                contact_info.get("has_contact_info", False),
                chunk_text[:100],
            )

            if source and source not in sources:
                sources.append(source)
            total_score += score

        # Apply context chunk limit while preserving order
        prioritized_chunks = self._apply_chunk_limit(
            all_chunks,
            context_config=context_config,
            max_context_chunks=max_context_length,
        )

        combined_context = self._combine_context_chunks(
            prioritized_chunks, max_context_length
        )

        product_links = self.chat_utilities.extract_product_links_from_documents(
            documents
        )
        if product_links:
            logger.info(
                "🛍️ Extracted %d product links from documents", len(product_links)
            )

        product_section = ""
        if product_links:
            product_section = self._build_product_section(product_links)

        contact_section = self._build_contact_section(
            all_phones, all_emails, all_addresses, all_demo_links
        )

        final_context_parts = [
            part
            for part in [product_section, contact_section, combined_context]
            if part
        ]
        context_text = "\n".join(final_context_parts)

        if all_phones or all_emails or all_demo_links:
            logger.info(
                "✅ Extracted contact info: phones=%d, emails=%d, demo_links=%d, addresses=%d",
                len(all_phones),
                len(all_emails),
                len(all_demo_links),
                len(all_addresses),
            )

        avg_score = total_score / len(documents) if documents else 0.0
        coverage_score = (
            min(len(context_text) / max_context_length, 1.0)
            if max_context_length
            else 0.0
        )

        return ContextBuildResult(
            context_text=context_text,
            sources=sources,
            contact_info={
                "phones": list(all_phones),
                "emails": list(all_emails),
                "addresses": all_addresses[:5],
                "demo_links": list(all_demo_links),
                "all_links": list(all_links),
            },
            product_links=product_links,
            demo_links=list(all_demo_links),
            context_quality={
                "coverage_score": coverage_score,
                "relevance_score": avg_score,
                "document_count": len(documents),
                "chunks_used": len(prioritized_chunks),
                "product_links_count": len(product_links),
            },
        )

    def _apply_chunk_limit(
        self,
        chunks: List[str],
        context_config: Optional[Any],
        max_context_chunks: int,
    ) -> List[str]:
        """Apply chunk limit while preserving relevance order"""
        final_context_chunks = None
        if context_config and hasattr(context_config, "final_context_chunks"):
            final_context_chunks = context_config.final_context_chunks
        elif isinstance(context_config, dict):
            final_context_chunks = context_config.get("final_context_chunks")

        if final_context_chunks and final_context_chunks > 0:
            # Truncate to limit while preserving order
            limited_chunks = chunks[:final_context_chunks]
            logger.info(
                "📊 Applied final_context_chunks limit: %d (from %d chunks)",
                final_context_chunks,
                len(chunks),
            )
        else:
            limited_chunks = chunks
            logger.debug(
                "No final_context_chunks limit applied, using all %d chunks",
                len(chunks),
            )

        return [self._compress_chunk(chunk) for chunk in limited_chunks]

    @staticmethod
    def _compress_chunk(chunk: str) -> str:
        if len(chunk) <= 800:
            return chunk
        return f"{chunk[:400]}... [compressed] ...{chunk[-400:]}"

    @staticmethod
    def _combine_context_chunks(chunks: List[str], max_context_length: int) -> str:
        if not chunks:
            return ""

        combined_parts: List[str] = []
        current_length = 0

        for chunk in chunks:
            chunk_length = len(chunk)
            if (
                max_context_length
                and current_length + chunk_length + 50 > max_context_length
            ):
                break

            combined_parts.append(chunk.strip())
            current_length += chunk_length + 7  # approximate separator length

        return "\n\n---\n\n".join(combined_parts)

    def _build_product_section(self, product_links: List[Dict[str, Any]]) -> str:
        if not product_links:
            return ""

        lines = ["PRODUCT CATALOG:"]
        for idx, product in enumerate(product_links, start=1):
            url = product.get("url", "")
            name = product.get("name", "Product")
            chunk_preview = product.get("chunk_preview", "")

            price = self._extract_price_from_chunk(chunk_preview)
            description = self._extract_description_from_chunk(chunk_preview, name)

            line = f"{idx}. **[{name}]({url})**"
            if description:
                line += f" - *{description}*"
            if price:
                line += f" - **Price**: {price}"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _build_contact_section(
        phones: set[str],
        emails: set[str],
        addresses: List[str],
        demo_links: set[str],
    ) -> str:
        if not (phones or emails or addresses or demo_links):
            return ""

        contact_info_parts = []
        if phones:
            contact_info_parts.append(f"Phone Numbers: {', '.join(sorted(phones))}")
        if emails:
            contact_info_parts.append(f"Email Addresses: {', '.join(sorted(emails))}")
        if addresses:
            contact_info_parts.append(f"Addresses: {' | '.join(addresses[:3])}")
        if demo_links:
            contact_info_parts.append(
                f"Demo/Booking Links: {', '.join(sorted(demo_links))}"
            )

        if not contact_info_parts:
            return ""

        return "\n\nCONTACT INFORMATION:\n" + "\n".join(contact_info_parts)

    @staticmethod
    def _extract_price_from_chunk(chunk: str) -> str:
        if not chunk:
            return ""

        price_patterns = [
            r"(?:price|cost|₹|Rs\.?|AED|Dhs\.?|\$)\s*:?\s*([\d,]+(?:\.\d{2})?)",
            r"(Dhs\.?\s*[\d,]+(?:\.\d{2})?)",
            r"(₹\s*[\d,]+(?:\.\d{2})?)",
            r"(AED\s*[\d,]+(?:\.\d{2})?)",
        ]

        for pattern in price_patterns:
            match = re.search(pattern, chunk, re.IGNORECASE)
            if match:
                return (
                    match.group(1).strip()
                    if len(match.groups()) >= 1
                    else match.group(0).strip()
                )
        return ""

    @staticmethod
    def _extract_description_from_chunk(chunk: str, product_name: str) -> str:
        if not chunk:
            return ""

        lines = chunk.strip().split("\n")
        for line in lines:
            if re.search(r"price|cost|₹|Rs\.|AED|Dhs\.|\$\d", line, re.IGNORECASE):
                continue

            clean_line = re.sub(
                re.escape(product_name), "", line.strip(), flags=re.IGNORECASE
            ).strip(" .-–—:")

            if len(clean_line) > 20:
                words = clean_line.split()
                if len(words) > 15:
                    clean_line = " ".join(words[:15]) + "..."
                return clean_line
        return ""
