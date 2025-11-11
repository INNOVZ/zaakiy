"""
Test script to verify collection page scraping works correctly.

This script tests the complete flow:
1. E-commerce URL detection
2. E-commerce scraper execution
3. Text extraction
4. Chunk creation
5. Chunk filtering

Run: python -m app.scripts.test_collection_scraping
"""

import asyncio
import os
import sys

# Setup path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.services.scraping.ingestion_worker import (
    filter_noise_chunks,
    smart_scrape_url,
    split_into_chunks,
)
from app.services.scraping.unified_scraper import UnifiedScraper
from app.services.scraping.url_utils import is_ecommerce_url, log_domain_safely


async def test_collection_page():
    """Test scraping a collection page"""

    # Test URL - use one of the failing collection pages
    test_url = "https://ambassadorscentworks.com/collections/new-arrivals"

    print("=" * 80)
    print("🧪 TESTING COLLECTION PAGE SCRAPING")
    print("=" * 80)
    print(f"\nTest URL: {test_url}\n")

    # Step 1: Check URL detection
    print("Step 1: E-commerce URL Detection")
    print("-" * 80)
    is_ecommerce = is_ecommerce_url(test_url)
    print(f"✅ Detected as e-commerce: {is_ecommerce}")
    if not is_ecommerce:
        print("❌ PROBLEM: URL not detected as e-commerce!")
        return
    print()

    # Step 2: Test UnifiedScraper directly
    print("Step 2: Testing UnifiedScraper")
    print("-" * 80)
    try:
        scraper = UnifiedScraper()
        result = await scraper.scrape(test_url, extract_products=True)

        print(f"Success: {result['success']}")
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Text length: {len(result.get('text', ''))}")
        print(f"Products found: {len(result.get('products', []))}")
        print(f"Product URLs found: {len(result.get('product_urls', []))}")
        print(f"Error: {result.get('error', 'None')}")

        if result.get("text"):
            text_preview = result["text"][:500]
            print(f"\nText preview (first 500 chars):")
            print("-" * 80)
            print(text_preview)
            print("-" * 80)
        else:
            print("\n❌ PROBLEM: No text extracted!")
            return

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    print()

    # Step 3: Test smart_scrape_url (full flow)
    print("Step 3: Testing smart_scrape_url (Full Flow)")
    print("-" * 80)
    try:
        text = await smart_scrape_url(test_url)
        print(f"✅ Text extracted: {len(text)} characters")
        print(f"\nText preview (first 500 chars):")
        print("-" * 80)
        print(text[:500])
        print("-" * 80)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    print()

    # Step 4: Test chunking
    print("Step 4: Testing Chunk Creation")
    print("-" * 80)
    try:
        chunks = split_into_chunks(text)
        pre_filter_count = len(chunks)
        print(f"✅ Chunks created: {pre_filter_count}")

        if pre_filter_count > 0:
            print(f"\nFirst chunk (first 200 chars):")
            print("-" * 80)
            print(chunks[0][:200])
            print("-" * 80)
        else:
            print("❌ PROBLEM: No chunks created!")
            return
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    print()

    # Step 5: Test chunk filtering
    print("Step 5: Testing Chunk Filtering (E-commerce)")
    print("-" * 80)
    try:
        # Use e-commerce filtering logic
        from app.services.scraping.text_cleaner import TextCleaner

        # First try standard filtering
        filtered_chunks = filter_noise_chunks(chunks)
        print(f"After standard filtering: {len(filtered_chunks)} chunks")

        # If all filtered out, try lenient
        if not filtered_chunks and pre_filter_count > 0:
            print("⚠️ All chunks filtered, trying lenient filtering...")
            for min_len in [30, 20, 10]:
                filtered_chunks = TextCleaner.filter_noise_chunks(
                    chunks, min_length=min_len
                )
                if filtered_chunks:
                    print(
                        f"✅ Lenient filtering (min_length={min_len}) kept {len(filtered_chunks)} chunks"
                    )
                    break

        # Last resort
        if not filtered_chunks and pre_filter_count > 0:
            print("⚠️ Even lenient filtering removed all chunks, using last resort...")
            import re

            strict_noise = [
                r"^\s*(Sign In|Sign Up|Log in)\s*$",
                r"^\s*(Add to cart|View Cart|Checkout)\s*$",
            ]
            compiled_strict = [re.compile(pat, re.IGNORECASE) for pat in strict_noise]

            filtered_chunks = []
            for chunk in chunks:
                chunk_stripped = chunk.strip()
                if len(chunk_stripped) >= 5:
                    is_strict_noise = any(
                        pat.search(chunk_stripped) for pat in compiled_strict
                    )
                    if not is_strict_noise:
                        filtered_chunks.append(chunk)

            if filtered_chunks:
                print(f"✅ Last resort filter kept {len(filtered_chunks)} chunks")

        # Absolute last resort
        if not filtered_chunks and len(text.strip()) > 10:
            print("⚠️ All chunks filtered, creating single chunk from full text...")
            filtered_chunks = [text.strip()]
            print(f"✅ Created single chunk: {len(filtered_chunks[0])} chars")

        print(f"\nFinal chunk count: {len(filtered_chunks)}")

        if filtered_chunks:
            print(f"\n✅ SUCCESS: {len(filtered_chunks)} chunks ready for embedding!")
            print(f"\nFirst filtered chunk (first 200 chars):")
            print("-" * 80)
            print(filtered_chunks[0][:200])
            print("-" * 80)
        else:
            print(f"\n❌ PROBLEM: No chunks after filtering!")
            print(f"Text length: {len(text)}")
            print(f"Pre-filter chunks: {pre_filter_count}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_collection_page())
