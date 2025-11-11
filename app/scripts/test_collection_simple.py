"""
Simple test script to verify collection page scraping.

This script tests the core scraping logic without full dependencies.

Run: python3 -m app.scripts.test_collection_simple
"""

import asyncio
import os
import re
import sys

# Setup path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Test URL
TEST_URL = "https://ambassadorscentworks.com/collections/new-arrivals"


async def test_ecommerce_scraper_directly():
    """Test the e-commerce scraper directly"""

    print("=" * 80)
    print("🧪 TESTING E-COMMERCE SCRAPER DIRECTLY")
    print("=" * 80)
    print(f"\nTest URL: {TEST_URL}\n")

    try:
        from app.services.scraping.ecommerce_scraper import (
            EnhancedEcommerceProductScraper,
        )

        print("Step 1: Initializing scraper...")
        async with EnhancedEcommerceProductScraper(
            headless=True, timeout=90000
        ) as scraper:
            print("✅ Scraper initialized")

            print("\nStep 2: Scraping collection page...")
            result = await scraper.scrape_product_collection(TEST_URL)

            print(f"\n✅ Scraping completed!")
            print(f"Text length: {len(result.get('text', ''))}")
            print(f"Products found: {len(result.get('products', []))}")
            print(f"Product URLs found: {len(result.get('product_urls', []))}")
            print(f"Error: {result.get('error', 'None')}")

            text = result.get("text", "")
            if text:
                print(f"\n📄 Text Preview (first 1000 chars):")
                print("-" * 80)
                print(text[:1000])
                print("-" * 80)

                # Test chunking
                print(f"\nStep 3: Testing chunk creation...")
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""],
                )
                chunks = splitter.split_text(text)
                chunks = [chunk for chunk in chunks if chunk.strip()]

                print(f"✅ Created {len(chunks)} chunks")

                if chunks:
                    print(f"\nFirst chunk (first 300 chars):")
                    print("-" * 80)
                    print(chunks[0][:300])
                    print("-" * 80)

                    # Test filtering
                    print(f"\nStep 4: Testing chunk filtering...")
                    from app.services.scraping.text_cleaner import TextCleaner

                    filtered = TextCleaner.filter_noise_chunks(chunks, min_length=60)
                    print(
                        f"After standard filtering (min_length=60): {len(filtered)} chunks"
                    )

                    if not filtered and chunks:
                        print("Trying lenient filtering...")
                        for min_len in [30, 20, 10]:
                            filtered = TextCleaner.filter_noise_chunks(
                                chunks, min_length=min_len
                            )
                            if filtered:
                                print(
                                    f"✅ Lenient filtering (min_length={min_len}) kept {len(filtered)} chunks"
                                )
                                break

                    if filtered:
                        print(f"\n✅ SUCCESS: {len(filtered)} chunks ready!")
                    else:
                        print(
                            f"\n⚠️ All chunks filtered, but text exists ({len(text)} chars)"
                        )
                        print(
                            "This should trigger the last resort chunk creation in ingestion_worker"
                        )
                else:
                    print("❌ No chunks created from text!")
            else:
                print("❌ No text extracted!")
                print(f"Error: {result.get('error', 'Unknown')}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()


async def test_url_detection():
    """Test URL detection"""
    print("\n" + "=" * 80)
    print("🧪 TESTING URL DETECTION")
    print("=" * 80)

    try:
        from app.services.scraping.url_utils import is_ecommerce_url

        test_urls = [
            "https://ambassadorscentworks.com/collections/new-arrivals",
            "https://ambassadorscentworks.com/products/vetiver-grande",
            "https://ambassadorscentworks.com/pages/about-us",
        ]

        for url in test_urls:
            is_ecom = is_ecommerce_url(url)
            print(f"{'✅' if is_ecom else '❌'} {url}: {is_ecom}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")


if __name__ == "__main__":
    print("\n")
    asyncio.run(test_url_detection())
    print("\n")
    asyncio.run(test_ecommerce_scraper_directly())
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
