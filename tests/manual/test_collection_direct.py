"""
Direct test of collection page scraping - bypasses __init__ imports.

Run: python3 test_collection_direct.py
"""

import asyncio
import os
import sys

# Ensure backend root is on the import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

TEST_URL = "https://ambassadorscentworks.com/collections/new-arrivals"


async def run_collection_scrape():
    print("=" * 80)
    print("🧪 TESTING COLLECTION PAGE SCRAPING")
    print("=" * 80)
    print(f"\nTest URL: {TEST_URL}\n")

    try:
        # Import directly to avoid __init__ issues
        from app.services.scraping.ecommerce_scraper import (
            EnhancedEcommerceProductScraper,
        )

        print("✅ Step 1: Scraper imported successfully")

        print("\n✅ Step 2: Initializing scraper...")
        async with EnhancedEcommerceProductScraper(
            headless=True, timeout=90000
        ) as scraper:
            print("✅ Scraper initialized")

            print(f"\n✅ Step 3: Scraping {TEST_URL}...")
            print("   (This may take 30-90 seconds for Playwright to load)...")

            result = await scraper.scrape_product_collection(TEST_URL)

            print(f"\n{'='*80}")
            print("📊 SCRAPING RESULTS")
            print("=" * 80)
            print(f"Text length: {len(result.get('text', ''))}")
            print(f"Products found: {len(result.get('products', []))}")
            print(f"Product URLs found: {len(result.get('product_urls', []))}")
            print(f"Error: {result.get('error', 'None')}")

            text = result.get("text", "")

            if text:
                print(f"\n{'='*80}")
                print("📄 EXTRACTED TEXT (first 1500 chars)")
                print("=" * 80)
                print(text[:1500])
                print("=" * 80)

                # Test chunking
                print(f"\n{'='*80}")
                print("📦 TESTING CHUNK CREATION")
                print("=" * 80)

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
                    print(f"\nFirst chunk preview (300 chars):")
                    print("-" * 80)
                    print(chunks[0][:300])
                    print("-" * 80)

                    # Test filtering
                    print(f"\n{'='*80}")
                    print("🔍 TESTING CHUNK FILTERING")
                    print("=" * 80)

                    from app.services.scraping.text_cleaner import TextCleaner

                    # Standard filtering
                    filtered = TextCleaner.filter_noise_chunks(chunks, min_length=60)
                    print(
                        f"Standard filter (min_length=60): {len(filtered)} chunks kept"
                    )

                    # Lenient filtering
                    if not filtered and chunks:
                        print("\nTrying lenient filtering...")
                        for min_len in [30, 20, 10]:
                            filtered = TextCleaner.filter_noise_chunks(
                                chunks, min_length=min_len
                            )
                            if filtered:
                                print(
                                    f"✅ Lenient (min_length={min_len}): {len(filtered)} chunks kept"
                                )
                                break

                    # Last resort
                    if not filtered and chunks:
                        print("\nTrying last resort filter...")
                        import re

                        strict_noise = [
                            r"^\s*(Sign In|Sign Up|Log in)\s*$",
                            r"^\s*(Add to cart|View Cart|Checkout)\s*$",
                        ]
                        compiled = [
                            re.compile(pat, re.IGNORECASE) for pat in strict_noise
                        ]

                        filtered = []
                        for chunk in chunks:
                            chunk_stripped = chunk.strip()
                            if len(chunk_stripped) >= 5:
                                is_noise = any(
                                    pat.search(chunk_stripped) for pat in compiled
                                )
                                if not is_noise:
                                    filtered.append(chunk)

                        if filtered:
                            print(f"✅ Last resort: {len(filtered)} chunks kept")

                    # Absolute last resort
                    if not filtered and len(text.strip()) > 10:
                        print(
                            "\nUsing absolute last resort: single chunk from full text"
                        )
                        filtered = [text.strip()]
                        print(f"✅ Created 1 chunk: {len(filtered[0])} chars")

                    print(f"\n{'='*80}")
                    print("✅ FINAL RESULT")
                    print("=" * 80)
                    if filtered:
                        print(f"✅ SUCCESS: {len(filtered)} chunks ready for embedding!")
                        print(f"\nFirst filtered chunk (300 chars):")
                        print("-" * 80)
                        print(filtered[0][:300])
                        print("-" * 80)
                    else:
                        print(f"❌ FAILED: No chunks after all filtering attempts")
                        print(f"   Text length: {len(text)}")
                        print(f"   Pre-filter chunks: {len(chunks)}")
                else:
                    print("❌ No chunks created from text!")
            else:
                print("\n❌ NO TEXT EXTRACTED!")
                print(f"Error: {result.get('error', 'Unknown')}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_collection_scrape())
