"""
Diagnostic script to trace Shopify URL upload failures.

This script will:
1. Test the actual URL scraping
2. Show what text is extracted
3. Show chunking results
4. Show filtering results
5. Identify where it fails

Run: python -m app.scripts.diagnose_shopify_upload
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.scraping.ingestion_worker import (
    filter_noise_chunks,
    smart_scrape_url,
    split_into_chunks,
)
from app.services.scraping.url_utils import is_ecommerce_url


async def diagnose_shopify_upload():
    """Diagnose why Shopify uploads are failing"""

    test_url = "https://ambassadorscentworks.com/"

    print("=" * 80)
    print("🔍 SHOPIFY UPLOAD DIAGNOSTIC")
    print("=" * 80)
    print()

    print(f"Testing URL: {test_url}")
    print(f"Is e-commerce URL: {is_ecommerce_url(test_url)}")
    print()

    # Step 1: Test scraping
    print("Step 1: Scraping URL...")
    print("-" * 80)
    try:
        text = await smart_scrape_url(test_url)
        print(f"✅ Scraping succeeded!")
        print(f"   Text length: {len(text)} characters")
        print(f"   Text preview (first 500 chars):")
        print(f"   {text[:500]}")
        print()

        if not text or len(text.strip()) < 10:
            print("❌ ERROR: Text extraction returned insufficient content!")
            print(f"   Length: {len(text.strip()) if text else 0} characters")
            return

    except Exception as e:
        print(f"❌ ERROR: Scraping failed!")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return

    # Step 2: Test chunking
    print("Step 2: Splitting into chunks...")
    print("-" * 80)
    try:
        chunks = split_into_chunks(text)
        print(f"✅ Chunking succeeded!")
        print(f"   Number of chunks: {len(chunks)}")
        if chunks:
            print(f"   First chunk length: {len(chunks[0])} characters")
            print(f"   First chunk preview:")
            print(f"   {chunks[0][:200]}")
            print()

            # Show chunk size distribution
            chunk_sizes = [len(c) for c in chunks]
            print(f"   Chunk size stats:")
            print(f"   - Min: {min(chunk_sizes)}")
            print(f"   - Max: {max(chunk_sizes)}")
            print(f"   - Avg: {sum(chunk_sizes) // len(chunk_sizes)}")
            print()
        else:
            print("❌ ERROR: No chunks created!")
            return

    except Exception as e:
        print(f"❌ ERROR: Chunking failed!")
        print(f"   Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Test filtering
    print("Step 3: Filtering noise chunks...")
    print("-" * 80)
    try:
        pre_filter_count = len(chunks)
        filtered_chunks = filter_noise_chunks(chunks)
        print(f"✅ Filtering completed!")
        print(f"   Chunks before filter: {pre_filter_count}")
        print(f"   Chunks after filter: {len(filtered_chunks)}")
        print()

        if not filtered_chunks:
            print("❌ ERROR: All chunks were filtered out!")
            print()
            print("   Analyzing why chunks were filtered...")

            # Check each chunk to see why it was filtered
            from app.services.scraping.text_cleaner import TextCleaner

            for i, chunk in enumerate(chunks[:10]):  # Check first 10 chunks
                print(f"\n   Chunk {i+1}:")
                print(f"   - Length: {len(chunk.strip())} chars")
                print(f"   - Preview: {chunk[:100]}...")

                # Test with lower min_length
                lenient_filtered = TextCleaner.filter_noise_chunks(
                    [chunk], min_length=30
                )
                if lenient_filtered:
                    print(f"   - ✅ Would pass with min_length=30")
                else:
                    print(f"   - ❌ Still filtered with min_length=30")

                    # Check if it's too short
                    if len(chunk.strip()) < 30:
                        print(f"   - Reason: Too short (< 30 chars)")

                    # Check for noise patterns
                    import re

                    noise_patterns = [
                        r"\b(Sign In|Sign Up|Log in|Create Account|Forgot your password)\b",
                        r"\b(Add to (cart|bag|wishlist)|Quick (view|buy)|View Cart|Checkout)\b",
                    ]
                    for pattern in noise_patterns:
                        if re.search(pattern, chunk, re.IGNORECASE):
                            print(f"   - Reason: Matches noise pattern: {pattern}")
                            break

                    # Check comma count
                    if chunk.count(",") >= 20:
                        print(f"   - Reason: Too many commas (likely country list)")

            # Try lenient filtering
            print()
            print("   Trying lenient filtering (min_length=30)...")
            lenient_filtered = TextCleaner.filter_noise_chunks(chunks, min_length=30)
            print(f"   Chunks with lenient filter: {len(lenient_filtered)}")

            if lenient_filtered:
                print("   ✅ Lenient filtering would work!")
            else:
                print("   ❌ Even lenient filtering removes all chunks!")

        else:
            print("✅ Filtering kept some chunks!")
            print(f"   First filtered chunk preview:")
            print(f"   {filtered_chunks[0][:200]}")

    except Exception as e:
        print(f"❌ ERROR: Filtering failed!")
        print(f"   Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if filtered_chunks:
        print("✅ All steps succeeded! Upload should work.")
    else:
        print("❌ Upload will fail - all chunks filtered out.")
        print()
        print("RECOMMENDATIONS:")
        print("1. Check if text extraction is getting actual content")
        print("2. Review noise filtering patterns")
        print("3. Consider adjusting min_length for e-commerce sites")
        print("4. Check if the site requires special handling")


if __name__ == "__main__":
    asyncio.run(diagnose_shopify_upload())
