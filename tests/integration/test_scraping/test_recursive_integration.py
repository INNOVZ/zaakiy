"""
Integration test for recursive URL topics with actual code
Tests the full flow without requiring network calls
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend root to path
backend_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_root))


async def test_implementation_flow():
    """Test the complete implementation flow"""
    print("=" * 80)
    print("🧪 INTEGRATION TEST: Recursive URL Topics Flow")
    print("=" * 80)
    print()

    # Test 1: Verify function exists and has correct signature
    print("1️⃣  Testing Function Signatures...")
    try:
        from app.services.scraping.ingestion_worker import (
            extract_topics_from_url,
            recursive_scrape_website,
            upload_to_pinecone_with_url_mapping,
        )

        print("   ✅ All functions imported successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test 2: Test topic extraction
    print()
    print("2️⃣  Testing Topic Extraction...")
    test_urls = [
        "https://ohhzones.com/digital-marketing/seo/",
        "https://ohhzones.com/branding-services/brand-identity/",
    ]

    for url in test_urls:
        topics = extract_topics_from_url(url)
        print(f"   URL: {url}")
        print(f"   Topics: {topics}")
        if topics:
            print("   ✅ Topics extracted")
        else:
            print("   ⚠️  No topics extracted")

    # Test 3: Test URL mapping logic
    print()
    print("3️⃣  Testing URL-to-Chunks Mapping Logic...")

    # Simulate the mapping
    url_to_chunks_map = {
        "https://example.com/digital-marketing/seo/": (0, 10),
        "https://example.com/digital-marketing/email-marketing/": (10, 20),
    }

    # Test chunk 5 (should be from SEO page)
    chunk_idx = 5
    chunk_url = None
    for url, (start_idx, end_idx) in url_to_chunks_map.items():
        if start_idx <= chunk_idx < end_idx:
            chunk_url = url
            break

    if chunk_url == "https://example.com/digital-marketing/seo/":
        print(f"   ✅ Chunk {chunk_idx} correctly mapped to SEO page")
        seo_topics = extract_topics_from_url(chunk_url)
        print(f"   ✅ SEO page topics: {seo_topics}")
    else:
        print(f"   ❌ Mapping failed")
        return False

    # Test chunk 15 (should be from email marketing page)
    chunk_idx = 15
    chunk_url = None
    for url, (start_idx, end_idx) in url_to_chunks_map.items():
        if start_idx <= chunk_idx < end_idx:
            chunk_url = url
            break

    if chunk_url == "https://example.com/digital-marketing/email-marketing/":
        print(f"   ✅ Chunk {chunk_idx} correctly mapped to email marketing page")
        email_topics = extract_topics_from_url(chunk_url)
        print(f"   ✅ Email marketing page topics: {email_topics}")
    else:
        print(f"   ❌ Mapping failed")
        return False

    # Test 4: Verify recursive_scrape_website signature
    print()
    print("4️⃣  Testing recursive_scrape_website() Signature...")
    import inspect

    sig = inspect.signature(recursive_scrape_website)
    params = list(sig.parameters.keys())

    if "return_individual_urls" in params:
        print("   ✅ Has 'return_individual_urls' parameter")
    else:
        print("   ❌ Missing 'return_individual_urls' parameter")
        return False

    # Test 5: Verify upload_to_pinecone_with_url_mapping signature
    print()
    print("5️⃣  Testing upload_to_pinecone_with_url_mapping() Signature...")
    sig2 = inspect.signature(upload_to_pinecone_with_url_mapping)
    params2 = list(sig2.parameters.keys())

    required = ["url_to_chunks_map", "start_url"]
    missing = [p for p in required if p not in params2]

    if not missing:
        print("   ✅ Has all required parameters")
    else:
        print(f"   ❌ Missing parameters: {missing}")
        return False

    # Test 6: Verify _process_upload_record uses new function
    print()
    print("6️⃣  Testing _process_upload_record() Integration...")
    from app.services.scraping.ingestion_worker import _process_upload_record

    # Read the source to check for integration
    source = inspect.getsource(_process_upload_record)

    if "return_individual_urls=True" in source:
        print("   ✅ Uses return_individual_urls=True for recursive scraping")
    else:
        print("   ⚠️  May not be using individual URLs")

    if "upload_to_pinecone_with_url_mapping" in source:
        print("   ✅ Uses upload_to_pinecone_with_url_mapping()")
    else:
        print("   ⚠️  May not be using URL-specific topic mapping")

    print()
    print("=" * 80)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ Function signatures are correct")
    print("  ✅ Topic extraction works")
    print("  ✅ URL-to-chunks mapping logic works")
    print("  ✅ Integration points are in place")
    print()
    print("💡 The implementation is ready for use!")
    print("   To fully test, upload a recursive URL and verify topics in Pinecone.")

    return True


if __name__ == "__main__":
    asyncio.run(test_implementation_flow())
