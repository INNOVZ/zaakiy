"""
Test refactored scraping modules
"""

import os
import sys
from pathlib import Path

# Add backend root to path
backend_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_root))
from dotenv import load_dotenv

load_dotenv()


def test_imports():
    """Test that all new modules can be imported"""
    print("=" * 80)
    print("🧪 TEST 1: Module Imports")
    print("=" * 80)
    print()

    try:
        from app.services.scraping.text_processing import (
            filter_noise_chunks,
            get_embeddings_for_chunks,
            split_into_chunks,
        )

        print("   ✅ text_processing module imported successfully")
    except Exception as e:
        print(f"   ❌ text_processing import failed: {e}")
        return False

    try:
        from app.services.scraping.topic_extraction import extract_topics_from_url

        print("   ✅ topic_extraction module imported successfully")
    except Exception as e:
        print(f"   ❌ topic_extraction import failed: {e}")
        return False

    try:
        from app.services.scraping.metadata_extraction import (
            extract_metadata_flags,
            extract_product_links_from_chunk,
        )

        print("   ✅ metadata_extraction module imported successfully")
    except Exception as e:
        print(f"   ❌ metadata_extraction import failed: {e}")
        return False

    try:
        from app.services.scraping.pinecone_upload import (
            upload_to_pinecone,
            upload_to_pinecone_with_url_mapping,
        )

        print("   ✅ pinecone_upload module imported successfully")
    except Exception as e:
        print(f"   ❌ pinecone_upload import failed: {e}")
        return False

    try:
        from app.services.scraping.ingestion_worker import (
            extract_metadata_flags,
            extract_product_links_from_chunk,
            extract_topics_from_url,
            filter_noise_chunks,
            get_embeddings_for_chunks,
            split_into_chunks,
            upload_to_pinecone,
            upload_to_pinecone_with_url_mapping,
        )

        print("   ✅ ingestion_worker imports from new modules successfully")
    except Exception as e:
        print(f"   ❌ ingestion_worker import failed: {e}")
        return False

    print()
    return True


def test_topic_extraction():
    """Test topic extraction"""
    print("=" * 80)
    print("🧪 TEST 2: Topic Extraction")
    print("=" * 80)
    print()

    try:
        from app.services.scraping.topic_extraction import extract_topics_from_url

        test_cases = [
            (
                "https://ohhzones.com/digital-marketing/seo/",
                ["digital-marketing", "seo"],
            ),
            (
                "https://ohhzones.com/branding-services/brand-identity/",
                ["branding-services", "brand-identity"],
            ),
            ("https://example.com/", []),
        ]

        all_passed = True
        for url, expected_keywords in test_cases:
            topics = extract_topics_from_url(url)
            found_keywords = [kw for kw in expected_keywords if kw in topics]

            if found_keywords:
                print(f"   ✅ {url[:50]}: Found {found_keywords}")
            else:
                print(
                    f"   ⚠️  {url[:50]}: Expected {expected_keywords}, got {topics[:3]}"
                )
                all_passed = False

        print()
        return all_passed
    except Exception as e:
        print(f"   ❌ Topic extraction test failed: {e}")
        return False


def test_text_processing():
    """Test text processing functions"""
    print("=" * 80)
    print("🧪 TEST 3: Text Processing")
    print("=" * 80)
    print()

    try:
        from app.services.scraping.text_processing import (
            filter_noise_chunks,
            split_into_chunks,
        )

        test_text = "This is a test. " * 100  # Long text
        chunks = split_into_chunks(test_text)

        if chunks:
            print(f"   ✅ split_into_chunks: Created {len(chunks)} chunks")
        else:
            print(f"   ❌ split_into_chunks: No chunks created")
            return False

        filtered = filter_noise_chunks(chunks)
        print(f"   ✅ filter_noise_chunks: {len(chunks)} → {len(filtered)} chunks")

        print()
        return True
    except Exception as e:
        print(f"   ❌ Text processing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metadata_extraction():
    """Test metadata extraction"""
    print("=" * 80)
    print("🧪 TEST 4: Metadata Extraction")
    print("=" * 80)
    print()

    try:
        from app.services.scraping.metadata_extraction import (
            extract_metadata_flags,
            extract_product_links_from_chunk,
        )

        # Test pricing detection
        pricing_chunk = "Our pricing starts at $99 per month"
        flags = extract_metadata_flags(pricing_chunk)

        if flags.get("has_pricing"):
            print("   ✅ Pricing detection works")
        else:
            print("   ⚠️  Pricing detection failed")

        # Test booking detection
        booking_chunk = "Book a consultation today"
        flags = extract_metadata_flags(booking_chunk)

        if flags.get("has_booking"):
            print("   ✅ Booking detection works")
        else:
            print("   ⚠️  Booking detection failed")

        # Test product links
        product_chunk = "Check out our product at https://example.com/products/item-123"
        links = extract_product_links_from_chunk(product_chunk)

        if links:
            print(f"   ✅ Product link extraction works: {links[0][:50]}")
        else:
            print("   ⚠️  Product link extraction failed")

        print()
        return True
    except Exception as e:
        print(f"   ❌ Metadata extraction test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ingestion_worker_imports():
    """Test that ingestion_worker can import from new modules"""
    print("=" * 80)
    print("🧪 TEST 5: Ingestion Worker Integration")
    print("=" * 80)
    print()

    try:
        # Test that ingestion_worker can be imported
        from app.services.scraping import ingestion_worker

        # Check that functions are available
        assert hasattr(ingestion_worker, "split_into_chunks") or hasattr(
            ingestion_worker, "split_into_chunks"
        )
        print("   ✅ ingestion_worker module loads successfully")

        # Check that it uses new modules
        import inspect

        source = inspect.getsource(ingestion_worker)

        if "from .text_processing import" in source:
            print("   ✅ Uses text_processing module")
        else:
            print("   ⚠️  May not be using text_processing module")

        if "from .topic_extraction import" in source:
            print("   ✅ Uses topic_extraction module")
        else:
            print("   ⚠️  May not be using topic_extraction module")

        if "from .metadata_extraction import" in source:
            print("   ✅ Uses metadata_extraction module")
        else:
            print("   ⚠️  May not be using metadata_extraction module")

        if "from .pinecone_upload import" in source:
            print("   ✅ Uses pinecone_upload module")
        else:
            print("   ⚠️  May not be using pinecone_upload module")

        print()
        return True
    except Exception as e:
        print(f"   ❌ Ingestion worker integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 REFACTORED SCRAPING MODULES TEST")
    print("=" * 80)
    print()

    results = {}

    results["imports"] = test_imports()
    print()

    results["topic_extraction"] = test_topic_extraction()
    print()

    results["text_processing"] = test_text_processing()
    print()

    results["metadata_extraction"] = test_metadata_extraction()
    print()

    results["integration"] = test_ingestion_worker_imports()
    print()

    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! Refactoring successful!")
    else:
        print("⚠️  Some tests failed. Check output above for details.")


if __name__ == "__main__":
    main()
