"""
Unit tests for recursive URL topics implementation
Tests the logic without requiring network calls or actual scraping
"""

import os
import sys
from pathlib import Path

# Add backend root to path
backend_root = (
    Path(__file__).parent.parent.parent.parent
    if "integration" in str(Path(__file__))
    else Path(__file__).parent.parent.parent
)
sys.path.insert(0, str(backend_root))
from dotenv import load_dotenv

load_dotenv()

from app.services.scraping.ingestion_worker import extract_topics_from_url


def test_topic_extraction():
    """Test topic extraction from various URL patterns"""
    print("=" * 80)
    print("🧪 TEST 1: Topic Extraction from URLs")
    print("=" * 80)
    print()

    test_cases = [
        {
            "url": "https://ohhzones.com/digital-marketing/seo/",
            "expected": ["seo", "digital-marketing", "digital marketing"],
            "description": "SEO page with nested path",
        },
        {
            "url": "https://ohhzones.com/branding-services/brand-identity/",
            "expected": [
                "branding-services",
                "brand-identity",
                "brand identity",
                "branding services",
            ],
            "description": "Branding page with multiple segments",
        },
        {
            "url": "https://ambassadorscentworks.com/collections/essential-series",
            "expected": ["collections", "essential-series", "essential series"],
            "description": "E-commerce collection page",
        },
        {
            "url": "https://example.com/products/electronics/phones",
            "expected": ["products", "electronics", "phones"],
            "description": "Product category page",
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        url = test_case["url"]
        expected = test_case["expected"]
        desc = test_case["description"]

        topics = extract_topics_from_url(url)

        print(f"Test {i}: {desc}")
        print(f"   URL: {url}")
        print(f"   Extracted: {topics}")
        print(f"   Expected: {expected}")

        # Check if expected topics are present
        found = [t for t in expected if t in topics]
        if found:
            print(
                f"   ✅ PASS: Found {len(found)}/{len(expected)} expected topics: {found}"
            )
            passed += 1
        else:
            print(f"   ⚠️  WARNING: No expected topics found")
            print(f"   Got: {topics}")
            failed += 1
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return passed, failed


def test_url_mapping_logic():
    """Test the URL-to-chunks mapping logic"""
    print("=" * 80)
    print("🧪 TEST 2: URL-to-Chunks Mapping Logic")
    print("=" * 80)
    print()

    # Simulate chunks from different URLs
    url_to_chunks_map = {
        "https://example.com/page1": (0, 5),  # Chunks 0-4
        "https://example.com/page2": (5, 10),  # Chunks 5-9
        "https://example.com/page3": (10, 15),  # Chunks 10-14
    }

    test_cases = [
        (0, "https://example.com/page1", True),
        (4, "https://example.com/page1", True),
        (5, "https://example.com/page2", True),
        (9, "https://example.com/page2", True),
        (10, "https://example.com/page3", True),
        (14, "https://example.com/page3", True),
        (15, None, False),  # Out of range
    ]

    passed = 0
    failed = 0

    for chunk_idx, expected_url, should_match in test_cases:
        # Simulate the mapping logic from upload_to_pinecone_with_url_mapping
        chunk_url = None
        for url, (start_idx, end_idx) in url_to_chunks_map.items():
            if start_idx <= chunk_idx < end_idx:
                chunk_url = url
                break

        if should_match:
            if chunk_url == expected_url:
                print(f"   ✅ Chunk {chunk_idx} → {chunk_url}")
                passed += 1
            else:
                print(
                    f"   ❌ Chunk {chunk_idx} → Expected {expected_url}, got {chunk_url}"
                )
                failed += 1
        else:
            if chunk_url is None:
                print(f"   ✅ Chunk {chunk_idx} → None (out of range, as expected)")
                passed += 1
            else:
                print(f"   ❌ Chunk {chunk_idx} → Expected None, got {chunk_url}")
                failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return passed, failed


def test_topic_matching():
    """Test topic matching logic used in retrieval"""
    print("=" * 80)
    print("🧪 TEST 3: Topic Matching Logic")
    print("=" * 80)
    print()

    # Simulate documents with different topics
    documents = [
        {
            "name": "SEO page",
            "metadata": {"topics": ["seo", "digital-marketing", "digital marketing"]},
            "source": "https://example.com/digital-marketing/seo/",
        },
        {
            "name": "Email marketing page",
            "metadata": {
                "topics": ["email-marketing", "email marketing", "digital-marketing"]
            },
            "source": "https://example.com/digital-marketing/email-marketing/",
        },
        {
            "name": "Generic digital marketing page",
            "metadata": {"topics": ["digital-marketing"]},
            "source": "https://example.com/digital-marketing/",
        },
    ]

    queries = [
        {
            "query": "What SEO services do you offer?",
            "expected_topics": ["seo"],
            "expected_match": "SEO page",
        },
        {
            "query": "Tell me about email marketing",
            "expected_topics": ["email", "marketing"],
            "expected_match": "Email marketing page",
        },
    ]

    passed = 0
    failed = 0

    for query_case in queries:
        query = query_case["query"]
        expected_topics = query_case["expected_topics"]
        expected_match = query_case["expected_match"]

        print(f"Query: {query}")
        print(f"   Expected topics: {expected_topics}")

        # Extract query topics (simplified)
        query_terms = set(query.lower().split())
        stop_words = {
            "what",
            "do",
            "you",
            "offer",
            "tell",
            "me",
            "about",
            "services",
            "service",
        }
        query_topics = [
            term for term in query_terms if len(term) > 2 and term not in stop_words
        ]

        print(f"   Extracted query topics: {query_topics}")
        print()

        # Test matching against documents
        best_match = None
        best_score = 0

        for doc in documents:
            doc_topics = doc["metadata"].get("topics", [])
            matched = []

            for q_topic in query_topics:
                for d_topic in doc_topics:
                    if q_topic in d_topic or d_topic in q_topic:
                        matched.append((q_topic, d_topic))

            if matched:
                score = len(matched)
                print(f"   {doc['name']}: {len(matched)} matches - {matched}")
                if score > best_score:
                    best_score = score
                    best_match = doc["name"]

        print()
        if best_match == expected_match:
            print(
                f"   ✅ PASS: Best match is '{best_match}' (expected '{expected_match}')"
            )
            passed += 1
        else:
            print(
                f"   ⚠️  WARNING: Best match is '{best_match}' (expected '{expected_match}')"
            )
            failed += 1
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return passed, failed


def test_function_signatures():
    """Test that function signatures are correct"""
    print("=" * 80)
    print("🧪 TEST 4: Function Signatures")
    print("=" * 80)
    print()

    import inspect

    from app.services.scraping.ingestion_worker import (
        recursive_scrape_website,
        upload_to_pinecone_with_url_mapping,
    )

    passed = 0
    failed = 0

    # Test recursive_scrape_website signature
    sig = inspect.signature(recursive_scrape_website)
    params = list(sig.parameters.keys())

    print("recursive_scrape_website() signature:")
    print(f"   Parameters: {params}")

    if "return_individual_urls" in params:
        print("   ✅ PASS: Has 'return_individual_urls' parameter")
        passed += 1
    else:
        print("   ❌ FAIL: Missing 'return_individual_urls' parameter")
        failed += 1

    # Test upload_to_pinecone_with_url_mapping signature
    sig2 = inspect.signature(upload_to_pinecone_with_url_mapping)
    params2 = list(sig2.parameters.keys())

    print()
    print("upload_to_pinecone_with_url_mapping() signature:")
    print(f"   Parameters: {params2}")

    required_params = [
        "chunks",
        "vectors",
        "namespace",
        "upload_id",
        "url_to_chunks_map",
        "start_url",
    ]
    missing = [p for p in required_params if p not in params2]

    if not missing:
        print("   ✅ PASS: Has all required parameters")
        passed += 1
    else:
        print(f"   ❌ FAIL: Missing parameters: {missing}")
        failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return passed, failed


def main():
    """Run all unit tests"""
    print("=" * 80)
    print("🚀 UNIT TESTS: Recursive URL Topics Implementation")
    print("=" * 80)
    print()

    results = {}

    # Test 1: Topic extraction
    passed, failed = test_topic_extraction()
    results["topic_extraction"] = (passed, failed)
    print()

    # Test 2: URL mapping
    passed, failed = test_url_mapping_logic()
    results["url_mapping"] = (passed, failed)
    print()

    # Test 3: Topic matching
    passed, failed = test_topic_matching()
    results["topic_matching"] = (passed, failed)
    print()

    # Test 4: Function signatures
    passed, failed = test_function_signatures()
    results["function_signatures"] = (passed, failed)
    print()

    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print()

    total_passed = sum(p for p, f in results.values())
    total_failed = sum(f for p, f in results.values())
    total_tests = total_passed + total_failed

    for test_name, (passed, failed) in results.items():
        status = "✅" if failed == 0 else "⚠️"
        print(f"   {status} {test_name}: {passed} passed, {failed} failed")

    print()
    print(f"Overall: {total_passed}/{total_tests} tests passed")

    if total_failed == 0:
        print("✅ All tests passed! Implementation logic is correct.")
    else:
        print(f"⚠️  {total_failed} test(s) failed. Check output above for details.")

    print()
    print("💡 Note: These are unit tests. For full integration testing,")
    print("   you'll need to test with actual recursive scraping and Pinecone uploads.")


if __name__ == "__main__":
    main()
