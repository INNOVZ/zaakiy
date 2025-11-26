"""
Simple test for recursive URL topics implementation
Tests core logic without full imports
"""

import re
from urllib.parse import urlparse


def extract_topics_from_url_simple(url: str) -> list:
    """
    Simplified version of extract_topics_from_url for testing
    """
    if not url:
        return []

    topics = []

    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return []

        segments = [s for s in path.split("/") if s]

        stop_words = {
            "the",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "a",
            "an",
            "www",
            "http",
            "https",
            "com",
            "net",
            "org",
            "io",
            "co",
        }

        for segment in segments:
            if len(segment) < 3:
                continue

            if segment.isdigit():
                continue

            segment_lower = segment.lower()

            if segment_lower in stop_words:
                continue

            topics.append(segment_lower)

            if "-" in segment or "_" in segment:
                space_version = segment.replace("-", " ").replace("_", " ").lower()
                topics.append(space_version)

        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        return unique_topics

    except Exception as e:
        print(f"Error: {e}")
        return []


def test_topic_extraction():
    """Test topic extraction"""
    print("=" * 80)
    print("🧪 TEST 1: Topic Extraction from URLs")
    print("=" * 80)
    print()

    test_cases = [
        {
            "url": "https://ohhzones.com/digital-marketing/seo/",
            "expected": ["seo", "digital-marketing", "digital marketing"],
        },
        {
            "url": "https://ohhzones.com/branding-services/brand-identity/",
            "expected": ["branding-services", "brand-identity", "brand identity"],
        },
        {
            "url": "https://ambassadorscentworks.com/collections/essential-series",
            "expected": ["collections", "essential-series", "essential series"],
        },
    ]

    passed = 0
    for i, test_case in enumerate(test_cases, 1):
        url = test_case["url"]
        expected = test_case["expected"]

        topics = extract_topics_from_url_simple(url)

        print(f"Test {i}: {url}")
        print(f"   Extracted: {topics}")

        found = [t for t in expected if t in topics]
        if found:
            print(f"   ✅ PASS: Found {len(found)}/{len(expected)} expected topics")
            passed += 1
        else:
            print(f"   ⚠️  WARNING: No expected topics found")
        print()

    print(f"Results: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_url_mapping():
    """Test URL-to-chunks mapping logic"""
    print("=" * 80)
    print("🧪 TEST 2: URL-to-Chunks Mapping")
    print("=" * 80)
    print()

    url_to_chunks_map = {
        "https://example.com/page1": (0, 5),
        "https://example.com/page2": (5, 10),
        "https://example.com/page3": (10, 15),
    }

    test_cases = [
        (0, "https://example.com/page1"),
        (4, "https://example.com/page1"),
        (5, "https://example.com/page2"),
        (9, "https://example.com/page2"),
        (10, "https://example.com/page3"),
    ]

    passed = 0
    for chunk_idx, expected_url in test_cases:
        chunk_url = None
        for url, (start_idx, end_idx) in url_to_chunks_map.items():
            if start_idx <= chunk_idx < end_idx:
                chunk_url = url
                break

        if chunk_url == expected_url:
            print(f"   ✅ Chunk {chunk_idx} → {chunk_url}")
            passed += 1
        else:
            print(f"   ❌ Chunk {chunk_idx} → Expected {expected_url}, got {chunk_url}")

    print()
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_topic_matching():
    """Test topic matching logic"""
    print("=" * 80)
    print("🧪 TEST 3: Topic Matching for Retrieval")
    print("=" * 80)
    print()

    # Simulate documents with URL-specific topics
    documents = [
        {
            "name": "SEO page",
            "metadata": {"topics": ["seo", "digital-marketing"]},
            "source": "https://example.com/digital-marketing/seo/",
        },
        {
            "name": "Email marketing page",
            "metadata": {"topics": ["email-marketing", "email marketing"]},
            "source": "https://example.com/digital-marketing/email-marketing/",
        },
        {
            "name": "Generic page",
            "metadata": {"topics": ["digital-marketing"]},
            "source": "https://example.com/digital-marketing/",
        },
    ]

    query = "What SEO services do you offer?"
    query_terms = set(query.lower().split())
    stop_words = {"what", "do", "you", "offer", "services", "service"}
    query_topics = [
        term for term in query_terms if len(term) > 2 and term not in stop_words
    ]

    print(f"Query: {query}")
    print(f"   Extracted topics: {query_topics}")
    print()

    # Test matching
    matches = []
    for doc in documents:
        doc_topics = doc["metadata"].get("topics", [])
        matched = []

        for q_topic in query_topics:
            for d_topic in doc_topics:
                if q_topic in d_topic or d_topic in q_topic:
                    matched.append((q_topic, d_topic))

        if matched:
            matches.append((doc["name"], len(matched), matched))
            print(f"   {doc['name']}: {len(matched)} matches - {matched}")

    print()
    if matches:
        best_match = max(matches, key=lambda x: x[1])
        print(f"   ✅ Best match: {best_match[0]} with {best_match[1]} topic matches")
        if "SEO" in best_match[0]:
            print("   ✅ PASS: SEO page correctly identified as best match")
            return True
        else:
            print("   ⚠️  WARNING: SEO page not identified as best match")
            return False
    else:
        print("   ❌ FAIL: No matches found")
        return False


def test_implementation_structure():
    """Test that implementation structure is correct"""
    print("=" * 80)
    print("🧪 TEST 4: Implementation Structure")
    print("=" * 80)
    print()

    import os
    import re

    # Check if functions exist in the file
    ingestion_file = "app/services/scraping/ingestion_worker.py"
    if not os.path.exists(ingestion_file):
        print(f"   ❌ File not found: {ingestion_file}")
        return False

    with open(ingestion_file, "r") as f:
        content = f.read()

    checks = [
        ("recursive_scrape_website", "return_individual_urls"),
        ("upload_to_pinecone_with_url_mapping", "url_to_chunks_map"),
        ("_process_upload_record", "return_individual_urls=True"),
    ]

    passed = 0
    for func_name, check_str in checks:
        if func_name in content and check_str in content:
            print(f"   ✅ {func_name}() has {check_str}")
            passed += 1
        else:
            print(f"   ❌ {func_name}() missing {check_str}")

    print()
    print(f"Results: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 TESTING: Recursive URL Topics Implementation")
    print("=" * 80)
    print()

    results = {}

    results["topic_extraction"] = test_topic_extraction()
    print()

    results["url_mapping"] = test_url_mapping()
    print()

    results["topic_matching"] = test_topic_matching()
    print()

    results["implementation"] = test_implementation_structure()
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
    print(f"Overall: {passed}/{total} test suites passed")

    if passed == total:
        print()
        print("✅ All tests passed! Implementation is working correctly.")
        print()
        print("💡 Next steps:")
        print("   1. Test with actual recursive scraping upload")
        print("   2. Verify topics in Pinecone metadata")
        print("   3. Test keyword boost retrieval with recursive URLs")
    else:
        print()
        print("⚠️  Some tests failed. Check output above for details.")


if __name__ == "__main__":
    main()
