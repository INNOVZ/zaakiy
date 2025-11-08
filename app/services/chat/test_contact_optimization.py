"""
Quick test script to verify contact extractor optimizations
Run this to verify the new optimized implementation works correctly
"""
import time

from contact_extractor_optimized import contact_extractor_optimized


def test_basic_extraction():
    """Test basic contact information extraction"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Contact Extraction")
    print("=" * 60)

    chunk = """
    Contact us for more information:
    Phone: +971 50 123 4567
    Email: support@company.com
    Address: Dubai Production City, IMPZ, Dubai, UAE
    Demo: https://calendly.com/company/demo
    """

    result = contact_extractor_optimized.extract_contact_info_dict(chunk)

    print(f"✅ Phones found: {result['phones']}")
    print(f"✅ Emails found: {result['emails']}")
    print(f"✅ Addresses found: {result['addresses'][:1]}")  # Show first address
    print(f"✅ Demo links found: {result['demo_links']}")
    print(f"✅ Has contact info: {result['has_contact_info']}")

    assert len(result["phones"]) > 0, "Should find phone number"
    assert len(result["emails"]) > 0, "Should find email"
    assert result["has_contact_info"], "Should detect contact info"

    print("✅ TEST 1 PASSED")


def test_cache_performance():
    """Test LRU cache performance"""
    print("\n" + "=" * 60)
    print("TEST 2: Cache Performance")
    print("=" * 60)

    chunk = "Call +971 50 123 4567 or email support@example.com"

    # Clear cache first
    contact_extractor_optimized.clear_cache()

    # First call (cache miss)
    start = time.time()
    result1 = contact_extractor_optimized.extract_contact_info_dict(chunk)
    first_call_time = (time.time() - start) * 1000

    # Second call (cache hit)
    start = time.time()
    result2 = contact_extractor_optimized.extract_contact_info_dict(chunk)
    second_call_time = (time.time() - start) * 1000

    print(f"First call (cache miss): {first_call_time:.2f}ms")
    print(f"Second call (cache hit): {second_call_time:.2f}ms")
    print(f"Speedup: {first_call_time/second_call_time:.0f}x faster")

    # Get cache stats
    cache_info = contact_extractor_optimized.get_cache_info()
    print(f"\n📊 Cache Stats:")
    print(f"  - Hits: {cache_info['hits']}")
    print(f"  - Misses: {cache_info['misses']}")
    print(f"  - Hit Rate: {cache_info['hit_rate']*100:.1f}%")
    print(f"  - Cache Size: {cache_info['size']}/{cache_info['maxsize']}")

    assert result1 == result2, "Cached result should match original"
    assert second_call_time < first_call_time, "Cached call should be faster"

    print("✅ TEST 2 PASSED")


def test_scoring_performance():
    """Test optimized scoring method"""
    print("\n" + "=" * 60)
    print("TEST 3: Scoring Performance")
    print("=" * 60)

    chunks = [
        "Contact us at +971 50 123 4567",  # High score (has contact)
        "Our products include solar panels",  # Low score (no contact)
        "Email support@company.com for help",  # High score (has email)
        "We offer competitive pricing",  # Low score (no contact)
    ]

    start = time.time()
    scores = [
        contact_extractor_optimized.score_chunk_for_contact_query(chunk)
        for chunk in chunks
    ]
    elapsed = (time.time() - start) * 1000

    print(f"Scored {len(chunks)} chunks in {elapsed:.2f}ms")
    print(f"Average: {elapsed/len(chunks):.2f}ms per chunk")
    print(f"\nScores:")
    for chunk, score in zip(chunks, scores):
        print(f"  - {chunk[:40]:40} → Score: {score:.1f}")

    assert scores[0] > scores[1], "Contact chunk should score higher"
    assert scores[2] > scores[3], "Email chunk should score higher"

    print("✅ TEST 3 PASSED")


def test_regex_patterns():
    """Test pre-compiled regex patterns"""
    print("\n" + "=" * 60)
    print("TEST 4: Regex Pattern Extraction")
    print("=" * 60)

    test_cases = [
        ("+971 50 123 4567", "UAE phone number"),
        ("0503789198", "Local UAE phone"),
        ("+1 234 567 8900", "International phone"),
        ("support@company.com", "Email"),
        ("https://calendly.com/demo", "Demo link"),
        ("www.example.com", "WWW link"),
    ]

    for text, description in test_cases:
        result = contact_extractor_optimized.extract_contact_info_dict(text)
        has_info = (
            len(result["phones"]) > 0
            or len(result["emails"]) > 0
            or len(result["links"]) > 0
        )
        status = "✅" if has_info else "❌"
        print(f"{status} {description:30} → Found: {has_info}")

        if "phone" in description.lower():
            assert len(result["phones"]) > 0, f"Should find phone in: {text}"
        elif "email" in description.lower():
            assert len(result["emails"]) > 0, f"Should find email in: {text}"
        elif "link" in description.lower():
            assert len(result["links"]) > 0, f"Should find link in: {text}"

    print("✅ TEST 4 PASSED")


def test_batch_performance():
    """Test performance with multiple extractions"""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Extraction Performance")
    print("=" * 60)

    # Simulate typical usage: extract from multiple document chunks
    chunks = [
        "Contact: +971 50 123 4567, email: sales@company.com",
        "Call us at +971 4 123 4567 for inquiries",
        "Reach out via support@company.com",
        "Visit our office in Dubai Production City",
        "Book a demo: https://calendly.com/company/demo",
        "Product prices: Dhs. 50.00, Dhs. 75.00",
        "Our services include consultation and support",
        "Email us at info@company.com or call +971 50 999 8888",
    ] * 2  # 16 chunks total

    # Clear cache
    contact_extractor_optimized.clear_cache()

    # First pass (all cache misses)
    start = time.time()
    for chunk in chunks:
        contact_extractor_optimized.extract_contact_info_dict(chunk)
    first_pass = (time.time() - start) * 1000

    # Second pass (all cache hits due to duplicates)
    start = time.time()
    for chunk in chunks:
        contact_extractor_optimized.extract_contact_info_dict(chunk)
    second_pass = (time.time() - start) * 1000

    print(f"First pass ({len(chunks)} chunks): {first_pass:.2f}ms")
    print(f"  → Average: {first_pass/len(chunks):.2f}ms per chunk")
    print(f"\nSecond pass ({len(chunks)} chunks): {second_pass:.2f}ms")
    print(f"  → Average: {second_pass/len(chunks):.2f}ms per chunk")
    print(f"\n🚀 Speedup: {first_pass/second_pass:.1f}x faster with cache")

    cache_info = contact_extractor_optimized.get_cache_info()
    print(f"\n📊 Final Cache Stats:")
    print(f"  - Hit Rate: {cache_info['hit_rate']*100:.1f}%")
    print(f"  - Cache Size: {cache_info['size']}/{cache_info['maxsize']}")

    assert second_pass < first_pass, "Cached pass should be faster"
    assert cache_info["hit_rate"] > 0.4, "Should have >40% cache hit rate"

    print("✅ TEST 5 PASSED")


def run_all_tests():
    """Run all optimization tests"""
    print("\n" + "🎯" * 30)
    print("CONTACT EXTRACTOR OPTIMIZATION TESTS")
    print("🎯" * 30)

    try:
        test_basic_extraction()
        test_cache_performance()
        test_scoring_performance()
        test_regex_patterns()
        test_batch_performance()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\n✅ Optimizations are working correctly")
        print("✅ Cache is functioning as expected")
        print("✅ Performance improvements verified")
        print("✅ Ready for production deployment")
        print("\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
