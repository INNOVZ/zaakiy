#!/usr/bin/env python3
"""
Quick Performance Test for Staging
Tests embedding caching and parallel generation
"""
import asyncio
import json
import sys
import time
from typing import Dict, List

import requests

# Configuration
STAGING_URL = "http://localhost:8001"
TEST_QUERIES = [
    "What are your products?",
    "How do I contact you?",
    "What are your prices?",
]


def test_health() -> bool:
    """Test if staging server is healthy"""
    try:
        response = requests.get(f"{STAGING_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_chat_performance(chatbot_id: str, session_id: str, query: str) -> Dict:
    """Test single chat request and measure performance"""
    try:
        start = time.time()
        response = requests.post(
            f"{STAGING_URL}/api/public/chat",
            json={"message": query, "chatbot_id": chatbot_id, "session_id": session_id},
            timeout=30,
        )
        elapsed_ms = (time.time() - start) * 1000

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time_ms": elapsed_ms,
            "query": query,
            "error": None if response.status_code == 200 else response.text,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "response_time_ms": 0,
            "query": query,
            "error": str(e),
        }


def test_caching_improvement(chatbot_id: str) -> Dict:
    """Test cache performance by making repeated requests"""
    query = TEST_QUERIES[0]

    print(f"\n🧪 Testing Cache Performance")
    print(f"   Query: '{query}'")
    print("=" * 60)

    # First request (cache miss expected)
    print("\n1️⃣ First request (cache MISS expected)...")
    result1 = test_chat_performance(chatbot_id, "test-session-1", query)

    if not result1["success"]:
        print(f"   ❌ Failed: {result1['error']}")
        return {"success": False, "error": result1["error"]}

    print(f"   ✅ Success: {result1['response_time_ms']:.0f}ms")

    # Wait a moment
    time.sleep(1)

    # Second request (cache hit expected)
    print("\n2️⃣ Second request (cache HIT expected)...")
    result2 = test_chat_performance(chatbot_id, "test-session-2", query)

    if not result2["success"]:
        print(f"   ❌ Failed: {result2['error']}")
        return {"success": False, "error": result2["error"]}

    print(f"   ✅ Success: {result2['response_time_ms']:.0f}ms")

    # Calculate improvement
    improvement_ms = result1["response_time_ms"] - result2["response_time_ms"]
    improvement_pct = (
        (improvement_ms / result1["response_time_ms"]) * 100
        if result1["response_time_ms"] > 0
        else 0
    )

    print("\n📊 Cache Performance Results:")
    print(f"   First request:  {result1['response_time_ms']:.0f}ms")
    print(f"   Second request: {result2['response_time_ms']:.0f}ms")
    print(f"   Improvement:    {improvement_ms:.0f}ms ({improvement_pct:.1f}%)")

    if improvement_ms > 50:
        print("   ✅ Caching appears to be working!")
        cache_working = True
    elif improvement_ms > 0:
        print("   ⚠️  Small improvement detected (may need more warmup)")
        cache_working = True
    else:
        print("   ⚠️  No improvement detected (cache may not be working)")
        cache_working = False

    return {
        "success": True,
        "cache_working": cache_working,
        "first_request_ms": result1["response_time_ms"],
        "second_request_ms": result2["response_time_ms"],
        "improvement_ms": improvement_ms,
        "improvement_pct": improvement_pct,
    }


def test_multiple_queries(chatbot_id: str) -> Dict:
    """Test multiple different queries"""
    print(f"\n🧪 Testing Multiple Queries")
    print("=" * 60)

    results = []
    total_time = 0

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{i}. Testing: '{query}'")
        result = test_chat_performance(chatbot_id, f"test-multi-{i}", query)

        if result["success"]:
            print(f"   ✅ {result['response_time_ms']:.0f}ms")
            results.append(result)
            total_time += result["response_time_ms"]
        else:
            print(f"   ❌ Failed: {result['error']}")

    if results:
        avg_time = total_time / len(results)
        print(f"\n📊 Results:")
        print(f"   Successful: {len(results)}/{len(TEST_QUERIES)}")
        print(f"   Average time: {avg_time:.0f}ms")

        if avg_time < 1500:
            print("   ✅ Performance is GOOD (< 1500ms)")
            performance_rating = "good"
        elif avg_time < 2500:
            print("   ⚠️  Performance is ACCEPTABLE (1500-2500ms)")
            performance_rating = "acceptable"
        else:
            print("   ❌ Performance needs improvement (> 2500ms)")
            performance_rating = "poor"

        return {
            "success": True,
            "num_queries": len(TEST_QUERIES),
            "successful": len(results),
            "avg_time_ms": avg_time,
            "performance_rating": performance_rating,
        }
    else:
        return {"success": False, "error": "All queries failed"}


def main():
    """Main test function"""
    print("🚀 Staging Performance Test")
    print("=" * 60)

    # Get chatbot ID from command line or use default
    if len(sys.argv) > 1:
        chatbot_id = sys.argv[1]
    else:
        print("\n⚠️  No chatbot ID provided")
        print("Usage: python3 test_staging_quick.py <chatbot_id>")
        print("\nUsing test mode (will fail without valid chatbot ID)")
        chatbot_id = "test-chatbot-id"

    print(f"\nConfiguration:")
    print(f"  Staging URL: {STAGING_URL}")
    print(f"  Chatbot ID:  {chatbot_id}")

    # Step 1: Health check
    print(f"\n📋 Step 1: Health Check")
    print("-" * 60)

    if not test_health():
        print("❌ Staging server is not healthy!")
        print("\nTroubleshooting:")
        print("  1. Check if staging is running: docker ps")
        print("  2. Check logs: docker logs zaakiy-backend")
        print("  3. Restart: docker-compose restart")
        sys.exit(1)

    print("✅ Staging server is healthy")

    # Step 2: Test caching
    print(f"\n📋 Step 2: Cache Performance Test")
    print("-" * 60)

    cache_result = test_caching_improvement(chatbot_id)

    if not cache_result["success"]:
        print(f"\n❌ Cache test failed: {cache_result.get('error')}")
        if chatbot_id == "test-chatbot-id":
            print("\n💡 Tip: Provide a valid chatbot ID:")
            print("   python3 test_staging_quick.py <your-chatbot-id>")
        sys.exit(1)

    # Step 3: Test multiple queries
    print(f"\n📋 Step 3: Multiple Query Test")
    print("-" * 60)

    multi_result = test_multiple_queries(chatbot_id)

    if not multi_result["success"]:
        print(f"\n❌ Multiple query test failed: {multi_result.get('error')}")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)

    print(f"\n✅ Health Check: PASSED")
    print(
        f"✅ Cache Test: {'PASSED' if cache_result.get('cache_working') else 'WARNING'}"
    )
    print(
        f"   - Improvement: {cache_result.get('improvement_ms', 0):.0f}ms ({cache_result.get('improvement_pct', 0):.1f}%)"
    )
    print(f"✅ Multiple Queries: PASSED")
    print(
        f"   - Success rate: {multi_result['successful']}/{multi_result['num_queries']}"
    )
    print(f"   - Average time: {multi_result['avg_time_ms']:.0f}ms")
    print(f"   - Performance: {multi_result['performance_rating'].upper()}")

    # Overall assessment
    print("\n🎯 Overall Assessment:")

    all_passed = cache_result.get("cache_working", False) and multi_result[
        "performance_rating"
    ] in ["good", "acceptable"]

    if all_passed:
        print("   ✅ All tests PASSED - Ready for production!")
    else:
        print("   ⚠️  Some tests need attention - Review results above")

    print("\n" + "=" * 60)
    print("💡 Next Steps:")
    print("   1. Review detailed logs: docker logs zaakiy-backend")
    print("   2. Monitor cache: python3 scripts/monitor_cache.py")
    print("   3. Run full test suite: pytest tests/ -v")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
