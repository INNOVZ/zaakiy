#!/usr/bin/env python3
"""
Live ngrok Testing Script
Tests your application through ngrok with cache performance monitoring
"""
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

import aiohttp


@dataclass
class TestResults:
    """Results from live testing"""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    response_times: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def success_rate(self) -> float:
        return (
            (self.successful / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )

    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0


class NgrokTester:
    """Test application through ngrok"""

    def __init__(self, ngrok_url: str, chatbot_id: str):
        self.ngrok_url = ngrok_url.rstrip("/")
        self.chatbot_id = chatbot_id
        self.results = TestResults()

    async def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ngrok_url}/health", timeout=10
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    async def send_chat_request(
        self, session: aiohttp.ClientSession, message: str, session_id: str
    ) -> Dict:
        """Send a chat request"""
        url = f"{self.ngrok_url}/api/public/chat"

        payload = {
            "message": message,
            "chatbot_id": self.chatbot_id,
            "session_id": session_id,
        }

        start = time.time()
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                elapsed_ms = (time.time() - start) * 1000

                if response.status == 200:
                    data = await response.json()

                    self.results.successful += 1
                    self.results.response_times.append(elapsed_ms)

                    return {
                        "success": True,
                        "time_ms": elapsed_ms,
                        "response": data.get("response", "")[:100],
                        "processing_time": data.get("processing_time", 0),
                    }
                else:
                    self.results.failed += 1
                    return {
                        "success": False,
                        "time_ms": elapsed_ms,
                        "error": f"Status {response.status}",
                    }
        except Exception as e:
            self.results.failed += 1
            return {
                "success": False,
                "time_ms": (time.time() - start) * 1000,
                "error": str(e),
            }

    async def test_cache_performance(self):
        """Test cache performance with repeated queries"""
        print(f"\n🔥 Testing Cache Performance")
        print("=" * 60)

        query = "What are your products?"

        async with aiohttp.ClientSession() as session:
            # First request (cache MISS expected)
            print(f"\n1️⃣ First request (cache MISS expected)...")
            result1 = await self.send_chat_request(session, query, "cache-test-1")

            if result1["success"]:
                print(f"   ✅ {result1['time_ms']:.0f}ms")
            else:
                print(f"   ❌ Failed: {result1.get('error')}")
                return

            # Wait a moment
            await asyncio.sleep(1)

            # Second request (cache HIT expected)
            print(f"\n2️⃣ Second request (cache HIT expected)...")
            result2 = await self.send_chat_request(session, query, "cache-test-2")

            if result2["success"]:
                print(f"   ✅ {result2['time_ms']:.0f}ms")
            else:
                print(f"   ❌ Failed: {result2.get('error')}")
                return

            # Calculate improvement
            improvement_ms = result1["time_ms"] - result2["time_ms"]
            improvement_pct = (
                (improvement_ms / result1["time_ms"] * 100)
                if result1["time_ms"] > 0
                else 0
            )

            print(f"\n📊 Cache Performance:")
            print(f"   First request:  {result1['time_ms']:.0f}ms")
            print(f"   Second request: {result2['time_ms']:.0f}ms")
            print(f"   Improvement:    {improvement_ms:.0f}ms ({improvement_pct:.1f}%)")

            if improvement_ms > 200:
                print(f"   ✅ Excellent cache performance!")
                self.results.cache_hits += 1
            elif improvement_ms > 100:
                print(f"   ✅ Good cache performance")
                self.results.cache_hits += 1
            elif improvement_ms > 0:
                print(f"   ⚠️  Moderate cache performance")
                self.results.cache_hits += 1
            else:
                print(f"   ⚠️  Cache may not be working")
                self.results.cache_misses += 1

    async def test_multiple_queries(self, num_queries: int = 5):
        """Test multiple different queries"""
        print(f"\n🧪 Testing Multiple Queries")
        print("=" * 60)

        queries = [
            "What are your products?",
            "How do I contact you?",
            "What are your prices?",
            "Tell me about your services",
            "Where are you located?",
        ]

        async with aiohttp.ClientSession() as session:
            for i, query in enumerate(queries[:num_queries], 1):
                print(f"\n{i}. Testing: '{query}'")
                result = await self.send_chat_request(session, query, f"multi-test-{i}")

                if result["success"]:
                    print(f"   ✅ {result['time_ms']:.0f}ms")
                else:
                    print(f"   ❌ Failed: {result.get('error')}")

    async def test_concurrent_requests(self, num_concurrent: int = 5):
        """Test concurrent requests"""
        print(f"\n⚡ Testing {num_concurrent} Concurrent Requests")
        print("=" * 60)

        query = "What are your products?"

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.send_chat_request(session, query, f"concurrent-{i}")
                for i in range(num_concurrent)
            ]

            start = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start

            successful = sum(1 for r in results if r["success"])

            print(f"\n📊 Results:")
            print(f"   Successful: {successful}/{num_concurrent}")
            print(f"   Total time: {elapsed:.2f}s")
            print(f"   Avg time per request: {elapsed/num_concurrent:.2f}s")

            if successful == num_concurrent:
                print(f"   ✅ All concurrent requests succeeded!")
            else:
                print(f"   ⚠️  Some requests failed")

    async def get_cache_metrics(self) -> Dict:
        """Get cache metrics from server"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ngrok_url}/api/cache/metrics"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            print(f"⚠️  Could not get cache metrics: {e}")
        return {}

    def print_summary(self):
        """Print test summary"""
        print(f"\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        print(f"\n📈 Request Statistics:")
        print(f"   Total Requests:  {self.results.total_requests}")
        print(
            f"   Successful:      {self.results.successful} ({self.results.success_rate:.1f}%)"
        )
        print(f"   Failed:          {self.results.failed}")

        if self.results.response_times:
            sorted_times = sorted(self.results.response_times)
            p95_index = int(len(sorted_times) * 0.95)

            print(f"\n⏱️  Response Times:")
            print(f"   Average:  {self.results.avg_response_time:.0f}ms")
            print(
                f"   Median:   {statistics.median(self.results.response_times):.0f}ms"
            )
            print(f"   Min:      {min(self.results.response_times):.0f}ms")
            print(f"   Max:      {max(self.results.response_times):.0f}ms")
            print(f"   P95:      {sorted_times[p95_index]:.0f}ms")

        print(f"\n" + "=" * 60)


async def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test application through ngrok")
    parser.add_argument(
        "ngrok_url", help="ngrok URL (e.g., https://abc123.ngrok-free.app)"
    )
    parser.add_argument("chatbot_id", help="Chatbot ID to test")
    parser.add_argument(
        "--cache-only", action="store_true", help="Only test cache performance"
    )
    parser.add_argument(
        "--concurrent", type=int, default=5, help="Number of concurrent requests"
    )

    args = parser.parse_args()

    print("🚀 ngrok Live Testing")
    print("=" * 60)
    print(f"URL: {args.ngrok_url}")
    print(f"Chatbot ID: {args.chatbot_id}")
    print("=" * 60)

    tester = NgrokTester(args.ngrok_url, args.chatbot_id)

    # Health check
    print(f"\n📋 Step 1: Health Check")
    print("-" * 60)

    if not await tester.test_health():
        print("❌ Health check failed! Is your app running?")
        sys.exit(1)

    print("✅ Server is healthy")

    # Cache performance test
    print(f"\n📋 Step 2: Cache Performance Test")
    print("-" * 60)

    await tester.test_cache_performance()

    if not args.cache_only:
        # Multiple queries test
        print(f"\n📋 Step 3: Multiple Queries Test")
        print("-" * 60)

        await tester.test_multiple_queries(num_queries=5)

        # Concurrent requests test
        print(f"\n📋 Step 4: Concurrent Requests Test")
        print("-" * 60)

        await tester.test_concurrent_requests(num_concurrent=args.concurrent)

    # Get server cache metrics
    print(f"\n📋 Step 5: Server Cache Metrics")
    print("-" * 60)

    metrics = await tester.get_cache_metrics()
    if metrics:
        print(f"\n💰 Cache Metrics from Server:")
        print(f"   Hits:       {metrics.get('hits', 0)}")
        print(f"   Misses:     {metrics.get('misses', 0)}")
        print(f"   Hit Rate:   {metrics.get('hit_rate', 0):.1f}%")
        print(f"   Total:      {metrics.get('total_requests', 0)}")

    # Print summary
    tester.print_summary()

    print("\n✅ Testing completed!")
    print("\n💡 Next steps:")
    print("   1. Check ngrok web interface: http://127.0.0.1:4040")
    print("   2. Review application logs")
    print("   3. Monitor cache hit rates over time")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
