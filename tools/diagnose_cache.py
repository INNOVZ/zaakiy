"""
Cache Hit Rate Troubleshooting Tool
Diagnoses and fixes low cache hit rate issues
"""
import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class CacheAnalysis:
    """Analysis results for cache performance"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_errors: int = 0

    # Key analysis
    unique_keys: int = 0
    duplicate_keys: int = 0
    key_patterns: Dict[str, int] = field(default_factory=dict)

    # TTL analysis
    expired_keys: int = 0
    avg_ttl_seconds: float = 0

    # Performance
    avg_hit_time_ms: float = 0
    avg_miss_time_ms: float = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0

    @property
    def error_rate(self) -> float:
        """Calculate cache error rate percentage"""
        return (
            (self.cache_errors / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )


class CacheDiagnostics:
    """Diagnostic tool for cache issues"""

    def __init__(self):
        self.analysis = CacheAnalysis()
        self.key_access_log: List[Dict] = []
        self.key_frequency: Dict[str, int] = defaultdict(int)

    async def analyze_cache_performance(
        self, num_requests: int = 100, test_queries: Optional[List[str]] = None
    ) -> CacheAnalysis:
        """
        Analyze cache performance with test queries

        Args:
            num_requests: Number of test requests to make
            test_queries: List of queries to test (if None, uses defaults)
        """
        print(f"\n🔍 Cache Performance Analysis")
        print(f"=" * 60)
        print(f"Test Requests: {num_requests}")
        print(f"=" * 60)

        if not test_queries:
            test_queries = self._get_default_test_queries()

        # Test cache with various patterns
        await self._test_cache_patterns(num_requests, test_queries)

        # Analyze results
        self._analyze_key_patterns()
        self._analyze_ttl_effectiveness()

        # Print diagnosis
        self._print_diagnosis()

        # Provide recommendations
        self._print_recommendations()

        return self.analysis

    def _get_default_test_queries(self) -> List[str]:
        """Get default test queries"""
        return [
            "What are your products?",
            "How do I contact you?",
            "What are your prices?",
            "Tell me about your services",
            "Where are you located?",
        ]

    async def _test_cache_patterns(self, num_requests: int, test_queries: List[str]):
        """Test various cache access patterns"""
        from app.services.shared import cache_service

        if not cache_service:
            print("❌ Cache service not available")
            return

        print(f"\n📊 Testing cache patterns...")

        # Pattern 1: Repeated queries (should have high hit rate)
        print(f"\n1️⃣ Testing repeated queries (expect high hit rate)...")
        await self._test_repeated_queries(test_queries[0], 20)

        # Pattern 2: Unique queries (should have low hit rate)
        print(f"\n2️⃣ Testing unique queries (expect low hit rate)...")
        await self._test_unique_queries(20)

        # Pattern 3: Mixed pattern (realistic usage)
        print(f"\n3️⃣ Testing mixed pattern (realistic usage)...")
        await self._test_mixed_pattern(test_queries, 60)

    async def _test_repeated_queries(self, query: str, count: int):
        """Test cache with repeated queries"""
        from app.services.shared import cache_service

        cache_key = f"test:{hashlib.md5(query.encode()).hexdigest()}"

        for i in range(count):
            self.analysis.total_requests += 1

            start = time.time()
            cached = await cache_service.get(cache_key)
            elapsed_ms = (time.time() - start) * 1000

            if cached:
                self.analysis.cache_hits += 1
                self.analysis.avg_hit_time_ms = (
                    self.analysis.avg_hit_time_ms * (self.analysis.cache_hits - 1)
                    + elapsed_ms
                ) / self.analysis.cache_hits
            else:
                self.analysis.cache_misses += 1
                # Set value for next iteration
                await cache_service.set(
                    cache_key, {"query": query, "result": "test"}, ttl=3600
                )
                self.analysis.avg_miss_time_ms = (
                    self.analysis.avg_miss_time_ms * (self.analysis.cache_misses - 1)
                    + elapsed_ms
                ) / self.analysis.cache_misses

            self.key_frequency[cache_key] += 1
            self.key_access_log.append(
                {
                    "key": cache_key,
                    "hit": bool(cached),
                    "time_ms": elapsed_ms,
                    "timestamp": datetime.now(),
                }
            )

        hit_rate = (self.analysis.cache_hits / count * 100) if count > 0 else 0
        print(f"   Hit rate: {hit_rate:.1f}% ({self.analysis.cache_hits}/{count})")

    async def _test_unique_queries(self, count: int):
        """Test cache with unique queries"""
        from app.services.shared import cache_service

        for i in range(count):
            self.analysis.total_requests += 1

            # Generate unique query
            query = f"Unique query {i} at {time.time()}"
            cache_key = f"test:{hashlib.md5(query.encode()).hexdigest()}"

            start = time.time()
            cached = await cache_service.get(cache_key)
            elapsed_ms = (time.time() - start) * 1000

            if cached:
                self.analysis.cache_hits += 1
            else:
                self.analysis.cache_misses += 1
                await cache_service.set(cache_key, {"query": query}, ttl=3600)

            self.key_frequency[cache_key] += 1
            self.key_access_log.append(
                {
                    "key": cache_key,
                    "hit": bool(cached),
                    "time_ms": elapsed_ms,
                    "timestamp": datetime.now(),
                }
            )

        hit_rate = self.analysis.cache_hits / self.analysis.total_requests * 100
        print(f"   Hit rate: {hit_rate:.1f}%")

    async def _test_mixed_pattern(self, queries: List[str], count: int):
        """Test cache with mixed access pattern"""
        import random

        from app.services.shared import cache_service

        for i in range(count):
            self.analysis.total_requests += 1

            # 70% repeated queries, 30% unique
            if random.random() < 0.7:
                query = random.choice(queries)
            else:
                query = f"Unique query {i}"

            cache_key = f"test:{hashlib.md5(query.encode()).hexdigest()}"

            start = time.time()
            cached = await cache_service.get(cache_key)
            elapsed_ms = (time.time() - start) * 1000

            if cached:
                self.analysis.cache_hits += 1
            else:
                self.analysis.cache_misses += 1
                await cache_service.set(cache_key, {"query": query}, ttl=3600)

            self.key_frequency[cache_key] += 1
            self.key_access_log.append(
                {
                    "key": cache_key,
                    "hit": bool(cached),
                    "time_ms": elapsed_ms,
                    "timestamp": datetime.now(),
                }
            )

        hit_rate = self.analysis.cache_hits / self.analysis.total_requests * 100
        print(f"   Hit rate: {hit_rate:.1f}%")

    def _analyze_key_patterns(self):
        """Analyze cache key patterns"""
        self.analysis.unique_keys = len(self.key_frequency)

        # Find keys accessed multiple times
        for key, count in self.key_frequency.items():
            if count > 1:
                self.analysis.duplicate_keys += 1

        # Analyze key patterns
        for key in self.key_frequency.keys():
            if "embedding:" in key:
                self.analysis.key_patterns["embedding"] = (
                    self.analysis.key_patterns.get("embedding", 0) + 1
                )
            elif "conversation:" in key:
                self.analysis.key_patterns["conversation"] = (
                    self.analysis.key_patterns.get("conversation", 0) + 1
                )
            elif "test:" in key:
                self.analysis.key_patterns["test"] = (
                    self.analysis.key_patterns.get("test", 0) + 1
                )

    def _analyze_ttl_effectiveness(self):
        """Analyze TTL effectiveness"""
        # Check if keys are expiring too quickly
        # This would require tracking key lifetimes
        # For now, we'll estimate based on access patterns

        if self.key_access_log:
            # Group accesses by key
            key_accesses = defaultdict(list)
            for access in self.key_access_log:
                key_accesses[access["key"]].append(access["timestamp"])

            # Check for keys that were accessed but expired
            for key, timestamps in key_accesses.items():
                if len(timestamps) > 1:
                    # Check time between accesses
                    for i in range(1, len(timestamps)):
                        time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
                        if time_diff > 3600:  # Assuming 1 hour TTL
                            self.analysis.expired_keys += 1

    def _print_diagnosis(self):
        """Print diagnostic results"""
        print(f"\n" + "=" * 60)
        print("📊 CACHE PERFORMANCE DIAGNOSIS")
        print("=" * 60)

        print(f"\n📈 Overall Statistics:")
        print(f"   Total Requests:    {self.analysis.total_requests}")
        print(f"   Cache Hits:        {self.analysis.cache_hits}")
        print(f"   Cache Misses:      {self.analysis.cache_misses}")
        print(f"   Cache Errors:      {self.analysis.cache_errors}")
        print(f"   Hit Rate:          {self.analysis.hit_rate:.1f}%")
        print(f"   Error Rate:        {self.analysis.error_rate:.1f}%")

        print(f"\n🔑 Key Analysis:")
        print(f"   Unique Keys:       {self.analysis.unique_keys}")
        print(f"   Repeated Keys:     {self.analysis.duplicate_keys}")
        print(f"   Key Patterns:")
        for pattern, count in self.analysis.key_patterns.items():
            print(f"      {pattern}: {count}")

        print(f"\n⏱️  Performance:")
        print(f"   Avg Hit Time:      {self.analysis.avg_hit_time_ms:.2f}ms")
        print(f"   Avg Miss Time:     {self.analysis.avg_miss_time_ms:.2f}ms")

        # Top accessed keys
        print(f"\n🔝 Top 10 Most Accessed Keys:")
        top_keys = sorted(self.key_frequency.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        for i, (key, count) in enumerate(top_keys, 1):
            key_short = key[:50] + "..." if len(key) > 50 else key
            print(f"   {i}. {key_short}: {count} accesses")

    def _print_recommendations(self):
        """Print recommendations based on analysis"""
        print(f"\n" + "=" * 60)
        print("💡 RECOMMENDATIONS")
        print("=" * 60)

        issues_found = []

        # Check hit rate
        if self.analysis.hit_rate < 20:
            issues_found.append("low_hit_rate")
            print(f"\n❌ CRITICAL: Very Low Hit Rate ({self.analysis.hit_rate:.1f}%)")
            print(f"   Causes:")
            print(f"   - Too many unique queries (no repetition)")
            print(f"   - Cache keys not consistent")
            print(f"   - TTL too short")
            print(f"   - Cache not being populated")
            print(f"\n   Solutions:")
            print(f"   1. Normalize queries before caching")
            print(f"   2. Increase cache TTL (current: 3600s)")
            print(f"   3. Implement cache warming")
            print(f"   4. Review cache key generation logic")

        elif self.analysis.hit_rate < 40:
            issues_found.append("moderate_hit_rate")
            print(f"\n⚠️  WARNING: Moderate Hit Rate ({self.analysis.hit_rate:.1f}%)")
            print(f"   Target: > 40% for good performance")
            print(f"\n   Solutions:")
            print(f"   1. Increase TTL for frequently accessed data")
            print(f"   2. Implement query normalization")
            print(f"   3. Add cache warming for common queries")

        # Check key diversity
        if self.analysis.unique_keys > self.analysis.total_requests * 0.8:
            issues_found.append("high_key_diversity")
            print(f"\n⚠️  WARNING: High Key Diversity")
            print(f"   Unique Keys: {self.analysis.unique_keys}")
            print(f"   Total Requests: {self.analysis.total_requests}")
            print(
                f"   Ratio: {self.analysis.unique_keys/self.analysis.total_requests*100:.1f}%"
            )
            print(f"\n   This indicates:")
            print(f"   - Most queries are unique (no cache benefit)")
            print(f"   - Cache keys may include timestamps or random data")
            print(f"\n   Solutions:")
            print(f"   1. Remove timestamps from cache keys")
            print(f"   2. Normalize query text (lowercase, trim)")
            print(f"   3. Use semantic similarity for similar queries")

        # Check error rate
        if self.analysis.error_rate > 1:
            issues_found.append("high_error_rate")
            print(
                f"\n❌ CRITICAL: High Cache Error Rate ({self.analysis.error_rate:.1f}%)"
            )
            print(f"\n   Solutions:")
            print(f"   1. Check Redis/cache service connectivity")
            print(f"   2. Review error logs")
            print(f"   3. Implement retry logic")
            print(f"   4. Add circuit breaker pattern")

        # Check performance
        if self.analysis.avg_miss_time_ms > 100:
            issues_found.append("slow_miss_time")
            print(
                f"\n⚠️  WARNING: Slow Cache Miss Time ({self.analysis.avg_miss_time_ms:.2f}ms)"
            )
            print(f"   Target: < 50ms")
            print(f"\n   Solutions:")
            print(f"   1. Check network latency to cache server")
            print(f"   2. Consider local in-memory cache")
            print(f"   3. Optimize cache key generation")

        # Summary
        if not issues_found:
            print(f"\n✅ Cache Performance Looks Good!")
            print(f"   Hit Rate: {self.analysis.hit_rate:.1f}%")
            print(f"   No critical issues detected")
        else:
            print(f"\n📋 Issues Found: {len(issues_found)}")
            for issue in issues_found:
                print(f"   - {issue}")

        print(f"\n" + "=" * 60)


async def run_cache_diagnostics():
    """Run comprehensive cache diagnostics"""
    diagnostics = CacheDiagnostics()

    print("🔍 Starting Cache Diagnostics")
    print("=" * 60)

    try:
        analysis = await diagnostics.analyze_cache_performance(num_requests=100)

        print(f"\n✅ Diagnostics completed")
        print(f"   Final Hit Rate: {analysis.hit_rate:.1f}%")

        return analysis

    except Exception as e:
        print(f"\n❌ Diagnostics failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main entry point"""
    print("🚀 Cache Hit Rate Troubleshooting Tool")
    print("=" * 60)

    try:
        asyncio.run(run_cache_diagnostics())
        print("\n✅ Analysis completed successfully")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
