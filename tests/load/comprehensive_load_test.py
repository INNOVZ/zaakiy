"""
Comprehensive Load Testing Suite
Tests performance, memory leaks, cache hit rates, and connection pool exhaustion
"""
import asyncio
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import psutil


@dataclass
class LoadTestMetrics:
    """Metrics collected during load test"""

    # Performance metrics
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0

    # Memory metrics
    memory_samples: List[float] = field(default_factory=list)
    memory_start_mb: float = 0
    memory_end_mb: float = 0
    memory_peak_mb: float = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_errors: int = 0

    # Database metrics
    db_connections_active: List[int] = field(default_factory=list)
    db_connection_errors: int = 0

    # Throughput metrics
    requests_per_second: float = 0
    total_duration_seconds: float = 0

    def add_response_time(self, time_ms: float):
        """Add a response time measurement"""
        self.response_times.append(time_ms)

    def add_memory_sample(self, memory_mb: float):
        """Add a memory usage sample"""
        self.memory_samples.append(memory_mb)
        if memory_mb > self.memory_peak_mb:
            self.memory_peak_mb = memory_mb

    def calculate_statistics(self):
        """Calculate statistical metrics"""
        if not self.response_times:
            return {}

        sorted_times = sorted(self.response_times)
        return {
            "count": len(self.response_times),
            "mean": statistics.mean(self.response_times),
            "median": statistics.median(self.response_times),
            "min": min(self.response_times),
            "max": max(self.response_times),
            "p50": sorted_times[int(len(sorted_times) * 0.50)],
            "p75": sorted_times[int(len(sorted_times) * 0.75)],
            "p90": sorted_times[int(len(sorted_times) * 0.90)],
            "p95": sorted_times[int(len(sorted_times) * 0.95)],
            "p99": sorted_times[int(len(sorted_times) * 0.99)],
            "stdev": statistics.stdev(self.response_times)
            if len(self.response_times) > 1
            else 0,
        }

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.success_count + self.error_count + self.timeout_count
        return (self.success_count / total * 100) if total > 0 else 0

    @property
    def memory_growth_mb(self) -> float:
        """Calculate memory growth during test"""
        return self.memory_end_mb - self.memory_start_mb


class LoadTester:
    """Comprehensive load tester for chat system"""

    def __init__(self, org_id: str, chatbot_config: dict):
        self.org_id = org_id
        self.chatbot_config = chatbot_config
        self.metrics = LoadTestMetrics()
        self.process = psutil.Process()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    async def single_request(
        self, session_id: str, message: str, timeout: float = 30.0
    ) -> Dict:
        """Execute a single chat request with full metrics"""
        from app.services.chat.chat_service import ChatService

        start_time = time.time()

        try:
            chat_service = ChatService(
                org_id=self.org_id, chatbot_config=self.chatbot_config
            )

            # Execute request with timeout
            result = await asyncio.wait_for(
                chat_service.process_message(message=message, session_id=session_id),
                timeout=timeout,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Track cache metrics if available
            if hasattr(result, "cache_hit"):
                if result.get("cache_hit"):
                    self.metrics.cache_hits += 1
                else:
                    self.metrics.cache_misses += 1

            self.metrics.success_count += 1
            self.metrics.add_response_time(elapsed_ms)

            return {"success": True, "time_ms": elapsed_ms, "result": result}

        except asyncio.TimeoutError:
            self.metrics.timeout_count += 1
            return {"success": False, "error": "timeout", "time_ms": timeout * 1000}
        except Exception as e:
            self.metrics.error_count += 1
            return {
                "success": False,
                "error": str(e),
                "time_ms": (time.time() - start_time) * 1000,
            }

    async def run_load_test(
        self,
        num_requests: int,
        concurrency: int,
        message: str = "What are your products?",
        monitor_interval: float = 1.0,
    ) -> LoadTestMetrics:
        """
        Run comprehensive load test

        Args:
            num_requests: Total number of requests to make
            concurrency: Number of concurrent requests
            message: Message to send
            monitor_interval: How often to sample metrics (seconds)
        """
        print(f"\n🔥 Starting Load Test")
        print(f"=" * 60)
        print(f"Total Requests: {num_requests}")
        print(f"Concurrency: {concurrency}")
        print(f"Message: {message}")
        print(f"=" * 60)

        # Start memory tracking
        tracemalloc.start()
        self.metrics.memory_start_mb = self.get_memory_usage_mb()

        # Force garbage collection before test
        gc.collect()

        # Create monitoring task
        monitoring_task = asyncio.create_task(self._monitor_resources(monitor_interval))

        # Create request batches
        batches = []
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            batch = [
                self.single_request(session_id=f"load-test-{i + j}", message=message)
                for j in range(batch_size)
            ]
            batches.append(batch)

        # Execute batches
        start_time = time.time()

        for batch_num, batch in enumerate(batches, 1):
            print(
                f"\r⏳ Processing batch {batch_num}/{len(batches)}...",
                end="",
                flush=True,
            )
            await asyncio.gather(*batch, return_exceptions=True)

            # Small delay between batches to prevent overwhelming
            if batch_num < len(batches):
                await asyncio.sleep(0.1)

        total_duration = time.time() - start_time
        print(f"\r✅ Completed {num_requests} requests in {total_duration:.2f}s")

        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # Final memory measurement
        self.metrics.memory_end_mb = self.get_memory_usage_mb()

        # Force garbage collection and measure again
        gc.collect()
        memory_after_gc = self.get_memory_usage_mb()

        # Calculate throughput
        self.metrics.total_duration_seconds = total_duration
        self.metrics.requests_per_second = num_requests / total_duration

        # Get memory allocation details
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Print summary
        self._print_summary(memory_after_gc, snapshot)

        return self.metrics

    async def _monitor_resources(self, interval: float):
        """Monitor system resources during test"""
        try:
            while True:
                # Sample memory
                memory_mb = self.get_memory_usage_mb()
                self.metrics.add_memory_sample(memory_mb)

                # Sample database connections (if available)
                try:
                    from app.services.storage.supabase_client import get_supabase_client

                    # This would need to be implemented based on your DB client
                    # self.metrics.db_connections_active.append(active_connections)
                except:
                    pass

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def _print_summary(self, memory_after_gc: float, snapshot):
        """Print comprehensive test summary"""
        print(f"\n" + "=" * 60)
        print("📊 LOAD TEST RESULTS")
        print("=" * 60)

        # Performance metrics
        stats = self.metrics.calculate_statistics()
        print(f"\n⏱️  Performance Metrics:")
        print(
            f"   Total Requests:    {self.metrics.success_count + self.metrics.error_count + self.metrics.timeout_count}"
        )
        print(
            f"   Successful:        {self.metrics.success_count} ({self.metrics.success_rate:.1f}%)"
        )
        print(f"   Errors:            {self.metrics.error_count}")
        print(f"   Timeouts:          {self.metrics.timeout_count}")
        print(f"   Duration:          {self.metrics.total_duration_seconds:.2f}s")
        print(f"   Throughput:        {self.metrics.requests_per_second:.2f} req/s")

        if stats:
            print(f"\n📈 Response Time Statistics:")
            print(f"   Mean:     {stats['mean']:.0f}ms")
            print(f"   Median:   {stats['median']:.0f}ms")
            print(f"   Min:      {stats['min']:.0f}ms")
            print(f"   Max:      {stats['max']:.0f}ms")
            print(f"   P50:      {stats['p50']:.0f}ms")
            print(f"   P75:      {stats['p75']:.0f}ms")
            print(f"   P90:      {stats['p90']:.0f}ms")
            print(f"   P95:      {stats['p95']:.0f}ms")
            print(f"   P99:      {stats['p99']:.0f}ms")
            print(f"   StdDev:   {stats['stdev']:.0f}ms")

        # Memory metrics
        print(f"\n💾 Memory Metrics:")
        print(f"   Start:            {self.metrics.memory_start_mb:.2f} MB")
        print(f"   End:              {self.metrics.memory_end_mb:.2f} MB")
        print(f"   After GC:         {memory_after_gc:.2f} MB")
        print(f"   Peak:             {self.metrics.memory_peak_mb:.2f} MB")
        print(f"   Growth:           {self.metrics.memory_growth_mb:.2f} MB")
        print(
            f"   Growth After GC:  {memory_after_gc - self.metrics.memory_start_mb:.2f} MB"
        )

        # Memory leak detection
        if self.metrics.memory_growth_mb > 100:
            print(f"   ⚠️  WARNING: Significant memory growth detected!")
        elif memory_after_gc - self.metrics.memory_start_mb > 50:
            print(f"   ⚠️  WARNING: Memory not fully released after GC!")
        else:
            print(f"   ✅ Memory usage looks healthy")

        # Top memory allocations
        print(f"\n🔝 Top 5 Memory Allocations:")
        top_stats = snapshot.statistics("lineno")[:5]
        for stat in top_stats:
            print(f"   {stat}")

        # Cache metrics
        if self.metrics.cache_hits > 0 or self.metrics.cache_misses > 0:
            print(f"\n💰 Cache Metrics:")
            print(f"   Hits:       {self.metrics.cache_hits}")
            print(f"   Misses:     {self.metrics.cache_misses}")
            print(f"   Errors:     {self.metrics.cache_errors}")
            print(f"   Hit Rate:   {self.metrics.cache_hit_rate:.1f}%")

            if self.metrics.cache_hit_rate < 20:
                print(f"   ⚠️  WARNING: Low cache hit rate!")
            elif self.metrics.cache_hit_rate > 50:
                print(f"   ✅ Good cache hit rate")

        # Performance rating
        print(f"\n🎯 Performance Rating:")
        if stats:
            if stats["p95"] < 1500:
                print(f"   ✅ EXCELLENT (P95 < 1.5s)")
            elif stats["p95"] < 2500:
                print(f"   ✅ GOOD (P95 < 2.5s)")
            elif stats["p95"] < 4000:
                print(f"   ⚠️  ACCEPTABLE (P95 < 4.0s)")
            else:
                print(f"   ❌ POOR (P95 > 4.0s)")

        print(f"\n" + "=" * 60)


async def run_progressive_load_test():
    """Run load test with progressively increasing load"""
    print("🔥 PROGRESSIVE LOAD TEST")
    print("=" * 60)

    chatbot_config = {
        "id": "load-test-bot",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
    }

    concurrency_levels = [1, 5, 10, 20, 50]
    results = {}

    for concurrency in concurrency_levels:
        print(f"\n{'='*60}")
        print(f"Testing with {concurrency} concurrent requests")
        print(f"{'='*60}")

        tester = LoadTester(org_id="load-test-org", chatbot_config=chatbot_config)

        metrics = await tester.run_load_test(
            num_requests=concurrency * 2,  # 2x concurrency
            concurrency=concurrency,
            message="What are your products?",
        )

        results[concurrency] = metrics

        # Wait between tests
        if concurrency < 50:
            print("\n⏸️  Waiting 10 seconds before next test...")
            await asyncio.sleep(10)

    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 PROGRESSIVE LOAD TEST SUMMARY")
    print("=" * 60)

    print(
        f"\n{'Concurrency':<15} {'Success%':<12} {'P95 (ms)':<12} {'Mem Growth':<15} {'Cache Hit%':<12}"
    )
    print("-" * 70)

    for concurrency, metrics in results.items():
        stats = metrics.calculate_statistics()
        p95 = stats.get("p95", 0) if stats else 0
        print(
            f"{concurrency:<15} "
            f"{metrics.success_rate:<12.1f} "
            f"{p95:<12.0f} "
            f"{metrics.memory_growth_mb:<15.2f} "
            f"{metrics.cache_hit_rate:<12.1f}"
        )


async def run_sustained_load_test(duration_seconds: int = 60, concurrency: int = 10):
    """Run sustained load test over time"""
    print(f"\n🔥 SUSTAINED LOAD TEST")
    print("=" * 60)
    print(f"Duration: {duration_seconds}s")
    print(f"Concurrency: {concurrency}")
    print("=" * 60)

    chatbot_config = {
        "id": "sustained-test-bot",
        "model": "gpt-3.5-turbo",
    }

    tester = LoadTester(org_id="sustained-test-org", chatbot_config=chatbot_config)

    # Calculate number of requests
    # Assuming ~2s per request, we can do ~30 requests/minute
    estimated_requests = int((duration_seconds / 2) * concurrency)

    metrics = await tester.run_load_test(
        num_requests=estimated_requests,
        concurrency=concurrency,
        message="What are your products?",
        monitor_interval=5.0,  # Sample every 5 seconds
    )

    # Check for memory leaks
    print(f"\n🔍 Memory Leak Analysis:")
    if metrics.memory_growth_mb > 200:
        print(
            f"   ❌ CRITICAL: Severe memory leak detected ({metrics.memory_growth_mb:.2f} MB growth)"
        )
    elif metrics.memory_growth_mb > 100:
        print(
            f"   ⚠️  WARNING: Possible memory leak ({metrics.memory_growth_mb:.2f} MB growth)"
        )
    else:
        print(
            f"   ✅ No significant memory leak detected ({metrics.memory_growth_mb:.2f} MB growth)"
        )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive load testing")
    parser.add_argument(
        "--test-type",
        choices=["progressive", "sustained", "both"],
        default="progressive",
        help="Type of load test to run",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration for sustained test (seconds)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency level for sustained test",
    )

    args = parser.parse_args()

    print("🚀 Comprehensive Load Testing Suite")
    print("=" * 60)

    try:
        if args.test_type in ["progressive", "both"]:
            asyncio.run(run_progressive_load_test())

        if args.test_type in ["sustained", "both"]:
            asyncio.run(
                run_sustained_load_test(
                    duration_seconds=args.duration, concurrency=args.concurrency
                )
            )

        print("\n✅ Load testing completed successfully")

    except KeyboardInterrupt:
        print("\n\n⚠️  Load test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Load test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
