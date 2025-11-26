"""
Load Testing Script for Concurrency
Stress test the chat system under high concurrent load
"""
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LoadTestResult:
    """Results from a load test run"""

    total_requests: int
    successful: int
    failed: int
    avg_response_time: float
    median_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors: List[str]


class LoadTester:
    """Load tester for chat system"""

    def __init__(self, org_id: str, chatbot_config: dict):
        self.org_id = org_id
        self.chatbot_config = chatbot_config
        self.results: List[float] = []
        self.errors: List[str] = []

    async def single_request(self, session_id: str, message: str) -> Dict:
        """Execute a single chat request and measure time"""
        from app.services.chat.chat_service import ChatService

        try:
            chat_service = ChatService(
                org_id=self.org_id, chatbot_config=self.chatbot_config
            )

            start_time = time.time()
            result = await chat_service.process_message(
                message=message, session_id=session_id
            )
            elapsed = time.time() - start_time

            return {"success": True, "time": elapsed, "result": result}
        except Exception as e:
            return {"success": False, "time": 0, "error": str(e)}

    async def run_concurrent_batch(
        self, num_requests: int, message: str = "What are your products?"
    ) -> LoadTestResult:
        """Run a batch of concurrent requests"""
        print(f"\n🚀 Running {num_requests} concurrent requests...")

        # Create tasks
        tasks = [
            self.single_request(session_id=f"load-test-{i}", message=message)
            for i in range(num_requests)
        ]

        # Execute all concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Analyze results
        successful = []
        failed = []
        response_times = []
        errors = []

        for result in results:
            if isinstance(result, Exception):
                failed.append(result)
                errors.append(str(result))
            elif result.get("success"):
                successful.append(result)
                response_times.append(result["time"])
            else:
                failed.append(result)
                errors.append(result.get("error", "Unknown error"))

        # Calculate statistics
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)

            return LoadTestResult(
                total_requests=num_requests,
                successful=len(successful),
                failed=len(failed),
                avg_response_time=statistics.mean(response_times),
                median_response_time=statistics.median(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p95_response_time=sorted_times[p95_index],
                p99_response_time=sorted_times[p99_index],
                requests_per_second=num_requests / total_time,
                errors=errors[:5],  # First 5 errors
            )
        else:
            return LoadTestResult(
                total_requests=num_requests,
                successful=0,
                failed=len(failed),
                avg_response_time=0,
                median_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                errors=errors[:5],
            )

    def print_results(self, result: LoadTestResult):
        """Print formatted results"""
        print("\n" + "=" * 60)
        print("📊 LOAD TEST RESULTS")
        print("=" * 60)

        print(f"\n📈 Request Statistics:")
        print(f"   Total Requests:  {result.total_requests}")
        print(
            f"   Successful:      {result.successful} ({result.successful/result.total_requests*100:.1f}%)"
        )
        print(
            f"   Failed:          {result.failed} ({result.failed/result.total_requests*100:.1f}%)"
        )
        print(f"   Requests/sec:    {result.requests_per_second:.2f}")

        if result.successful > 0:
            print(f"\n⏱️  Response Times:")
            print(f"   Average:  {result.avg_response_time:.3f}s")
            print(f"   Median:   {result.median_response_time:.3f}s")
            print(f"   Min:      {result.min_response_time:.3f}s")
            print(f"   Max:      {result.max_response_time:.3f}s")
            print(f"   P95:      {result.p95_response_time:.3f}s")
            print(f"   P99:      {result.p99_response_time:.3f}s")

            # Performance rating
            if result.avg_response_time < 1.5:
                print(f"\n   ✅ Performance: EXCELLENT (< 1.5s)")
            elif result.avg_response_time < 2.5:
                print(f"\n   ✅ Performance: GOOD (< 2.5s)")
            elif result.avg_response_time < 4.0:
                print(f"\n   ⚠️  Performance: ACCEPTABLE (< 4.0s)")
            else:
                print(f"\n   ❌ Performance: POOR (> 4.0s)")

        if result.errors:
            print(f"\n❌ Sample Errors:")
            for i, error in enumerate(result.errors, 1):
                print(f"   {i}. {error[:100]}")

        print("\n" + "=" * 60)


async def run_progressive_load_test():
    """Run load test with progressively increasing concurrency"""
    print("🔥 PROGRESSIVE LOAD TEST")
    print("=" * 60)
    print("Testing system under increasing concurrent load")
    print("=" * 60)

    # Configuration
    chatbot_config = {
        "id": "load-test-bot",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
    }

    tester = LoadTester(org_id="load-test-org", chatbot_config=chatbot_config)

    # Test with increasing concurrency
    concurrency_levels = [1, 5, 10, 20, 50, 100]

    all_results = {}

    for concurrency in concurrency_levels:
        print(f"\n{'='*60}")
        print(f"Testing with {concurrency} concurrent requests")
        print(f"{'='*60}")

        result = await tester.run_concurrent_batch(concurrency)
        tester.print_results(result)

        all_results[concurrency] = result

        # Wait between tests
        if concurrency < 100:
            print("\nWaiting 5 seconds before next test...")
            await asyncio.sleep(5)

    # Summary
    print("\n" + "=" * 60)
    print("📊 PROGRESSIVE LOAD TEST SUMMARY")
    print("=" * 60)

    print(
        f"\n{'Concurrency':<15} {'Success Rate':<15} {'Avg Time':<15} {'P95 Time':<15}"
    )
    print("-" * 60)

    for concurrency, result in all_results.items():
        success_rate = f"{result.successful/result.total_requests*100:.1f}%"
        avg_time = f"{result.avg_response_time:.2f}s"
        p95_time = f"{result.p95_response_time:.2f}s"
        print(f"{concurrency:<15} {success_rate:<15} {avg_time:<15} {p95_time:<15}")

    # Performance degradation analysis
    if 1 in all_results and 50 in all_results:
        baseline = all_results[1].avg_response_time
        high_load = all_results[50].avg_response_time
        degradation = high_load / baseline if baseline > 0 else 0

        print(f"\n📈 Performance Degradation:")
        print(f"   Baseline (1 request):  {baseline:.2f}s")
        print(f"   High load (50 requests): {high_load:.2f}s")
        print(f"   Degradation factor:    {degradation:.2f}x")

        if degradation < 2.0:
            print("   ✅ System scales well under load")
        elif degradation < 3.0:
            print("   ⚠️  Moderate degradation under load")
        else:
            print("   ❌ Significant degradation - scaling issues detected")


async def run_sustained_load_test(duration_seconds: int = 60, concurrency: int = 10):
    """Run sustained load test over time"""
    print(f"\n🔥 SUSTAINED LOAD TEST")
    print("=" * 60)
    print(f"Duration: {duration_seconds}s")
    print(f"Concurrency: {concurrency} requests/batch")
    print("=" * 60)

    chatbot_config = {
        "id": "sustained-test-bot",
        "model": "gpt-3.5-turbo",
    }

    tester = LoadTester(org_id="sustained-test-org", chatbot_config=chatbot_config)

    start_time = time.time()
    batch_count = 0
    all_results = []

    while (time.time() - start_time) < duration_seconds:
        batch_count += 1
        print(f"\n⏱️  Batch {batch_count} (elapsed: {time.time() - start_time:.0f}s)")

        result = await tester.run_concurrent_batch(concurrency)
        all_results.append(result)

        print(f"   Success: {result.successful}/{result.total_requests}")
        print(f"   Avg time: {result.avg_response_time:.2f}s")

        # Small delay between batches
        await asyncio.sleep(1)

    # Aggregate results
    total_requests = sum(r.total_requests for r in all_results)
    total_successful = sum(r.successful for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    all_times = []
    for r in all_results:
        if r.successful > 0:
            # Approximate - we don't have individual times
            all_times.extend([r.avg_response_time] * r.successful)

    print("\n" + "=" * 60)
    print("📊 SUSTAINED LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"\n   Duration:        {duration_seconds}s")
    print(f"   Batches:         {batch_count}")
    print(f"   Total Requests:  {total_requests}")
    print(
        f"   Successful:      {total_successful} ({total_successful/total_requests*100:.1f}%)"
    )
    print(
        f"   Failed:          {total_failed} ({total_failed/total_requests*100:.1f}%)"
    )

    if all_times:
        print(f"   Avg Time:        {statistics.mean(all_times):.2f}s")
        print(f"   Median Time:     {statistics.median(all_times):.2f}s")

    print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Load test the chat system")
    parser.add_argument(
        "--test-type",
        choices=["progressive", "sustained"],
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

    print("🚀 Chat System Load Tester")
    print("=" * 60)
    print(f"Test Type: {args.test_type}")
    print("=" * 60)

    try:
        if args.test_type == "progressive":
            asyncio.run(run_progressive_load_test())
        else:
            asyncio.run(
                run_sustained_load_test(
                    duration_seconds=args.duration, concurrency=args.concurrency
                )
            )

        print("\n✅ Load test completed successfully")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Load test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Load test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
