"""
Database Connection Pool Monitor
Monitors database connection pool usage and detects exhaustion
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class ConnectionPoolMetrics:
    """Metrics for database connection pool"""

    timestamp: datetime
    active_connections: int
    idle_connections: int
    total_connections: int
    max_connections: int
    waiting_requests: int = 0
    connection_errors: int = 0

    @property
    def utilization_percent(self) -> float:
        """Calculate pool utilization percentage"""
        return (
            (self.total_connections / self.max_connections * 100)
            if self.max_connections > 0
            else 0
        )

    @property
    def is_exhausted(self) -> bool:
        """Check if pool is exhausted"""
        return self.total_connections >= self.max_connections


class ConnectionPoolMonitor:
    """Monitor database connection pool during load tests"""

    def __init__(self):
        self.metrics_history: List[ConnectionPoolMetrics] = []
        self.exhaustion_events: List[datetime] = []

    async def monitor_pool(
        self, duration_seconds: int = 60, sample_interval: float = 1.0
    ):
        """
        Monitor connection pool for specified duration

        Args:
            duration_seconds: How long to monitor
            sample_interval: How often to sample (seconds)
        """
        print(f"\n🔍 Monitoring Database Connection Pool")
        print(f"=" * 60)
        print(f"Duration: {duration_seconds}s")
        print(f"Sample Interval: {sample_interval}s")
        print(f"=" * 60)

        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            metrics = await self._sample_pool_metrics()
            self.metrics_history.append(metrics)

            # Check for exhaustion
            if metrics.is_exhausted:
                self.exhaustion_events.append(metrics.timestamp)
                print(
                    f"\n⚠️  WARNING: Connection pool exhausted at {metrics.timestamp}"
                )

            # Print current status
            self._print_status(metrics)

            await asyncio.sleep(sample_interval)

        # Print summary
        self._print_summary()

    async def _sample_pool_metrics(self) -> ConnectionPoolMetrics:
        """Sample current connection pool metrics"""
        try:
            from app.services.storage.supabase_client import get_supabase_client

            # This is a placeholder - actual implementation depends on your DB client
            # For Supabase/PostgreSQL, you might query pg_stat_activity
            # Example query to get connection stats:
            # SELECT count(*) FROM pg_stat_activity WHERE datname = 'your_db';
            # For now, return mock data
            # In production, replace with actual pool stats
            return ConnectionPoolMetrics(
                timestamp=datetime.now(),
                active_connections=5,  # Replace with actual
                idle_connections=3,  # Replace with actual
                total_connections=8,  # Replace with actual
                max_connections=20,  # Replace with actual
                waiting_requests=0,  # Replace with actual
                connection_errors=0,  # Replace with actual
            )

        except Exception as e:
            print(f"Error sampling pool metrics: {e}")
            return ConnectionPoolMetrics(
                timestamp=datetime.now(),
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                max_connections=0,
            )

    def _print_status(self, metrics: ConnectionPoolMetrics):
        """Print current pool status"""
        status_line = (
            f"\r⏱️  {metrics.timestamp.strftime('%H:%M:%S')} | "
            f"Active: {metrics.active_connections:2d} | "
            f"Idle: {metrics.idle_connections:2d} | "
            f"Total: {metrics.total_connections:2d}/{metrics.max_connections:2d} | "
            f"Utilization: {metrics.utilization_percent:5.1f}% | "
            f"Waiting: {metrics.waiting_requests:2d}"
        )
        print(status_line, end="", flush=True)

    def _print_summary(self):
        """Print monitoring summary"""
        if not self.metrics_history:
            print("\nNo metrics collected")
            return

        print(f"\n\n" + "=" * 60)
        print("📊 CONNECTION POOL MONITORING SUMMARY")
        print("=" * 60)

        # Calculate statistics
        active_conns = [m.active_connections for m in self.metrics_history]
        total_conns = [m.total_connections for m in self.metrics_history]
        utilization = [m.utilization_percent for m in self.metrics_history]

        print(f"\n📈 Connection Statistics:")
        print(f"   Samples Collected:     {len(self.metrics_history)}")
        print(f"   Max Connections:       {self.metrics_history[0].max_connections}")
        print(f"   Peak Active:           {max(active_conns)}")
        print(f"   Peak Total:            {max(total_conns)}")
        print(f"   Avg Active:            {sum(active_conns)/len(active_conns):.1f}")
        print(f"   Avg Total:             {sum(total_conns)/len(total_conns):.1f}")
        print(f"   Peak Utilization:      {max(utilization):.1f}%")
        print(f"   Avg Utilization:       {sum(utilization)/len(utilization):.1f}%")

        # Exhaustion analysis
        print(f"\n⚠️  Exhaustion Analysis:")
        print(f"   Exhaustion Events:     {len(self.exhaustion_events)}")

        if self.exhaustion_events:
            print(f"   First Exhaustion:      {self.exhaustion_events[0]}")
            print(f"   Last Exhaustion:       {self.exhaustion_events[-1]}")
            print(f"   ❌ CRITICAL: Pool exhaustion detected!")
        else:
            print(f"   ✅ No pool exhaustion detected")

        # Recommendations
        print(f"\n💡 Recommendations:")
        peak_util = max(utilization)
        if peak_util > 90:
            print(
                f"   ⚠️  Increase max_connections (current: {self.metrics_history[0].max_connections})"
            )
            print(f"   ⚠️  Consider connection pooling optimization")
        elif peak_util > 70:
            print(f"   ⚠️  Monitor closely - approaching capacity")
        else:
            print(f"   ✅ Connection pool size is adequate")

        print(f"\n" + "=" * 60)


async def run_load_test_with_pool_monitoring(
    num_requests: int = 100, concurrency: int = 20
):
    """Run load test while monitoring connection pool"""
    print("🔥 LOAD TEST WITH CONNECTION POOL MONITORING")
    print("=" * 60)

    # Start pool monitoring
    monitor = ConnectionPoolMonitor()
    monitoring_task = asyncio.create_task(
        monitor.monitor_pool(duration_seconds=120, sample_interval=1.0)
    )

    # Wait a bit for monitoring to start
    await asyncio.sleep(2)

    # Run load test
    print(f"\n🚀 Starting load test...")
    print(f"   Requests: {num_requests}")
    print(f"   Concurrency: {concurrency}")

    from tests.load.comprehensive_load_test import LoadTester

    chatbot_config = {
        "id": "pool-test-bot",
        "model": "gpt-3.5-turbo",
    }

    tester = LoadTester(org_id="pool-test-org", chatbot_config=chatbot_config)

    await tester.run_load_test(num_requests=num_requests, concurrency=concurrency)

    # Wait for monitoring to complete
    await monitoring_task


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor database connection pool")
    parser.add_argument(
        "--mode",
        choices=["monitor-only", "load-test"],
        default="monitor-only",
        help="Monitoring mode",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Monitoring duration (seconds)"
    )
    parser.add_argument(
        "--requests", type=int, default=100, help="Number of requests for load test"
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Concurrency level for load test"
    )

    args = parser.parse_args()

    try:
        if args.mode == "monitor-only":
            monitor = ConnectionPoolMonitor()
            asyncio.run(
                monitor.monitor_pool(
                    duration_seconds=args.duration, sample_interval=1.0
                )
            )
        else:
            asyncio.run(
                run_load_test_with_pool_monitoring(
                    num_requests=args.requests, concurrency=args.concurrency
                )
            )

        print("\n✅ Monitoring completed successfully")

    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Monitoring failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
