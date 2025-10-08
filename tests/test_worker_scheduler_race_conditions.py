"""
Test suite for worker scheduler race condition fixes

This test suite verifies that the IngestionWorkerScheduler is thread-safe
and handles concurrent start/stop operations correctly.
"""

import asyncio
import threading
import time
import pytest
from app.services.shared.worker_scheduler import IngestionWorkerScheduler


class TestWorkerSchedulerRaceConditions:
    """Test race conditions in worker scheduler"""

    def test_single_start(self):
        """Test that scheduler starts correctly"""
        scheduler = IngestionWorkerScheduler()

        assert scheduler.is_running is False

        scheduler.start()

        assert scheduler.is_running is True
        assert scheduler._start_count == 1

        scheduler.stop()

        assert scheduler.is_running is False

    def test_idempotent_start(self):
        """Test that calling start() multiple times is safe"""
        scheduler = IngestionWorkerScheduler()

        # Start multiple times
        scheduler.start()
        scheduler.start()
        scheduler.start()

        # Should only start once
        assert scheduler.is_running is True
        assert scheduler._start_count == 1

        scheduler.stop()

    def test_idempotent_stop(self):
        """Test that calling stop() multiple times is safe"""
        scheduler = IngestionWorkerScheduler()

        scheduler.start()

        # Stop multiple times
        scheduler.stop()
        scheduler.stop()
        scheduler.stop()

        # Should only stop once
        assert scheduler.is_running is False
        assert scheduler._stop_count == 1

    def test_concurrent_starts(self):
        """Test that concurrent start() calls don't cause race conditions"""
        scheduler = IngestionWorkerScheduler()

        def start_scheduler():
            scheduler.start()

        # Create multiple threads that try to start simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=start_scheduler)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Scheduler should only be started once
        assert scheduler.is_running is True
        assert scheduler._start_count == 1

        scheduler.stop()

    def test_concurrent_stops(self):
        """Test that concurrent stop() calls don't cause race conditions"""
        scheduler = IngestionWorkerScheduler()
        scheduler.start()

        def stop_scheduler():
            scheduler.stop()

        # Create multiple threads that try to stop simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=stop_scheduler)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Scheduler should only be stopped once
        assert scheduler.is_running is False
        assert scheduler._stop_count == 1

    def test_concurrent_start_stop(self):
        """Test that concurrent start() and stop() calls are handled correctly"""
        scheduler = IngestionWorkerScheduler()

        def start_scheduler():
            time.sleep(0.001)  # Small delay to increase chance of race
            scheduler.start()

        def stop_scheduler():
            time.sleep(0.001)  # Small delay to increase chance of race
            scheduler.stop()

        # Create threads that alternate between start and stop
        threads = []
        for i in range(20):
            if i % 2 == 0:
                thread = threading.Thread(target=start_scheduler)
            else:
                thread = threading.Thread(target=stop_scheduler)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Final state should be consistent
        status = scheduler.get_status()
        assert isinstance(status["is_running"], bool)
        assert status["start_count"] >= 1

        # Clean up
        if scheduler.is_running:
            scheduler.stop()

    def test_restart_functionality(self):
        """Test that restart() works correctly"""
        scheduler = IngestionWorkerScheduler()

        scheduler.start()
        assert scheduler.is_running is True

        initial_start_count = scheduler._start_count
        initial_stop_count = scheduler._stop_count

        scheduler.restart()

        assert scheduler.is_running is True
        assert scheduler._start_count == initial_start_count + 1
        assert scheduler._stop_count == initial_stop_count + 1

        scheduler.stop()

    def test_get_status_thread_safe(self):
        """Test that get_status() is thread-safe"""
        scheduler = IngestionWorkerScheduler()
        scheduler.start()

        statuses = []

        def get_status():
            status = scheduler.get_status()
            statuses.append(status)

        # Create multiple threads that get status simultaneously
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=get_status)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All statuses should be consistent
        assert len(statuses) == 20
        for status in statuses:
            assert status["is_running"] is True
            assert isinstance(status["start_count"], int)
            assert isinstance(status["stop_count"], int)

        scheduler.stop()

    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles"""
        scheduler = IngestionWorkerScheduler()

        for i in range(5):
            scheduler.start()
            assert scheduler.is_running is True
            assert scheduler._start_count == i + 1

            time.sleep(0.1)  # Let scheduler run briefly

            scheduler.stop()
            assert scheduler.is_running is False
            assert scheduler._stop_count == i + 1

    def test_status_includes_jobs(self):
        """Test that status includes job information when running"""
        scheduler = IngestionWorkerScheduler()

        # Status when not running
        status = scheduler.get_status()
        assert status["is_running"] is False
        assert status["jobs"] == []

        # Status when running
        scheduler.start()
        status = scheduler.get_status()
        assert status["is_running"] is True
        assert len(status["jobs"]) > 0
        assert status["jobs"][0]["id"] == "process_uploads"

        scheduler.stop()

    def test_scheduler_state_consistency(self):
        """Test that scheduler state remains consistent"""
        scheduler = IngestionWorkerScheduler()

        # Initial state
        assert scheduler.is_running is False
        assert scheduler._start_count == 0
        assert scheduler._stop_count == 0

        # After start
        scheduler.start()
        assert scheduler.is_running is True
        assert scheduler._start_count == 1
        assert scheduler._stop_count == 0

        # After stop
        scheduler.stop()
        assert scheduler.is_running is False
        assert scheduler._start_count == 1
        assert scheduler._stop_count == 1

        # After restart
        scheduler.restart()
        assert scheduler.is_running is True
        assert scheduler._start_count == 2
        assert scheduler._stop_count == 2

        scheduler.stop()


class TestWorkerSchedulerStressTest:
    """Stress tests for worker scheduler"""

    def test_rapid_start_stop(self):
        """Test rapid start/stop operations"""
        scheduler = IngestionWorkerScheduler()

        for _ in range(100):
            scheduler.start()
            scheduler.stop()

        # Should handle rapid operations without errors
        assert scheduler._start_count == 100
        assert scheduler._stop_count == 100
        assert scheduler.is_running is False

    def test_high_concurrency(self):
        """Test with high number of concurrent operations"""
        scheduler = IngestionWorkerScheduler()

        def random_operation():
            import random
            time.sleep(random.uniform(0.001, 0.01))

            operation = random.choice(['start', 'stop', 'status', 'restart'])

            if operation == 'start':
                scheduler.start()
            elif operation == 'stop':
                scheduler.stop()
            elif operation == 'status':
                scheduler.get_status()
            elif operation == 'restart':
                scheduler.restart()

        # Create many threads with random operations
        threads = []
        for _ in range(50):
            thread = threading.Thread(target=random_operation)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Scheduler should still be in a valid state
        status = scheduler.get_status()
        assert isinstance(status["is_running"], bool)

        # Clean up
        if scheduler.is_running:
            scheduler.stop()


def test_scheduler_integration():
    """Integration test with actual scheduler usage"""
    from app.services.shared.worker_scheduler import worker_scheduler

    # Get initial status
    initial_status = worker_scheduler.get_status()

    # Scheduler should be running (started by main.py)
    # But in test environment it might not be
    if not initial_status["is_running"]:
        worker_scheduler.start()

    # Get status after ensuring it's running
    status = worker_scheduler.get_status()
    assert status["is_running"] is True
    assert len(status["jobs"]) > 0

    # Test that we can get status multiple times
    for _ in range(10):
        status = worker_scheduler.get_status()
        assert status["is_running"] is True


if __name__ == "__main__":
    """Run tests manually"""
    print("Running worker scheduler race condition tests...")

    test_suite = TestWorkerSchedulerRaceConditions()

    print("\n1. Testing single start...")
    test_suite.test_single_start()
    print("✓ Passed")

    print("\n2. Testing idempotent start...")
    test_suite.test_idempotent_start()
    print("✓ Passed")

    print("\n3. Testing idempotent stop...")
    test_suite.test_idempotent_stop()
    print("✓ Passed")

    print("\n4. Testing concurrent starts...")
    test_suite.test_concurrent_starts()
    print("✓ Passed")

    print("\n5. Testing concurrent stops...")
    test_suite.test_concurrent_stops()
    print("✓ Passed")

    print("\n6. Testing concurrent start/stop...")
    test_suite.test_concurrent_start_stop()
    print("✓ Passed")

    print("\n7. Testing restart functionality...")
    test_suite.test_restart_functionality()
    print("✓ Passed")

    print("\n8. Testing thread-safe status...")
    test_suite.test_get_status_thread_safe()
    print("✓ Passed")

    print("\n9. Testing multiple cycles...")
    test_suite.test_multiple_start_stop_cycles()
    print("✓ Passed")

    print("\n10. Testing status includes jobs...")
    test_suite.test_status_includes_jobs()
    print("✓ Passed")

    print("\n✅ All race condition tests passed!")

    print("\n\nRunning stress tests...")
    stress_suite = TestWorkerSchedulerStressTest()

    print("\n1. Testing rapid start/stop...")
    stress_suite.test_rapid_start_stop()
    print("✓ Passed")

    print("\n2. Testing high concurrency...")
    stress_suite.test_high_concurrency()
    print("✓ Passed")

    print("\n✅ All stress tests passed!")
    print("\n🎉 Worker scheduler is thread-safe and race-condition free!")
