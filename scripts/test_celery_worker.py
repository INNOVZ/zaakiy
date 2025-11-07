#!/usr/bin/env python
"""
Test Celery worker with actual task execution.

This script:
1. Sends test tasks to an existing Celery worker (worker must be running separately)
2. Executes test tasks via the worker
3. Verifies task completion
4. Reports test results

Note: This script does NOT start a worker. You must start a Celery worker
separately before running this test:
    python3 scripts/start_celery_worker.py
"""
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_worker_execution():
    """Test task execution with a worker"""
    print("=" * 60)
    print("CELERY WORKER EXECUTION TEST")
    print("=" * 60)
    print()

    try:
        from app.services.celery_app import celery_app
        from app.services.tasks import health_check

        print("📋 Test Plan:")
        print("   1. Send health_check task to queue")
        print("   2. Wait for worker to process it")
        print("   3. Verify result")
        print()

        print("⚠️  Note: This test requires a Celery worker to be running")
        print("   Start one in another terminal:")
        print("   python3 scripts/start_celery_worker.py")
        print()

        input("Press Enter when worker is running...")

        print("\n📤 Sending health_check task...")
        task = health_check.delay()

        print(f"   Task ID: {task.id}")
        print(f"   Status: {task.status}")

        print("\n⏳ Waiting for task to complete (max 10 seconds)...")
        start_time = time.time()

        try:
            result = task.get(timeout=10)
            elapsed = time.time() - start_time

            print(f"\n✅ Task completed in {elapsed:.2f} seconds")
            print(f"   Result: {result}")
            print(f"   Status: {task.status}")

            if result and result.get("status") == "healthy":
                print("\n🎉 SUCCESS! Worker is processing tasks correctly.")
                return True
            else:
                print(f"\n⚠️  Unexpected result: {result}")
                return False

        except Exception as e:
            print(f"\n❌ Task failed or timed out: {e}")
            print(f"   Make sure worker is running and can connect to Redis")
            return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pending_uploads_task():
    """Test the actual process_pending_uploads task"""
    print("\n" + "=" * 60)
    print("PROCESS PENDING UPLOADS TASK TEST")
    print("=" * 60)
    print()

    try:
        from app.services.tasks import process_pending_uploads

        print("📋 This will test the actual upload processing task")
        print("   (This may take longer as it processes real uploads)")
        print()

        response = input("Do you want to run this test? (y/n): ")
        if response.lower() != "y":
            print("   Skipping...")
            return True

        print("\n📤 Sending process_pending_uploads task...")
        task = process_pending_uploads.delay()

        print(f"   Task ID: {task.id}")

        print("\n⏳ Waiting for task to complete (max 60 seconds)...")
        print("   This task processes all pending uploads...")

        try:
            result = task.get(timeout=60)

            print(f"\n✅ Task completed")
            print(f"   Result: {result}")

            if result and result.get("status") == "success":
                print("\n🎉 SUCCESS! Upload processing task works correctly.")
                return True
            else:
                print(f"\n⚠️  Task completed but with unexpected result: {result}")
                # Report as failure if result doesn't match expected format
                return False

        except Exception as e:
            print(f"\n❌ Task failed or timed out: {e}")
            print(f"   Check worker logs for details")
            # Report as failure - don't mask errors
            return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run worker tests"""
    print("\n")
    print("🧪 CELERY WORKER EXECUTION TESTS")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    # Test 1: Health check
    test1 = test_worker_execution()

    # Test 2: Process pending uploads (optional)
    test2 = test_pending_uploads_task()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if test1:
        print("✅ Health check task: PASS")
    else:
        print("❌ Health check task: FAIL")

    if test2:
        print("✅ Process pending uploads task: PASS")
    else:
        print("❌ Process pending uploads task: FAIL")

    if test1 and test2:
        print("\n🎉 All worker tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
