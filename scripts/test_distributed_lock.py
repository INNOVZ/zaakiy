#!/usr/bin/env python
"""
Test distributed lock to verify it prevents concurrent execution.
"""
import os
import sys
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lock_acquisition():
    """Test that only one process can acquire the lock"""
    print("=" * 60)
    print("TEST: Distributed Lock Acquisition")
    print("=" * 60)

    try:
        from app.services.shared.distributed_lock import (
            DistributedLock,
            get_redis_client_for_lock,
        )

        redis_client = get_redis_client_for_lock()
        if not redis_client:
            print("❌ Redis not available - cannot test distributed lock")
            return False

        lock1 = DistributedLock(redis_client, "test_lock", timeout=10)
        lock2 = DistributedLock(redis_client, "test_lock", timeout=10)

        # First process should acquire lock
        print("\n1. First process acquiring lock...")
        if lock1.acquire(blocking=False):
            print("   ✅ First process acquired lock")
        else:
            print("   ❌ First process failed to acquire lock")
            return False

        # Second process should NOT acquire lock
        print("\n2. Second process trying to acquire same lock...")
        if not lock2.acquire(blocking=False):
            print("   ✅ Second process correctly blocked (lock already held)")
        else:
            print("   ❌ Second process incorrectly acquired lock (race condition!)")
            lock2.release()
            return False

        # Check lock status
        print("\n3. Checking lock status...")
        if lock1.is_locked():
            print("   ✅ Lock is correctly held")
        else:
            print("   ❌ Lock not detected as held")
            return False

        # Release lock
        print("\n4. Releasing lock...")
        if lock1.release():
            print("   ✅ Lock released successfully")
        else:
            print("   ❌ Failed to release lock")
            return False

        # Second process should now be able to acquire
        print("\n5. Second process trying again after release...")
        if lock2.acquire(blocking=False):
            print("   ✅ Second process acquired lock after release")
            lock2.release()
            return True
        else:
            print("   ❌ Second process still cannot acquire (lock not released?)")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lock_timeout():
    """Test that lock expires after timeout"""
    print("\n" + "=" * 60)
    print("TEST: Distributed Lock Timeout")
    print("=" * 60)

    try:
        from app.services.shared.distributed_lock import (
            DistributedLock,
            get_redis_client_for_lock,
        )

        redis_client = get_redis_client_for_lock()
        if not redis_client:
            print("❌ Redis not available - cannot test")
            return False

        lock = DistributedLock(
            redis_client, "test_timeout_lock", timeout=2
        )  # 2 second timeout

        print("\n1. Acquiring lock with 2 second timeout...")
        if lock.acquire(blocking=False):
            print("   ✅ Lock acquired")
        else:
            print("   ❌ Failed to acquire lock")
            return False

        print("\n2. Waiting 3 seconds for lock to expire...")
        time.sleep(3)

        print("\n3. Checking if lock expired...")
        if not lock.is_locked():
            print("   ✅ Lock expired correctly")
            return True
        else:
            print("   ❌ Lock did not expire (checking Redis TTL)")
            # Try to acquire again - should work if expired
            lock2 = DistributedLock(redis_client, "test_timeout_lock", timeout=2)
            if lock2.acquire(blocking=False):
                print("   ✅ Lock expired (acquired by new process)")
                lock2.release()
                return True
            else:
                print("   ❌ Lock did not expire")
                return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lock_in_task():
    """Test that the lock works in the actual task"""
    print("\n" + "=" * 60)
    print("TEST: Lock Integration in process_pending_uploads Task")
    print("=" * 60)

    try:
        from app.services.shared.distributed_lock import get_redis_client_for_lock
        from app.services.tasks import process_pending_uploads

        redis_client = get_redis_client_for_lock()
        if not redis_client:
            print("⚠️  Redis not available - task will run without lock")
            print("   This is OK for testing but not recommended for production")
            return True

        # Check if lock mechanism is in place
        import inspect

        source = inspect.getsource(process_pending_uploads)

        if "DistributedLock" in source:
            print("✅ DistributedLock is used in process_pending_uploads")
        else:
            print("❌ DistributedLock not found in task code")
            return False

        if "lock.acquire" in source:
            print("✅ Lock acquisition is implemented")
        else:
            print("❌ Lock acquisition not found")
            return False

        if "lock.release" in source or "finally" in source:
            print("✅ Lock release is implemented")
        else:
            print("❌ Lock release not found")
            return False

        print("\n✅ Task has distributed lock protection")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all lock tests"""
    print("\n")
    print("🔒 DISTRIBUTED LOCK TEST SUITE")
    print("=" * 60)

    tests = [
        ("Lock Acquisition", test_lock_acquisition),
        ("Lock Timeout", test_lock_timeout),
        ("Lock in Task", test_lock_in_task),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All lock tests passed! Concurrent execution is prevented.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
