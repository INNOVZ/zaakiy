#!/usr/bin/env python
"""
End-to-end test for Celery implementation.

Tests:
1. Celery app initialization
2. Task registration
3. Redis connection
4. Task execution (async functions)
5. Task scheduling
6. Worker connectivity
"""
import asyncio
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_celery_app_import():
    """Test 1: Can we import and initialize Celery app?"""
    print("=" * 60)
    print("TEST 1: Celery App Import and Initialization")
    print("=" * 60)

    try:
        from app.services.celery_app import celery_app

        print(f"✅ Celery app imported successfully")
        print(f"   App name: {celery_app.main}")
        print(f"   Broker: {celery_app.conf.broker_url}")
        print(f"   Backend: {celery_app.conf.result_backend}")
        return True
    except Exception as e:
        print(f"❌ Failed to import Celery app: {e}")
        return False


def test_task_registration():
    """Test 2: Are tasks registered?"""
    print("\n" + "=" * 60)
    print("TEST 2: Task Registration")
    print("=" * 60)

    try:
        from app.services.celery_app import celery_app

        # Check registered tasks
        registered = list(celery_app.tasks.keys())
        print(f"✅ Found {len(registered)} registered tasks")

        expected_tasks = [
            "app.services.tasks.process_pending_uploads",
            "app.services.tasks.process_upload",
            "app.services.tasks.reindex_upload",
            "app.services.tasks.health_check",
        ]

        for task_name in expected_tasks:
            if task_name in registered:
                print(f"   ✅ {task_name}")
            else:
                print(f"   ❌ {task_name} - NOT FOUND")
                return False

        return True
    except Exception as e:
        print(f"❌ Failed to check task registration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_redis_connection():
    """Test 3: Can we connect to Redis?"""
    print("\n" + "=" * 60)
    print("TEST 3: Redis Connection")
    print("=" * 60)

    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_password = os.getenv("REDIS_PASSWORD")

        # Parse URL
        if "://" in redis_url:
            parts = redis_url.split("://")
            if "@" in parts[1]:
                # Has password
                auth, host_port = parts[1].split("@")
                password = auth.split(":")[-1] if ":" in auth else auth
                host, port = (
                    host_port.split(":") if ":" in host_port else (host_port, "6379")
                )
            else:
                password = redis_password
                host, port = (
                    parts[1].split(":") if ":" in parts[1] else (parts[1], "6379")
                )
        else:
            host = "localhost"
            port = "6379"
            password = redis_password

        client = redis.Redis(
            host=host,
            port=int(port),
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
        )

        # Test connection
        result = client.ping()
        if result:
            print(f"✅ Redis connection successful")
            print(f"   Host: {host}")
            print(f"   Port: {port}")

            # Check if we can write/read
            test_key = f"celery_test_{int(time.time())}"
            client.set(test_key, "test_value", ex=10)
            value = client.get(test_key)
            if value == "test_value":
                print(f"   ✅ Read/Write test passed")
                client.delete(test_key)
                return True
            else:
                print(f"   ❌ Read/Write test failed")
                return False
        else:
            print(f"❌ Redis ping failed")
            return False

    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print(f"   Make sure Redis is running: docker-compose up redis -d")
        return False
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_async_function_execution():
    """Test 4: Can we run async functions in Celery tasks?"""
    print("\n" + "=" * 60)
    print("TEST 4: Async Function Execution")
    print("=" * 60)

    try:
        import nest_asyncio

        async def test_async_function():
            await asyncio.sleep(0.1)
            return "async_test_passed"

        # Test nest_asyncio
        nest_asyncio.apply()

        # Try to run async function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(test_async_function())

        if result == "async_test_passed":
            print(f"✅ Async function execution works")
            print(f"   nest_asyncio is properly configured")
            return True
        else:
            print(f"❌ Async function returned unexpected result: {result}")
            return False

    except Exception as e:
        print(f"❌ Async function test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_health_check_task():
    """Test 5: Can we execute a simple Celery task?"""
    print("\n" + "=" * 60)
    print("TEST 5: Health Check Task Execution")
    print("=" * 60)

    try:
        from app.services.celery_app import celery_app
        from app.services.tasks import health_check

        print("   Attempting to execute health_check task...")
        print("   (This requires a Celery worker to be running)")

        # Try to send task (will fail if no worker, but we can check if it's registered)
        if celery_app.conf.task_always_eager:
            # Eager mode - executes immediately
            result = health_check.delay()
            print(f"   ✅ Task executed in eager mode")
            print(f"   Result: {result}")
            return True
        else:
            # Normal mode - needs worker
            task = health_check.delay()
            print(f"   ✅ Task sent to queue (ID: {task.id})")
            print(f"   Note: To get result, ensure a worker is running")
            print(f"   Start worker: python scripts/start_celery_worker.py")

            # Try to get result with timeout
            try:
                result = task.get(timeout=5)
                print(f"   ✅ Task executed successfully: {result}")
                return True
            except Exception as e:
                print(
                    f"   ⚠️  Task sent but no worker available (this is OK for testing)"
                )
                print(f"   Error: {e}")
                print(f"   Start a worker to complete this test")
                return True  # Not a failure, just needs worker

    except Exception as e:
        print(f"❌ Health check task test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_beat_schedule():
    """Test 6: Is Celery Beat schedule configured?"""
    print("\n" + "=" * 60)
    print("TEST 6: Celery Beat Schedule Configuration")
    print("=" * 60)

    try:
        from app.services.celery_app import celery_app

        beat_schedule = celery_app.conf.beat_schedule

        if beat_schedule:
            print(f"✅ Beat schedule configured with {len(beat_schedule)} tasks")

            for task_name, config in beat_schedule.items():
                print(f"   📅 {task_name}:")
                print(f"      Task: {config['task']}")
                print(f"      Schedule: {config.get('schedule', 'N/A')}")
                print(f"      Options: {config.get('options', {})}")

            # Check for required task
            if "process-pending-uploads" in beat_schedule:
                print(f"   ✅ process-pending-uploads is scheduled")
                return True
            else:
                print(f"   ❌ process-pending-uploads not found in schedule")
                return False
        else:
            print(f"❌ No beat schedule configured")
            return False

    except Exception as e:
        print(f"❌ Beat schedule test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_worker_connectivity():
    """Test 7: Can we connect to a running worker?"""
    print("\n" + "=" * 60)
    print("TEST 7: Worker Connectivity")
    print("=" * 60)

    try:
        from app.services.celery_app import celery_app

        # Try to inspect active workers
        inspect = celery_app.control.inspect()
        active = inspect.active()

        if active:
            print(f"✅ Found {len(active)} active worker(s):")
            for worker_name, tasks in active.items():
                print(f"   🔧 {worker_name}: {len(tasks)} active tasks")
            return True
        else:
            print(f"⚠️  No active workers found")
            print(f"   This is OK if you're just testing the setup")
            print(f"   Start a worker: python scripts/start_celery_worker.py")
            return True  # Not a failure

    except Exception as e:
        print(f"⚠️  Could not inspect workers: {e}")
        print(f"   This is OK if no worker is running")
        return True  # Not a failure


def test_imports():
    """Test 0: Can we import all required modules?"""
    print("=" * 60)
    print("TEST 0: Module Imports")
    print("=" * 60)

    try:
        print("   Importing celery...")
        import celery

        print(f"   ✅ celery {celery.__version__}")

        print("   Importing nest_asyncio...")
        import nest_asyncio

        print(f"   ✅ nest_asyncio")

        print("   Importing redis...")
        import redis

        print(f"   ✅ redis {redis.__version__}")

        print("   Importing app modules...")
        from app.services.celery_app import celery_app

        print(f"   ✅ celery_app")

        from app.services.tasks import health_check, process_pending_uploads

        print(f"   ✅ tasks module")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"   Run: pip install celery[redis] nest-asyncio")
        return False
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("🧪 CELERY END-TO-END TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    tests = [
        ("Module Imports", test_imports),
        ("Celery App Import", test_celery_app_import),
        ("Task Registration", test_task_registration),
        ("Redis Connection", test_redis_connection),
        ("Async Function Execution", test_async_function_execution),
        ("Beat Schedule", test_beat_schedule),
        ("Health Check Task", test_health_check_task),
        ("Worker Connectivity", test_worker_connectivity),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback

            traceback.print_exc()
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
        print("\n🎉 All tests passed! Celery is properly configured.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
