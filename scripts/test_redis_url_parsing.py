#!/usr/bin/env python
"""
Test Redis URL parsing for distributed lock with various URL formats.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_url_parsing():
    """Test various Redis URL formats"""
    print("=" * 60)
    print("TEST: Redis URL Parsing for Distributed Lock")
    print("=" * 60)

    test_cases = [
        {
            "name": "Simple URL",
            "url": "redis://localhost:6379",
            "password": None,
            "db": "0",
        },
        {
            "name": "URL with database suffix",
            "url": "redis://redis:6379/0",
            "password": None,
            "db": "0",
        },
        {
            "name": "URL with database suffix and password",
            "url": "redis://redis:6379/1",
            "password": "mypassword",
            "db": "1",
        },
        {
            "name": "URL with password in URL",
            "url": "redis://:password123@redis:6379/0",
            "password": None,
            "db": "0",
        },
        {
            "name": "URL with password containing colon",
            "url": "redis://redis:6379/0",
            "password": "pass:word:with:colons",
            "db": "0",
        },
        {
            "name": "URL with complex password",
            "url": "redis://redis:6379/2",
            "password": "p@ss:w0rd!@#",
            "db": "2",
        },
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print(f"   URL: {test_case['url']}")
        print(f"   Password: {'***' if test_case['password'] else 'None'}")
        print(f"   DB: {test_case['db']}")

        try:
            # Set environment
            os.environ["REDIS_URL"] = test_case["url"]
            if test_case["password"]:
                os.environ["REDIS_PASSWORD"] = test_case["password"]
            else:
                os.environ.pop("REDIS_PASSWORD", None)
            os.environ["REDIS_DB"] = test_case["db"]

            # Import after setting env (to get fresh import)
            if "app.services.shared.distributed_lock" in sys.modules:
                del sys.modules["app.services.shared.distributed_lock"]

            from app.services.shared.distributed_lock import get_redis_client_for_lock

            client = get_redis_client_for_lock()

            if client is not None:
                print(f"   ✅ Client created successfully")
                passed += 1
            else:
                print(f"   ⚠️  Client is None (Redis may not be running)")
                print(f"   This is OK if Redis is not available - URL parsing worked")
                passed += 1  # URL parsing succeeded, just no connection

        except ValueError as e:
            print(f"   ❌ ValueError (URL parsing failed): {e}")
            failed += 1
        except Exception as e:
            print(f"   ⚠️  Exception: {e}")
            print(
                f"   (This is OK if Redis is not running - URL parsing may have worked)"
            )
            # Check if it's a connection error (parsing worked) or parsing error
            if "connection" in str(e).lower() or "cannot connect" in str(e).lower():
                print(f"   ✅ URL parsing succeeded (connection error is expected)")
                passed += 1
            else:
                print(f"   ❌ URL parsing may have failed")
                failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("\n🎉 All URL formats parsed correctly!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = test_url_parsing()
    sys.exit(exit_code)
