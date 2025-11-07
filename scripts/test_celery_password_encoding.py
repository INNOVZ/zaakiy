#!/usr/bin/env python
"""
Test Celery password URL encoding with special characters.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_password_encoding():
    """Test various password formats"""
    print("=" * 60)
    print("TEST: Celery Password URL Encoding")
    print("=" * 60)

    test_cases = [
        {
            "name": "Password with @",
            "url": "redis://redis:6379/0",
            "password": "pass@word",
        },
        {
            "name": "Password with :",
            "url": "redis://redis:6379/0",
            "password": "pass:word",
        },
        {
            "name": "Password with %",
            "url": "redis://redis:6379/0",
            "password": "pass%word",
        },
        {
            "name": "Password with /",
            "url": "redis://redis:6379/0",
            "password": "pass/word",
        },
        {
            "name": "Password with multiple special chars",
            "url": "redis://redis:6379/0",
            "password": "pass:word@complex%special/chars",
        },
        {
            "name": "Password with #",
            "url": "redis://redis:6379/0",
            "password": "pass#word",
        },
        {
            "name": "Password with ?",
            "url": "redis://redis:6379/0",
            "password": "pass?word",
        },
        {
            "name": "Password with &",
            "url": "redis://redis:6379/0",
            "password": "pass&word",
        },
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print(f"   URL: {test_case['url']}")
        print(f"   Password: {test_case['password']}")

        try:
            # Set environment
            os.environ["REDIS_URL"] = test_case["url"]
            os.environ["REDIS_PASSWORD"] = test_case["password"]

            # Import after setting env (to get fresh import)
            if "app.services.celery_app" in sys.modules:
                del sys.modules["app.services.celery_app"]

            from app.services.celery_app import (
                CELERY_BROKER_URL,
                CELERY_RESULT_BACKEND,
                add_password_to_redis_url,
            )

            # Test the function directly
            result_url = add_password_to_redis_url(
                test_case["url"], test_case["password"]
            )

            # Verify URL can be parsed
            from urllib.parse import urlparse

            parsed = urlparse(result_url)

            if parsed.scheme and parsed.hostname:
                print(f"   ✅ URL encoded successfully")
                print(f"   Result: {result_url[:50]}...")
                print(f"   Parsed: scheme={parsed.scheme}, host={parsed.hostname}")
                passed += 1
            else:
                print(f"   ❌ URL parsing failed")
                failed += 1

        except ValueError as e:
            print(f"   ❌ ValueError: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("\n🎉 All password encoding tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = test_password_encoding()
    sys.exit(exit_code)
