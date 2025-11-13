#!/usr/bin/env python3
"""
Quick test script to verify intent analytics endpoint is accessible
"""
import sys

import requests


def run_endpoint_check(base_url: str = "http://localhost:8001"):
    """Test if the intent analytics endpoint exists"""
    endpoint = f"{base_url}/api/chat/analytics/intent?days=7"

    print(f"Testing endpoint: {endpoint}")
    print("-" * 50)

    try:
        # Test without auth (should get 401 or 403, not 404)
        response = requests.get(endpoint, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}")

        if response.status_code == 404:
            print("❌ ERROR: Endpoint not found (404)")
            print("   This means the route is not registered.")
            print("   Please check:")
            print("   1. Backend server is running")
            print("   2. Backend server is on the correct port")
            print("   3. Backend server was restarted after adding the route")
            return False
        elif response.status_code in [401, 403]:
            print("✅ Endpoint exists (got auth error, which is expected)")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return True

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to backend server")
        print(f"   Make sure the backend is running on {base_url}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    # Allow custom base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = run_endpoint_check(base_url)
    sys.exit(0 if success else 1)
