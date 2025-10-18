#!/usr/bin/env python3
"""
Test script to debug JWT authentication issues
"""

import json
import os

import requests

# Backend URL
BASE_URL = "http://localhost:8001"


def test_health_endpoint():
    """Test if backend is accessible"""
    print("🔍 TESTING BACKEND HEALTH")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Backend is running")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend is not accessible: {e}")
        return False


def test_auth_endpoint():
    """Test authentication endpoint"""
    print(f"\n🔍 TESTING AUTH ENDPOINT")
    print("=" * 50)

    # Test without auth
    try:
        response = requests.get(f"{BASE_URL}/api/auth/me", timeout=5)
        print(f"✅ Auth endpoint accessible")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Auth endpoint failed: {e}")
        return False


def test_subscription_without_auth():
    """Test subscription endpoint without authentication"""
    print(f"\n🔍 TESTING SUBSCRIPTION WITHOUT AUTH")
    print("=" * 50)

    # Test organization subscription
    try:
        response = requests.get(
            f"{BASE_URL}/api/onboarding/subscription/organization/test-org-id",
            timeout=5,
        )
        print(f"✅ Organization subscription endpoint accessible")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Organization subscription endpoint failed: {e}")

    # Test user subscription
    try:
        response = requests.get(
            f"{BASE_URL}/api/onboarding/subscription/user/test-user-id", timeout=5
        )
        print(f"✅ User subscription endpoint accessible")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ User subscription endpoint failed: {e}")


def test_subscription_with_invalid_auth():
    """Test subscription endpoint with invalid authentication"""
    print(f"\n🔍 TESTING SUBSCRIPTION WITH INVALID AUTH")
    print("=" * 50)

    headers = {
        "Authorization": "Bearer invalid-token",
        "Content-Type": "application/json",
    }

    # Test organization subscription
    try:
        response = requests.get(
            f"{BASE_URL}/api/onboarding/subscription/organization/test-org-id",
            headers=headers,
            timeout=5,
        )
        print(f"✅ Organization subscription with invalid auth")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Organization subscription with invalid auth failed: {e}")

    # Test user subscription
    try:
        response = requests.get(
            f"{BASE_URL}/api/onboarding/subscription/user/test-user-id",
            headers=headers,
            timeout=5,
        )
        print(f"✅ User subscription with invalid auth")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ User subscription with invalid auth failed: {e}")


def check_environment_variables():
    """Check if required environment variables are set"""
    print(f"\n🔍 CHECKING ENVIRONMENT VARIABLES")
    print("=" * 50)

    required_vars = [
        "SUPABASE_JWT_SECRET",
        "SUPABASE_PROJECT_ID",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
    ]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: SET")
        else:
            print(f"❌ {var}: NOT SET")


def main():
    """Main test function"""
    print("🚀 JWT AUTHENTICATION DEBUG")
    print("=" * 60)

    # Check environment variables
    check_environment_variables()

    # Test backend health
    if not test_health_endpoint():
        return

    # Test auth endpoint
    test_auth_endpoint()

    # Test subscription endpoints without auth
    test_subscription_without_auth()

    # Test subscription endpoints with invalid auth
    test_subscription_with_invalid_auth()

    print(f"\n{'='*60}")
    print("🔍 SUMMARY")
    print(f"{'='*60}")
    print("The subscription endpoints require valid JWT authentication.")
    print("The frontend needs to send a valid Supabase JWT token.")
    print("Check if:")
    print("1. User is logged in to Supabase")
    print("2. JWT token is valid and not expired")
    print("3. Frontend is sending the token correctly")
    print("4. Backend JWT validation is working")


if __name__ == "__main__":
    main()
