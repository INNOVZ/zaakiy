#!/usr/bin/env python3
"""
Test script to check subscription API endpoints for both users
"""

import json
from datetime import datetime

import requests

# Backend URL
BASE_URL = "http://localhost:8001"

# User data from the database schema
USERS = [
    {
        "id": "70a4553c-e88c-4939-98cc-5791e53dcaf6",
        "email": "innovzhub@gmail.com",
        "org_id": "21bcca33-a10c-442c-9bc5-7a0208b5928f",
        "status": "working",
    },
    {
        "id": "09294a59-4100-4854-bca3-b23bbe39055",
        "email": "jithinkjacob@live.com",
        "org_id": None,
        "status": "not_working",
    },
]


def test_backend_health():
    """Test if backend is running"""
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


def test_subscription_endpoint(user_id, entity_type, entity_id, auth_token=None):
    """Test subscription endpoint for a specific user"""
    print(f"\n🔍 TESTING SUBSCRIPTION ENDPOINT")
    print(f"Entity Type: {entity_type}")
    print(f"Entity ID: {entity_id}")
    print("=" * 50)

    url = f"{BASE_URL}/api/onboarding/subscription/{entity_type}/{entity_id}"

    headers = {"Content-Type": "application/json"}

    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"✅ Request successful")
        print(f"   Status: {response.status_code}")
        print(f"   URL: {url}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"   Error: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def test_plans_endpoint():
    """Test plans endpoint"""
    print(f"\n🔍 TESTING PLANS ENDPOINT")
    print("=" * 50)

    url = f"{BASE_URL}/api/onboarding/plans"

    try:
        response = requests.get(url, timeout=10)
        print(f"✅ Request successful")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"   Error: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def simulate_frontend_logic(user):
    """Simulate the frontend subscription detection logic"""
    print(f"\n🔍 SIMULATING FRONTEND LOGIC FOR {user['email']}")
    print("=" * 60)

    user_id = user["id"]
    org_id = user["org_id"]

    print(f"User ID: {user_id}")
    print(f"Org ID: {org_id}")

    # Try organization subscription first (if orgId exists)
    if org_id:
        print(f"\n1. Trying organization subscription...")
        result = test_subscription_endpoint(user_id, "organization", org_id)
        if result and result.get("has_subscription"):
            print(f"✅ Organization subscription found!")
            return result
        else:
            print(f"❌ Organization subscription not found")

    # Fall back to user subscription
    print(f"\n2. Trying user subscription...")
    result = test_subscription_endpoint(user_id, "user", user_id)
    if result and result.get("has_subscription"):
        print(f"✅ User subscription found!")
        return result
    else:
        print(f"❌ User subscription not found")
        return None


def main():
    """Main test function"""
    print("🚀 SUBSCRIPTION API TEST - LIVE BACKEND")
    print("=" * 60)

    # Test backend health
    if not test_backend_health():
        return

    # Test plans endpoint
    plans = test_plans_endpoint()

    # Test both users
    results = {}

    for user in USERS:
        print(f"\n{'='*60}")
        print(f"TESTING USER: {user['email']} ({user['status']})")
        print(f"{'='*60}")

        result = simulate_frontend_logic(user)
        results[user["email"]] = {"user": user, "result": result}

    # Summary
    print(f"\n{'='*60}")
    print("🔍 SUMMARY")
    print(f"{'='*60}")

    for email, data in results.items():
        user = data["user"]
        result = data["result"]

        print(f"\n📧 {email} ({user['status']}):")
        print(f"   User ID: {user['id']}")
        print(f"   Org ID: {user['org_id']}")

        if result:
            print(f"   ✅ SUBSCRIPTION FOUND!")
            print(f"   Plan: {result.get('plan_name', 'Unknown')}")
            print(
                f"   Tokens: {result.get('tokens_used_this_month', 0)}/{result.get('monthly_limit', 0)}"
            )
            print(f"   Remaining: {result.get('tokens_remaining', 0)}")
            print(f"   Usage: {result.get('usage_percentage', 0):.2f}%")
        else:
            print(f"   ❌ NO SUBSCRIPTION FOUND!")
            print(f"   This explains why the dashboard shows 'No subscription found'")

    # Analysis
    print(f"\n{'='*60}")
    print("🔍 ANALYSIS")
    print(f"{'='*60}")

    working_user = results[USERS[0]["email"]]
    not_working_user = results[USERS[1]["email"]]

    print(f"✅ Working user ({USERS[0]['email']}):")
    print(f"   - Has organization: {working_user['user']['org_id'] is not None}")
    print(f"   - Subscription found: {'✅ Yes' if working_user['result'] else '❌ No'}")

    print(f"\n❌ Not working user ({USERS[1]['email']}):")
    print(f"   - Has organization: {not_working_user['user']['org_id'] is not None}")
    print(
        f"   - Subscription found: {'✅ Yes' if not_working_user['result'] else '❌ No'}"
    )

    print(f"\n🔍 DIAGNOSIS:")
    if not_working_user["result"] is None:
        print(f"   The second user's subscription endpoint is failing")
        print(f"   This could be due to:")
        print(f"   1. Authentication issues")
        print(f"   2. Database connection problems")
        print(f"   3. User access permissions")
        print(f"   4. Missing subscription record")
    else:
        print(f"   Both users have working subscriptions")
        print(f"   The issue might be in the frontend or caching")


if __name__ == "__main__":
    main()
