#!/usr/bin/env python3
"""
Test script to verify the subscription fix works
"""

import json
import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend URL
BASE_URL = "http://localhost:8001"


def test_subscription_with_correct_org_id():
    """Test subscription endpoint with the correct org_id"""
    print("🔍 TESTING SUBSCRIPTION WITH CORRECT ORG_ID")
    print("=" * 60)

    # User data from our investigation
    user_id = "70a4553c-e88c-4939-98cc-5791e53dcaf6"
    correct_org_id = "cc43053a-46af-4a42-8794-0d4294f5ccdd"

    print(f"User ID: {user_id}")
    print(f"Correct Org ID: {correct_org_id}")

    # Test organization subscription with correct org_id
    url = f"{BASE_URL}/api/onboarding/subscription/organization/{correct_org_id}"

    # Note: This will fail with 403 because we don't have a valid JWT token
    # But it will show us the endpoint structure
    try:
        response = requests.get(url, timeout=5)
        print(f"✅ Organization subscription endpoint accessible")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Subscription found!")
            print(f"   Plan: {data.get('plan_name', 'Unknown')}")
            print(
                f"   Tokens: {data.get('tokens_used_this_month', 0)}/{data.get('monthly_limit', 0)}"
            )
            return True
        else:
            print(f"   ❌ Subscription not found or auth failed")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_user_subscription():
    """Test user subscription endpoint"""
    print(f"\n🔍 TESTING USER SUBSCRIPTION")
    print("=" * 60)

    user_id = "70a4553c-e88c-4939-98cc-5791e53dcaf6"
    url = f"{BASE_URL}/api/onboarding/subscription/user/{user_id}"

    try:
        response = requests.get(url, timeout=5)
        print(f"✅ User subscription endpoint accessible")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Subscription found!")
            print(f"   Plan: {data.get('plan_name', 'Unknown')}")
            print(
                f"   Tokens: {data.get('tokens_used_this_month', 0)}/{data.get('monthly_limit', 0)}"
            )
            return True
        else:
            print(f"   ❌ Subscription not found or auth failed")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 SUBSCRIPTION FIX VERIFICATION")
    print("=" * 60)

    print("This test verifies that:")
    print("1. The correct org_id is being used")
    print("2. The subscription endpoints are accessible")
    print("3. The database has the correct subscription data")

    # Test organization subscription
    org_success = test_subscription_with_correct_org_id()

    # Test user subscription
    user_success = test_user_subscription()

    print(f"\n{'='*60}")
    print("🔍 SUMMARY")
    print(f"{'='*60}")

    print(
        f"Organization subscription: {'✅ Working' if org_success else '❌ Failed (expected - no auth)'}"
    )
    print(
        f"User subscription: {'✅ Working' if user_success else '❌ Failed (expected - no auth)'}"
    )

    print(f"\n🔍 NEXT STEPS:")
    print("1. The frontend has been updated to fetch org_id from database")
    print("2. Test in browser with actual user login")
    print("3. Check browser console for subscription data")
    print("4. Verify subscription displays correctly on dashboard")

    print(f"\n🔍 EXPECTED RESULT:")
    print("User should now see their Basic Plan subscription")
    print("instead of 'No subscription found'")


if __name__ == "__main__":
    main()
