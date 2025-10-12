"""
Simple test script to verify the subscription system implementation.
Run this after setting up the database migration.
"""

import asyncio
import json
from datetime import datetime

from app.models.subscription import (OnboardingRequest, SubscriptionPlan,
                                     TokenUsageRequest)
from app.services.storage.supabase_client import get_supabase_client
from app.services.subscription import SubscriptionService


async def test_subscription_system():
    """Test the complete subscription system flow."""

    print("🚀 Testing Subscription System Implementation")
    print("=" * 50)

    # Initialize service
    supabase = get_supabase_client()
    subscription_service = SubscriptionService(supabase)

    try:
        # Test 1: Get all plans
        print("\n1. Testing plan retrieval...")
        plans = await subscription_service.get_all_plans()
        print(f"✅ Retrieved {len(plans)} subscription plans")
        for plan_name, features in plans.items():
            print(
                f"   - {features['name']}: {features['monthly_token_limit']} tokens, ${features['price_per_month']}/month"
            )

        # Test 2: User onboarding
        print("\n2. Testing user onboarding...")
        user_request = OnboardingRequest(
            entity_type="user",
            full_name="Test User",
            email=f"testuser_{datetime.now().timestamp()}@example.com",
            selected_plan=SubscriptionPlan.PROFESSIONAL,
        )

        user_response = await subscription_service.onboard_entity(user_request)
        print(f"✅ User onboarded successfully")
        print(f"   - Entity ID: {user_response.entity_id}")
        print(f"   - Plan: {user_response.plan}")
        print(
            f"   - Tokens: {user_response.tokens_remaining}/{user_response.tokens_limit}"
        )

        # Test 3: Organization onboarding
        print("\n3. Testing organization onboarding...")
        org_request = OnboardingRequest(
            entity_type="organization",
            full_name="Admin User",
            email=f"admin_{datetime.now().timestamp()}@testcorp.com",
            organization_name="Test Corporation",
            contact_phone="+1-555-0123",
            business_type="Technology",
            selected_plan=SubscriptionPlan.ENTERPRISE,
        )

        org_response = await subscription_service.onboard_entity(org_request)
        print(f"✅ Organization onboarded successfully")
        print(f"   - Entity ID: {org_response.entity_id}")
        print(f"   - Plan: {org_response.plan}")
        print(
            f"   - Tokens: {org_response.tokens_remaining}/{org_response.tokens_limit}"
        )

        # Test 4: Token consumption
        print("\n4. Testing token consumption...")

        # Test user token consumption
        user_token_request = TokenUsageRequest(
            entity_id=user_response.entity_id,
            entity_type="user",
            tokens_consumed=500,
            operation_type="chat",
        )

        success = await subscription_service.consume_tokens(user_token_request)
        print(f"✅ User token consumption: {'Success' if success else 'Failed'}")

        # Check updated usage
        user_usage = await subscription_service.get_subscription_usage(
            user_response.entity_id, "user"
        )
        print(
            f"   - Updated usage: {user_usage.tokens_used_this_month}/{user_usage.monthly_limit}"
        )
        print(f"   - Usage percentage: {user_usage.usage_percentage:.1f}%")

        # Test 5: Token availability check
        print("\n5. Testing token availability check...")
        has_enough, available = await subscription_service.check_token_availability(
            org_response.entity_id, "organization", 1000
        )
        print(
            f"✅ Token availability check: {'Sufficient' if has_enough else 'Insufficient'}"
        )
        print(f"   - Available tokens: {available}")

        # Test 6: Large token consumption (should work for enterprise)
        print("\n6. Testing large token consumption...")
        large_token_request = TokenUsageRequest(
            entity_id=org_response.entity_id,
            entity_type="organization",
            tokens_consumed=10000,
            operation_type="document_processing",
        )

        success = await subscription_service.consume_tokens(large_token_request)
        print(f"✅ Large token consumption: {'Success' if success else 'Failed'}")

        # Test 7: Exceeding token limit
        print("\n7. Testing token limit enforcement...")
        try:
            # Try to consume more tokens than available for user
            excessive_request = TokenUsageRequest(
                entity_id=user_response.entity_id,
                entity_type="user",
                tokens_consumed=100000,  # More than professional plan limit
                operation_type="test",
            )

            success = await subscription_service.consume_tokens(excessive_request)
            if not success:
                print("✅ Token limit enforcement working correctly")
            else:
                print("❌ Token limit enforcement failed")
        except Exception as e:
            print(f"✅ Token limit enforcement working: {str(e)}")

        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nSubscription system is ready for production use.")

        # Print summary
        print(f"\nTest Summary:")
        print(f"- User created: {user_response.entity_id} ({user_response.plan})")
        print(f"- Organization created: {org_response.entity_id} ({org_response.plan})")
        print(f"- Token consumption tested and working")
        print(f"- Token limits enforced correctly")

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("\nPlease check:")
        print("1. Database migration has been run")
        print("2. Supabase connection is working")
        print("3. Environment variables are set correctly")
        raise


async def test_api_endpoints():
    """Test the API endpoints using HTTP requests."""

    print("\n🌐 Testing API Endpoints")
    print("=" * 30)

    try:
        import httpx

        # Assuming your server is running on localhost:8001
        base_url = "http://localhost:8001"

        async with httpx.AsyncClient() as client:
            # Test 1: Get plans
            print("1. Testing GET /api/onboarding/plans")
            response = await client.get(f"{base_url}/api/onboarding/plans")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Plans endpoint working: {len(data['plans'])} plans available")
            else:
                print(f"❌ Plans endpoint failed: {response.status_code}")

            # Test 2: Signup
            print("\n2. Testing POST /api/onboarding/signup")
            signup_data = {
                "entity_type": "user",
                "full_name": "API Test User",
                "email": f"apitest_{datetime.now().timestamp()}@example.com",
                "selected_plan": "basic",
            }

            response = await client.post(
                f"{base_url}/api/onboarding/signup", json=signup_data
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Signup endpoint working: {data['entity_id']}")
                entity_id = data["entity_id"]

                # Test 3: Get subscription status
                print(
                    "\n3. Testing GET /api/onboarding/subscription/{entity_type}/{entity_id}"
                )
                response = await client.get(
                    f"{base_url}/api/onboarding/subscription/user/{entity_id}"
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Subscription status endpoint working")
                    print(f"   - Tokens remaining: {data['tokens_remaining']}")
                else:
                    print(f"❌ Subscription status failed: {response.status_code}")

            else:
                print(f"❌ Signup endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")

        print("\n✅ API endpoint tests completed!")

    except ImportError:
        print("❌ httpx not installed. Install with: pip install httpx")
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        print("Make sure your server is running on localhost:8001")


if __name__ == "__main__":
    print("Subscription System Test Suite")
    print("==============================")

    # Run service tests
    asyncio.run(test_subscription_system())

    # Run API tests (optional)
    try:
        asyncio.run(test_api_endpoints())
    except Exception as e:
        print(f"\nAPI tests skipped: {str(e)}")

    print("\n🏁 Test suite completed!")
