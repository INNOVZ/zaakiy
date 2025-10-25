#!/usr/bin/env python3
"""
Helper script to fetch real IDs from your database for load testing

Usage:
    python tests/load/get_test_data.py

This will:
1. Connect to your Supabase database
2. Fetch real chatbot and organization IDs
3. Display them so you can update locustfile.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.storage.supabase_client import get_supabase_client


def main():
    print("🔍 Fetching real IDs from database for load testing...\n")

    try:
        supabase = get_supabase_client()

        # Get real chatbot IDs
        print("📊 Fetching Chatbot IDs...")
        chatbots = (
            supabase.table("chatbots").select("id, name, org_id").limit(5).execute()
        )

        if chatbots.data and len(chatbots.data) > 0:
            print(f"✅ Found {len(chatbots.data)} chatbots:\n")
            chatbot_ids = []
            for bot in chatbots.data:
                print(f"   ID: {bot['id']}")
                print(f"   Name: {bot.get('name', 'N/A')}")
                print(f"   Org ID: {bot.get('org_id', 'N/A')}")
                print()
                chatbot_ids.append(bot["id"])
        else:
            print("⚠️  No chatbots found in database")
            chatbot_ids = []

        # Get real organization IDs
        print("\n📊 Fetching Organization IDs...")
        orgs = supabase.table("organizations").select("id, name").limit(5).execute()

        if orgs.data and len(orgs.data) > 0:
            print(f"✅ Found {len(orgs.data)} organizations:\n")
            org_ids = []
            for org in orgs.data:
                print(f"   ID: {org['id']}")
                print(f"   Name: {org.get('name', 'N/A')}")
                print()
                org_ids.append(org["id"])
        else:
            print("⚠️  No organizations found in database")
            org_ids = []

        # Generate code to copy-paste
        print("\n" + "=" * 70)
        print("📝 Copy and paste this into tests/load/locustfile.py:")
        print("=" * 70)
        print()

        if chatbot_ids:
            print("SAMPLE_CHATBOT_IDS = [")
            for cid in chatbot_ids[:3]:  # Use max 3 for testing
                print(f'    "{cid}",')
            print("]")
            print()
        else:
            print("# ⚠️  No chatbots found - you may need to create one first")
            print("SAMPLE_CHATBOT_IDS = []")
            print()

        if org_ids:
            print("SAMPLE_ORG_IDS = [")
            for oid in org_ids[:3]:  # Use max 3 for testing
                print(f'    "{oid}",')
            print("]")
            print()
        else:
            print("# ⚠️  No organizations found")
            print("SAMPLE_ORG_IDS = []")
            print()

        print("=" * 70)
        print()

        # Provide guidance
        if not chatbot_ids or not org_ids:
            print("⚠️  WARNING: Missing data in database")
            print("\nTo create test data:")
            print("  1. Log in to your application")
            print("  2. Create an organization")
            print("  3. Create a chatbot")
            print("  4. Run this script again")
            print()
        else:
            print("✅ You're all set! Update locustfile.py and run your test again.")
            print()
            print("Next steps:")
            print("  1. Update SAMPLE_CHATBOT_IDS and SAMPLE_ORG_IDS in locustfile.py")
            print("  2. Run: make load-test-quick")
            print("  3. Check results in reports/")
            print()

    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print()
        print("Make sure:")
        print("  1. Your .env file is configured correctly")
        print("  2. Database credentials are valid")
        print("  3. You're in the backend directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
