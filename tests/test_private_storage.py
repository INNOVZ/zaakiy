"""
Test script to verify private storage authentication works
"""
import os

import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def test_private_access():
    """Test that private files require authentication"""
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )

        # List files
        files = supabase.storage.from_("uploads").list()

        if not files:
            print("❌ No files found to test with")
            return

        test_file = files[0]["name"]
        base_url = os.getenv("SUPABASE_URL")

        print(f"🧪 Testing access to file: {test_file}")

        # Test 1: Public access (should fail)
        public_url = f"{base_url}/storage/v1/object/public/uploads/{test_file}"
        response = requests.head(public_url, timeout=10)
        print(f"Public access: {response.status_code} (should be 400/403)")

        # Test 2: Authenticated access (should work)
        auth_url = f"{base_url}/storage/v1/object/uploads/{test_file}"
        headers = {
            "Authorization": f'Bearer {os.getenv("SUPABASE_SERVICE_ROLE_KEY")}',
            "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        }

        auth_response = requests.head(auth_url, headers=headers, timeout=10)
        print(f"Authenticated access: {auth_response.status_code} (should be 200)")

        if auth_response.status_code == 200:
            print("✅ Private storage authentication working correctly!")
        else:
            print("❌ Authenticated access failed")

    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_private_access()
