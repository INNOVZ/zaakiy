#!/usr/bin/env python3
"""
Script to check and configure Supabase storage bucket permissions
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def check_bucket_config():
    """Check current bucket configuration"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase environment variables")
            return

        supabase = create_client(supabase_url, supabase_key)

        # List buckets
        print("📁 Checking storage buckets...")
        buckets = supabase.storage.list_buckets()

        uploads_bucket = None
        for bucket in buckets:
            print(f"   Bucket: {bucket.name} - Public: {bucket.public}")
            if bucket.name == "uploads":
                uploads_bucket = bucket

        if uploads_bucket:
            print("\n✅ Uploads bucket found")
            print(f"   Public: {uploads_bucket.public}")
            print(f"   ID: {uploads_bucket.id}")

            # Test file list
            try:
                files = supabase.storage.from_("uploads").list()
                print(f"   Files in bucket: {len(files)}")

                # Show first few files
                for file in files[:3]:
                    print(f"      - {file['name']}")

            except (KeyError, AttributeError, TypeError, ValueError, OSError) as e:
                print(f"   ⚠️  Could not list files: {e}")
        else:
            print("❌ Uploads bucket not found")

            # Try to create the bucket
            print("\n🔧 Attempting to create uploads bucket...")
            try:
                result = supabase.storage.create_bucket(
                    "uploads", {"public": True})
                print(f"✅ Bucket created successfully: {result}")
            except (KeyError, AttributeError, TypeError, ValueError, OSError) as e:
                print(f"❌ Failed to create bucket: {e}")

    except (KeyError, AttributeError, TypeError, ValueError, OSError) as e:
        print(f"❌ Error checking bucket config: {e}")


def test_file_access():
    """Test file access from bucket"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        supabase = create_client(supabase_url, supabase_key)

        print("\n🔗 Testing file access...")

        # List some files and try to get their URLs
        files = supabase.storage.from_("uploads").list()

        if files:
            test_file = files[0]["name"]

            # Get public URL
            try:
                public_url = supabase.storage.from_(
                    "uploads").get_public_url(test_file)
                print(f"   Public URL for {test_file}:")
                print(f"   {public_url}")

                # Test if URL is accessible
                import requests
                response = requests.head(public_url, timeout=10)
                print(f"   URL Status: {response.status_code}")

            except (KeyError, AttributeError, TypeError, ValueError, ImportError, OSError) as e:
                print(f"   ❌ Error getting public URL: {e}")
        else:
            print("   No files found to test")

    except (KeyError, AttributeError, TypeError, ValueError, ImportError, OSError) as e:
        print(f"❌ Error testing file access: {e}")


def main():
    print("🔧 Supabase Storage Configuration Check")
    print("=" * 50)

    check_bucket_config()
    test_file_access()

    print("\n💡 If the uploads bucket is not public, you may need to:")
    print("   1. Go to your Supabase dashboard")
    print("   2. Navigate to Storage > uploads bucket")
    print("   3. Make the bucket public")
    print("   4. Or configure RLS policies for authenticated access")


if __name__ == "__main__":
    main()
