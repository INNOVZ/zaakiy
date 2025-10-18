#!/usr/bin/env python3
"""
Script to configure Supabase storage bucket for private access
"""

import os

import requests
from supabase import create_client


def configure_private_bucket():
    """Configure uploads bucket to be private"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase environment variables")
            return

        supabase = create_client(supabase_url, supabase_key)

        print("🔧 Configuring uploads bucket for private access...")

        # Check if bucket exists
        buckets = supabase.storage.list_buckets()
        uploads_bucket = None

        for bucket in buckets:
            if bucket.name == "uploads":
                uploads_bucket = bucket
                break

        if not uploads_bucket:
            print("📁 Creating private uploads bucket...")
            try:
                result = supabase.storage.create_bucket("uploads", {"public": False})
                print(f"✅ Private bucket created successfully: {result}")
            except Exception as e:
                print(f"❌ Failed to create bucket: {e}")
                return
        else:
            # Make existing bucket private if it's public
            if uploads_bucket.public:
                print("🔒 Making existing bucket private...")
                try:
                    # Note: You might need to do this via Supabase dashboard or SQL
                    print("⚠️  Please make the bucket private via Supabase dashboard:")
                    print("   1. Go to Storage > uploads bucket")
                    print("   2. Click Settings")
                    print("   3. Turn OFF 'Public bucket'")
                except RuntimeError as e:
                    print(f"❌ Failed to update bucket: {e}")
            else:
                print("✅ Bucket is already private")

        # Test authenticated access
        print("\n🧪 Testing authenticated file access...")
        test_authenticated_access(supabase)

    except Exception as e:
        print(f"❌ Error configuring bucket: {e}")


def test_authenticated_access(supabase):
    """Test that we can access files with authentication"""
    try:
        # List files in the bucket
        files = supabase.storage.from_("uploads").list()
        print(f"   Found {len(files)} files in uploads bucket")

        if files:
            # Test getting a signed URL for the first file
            test_file = files[0]["name"]
            try:
                # Create a signed URL (temporary authenticated access)
                signed_url = supabase.storage.from_("uploads").create_signed_url(
                    test_file, 3600  # Valid for 1 hour
                )
                print(f"✅ Successfully created signed URL for {test_file}")
                print(f"   URL: [SIGNED_URL_REDACTED]")

                # Test accessing the file with service role key
                headers = {
                    "Authorization": f'Bearer {os.getenv("SUPABASE_SERVICE_ROLE_KEY")}',
                    "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
                }

                file_url = (
                    f"{os.getenv('SUPABASE_URL')}/storage/v1/object/uploads/{test_file}"
                )
                response = requests.head(file_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    print("✅ Authenticated access to files working correctly")
                else:
                    print(
                        f"⚠️  Authenticated access returned status: {response.status_code}"
                    )

            except requests.RequestException as e:
                print(f"⚠️  Error testing file access: {e}")
        else:
            print("   No files to test with")

    except requests.RequestException as e:
        print(f"❌ Error testing authenticated access: {e}")


def create_test_upload():
    """Create a test upload to verify the system works with private bucket"""
    try:
        print("\n📝 Creating test upload...")

        supabase = create_client(
            os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )

        # Get first organization
        orgs_result = supabase.table("organizations").select("*").limit(1).execute()
        if not orgs_result.data:
            print("❌ No organizations found")
            return

        org_id = orgs_result.data[0]["id"]

        # Create test URL upload (doesn't require storage)
        test_upload = {
            "org_id": org_id,
            "type": "url",
            "source": "https://httpbin.org/json",  # Simple JSON endpoint
            "pinecone_namespace": f"org-{org_id}",
            "status": "pending",
        }

        result = supabase.table("uploads").insert(test_upload).execute()
        upload_id = result.data[0]["id"]
        print(f"✅ Test upload created: {upload_id}")
        print("   This will be processed by the background worker in ~30 seconds")

    except Exception as e:
        print(f"❌ Failed to create test upload: {e}")


def main():
    print("🔒 Supabase Private Storage Configuration")
    print("=" * 50)

    configure_private_bucket()
    create_test_upload()

    print("\n✅ Configuration complete!")
    print("\n📋 Next steps:")
    print("   1. Verify bucket is private in Supabase dashboard")
    print("   2. Upload files via your application")
    print("   3. Monitor background worker logs for processing")
    print("   4. Files will be accessed securely using service role authentication")


if __name__ == "__main__":
    main()
