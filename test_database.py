"""
Simple test script to check database connection and create a test upload
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)


def test_database_connection():
    """Test basic database connection"""
    try:
        # Try to query organizations table
        result = supabase.table("organizations").select("*").limit(5).execute()
        print(f"Organizations table: {len(result.data)} records found")

        # Try to query users table
        result = supabase.table("users").select("*").limit(5).execute()
        print(f"Users table: {len(result.data)} records found")

        # Try to query uploads table
        result = supabase.table("uploads").select("*").limit(5).execute()
        print(f"Uploads table: {len(result.data)} records found")

        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


def create_test_upload():
    """Create a test upload record"""
    try:
        # Get the first organization
        orgs_result = supabase.table(
            "organizations").select("*").limit(1).execute()
        if not orgs_result.data:
            print("No organizations found")
            return False

        org_id = orgs_result.data[0]["id"]
        print(f"Using organization ID: {org_id}")

        # Create test upload
        test_upload = {
            "org_id": org_id,
            "type": "url",
            "source": "https://example.com",
            "pinecone_namespace": f"org-{org_id}",
            "status": "pending"
        }

        result = supabase.table("uploads").insert(test_upload).execute()
        print(f"Test upload created: {result.data}")
        return True

    except Exception as e:
        print(f"Failed to create test upload: {e}")
        return False


if __name__ == "__main__":
    print("=== Database Connection Test ===")
    if test_database_connection():
        print("\n=== Creating Test Upload ===")
        create_test_upload()
