"""
Test script to verify the upload and ingestion process works
Run this after updating your OpenAI API key
"""
import asyncio

from app.services.scraping.ingestion_worker import process_pending_uploads, supabase


async def create_test_upload_and_process():
    """Create a test upload and process it"""
    try:
        # Get organization ID
        orgs_result = supabase.table("organizations").select("*").limit(1).execute()
        org_id = orgs_result.data[0]["id"]

        # Create a simple text-based test upload
        test_upload = {
            "org_id": org_id,
            "type": "url",
            "source": "https://jsonplaceholder.typicode.com/posts/1",  # Simple JSON API
            "pinecone_namespace": f"org-{org_id}",
            "status": "pending",
        }

        # Insert test upload
        result = supabase.table("uploads").insert(test_upload).execute()
        upload_id = result.data[0]["id"]
        print(f"Created test upload: {upload_id}")

        # Process it
        print("Processing uploads...")
        await process_pending_uploads()

        # Check final status
        check_result = (
            supabase.table("uploads").select("*").eq("id", upload_id).execute()
        )
        final_status = check_result.data[0]
        print(f"Final status: {final_status['status']}")
        if final_status["error_message"]:
            print(f"Error: {final_status['error_message']}")
        else:
            print("SUCCESS: Upload processed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    print("=== Testing Upload and Ingestion Process ===")
    print("Make sure you have a valid OpenAI API key set!")
    asyncio.run(create_test_upload_and_process())
