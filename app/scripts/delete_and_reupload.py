"""
Delete old unwanted chunks from Pinecone and trigger re-upload.

Run: python -m app.scripts.delete_and_reupload
"""

import os
import sys

# Setup path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.services.storage.pinecone_client import get_pinecone_index
from app.services.storage.supabase_client import get_supabase_client
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


async def delete_and_mark_for_reupload(upload_id: str, namespace: str):
    """Delete Pinecone vectors and mark upload as pending for re-processing"""

    supabase = get_supabase_client()
    index = get_pinecone_index()

    print(f"\n🔍 Finding vectors for upload: {upload_id}")

    # Query to find all vectors for this upload
    try:
        # Get metadata filter for this upload_id
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=1000,
            include_metadata=True,
            namespace=namespace,
            filter={"upload_id": upload_id},
        )

        matches = results.get("matches", [])
        print(f"Found {len(matches)} vectors to delete")

        if matches:
            # Delete them
            ids_to_delete = [match["id"] for match in matches]

            # Delete in batches of 100
            for i in range(0, len(ids_to_delete), 100):
                batch = ids_to_delete[i : i + 100]
                index.delete(ids=batch, namespace=namespace)
                print(f"Deleted batch {i//100 + 1}: {len(batch)} vectors")

            print(f"✅ Deleted {len(ids_to_delete)} vectors from Pinecone")

        # Mark upload as pending
        supabase.table("uploads").update(
            {"status": "pending", "error_message": None}
        ).eq("id", upload_id).execute()

        print(f"✅ Marked upload {upload_id} as 'pending' for re-processing")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Failed to delete and re-upload: {e}")


async def main():
    import asyncio

    # The upload ID from your Pinecone record
    upload_id = input("Enter the upload_id to delete and re-process: ").strip()

    # Get namespace from upload
    supabase = get_supabase_client()
    result = (
        supabase.table("uploads")
        .select("pinecone_namespace")
        .eq("id", upload_id)
        .execute()
    )

    if not result.data:
        print(f"❌ Upload {upload_id} not found")
        return

    namespace = result.data[0]["pinecone_namespace"]
    print(f"📦 Namespace: {namespace}")

    await delete_and_mark_for_reupload(upload_id, namespace)

    print("\n✅ Upload marked as 'pending' - backend will automatically re-process it.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
