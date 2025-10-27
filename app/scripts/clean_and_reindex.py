"""
Simple script to delete Pinecone vectors for a specific upload and re-process.

Usage: python -m app.scripts.clean_and_reindex <upload_id> <namespace>
"""

import asyncio
import sys

from app.services.storage.pinecone_client import get_pinecone_index
from app.services.storage.supabase_client import get_supabase_client


async def clean_and_reindex(upload_id: str, namespace: str):
    """Delete vectors and mark upload as pending"""

    supabase = get_supabase_client()
    index = get_pinecone_index()

    print(f"\n🔍 Finding vectors for upload: {upload_id} in namespace: {namespace}")

    try:
        # Query to find all vectors for this upload
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10000,
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
        print(
            "\nThe backend will automatically re-process this upload with the new cleaning logic."
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m app.scripts.clean_and_reindex <upload_id> <namespace>")
        print("\nExample:")
        print(
            '  python -m app.scripts.clean_and_reindex 2e256d5c-52e4-47f5-aeeb-1bb6dc7f1604 "org-0cfda352-0446-4128-9fef-73ae0c706bdb"'
        )
        sys.exit(1)

    upload_id = sys.argv[1]
    namespace = sys.argv[2]

    asyncio.run(clean_and_reindex(upload_id, namespace))
