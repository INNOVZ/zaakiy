"""
Clear E-Commerce Cache and Re-Index

This script is part of the app.scripts module.

Run from backend directory:
    python -m app.scripts.clear_cache_and_reindex
"""

import asyncio

from app.services.scraping.scraping_cache_service import scraping_cache_service
from app.services.storage.pinecone_client import get_pinecone_index
from app.services.storage.supabase_client import get_supabase_client


async def clear_cache_and_reindex():
    """Clear cache and re-index ALL e-commerce URLs"""

    supabase = get_supabase_client()
    index = get_pinecone_index()

    print("=" * 80)
    print("🧹 CLEAR CACHE + RE-INDEX ALL E-COMMERCE URLs")
    print("=" * 80)
    print()

    # Step 1: Find ALL e-commerce uploads
    print("Step 1: Finding ALL e-commerce uploads...")

    # Get ALL URL uploads
    result = supabase.table("uploads").select("*").eq("type", "url").execute()

    if not result.data:
        print("❌ No URL uploads found in database")
        return

    # Filter for e-commerce URLs
    ecommerce_patterns = [
        "ambassadorscentworks.com",
        "shopify",
        "/products/",
        "/collections/",
        "/shop/",
        "/store/",
    ]

    uploads = []
    for upload in result.data:
        source = upload.get("source", "")
        if any(pattern in source.lower() for pattern in ecommerce_patterns):
            uploads.append(upload)

    if not uploads:
        print("❌ No e-commerce uploads found")
        print("\nℹ️  Upload an e-commerce URL through the dashboard first.")
        return

    print(f"✅ Found {len(uploads)} e-commerce upload(s):")
    for i, upload in enumerate(uploads, 1):
        print(f"   {i}. {upload['source'][:80]}... (status: {upload['status']})")
    print()

    for upload in uploads:
        upload_id = upload["id"]
        org_id = upload["org_id"]
        namespace = upload["pinecone_namespace"]

        print(f"\n{'='*80}")
        print(f"Processing Upload: {upload_id}")
        print(f"Organization: {org_id}")
        print(f"Namespace: {namespace}")
        print(f"{'='*80}")

        # Step 2: CLEAR CACHE (THE KEY!)
        print("\n🔥 Step 2: Clearing cached scraped data...")
        source_url = upload["source"]
        try:
            deleted_count = await scraping_cache_service.invalidate_url(
                url=source_url, content_type="url", org_id=org_id
            )
            print(f"✅ Cleared {deleted_count} cached entries")
            print(f"   Cache is now empty - next scrape will be FRESH!")
        except Exception as e:
            print(f"⚠️  Error clearing cache: {e}")
            print(f"   Continuing anyway...")

        # Step 3: Delete Pinecone vectors
        print("\n🗑️  Step 3: Deleting old unstructured vectors from Pinecone...")
        try:
            index.delete(filter={"upload_id": upload_id}, namespace=namespace)
            print(f"✅ Deleted old vectors for upload {upload_id}")
        except Exception as e:
            print(f"⚠️  Error deleting vectors: {e}")

        # Step 4: Mark for re-processing
        print("\n♻️  Step 4: Marking upload for re-processing...")
        try:
            supabase.table("uploads").update(
                {"status": "pending", "error_message": None}
            ).eq("id", upload_id).execute()
            print(f"✅ Upload {upload_id} marked as 'pending'")
        except Exception as e:
            print(f"❌ Error updating upload: {e}")
            continue

        print(f"\n✅ Upload {upload_id} ready for FRESH scrape with e-commerce scraper!")

    print("\n" + "=" * 80)
    print("✅ CACHE CLEARED + RE-INDEX SETUP COMPLETE!")
    print("=" * 80)
    print(
        """
What Just Happened:
━━━━━━━━━━━━━━━━━━
1. ✅ CLEARED cached unstructured data (Redis/cache)
2. ✅ DELETED old vectors from Pinecone
3. ✅ MARKED uploads as 'pending' for re-processing

Next Steps:
━━━━━━━━━━
Backend worker will automatically pick up pending uploads.
Since cache is CLEARED, it will perform a FRESH scrape.
E-commerce scraper will detect the URL and extract structured data.

Monitor Progress:
━━━━━━━━━━━━━━━
    tail -f logs/app.log | grep "E-commerce"

Look for:
    Cache MISS - Performing fresh scrape  👈 KEY INDICATOR!
    [E-commerce] ✅ Structured scraping succeeded

Wait Time: 2-5 minutes

Then check Pinecone for structured data! 🎉
    """
    )


if __name__ == "__main__":
    asyncio.run(clear_cache_and_reindex())
