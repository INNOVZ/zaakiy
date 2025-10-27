"""
Clear E-Commerce Cache and Re-Index (Backend Version)

Run from backend directory:
    cd backend
    source .venv/bin/activate  # Activate venv first!
    python clear_cache_reindex.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to Python path so it can find 'app' module
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app.services.scraping.scraping_cache_service import scraping_cache_service
from app.storage.pinecone_client import get_pinecone_index
from app.storage.supabase_client import get_supabase_client


async def clear_cache_and_reindex():
    """Clear cache and re-index e-commerce URLs"""

    supabase = get_supabase_client()
    index = get_pinecone_index()

    # Target URL
    target_url = "https://ambassadorscentworks.com/collections/talisman-series"

    print("=" * 80)
    print("🧹 CLEAR CACHE + RE-INDEX E-COMMERCE URLs")
    print("=" * 80)
    print(f"\nTarget URL: {target_url}\n")

    # Step 1: Find existing uploads
    print("Step 1: Finding existing uploads...")
    result = supabase.table("uploads").select("*").eq("source", target_url).execute()

    if not result.data:
        print(f"❌ No existing uploads found for {target_url}")
        print("\nℹ️  Upload this URL through the dashboard first.")
        return

    uploads = result.data
    print(f"✅ Found {len(uploads)} upload(s)")

    for upload in uploads:
        upload_id = upload["id"]
        org_id = upload["org_id"]
        namespace = upload["pinecone_namespace"]

        print(f"\n{'='*80}")
        print(f"Processing Upload: {upload_id}")
        print(f"Organization: {org_id}")
        print(f"Namespace: {namespace}")
        print(f"{'='*80}")

        # Step 2: CLEAR CACHE
        print("\n🔥 Step 2: Clearing cached scraped data...")
        try:
            deleted_count = await scraping_cache_service.invalidate_url(
                url=target_url, content_type="url", org_id=org_id
            )
            print(f"✅ Cleared {deleted_count} cached entries for {target_url}")
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
    Cache MISS - Performing fresh scrape  👈 KEY!
    [E-commerce] ✅ Structured scraping succeeded

Wait Time: 2-5 minutes

Then check Pinecone for structured data! 🎉
    """
    )


async def check_cache_status(url: str, org_id: str = None):
    """Check if a URL is currently cached"""

    print("=" * 80)
    print("🔍 CHECKING CACHE STATUS")
    print("=" * 80)
    print(f"\nURL: {url}")
    print(f"Org: {org_id or 'global'}\n")

    try:
        cached = await scraping_cache_service.get_cached_content(
            url=url, content_type="url", org_id=org_id
        )

        if cached:
            print("❌ URL IS CACHED (Old data will be served)")
            print(f"\nCache Details:")
            print(f"  - Cached at: {cached.cached_at}")
            print(f"  - Content size: {cached.content_size} bytes")
            print(f"  - Scrape time: {cached.scrape_time_ms}ms")
            print(f"\n  First 200 chars of cached content:")
            print(f"  {cached.content[:200]}...")

            if "Log in" in cached.content or "Sign Up" in cached.content:
                print("\n  ⚠️  WARNING: This is UNSTRUCTURED old data!")
                print("  ⚠️  Clear cache to get structured data.")
            else:
                print("\n  ✅ This appears to be structured data.")
        else:
            print("✅ URL IS NOT CACHED")
            print("   Next scrape will be FRESH with e-commerce scraper!")
    except Exception as e:
        print(f"❌ Error checking cache: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            url = input("Enter URL to check: ")
            org_id = input("Enter org_id (press Enter for none): ").strip() or None
            asyncio.run(check_cache_status(url, org_id))
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  python clear_cache_reindex.py          # Clear cache and re-index")
            print("  python clear_cache_reindex.py check    # Check cache status")
    else:
        asyncio.run(clear_cache_and_reindex())
