"""
Test Background Worker - See What's Actually Happening

This will show you:
1. What pending uploads exist
2. What the worker is actually doing
3. Whether e-commerce scraper is being called

Run: python -m app.scripts.test_worker
"""

import asyncio

from app.services.scraping.ingestion_worker import process_pending_uploads
from app.services.storage.supabase_client import get_supabase_client


async def test_worker():
    """Test what the background worker is actually doing"""

    supabase = get_supabase_client()

    print("=" * 80)
    print("🔍 TESTING BACKGROUND WORKER")
    print("=" * 80)
    print()

    # Step 1: Check pending uploads
    print("Step 1: Checking pending uploads...")
    result = supabase.table("uploads").select("*").eq("status", "pending").execute()

    if not result.data:
        print("❌ No pending uploads found")
        print("\nThis means:")
        print("  1. Either no uploads need processing")
        print("  2. Or they've all been processed")
        print("  3. Or uploads have other statuses")

        # Show recent uploads
        print("\nRecent URL uploads:")
        recent = (
            supabase.table("uploads")
            .select("*")
            .eq("type", "url")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        for upload in recent.data:
            emoji = (
                "✅"
                if upload["status"] == "completed"
                else "⏳"
                if upload["status"] == "pending"
                else "❌"
            )
            print(f"  {emoji} {upload['source'][:70]}... (status: {upload['status']})")
        return

    print(f"✅ Found {len(result.data)} pending upload(s)")

    for upload in result.data:
        print(f"\n  - {upload['source']}")
        print(f"    ID: {upload['id']}")
        print(f"    Type: {upload['type']}")
        print(f"    Org: {upload['org_id']}")

    # Step 2: Run worker manually
    print("\n" + "=" * 80)
    print("Step 2: Running worker manually...")
    print("=" * 80)
    print()

    print("Calling process_pending_uploads()...")
    await process_pending_uploads()

    print("\n" + "=" * 80)
    print("Worker finished!")
    print("=" * 80)
    print("\nCheck backend logs for scraping activity.")
    print("Look for: [E-commerce] or Cache MISS messages")


if __name__ == "__main__":
    asyncio.run(test_worker())
