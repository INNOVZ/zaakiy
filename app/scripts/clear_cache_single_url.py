"""
Clear Cache and Re-Index SINGLE URL (Production Safe)

This version processes ONE URL at a time for safety in production.

Usage:
    python -m app.scripts.clear_cache_single_url
"""

import asyncio
from typing import Any, Dict, List

from app.services.scraping.scraping_cache_service import scraping_cache_service
from app.services.storage.pinecone_client import get_pinecone_index
from app.services.storage.supabase_client import (
    get_supabase_client,
    get_supabase_http_client,
)

HTTP_ARG_ERROR = "http_client"


async def _fetch_url_uploads() -> List[Dict[str, Any]]:
    """
    Fetch URL uploads. Falls back to direct REST calls when the supabase
    python client is out of sync with the installed PostgREST version.
    """
    supabase = get_supabase_client()
    try:
        result = supabase.table("uploads").select("*").eq("type", "url").execute()
        return result.data or []
    except TypeError as exc:
        if HTTP_ARG_ERROR not in str(exc):
            raise
        client = get_supabase_http_client()
        if client is None:
            raise RuntimeError("Supabase HTTP client unavailable") from exc
        response = await client.get(
            "/uploads",
            params={"select": "*", "type": "eq.url", "order": "created_at.desc"},
        )
        response.raise_for_status()
        return response.json()


async def _mark_upload_pending(upload_id: str) -> None:
    """Update upload status with graceful fallback when supabase client mismatches."""
    supabase = get_supabase_client()
    payload = {"status": "pending", "error_message": None}
    try:
        supabase.table("uploads").update(payload).eq("id", upload_id).execute()
    except TypeError as exc:
        if HTTP_ARG_ERROR not in str(exc):
            raise
        client = get_supabase_http_client()
        if client is None:
            raise RuntimeError("Supabase HTTP client unavailable") from exc
        response = await client.patch(
            "/uploads",
            params={"id": f"eq.{upload_id}"},
            json=payload,
            headers={"Prefer": "return=representation"},
        )
        response.raise_for_status()


async def clear_cache_single_url():
    """Clear cache and re-index a single e-commerce URL"""

    index = get_pinecone_index()

    print("=" * 80)
    print("🧹 CLEAR CACHE + RE-INDEX SINGLE URL (Production Safe)")
    print("=" * 80)
    print()

    # List all e-commerce uploads
    uploads = await _fetch_url_uploads()

    if not uploads:
        print("❌ No URL uploads found")
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

    ecommerce_uploads = []
    for upload in uploads:
        source = upload.get("source", "")
        if any(pattern in source.lower() for pattern in ecommerce_patterns):
            ecommerce_uploads.append(upload)

    if not ecommerce_uploads:
        print("❌ No e-commerce uploads found")
        return

    print(f"Found {len(ecommerce_uploads)} e-commerce upload(s):\n")
    for i, upload in enumerate(ecommerce_uploads, 1):
        status_emoji = "✅" if upload["status"] == "completed" else "⏳"
        print(f"{i}. {status_emoji} {upload['source']}")
        print(f"   Status: {upload['status']}, ID: {upload['id'][:8]}...")

    print("\n" + "=" * 80)
    choice = input("\nEnter the number to re-index (or 'q' to quit): ").strip()

    if choice.lower() == "q":
        print("Cancelled.")
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(ecommerce_uploads):
            print("❌ Invalid choice")
            return
    except ValueError:
        print("❌ Invalid input")
        return

    upload = ecommerce_uploads[idx]
    upload_id = upload["id"]
    org_id = upload["org_id"]
    namespace = upload["pinecone_namespace"]
    source_url = upload["source"]

    print("\n" + "=" * 80)
    print(f"Processing: {source_url}")
    print(f"Upload ID: {upload_id}")
    print(f"Organization: {org_id}")
    print("=" * 80)

    # Confirm
    confirm = input(
        "\n⚠️  This will delete old vectors and clear cache. Continue? (yes/no): "
    ).strip()
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    # Step 1: Clear cache
    print("\n🔥 Step 1: Clearing cached data...")
    try:
        deleted_count = await scraping_cache_service.invalidate_url(
            url=source_url, content_type="url", org_id=org_id
        )
        print(f"✅ Cleared {deleted_count} cached entries")
    except Exception as e:
        print(f"⚠️  Error clearing cache: {e}")

    # Step 2: Delete Pinecone vectors
    print("\n🗑️  Step 2: Deleting old vectors from Pinecone...")
    try:
        index.delete(filter={"upload_id": upload_id}, namespace=namespace)
        print(f"✅ Deleted old vectors")
    except Exception as e:
        print(f"⚠️  Error deleting vectors: {e}")

    # Step 3: Mark as pending
    print("\n♻️  Step 3: Marking for re-processing...")
    try:
        await _mark_upload_pending(upload_id)
        print(f"✅ Marked as 'pending'")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(
        """
Next Steps:
━━━━━━━━━━
1. Background worker will pick this up within 30 seconds
2. Fresh scrape with e-commerce scraper
3. Structured data indexed to Pinecone

Monitor:
━━━━━━━━
    tail -f logs/app.log | grep "E-commerce"

Look for:
    Cache MISS - Performing fresh scrape
    [E-commerce] ✅ Structured scraping succeeded

Wait 2-5 minutes, then check Pinecone! 🎉
    """
    )


if __name__ == "__main__":
    asyncio.run(clear_cache_single_url())
