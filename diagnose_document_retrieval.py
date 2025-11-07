#!/usr/bin/env python3
"""
Diagnostic script to identify why document retrieval is failing
"""
import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.shared import get_openai_client
from app.services.storage.pinecone_client import get_pinecone_index
from app.services.storage.supabase_client import get_supabase_client, run_supabase


async def diagnose_retrieval(org_id: str, chatbot_id: str = None):
    """Comprehensive diagnosis of document retrieval issues"""
    namespace = f"org-{org_id}"

    print(f"\n{'='*80}")
    print(f"DOCUMENT RETRIEVAL DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Organization ID: {org_id}")
    print(f"Namespace: {namespace}\n")

    # Step 1: Check Pinecone Index
    print("Step 1: Checking Pinecone Index...")
    try:
        index = get_pinecone_index()
        if index:
            print("✅ Pinecone index is available")
            print(f"   Index name: {getattr(index, '_index_name', 'unknown')}")
        else:
            print("❌ Pinecone index is None")
            print("   Check PINECONE_API_KEY and PINECONE_INDEX environment variables")
            return
    except Exception as e:
        print(f"❌ Failed to get Pinecone index: {e}")
        return

    # Step 2: Check OpenAI Client
    print("\nStep 2: Checking OpenAI Client...")
    try:
        openai_client = get_openai_client()
        if openai_client:
            print("✅ OpenAI client is available")
        else:
            print("❌ OpenAI client is None")
            print("   Check OPENAI_API_KEY environment variable")
            return
    except Exception as e:
        print(f"❌ Failed to get OpenAI client: {e}")
        return

    # Step 3: Check if documents exist in namespace
    print("\nStep 3: Checking if documents exist in namespace...")
    try:
        # Generate a very generic embedding to test if ANY documents exist
        test_query = "test"
        print(f"   Generating embedding for test query: '{test_query}'")

        response = openai_client.embeddings.create(
            model="text-embedding-3-small", input=test_query
        )
        test_embedding = response.data[0].embedding
        print(f"   ✅ Embedding generated ({len(test_embedding)} dimensions)")

        # Query with a very high top_k to see if anything exists
        print(f"   Querying Pinecone namespace '{namespace}' with top_k=100...")
        results = index.query(
            vector=test_embedding, top_k=100, namespace=namespace, include_metadata=True
        )

        if results.matches and len(results.matches) > 0:
            print(f"   ✅ Found {len(results.matches)} documents in namespace!")
            print(f"\n   Sample documents:")
            for i, match in enumerate(results.matches[:5], 1):
                source = (
                    match.metadata.get("source", "unknown")
                    if match.metadata
                    else "unknown"
                )
                chunk_preview = ""
                if match.metadata:
                    chunk_preview = match.metadata.get(
                        "chunk", match.metadata.get("text", "")
                    )[:100]
                print(f"      {i}. Score: {match.score:.3f}, Source: {source}")
                if chunk_preview:
                    print(f"         Preview: {chunk_preview}...")
        else:
            print(f"   ❌ NO DOCUMENTS FOUND in namespace '{namespace}'")
            print(f"\n   This is the root cause! Documents need to be indexed.")
            print(f"\n   To fix:")
            print(f"   1. Check if uploads exist for this org in Supabase")
            print(f"   2. Verify documents were ingested with namespace: {namespace}")
            print(f"   3. Re-run ingestion if needed")
    except Exception as e:
        print(f"   ❌ Error checking namespace: {e}")
        import traceback

        traceback.print_exc()

    # Step 4: Check Supabase for uploads
    print("\nStep 4: Checking Supabase for uploads...")
    try:
        supabase = get_supabase_client()
        uploads_response = await run_supabase(
            lambda: (
                supabase.table("uploads")
                .select("id, source, status, org_id, pinecone_namespace, created_at")
                .eq("org_id", org_id)
                .order("created_at", desc=True)
                .limit(10)
                .execute()
            )
        )

        if uploads_response.data:
            print(f"   ✅ Found {len(uploads_response.data)} uploads for this org")
            print(f"\n   Recent uploads:")
            for upload in uploads_response.data[:5]:
                status = upload.get("status", "unknown")
                namespace_db = upload.get("pinecone_namespace", "not set")
                source = upload.get("source", "unknown")
                print(f"      - {source}")
                print(f"        Status: {status}, Namespace: {namespace_db}")
                if namespace_db != namespace:
                    print(
                        f"        ⚠️  WARNING: Namespace mismatch! Expected: {namespace}"
                    )
        else:
            print(f"   ⚠️  No uploads found for org_id: {org_id}")
            print(f"      You need to upload documents first!")
    except Exception as e:
        print(f"   ❌ Error checking Supabase: {e}")
        import traceback

        traceback.print_exc()

    # Step 5: Check chatbot config
    if chatbot_id:
        print(f"\nStep 5: Checking chatbot configuration...")
        try:
            supabase = get_supabase_client()
            chatbot_response = await run_supabase(
                lambda: (
                    supabase.table("chatbots")
                    .select("id, name, org_id, chain_status")
                    .eq("id", chatbot_id)
                    .execute()
                )
            )

            if chatbot_response.data:
                chatbot = chatbot_response.data[0]
                print(f"   ✅ Chatbot found: {chatbot.get('name', 'unknown')}")
                print(f"      Org ID: {chatbot.get('org_id')}")
                print(f"      Status: {chatbot.get('chain_status', 'unknown')}")
                if chatbot.get("org_id") != org_id:
                    print(f"      ⚠️  WARNING: Chatbot org_id doesn't match!")
            else:
                print(f"   ⚠️  Chatbot not found: {chatbot_id}")
        except Exception as e:
            print(f"   ❌ Error checking chatbot: {e}")

    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}\n")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_document_retrieval.py <org_id> [chatbot_id]")
        print("\nExample:")
        print(
            "  python diagnose_document_retrieval.py 09294a59-4100-4854-bca3-b23bbe390557"
        )
        print(
            "  python diagnose_document_retrieval.py 09294a59-4100-4854-bca3-b23bbe390557 057aa048-f77d-44ec-aba2-a8e0d2abd11f"
        )
        sys.exit(1)

    org_id = sys.argv[1]
    chatbot_id = sys.argv[2] if len(sys.argv) > 2 else None

    await diagnose_retrieval(org_id, chatbot_id)


if __name__ == "__main__":
    asyncio.run(main())
