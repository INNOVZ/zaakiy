import os
from typing import List

from app.services.storage.pinecone_client import get_pinecone_index


def delete_by_sources(namespace: str, sources: List[str]) -> None:
    index = get_pinecone_index()
    if not sources:
        print("❌ No sources provided")
        return
    # Pinecone supports equality and $in filters on metadata
    filter_expr = {"source": {"$in": sources}}
    print(
        f"Deleting vectors in namespace '{namespace}' for {len(sources)} source(s)..."
    )
    index.delete(namespace=namespace, filter=filter_expr)
    print("✅ Delete request submitted")


def main():
    print("=" * 80)
    print("🗑️  PINECONE DELETE BY SOURCE URL(S)")
    print("=" * 80)
    namespace = input("Enter Pinecone namespace (e.g., org-xxxxxxxx): ").strip()
    urls_raw = input(
        "Enter comma-separated source URL(s) to delete (exact match): "
    ).strip()
    urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
    delete_by_sources(namespace, urls)


if __name__ == "__main__":
    main()
