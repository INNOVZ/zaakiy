from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from ..services.auth import verify_jwt_token, get_user_with_org

load_dotenv()

router = APIRouter()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
embedder = OpenAIEmbeddings()


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    namespace: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    upload_id: str
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user=Depends(verify_jwt_token)
):
    """
    Search through uploaded documents using vector similarity
    """
    try:
        # Get user's organization
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data.get("org_id")

        if not org_id:
            raise HTTPException(
                status_code=400, detail="User not associated with an organization")

        # Use organization-specific namespace
        namespace = request.namespace or f"org-{org_id}"

        print(f"[Info] Searching in namespace: {namespace}")
        print(f"[Info] Query: {request.query}")

        # Convert query to embedding
        query_embedding = embedder.embed_query(request.query)

        # Search Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )

        # Format results
        formatted_results = []
        for match in search_results.matches:
            result = SearchResult(
                id=match.id,
                score=float(match.score),
                content=match.metadata.get("chunk", ""),
                upload_id=match.metadata.get("upload_id", ""),
                metadata=match.metadata
            )
            formatted_results.append(result)

        print(f"[Info] Found {len(formatted_results)} results")

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_found=len(formatted_results)
        )

    except Exception as e:
        print(f"[Error] Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/health")
async def search_health():
    """Check if search service is healthy"""
    try:
        # Test Pinecone connection
        stats = index.describe_index_stats()

        return {
            "status": "healthy",
            "pinecone_connected": True,
            "total_vectors": stats.total_vector_count,
            "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Search service unhealthy: {str(e)}")


@router.post("/similar")
async def find_similar_content(
    request: SearchRequest,
    user=Depends(verify_jwt_token)
):
    """
    Find content similar to a given text (for recommendation systems)
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data.get("org_id")
        namespace = f"org-{org_id}"

        # Get embedding for the query
        query_embedding = embedder.embed_query(request.query)

        # Search for similar content
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )

        similar_content = []
        for match in results.matches:
            similar_content.append({
                "content": match.metadata.get("chunk", ""),
                "similarity_score": float(match.score),
                "source": match.metadata.get("upload_id", "unknown")
            })

        return {
            "query": request.query,
            "similar_content": similar_content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Similar search failed: {str(e)}")


@router.get("/stats")
async def get_search_stats(user=Depends(verify_jwt_token)):
    """Get statistics about the organization's uploaded content"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data.get("org_id")
        namespace = f"org-{org_id}"

        # Get index statistics
        stats = index.describe_index_stats()
        namespace_stats = stats.namespaces.get(namespace, {})

        return {
            "organization_id": org_id,
            "namespace": namespace,
            "total_vectors": namespace_stats.get("vector_count", 0),
            "index_total_vectors": stats.total_vector_count,
            "available_namespaces": list(stats.namespaces.keys())
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Stats retrieval failed: {str(e)}")
