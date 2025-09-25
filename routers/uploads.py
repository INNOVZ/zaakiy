import os
import uuid
from supabase import create_client
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org

router = APIRouter()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


class URLIngestRequest(BaseModel):
    """Request model for ingesting a URL."""

    url: str


class UpdateRequest(BaseModel):
    """Request model for updating an upload with a new URL."""

    url: str


class SearchRequest(BaseModel):
    """Request model for searching uploads."""

    query: str
    top_k: int = 5
    filter_upload_ids: list[str] = None  # Optional: filter by specific uploads


def delete_vectors_from_pinecone(upload_id: str, namespace: str):
    """Delete all vectors associated with an upload from Pinecone"""
    try:
        # First, get all vector IDs for this upload
        vector_ids = []

        # Use a dummy vector for metadata-only query
        dummy_vector = [0.0] * 1536  # OpenAI embeddings are 1536 dimensions

        # Query in batches to handle large numbers of vectors
        batch_size = 1000
        has_more = True

        while has_more:
            query_result = index.query(
                vector=dummy_vector,
                filter={"upload_id": upload_id},
                namespace=namespace,
                top_k=batch_size,
                include_metadata=False,
                include_values=False
            )

            batch_ids = [match.id for match in query_result.matches]
            vector_ids.extend(batch_ids)

            # If we got fewer results than batch_size, we're done
            has_more = len(batch_ids) == batch_size

        if vector_ids:
            # Delete vectors in batches (Pinecone has limits on batch operations)
            delete_batch_size = 100
            for i in range(0, len(vector_ids), delete_batch_size):
                batch = vector_ids[i:i + delete_batch_size]
                index.delete(ids=batch, namespace=namespace)

            print(
                f"[Info] Deleted {len(vector_ids)} vectors from Pinecone for upload {upload_id}")
        else:
            print(
                f"[Info] No vectors found in Pinecone for upload {upload_id}")

    except Exception as e:
        print(f"[Error] Failed to delete vectors from Pinecone: {e}")
        # Don't raise exception - we still want to continue with other operations


@router.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
    """
    Upload a PDF file to Supabase storage and insert its record into the database.

    Args:
        file (UploadFile): The PDF file to be uploaded.
        user: The authenticated user, verified via JWT token.

    Returns:
        dict: A dictionary containing the upload ID, file path, and a success message.
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Generate unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.pdf"

        # Upload to Supabase Storage
        file_content = await file.read()
        try:
            storage_result = supabase.storage.from_(
                "uploads").upload(supabase_path, file_content)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Storage upload failed: {str(e)}") from e

        # Insert record directly into database
        db_result = supabase.table("uploads").insert({
            "org_id": org_id,
            "type": "pdf",
            "source": supabase_path,
            "pinecone_namespace": f"org-{org_id}",
            "status": "pending"
        }).execute()

        return {
            "message": "PDF uploaded successfully",
            "upload_id": db_result.data[0]["id"],
            "path": supabase_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/json")
async def upload_json(
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
    """
    Upload a JSON file to Supabase storage and insert its record into the database.


    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Generate unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.json"

        # Upload to Supabase Storage
        file_content = await file.read()
        try:
            storage_result = supabase.storage.from_("uploads").upload(supabase_path, file_content)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Storage upload failed: {str(e)}") from e

        # Insert record directly into database
        db_result = supabase.table("uploads").insert({
            "org_id": org_id,
            "type": "json",
            "source": supabase_path,
            "pinecone_namespace": f"org-{org_id}",
            "status": "pending"
        }).execute()

        return {
            "message": "JSON uploaded successfully",
            "upload_id": db_result.data[0]["id"],
            "path": supabase_path
        }

        # return {
        #     "message": "JSON uploaded successfully",
        #     "upload_id": db_result.data[0]["id"],
        #     "path": supabase_path
        # }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/url")
async def ingest_url(
    request: URLIngestRequest,
    user=Depends(verify_jwt_token)
):
    """
    Ingest a URL by adding it to the Supabase database for processing.

    Args:
        request (URLIngestRequest): The request containing the URL to ingest.
        user: The authenticated user, verified via JWT token.

    Returns:
        dict: A dictionary containing the upload ID and a success message.
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Insert URL record directly into database
        db_result = supabase.table("uploads").insert({
            "org_id": org_id,
            "type": "url",
            "source": request.url,
            "pinecone_namespace": f"org-{org_id}",
            "status": "pending"
        }).execute()

        return {
            "message": "URL registered for ingestion",
            "upload_id": db_result.data[0]["id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/")
async def list_uploads(user=Depends(verify_jwt_token)):
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        result = supabase.table("uploads").select(
            "*").eq("org_id", org_id).order("created_at", desc=True).execute()

        return {"uploads": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{upload_id}/status")
async def get_upload_status(upload_id: str, user=Depends(verify_jwt_token)):
    try:
        result = supabase.table("uploads").select(
            "*").eq("id", upload_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        return result.data[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str, user=Depends(verify_jwt_token)):
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Fetch the complete upload record
        upload_result = supabase.table("uploads").select(
            "*").eq("id", upload_id).eq("org_id", org_id).execute()

        if not upload_result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        upload_record = upload_result.data[0]
        namespace = upload_record["pinecone_namespace"]
        file_type = upload_record["type"]
        source = upload_record["source"]

        # Delete vectors from Pinecone
        delete_vectors_from_pinecone(upload_id, namespace)

        # Delete file from Supabase storage if it's a file upload (not URL)
        if file_type in ["pdf", "json"] and source:
            try:
                supabase.storage.from_("uploads").remove([source])
                print(f"[Info] Deleted file from storage: {source}")
            except Exception as e:
                print(f"[Warning] Failed to delete file from storage: {e}")
                # Continue with database deletion even if storage deletion fails

        # Delete the upload record from the database
        supabase.table("uploads").delete().eq(
            "id", upload_id).eq("org_id", org_id).execute()

        return {
            "message": "Upload deleted successfully",
            "upload_id": upload_id,
            "type": file_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{upload_id}/status")
async def update_upload_status(upload_id: str, status: str, user=Depends(verify_jwt_token)):
    try:
        # Update the upload status in the database
        result = supabase.table("uploads").update({
            "status": status
        }).eq("id", upload_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        return {"message": "Upload status updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{upload_id}/pdf")
async def update_pdf_upload(
    upload_id: str,
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Fetch the existing upload record
        upload_result = supabase.table("uploads").select(
            "*").eq("id", upload_id).eq("org_id", org_id).execute()

        if not upload_result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        upload_record = upload_result.data[0]
        namespace = upload_record["pinecone_namespace"]
        old_source = upload_record["source"]

        # Delete existing vectors from Pinecone
        delete_vectors_from_pinecone(upload_id, namespace)

        # Delete old file from storage if it exists
        if old_source:
            try:
                supabase.storage.from_("uploads").remove([old_source])
                print(f"[Info] Deleted old file from storage: {old_source}")
            except Exception as e:
                print(f"[Warning] Failed to delete old file from storage: {e}")

        # Generate new unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.pdf"

        # Upload new file to Supabase Storage
        file_content = await file.read()
        try:
            storage_result = supabase.storage.from_(
                "uploads").upload(supabase_path, file_content)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Storage upload failed: {str(e)}") from e

        # Update the upload record in database
        db_result = supabase.table("uploads").update({
            "source": supabase_path,
            "status": "pending",
            "error_message": None,
            "updated_at": "now()"
        }).eq("id", upload_id).eq("org_id", org_id).execute()

        return {
            "message": "PDF updated successfully",
            "upload_id": upload_id,
            "new_path": supabase_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{upload_id}/json")
async def update_json_upload(
    upload_id: str,
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Fetch the existing upload record
        upload_result = supabase.table("uploads").select(
            "*").eq("id", upload_id).eq("org_id", org_id).execute()

        if not upload_result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        upload_record = upload_result.data[0]
        namespace = upload_record["pinecone_namespace"]
        old_source = upload_record["source"]

        # Delete existing vectors from Pinecone
        delete_vectors_from_pinecone(upload_id, namespace)

        # Delete old file from storage if it exists
        if old_source:
            try:
                supabase.storage.from_("uploads").remove([old_source])
                print(f"[Info] Deleted old file from storage: {old_source}")
            except Exception as e:
                print(f"[Warning] Failed to delete old file from storage: {e}")

        # Generate new unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.json"

        # Upload new file to Supabase Storage
        file_content = await file.read()
        try:
            storage_result = supabase.storage.from_(
                "uploads").upload(supabase_path, file_content)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Storage upload failed: {str(e)}") from e

        # Update the upload record in database
        db_result = supabase.table("uploads").update({
            "source": supabase_path,
            "status": "pending",
            "error_message": None,
            "updated_at": "now()"
        }).eq("id", upload_id).eq("org_id", org_id).execute()

        return {
            "message": "JSON updated successfully",
            "upload_id": upload_id,
            "new_path": supabase_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{upload_id}/url")
async def update_url_upload(
    upload_id: str,
    request: UpdateRequest,
    user=Depends(verify_jwt_token)
):
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Fetch the existing upload record
        upload_result = supabase.table("uploads").select(
            "*").eq("id", upload_id).eq("org_id", org_id).execute()

        if not upload_result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        upload_record = upload_result.data[0]
        namespace = upload_record["pinecone_namespace"]

        # Delete existing vectors from Pinecone
        delete_vectors_from_pinecone(upload_id, namespace)

        # Update the upload record in database with new URL
        db_result = supabase.table("uploads").update({
            "source": request.url,
            "status": "pending",
            "error_message": None,
            "updated_at": "now()"
        }).eq("id", upload_id).eq("org_id", org_id).execute()

        return {
            "message": "URL updated successfully",
            "upload_id": upload_id,
            "new_url": request.url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search")
async def search_uploads(
    request: SearchRequest,
    user=Depends(verify_jwt_token)
):
    try:
        from langchain_openai import OpenAIEmbeddings

        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]
        namespace = f"org-{org_id}"

        # Generate embedding for the query
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        query_vector = embeddings.embed_query(request.query)

        # Build filter if specific upload IDs are provided
        filter_dict = {}
        if request.filter_upload_ids:
            filter_dict["upload_id"] = {"$in": request.filter_upload_ids}

        # Search in Pinecone
        search_result = index.query(
            vector=query_vector,
            namespace=namespace,
            top_k=request.top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        # Format results
        results = []
        for match in search_result.matches:
            result = {
                "id": match.id,
                "score": match.score,
                "content": match.metadata.get("text", ""),
                "upload_id": match.metadata.get("upload_id"),
                "source": match.metadata.get("source"),
                "type": match.metadata.get("type")
            }
            results.append(result)

        return {
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{upload_id}/vectors")
async def get_upload_vectors(upload_id: str, user=Depends(verify_jwt_token)):
    """Debug endpoint to see what vectors exist for an upload"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]
        namespace = f"org-{org_id}"

        # Query vectors for this upload
        dummy_vector = [0.0] * 1536
        query_result = index.query(
            vector=dummy_vector,
            filter={"upload_id": upload_id},
            namespace=namespace,
            top_k=100,
            include_metadata=True,
            include_values=False
        )

        vectors_info = []
        for match in query_result.matches:
            vector_info = {
                "id": match.id,
                "metadata": match.metadata
            }
            vectors_info.append(vector_info)

        return {
            "upload_id": upload_id,
            "namespace": namespace,
            "vector_count": len(vectors_info),
            "vectors": vectors_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{upload_id}/reprocess")
async def reprocess_upload(upload_id: str, user=Depends(verify_jwt_token)):
    """Force reprocessing of an upload"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Check if upload exists and belongs to user's org
        upload_result = supabase.table("uploads").select(
            "*").eq("id", upload_id).eq("org_id", org_id).execute()

        if not upload_result.data:
            raise HTTPException(status_code=404, detail="Upload not found")

        upload_record = upload_result.data[0]
        namespace = upload_record["pinecone_namespace"]

        # Delete existing vectors from Pinecone
        delete_vectors_from_pinecone(upload_id, namespace)

        # Reset status to pending
        supabase.table("uploads").update({
            "status": "pending",
            "error_message": None,
            "updated_at": "now()"
        }).eq("id", upload_id).eq("org_id", org_id).execute()

        return {
            "message": "Upload queued for reprocessing",
            "upload_id": upload_id,
            "status": "pending"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats")
async def get_upload_stats(user=Depends(verify_jwt_token)):
    """Get statistics about uploads for the user's organization"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Get upload counts by status
        uploads = supabase.table("uploads").select(
            "status, type").eq("org_id", org_id).execute()

        stats = {
            "total_uploads": len(uploads.data),
            "by_status": {},
            "by_type": {}
        }

        for upload in uploads.data:
            status = upload.get("status", "unknown")
            upload_type = upload.get("type", "unknown")

            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["by_type"][upload_type] = stats["by_type"].get(
                upload_type, 0) + 1

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
