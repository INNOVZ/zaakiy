import os
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from ..models import URLIngestRequest, UpdateRequest, SearchRequest
from ..services.auth import verify_jwt_token, get_user_with_org
from ..services.storage.supabase_client import get_supabase_client
from ..services.storage.pinecone_client import get_pinecone_index
from ..utils.validators import validate_file_size, validate_file_type
from ..utils.rate_limiter import rate_limit, get_rate_limit_config
from ..utils.vector_operations import efficient_delete_by_upload_id

router = APIRouter()

# Get centralized clients
supabase = get_supabase_client()
index = get_pinecone_index()


async def _update_upload_helper(
    upload_id: str,
    org_id: str,
    file_type: str,
    new_source: str,
    old_source: str = None,
    file_content: bytes = None
) -> dict:
    """
    Helper function to update an upload (reduces code duplication)

    Args:
        upload_id: Upload ID
        org_id: Organization ID
        file_type: Type of upload (pdf, json, url)
        new_source: New source path or URL
        old_source: Old source path (for cleanup)
        file_content: File content for file uploads (None for URLs)

    Returns:
        Success response dictionary
    """
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

    # Handle file uploads (pdf, json)
    if file_content is not None:
        # Delete old file from storage if it exists
        if old_source:
            try:
                supabase.storage.from_("uploads").remove([old_source])
                print(f"[Info] Deleted old file from storage: {old_source}")
            except Exception as e:
                print(f"[Warning] Failed to delete old file from storage: {e}")

        # Upload new file to Supabase Storage
        try:
            storage_result = supabase.storage.from_(
                "uploads").upload(new_source, file_content)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Storage upload failed: {str(e)}") from e

    # Update the upload record in database
    db_result = supabase.table("uploads").update({
        "source": new_source,
        "status": "pending",
        "error_message": None,
        "updated_at": "now()"
    }).eq("id", upload_id).eq("org_id", org_id).execute()

    return {
        "message": f"{file_type.upper()} updated successfully",
        "upload_id": upload_id,
        "new_source": new_source
    }


def delete_vectors_from_pinecone(upload_id: str, namespace: str):
    """
    Delete all vectors associated with an upload from Pinecone

    Uses Pinecone's delete_by_metadata feature for efficient deletion
    without needing to query for vector IDs first.
    """
    try:
        import logging
        logger = logging.getLogger(__name__)

        logger.info(
            f"Deleting vectors for upload {upload_id} from namespace {namespace}")

        # Use Pinecone's delete by metadata filter - much more efficient!
        # This deletes all vectors matching the filter without needing to query first
        try:
            # Try the new delete method with filter (Pinecone v3+)
            delete_response = index.delete(
                filter={"upload_id": upload_id},
                namespace=namespace
            )

            logger.info(
                f"Successfully deleted vectors for upload {upload_id} using metadata filter"
            )

        except (TypeError, AttributeError) as filter_error:
            # Fallback for older Pinecone versions that don't support filter in delete
            logger.warning(
                f"Metadata filter delete not supported, falling back to query-then-delete: {filter_error}"
            )

            # Fallback: Query for IDs then delete (less efficient but works)
            vector_ids = []
            dummy_vector = [0.0] * 1536  # OpenAI embeddings dimension

            # Query in batches
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
                # Delete in batches
                delete_batch_size = 100
                for i in range(0, len(vector_ids), delete_batch_size):
                    batch = vector_ids[i:i + delete_batch_size]
                    index.delete(ids=batch, namespace=namespace)

                logger.info(
                    f"Deleted {len(vector_ids)} vectors for upload {upload_id} (fallback method)"
                )
            else:
                logger.info(f"No vectors found for upload {upload_id}")

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"Failed to delete vectors from Pinecone for upload {upload_id}: {e}",
            exc_info=True
        )
        # Don't raise exception - we still want to continue with other operations


@router.post("/pdf")
@rate_limit(**get_rate_limit_config("upload"))
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

        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        try:
            validate_file_type(file.filename, ['.pdf'])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Read file content
        file_content = await file.read()

        # Validate file size (50MB max)
        try:
            validate_file_size(len(file_content), max_size_mb=50)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate it's actually a PDF
        if not file_content.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400,
                detail="File is not a valid PDF"
            )

        # Generate unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.pdf"

        # Upload to Supabase Storage
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
@rate_limit(**get_rate_limit_config("upload"))
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

        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        try:
            validate_file_type(file.filename, ['.json'])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Read file content
        file_content = await file.read()

        # Validate file size (10MB max for JSON)
        try:
            validate_file_size(len(file_content), max_size_mb=10)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate it's actually valid JSON
        import json
        try:
            json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File is not valid JSON"
            )
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File encoding is not UTF-8"
            )

        # Generate unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.json"

        # Upload to Supabase Storage
        try:
            storage_result = supabase.storage.from_(
                "uploads").upload(supabase_path, file_content)
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
@rate_limit(**get_rate_limit_config("upload"))
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
async def list_uploads(
    page: int = 1,
    page_size: int = 20,
    status: str = None,
    type: str = None,
    user=Depends(verify_jwt_token)
):
    """
    List uploads with pagination and filtering

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        status: Filter by status (pending, completed, failed)
        type: Filter by type (pdf, json, url)
    """
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")

        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=400, detail="Page size must be between 1 and 100")

        # Calculate offset
        offset = (page - 1) * page_size

        # Build query
        query = supabase.table("uploads").select(
            "*", count="exact"
        ).eq("org_id", org_id)

        # Apply filters
        if status:
            allowed_statuses = ["pending", "completed", "failed"]
            if status not in allowed_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Status must be one of: {', '.join(allowed_statuses)}"
                )
            query = query.eq("status", status)

        if type:
            allowed_types = ["pdf", "json", "url"]
            if type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Type must be one of: {', '.join(allowed_types)}"
                )
            query = query.eq("type", type)

        # Apply pagination and ordering
        result = query.order("created_at", desc=True).range(
            offset, offset + page_size - 1
        ).execute()

        # Get total count
        total_count = result.count if hasattr(
            result, 'count') else len(result.data)

        # Calculate pagination metadata
        total_pages = (total_count + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1

        return {
            "uploads": result.data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }

    except HTTPException:
        raise
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
    """Update a PDF upload with a new file"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Validate file type
        if not file.filename or not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Generate new unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.pdf"

        # Read file content
        file_content = await file.read()

        # Validate file size
        try:
            validate_file_size(len(file_content), max_size_mb=50)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Use helper function to update
        return await _update_upload_helper(
            upload_id=upload_id,
            org_id=org_id,
            file_type="pdf",
            new_source=supabase_path,
            file_content=file_content
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{upload_id}/json")
async def update_json_upload(
    upload_id: str,
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
    """Update a JSON upload with a new file"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Validate file type
        if not file.filename or not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be JSON")

        # Generate new unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.json"

        # Read file content
        file_content = await file.read()

        # Validate file size
        try:
            validate_file_size(len(file_content), max_size_mb=10)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate JSON format
        import json
        try:
            json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        # Use helper function to update
        return await _update_upload_helper(
            upload_id=upload_id,
            org_id=org_id,
            file_type="json",
            new_source=supabase_path,
            file_content=file_content
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/{upload_id}/url")
async def update_url_upload(
    upload_id: str,
    request: UpdateRequest,
    user=Depends(verify_jwt_token)
):
    """Update a URL upload with a new URL"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Use helper function to update (no file content for URLs)
        return await _update_upload_helper(
            upload_id=upload_id,
            org_id=org_id,
            file_type="url",
            new_source=request.url,
            file_content=None  # URLs don't have file content
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search")
@rate_limit(**get_rate_limit_config("search"))
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
