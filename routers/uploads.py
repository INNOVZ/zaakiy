import os
import uuid
from typing import List, Dict, Any
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from pinecone import Pinecone
from supabase import create_client
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.vector_management import vector_management_service
from utils.logging_config import LogContext, PerformanceLogger, get_logger, log_api_request, log_api_response

router = APIRouter()
logger = get_logger("uploads")


# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


class URLIngestRequest(BaseModel):
    """
This module handles vector data ingestion from URLs.
"""
    url: str


class UpdateRequest(BaseModel):
    """
This module handles upload vector update functionality.
"""
    url: str


class SearchRequest(BaseModel):
    """
This module handles upload vector search functionality.
"""
    query: str
    top_k: int = 5
    filter_upload_ids: list[str] = None  # Optional: filter by specific uploads


def delete_vectors_from_pinecone(upload_id: str, namespace: str):
    """Delete all vectors associated with an upload from Pinecone - IMPROVED VERSION"""
    try:
        logging.info(
            f"Starting vector deletion for upload {upload_id} in namespace {namespace}")

        # Method 1: Try deletion by metadata filter (most efficient)
        try:
            delete_response = index.delete(
                filter={"upload_id": upload_id},
                namespace=namespace
            )
            logging.info(
                f"Deleted vectors using filter for upload {upload_id}: {delete_response}")
            return
        except Exception as filter_error:
            logging.warning(
                f"Filter-based deletion failed, falling back to query method: {filter_error}")

        # Method 2: Fallback to query-based deletion
        vector_ids = []
        batch_size = 1000

        # Use sparse vector query instead of dummy dense vector (more efficient)
        try:
            # Try to get vector IDs without dummy vector
            query_result = index.query(
                filter={"upload_id": upload_id},
                namespace=namespace,
                top_k=batch_size,
                include_metadata=False,
                include_values=False
            )

            batch_ids = [match.id for match in query_result.matches]
            vector_ids.extend(batch_ids)

        except Exception as query_error:
            logging.warning(
                f"Efficient query failed, using dummy vector: {query_error}")

            # Last resort: use dummy vector approach (but with smaller dimension check)
            try:
                # Get index stats to determine dimension
                stats = index.describe_index_stats()
                dimension = 1536  # Default OpenAI dimension

                dummy_vector = [0.0] * dimension

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
                    has_more = len(batch_ids) == batch_size

            except Exception as dummy_error:
                logging.error(f"All deletion methods failed: {dummy_error}")
                raise

        if vector_ids:
            # Delete vectors in optimal batches
            delete_batch_size = 100
            total_deleted = 0

            for i in range(0, len(vector_ids), delete_batch_size):
                batch = vector_ids[i:i + delete_batch_size]
                try:
                    delete_result = index.delete(
                        ids=batch, namespace=namespace)
                    total_deleted += len(batch)
                    logging.info(
                        f"Deleted batch {i//delete_batch_size + 1}: {len(batch)} vectors")
                except Exception as batch_error:
                    logging.error(
                        f"Failed to delete batch {i//delete_batch_size + 1}: {batch_error}")
                    # Continue with other batches

            logging.info(
                f"Successfully deleted {total_deleted} vectors from Pinecone for upload {upload_id}")
        else:
            logging.info(
                f"No vectors found in Pinecone for upload {upload_id}")

    except Exception as e:
        logging.error(
            f"Critical error in vector deletion for upload {upload_id}: {e}")
        # Don't raise exception - we still want to continue with other operations
        # But log it for monitoring


async def batch_delete_uploads(upload_ids: List[str], org_id: str) -> Dict[str, Any]:
    """Efficiently delete multiple uploads in batch"""
    try:
        namespace = f"org-{org_id}"
        results = {"success": [], "failed": []}

        # Get all upload records first
        upload_records = supabase.table("uploads").select("*").in_(
            "id", upload_ids
        ).eq("org_id", org_id).execute()

        if not upload_records.data:
            return {"success": [], "failed": upload_ids, "error": "No uploads found"}

        # Delete vectors for all uploads in batch
        for upload_record in upload_records.data:
            upload_id = upload_record["id"]
            try:
                delete_vectors_from_pinecone(upload_id, namespace)

                # Delete file from storage if applicable
                if upload_record["type"] in ["pdf", "json"] and upload_record["source"]:
                    try:
                        supabase.storage.from_("uploads").remove(
                            [upload_record["source"]])
                    except Exception as storage_error:
                        logging.warning(
                            f"Storage deletion failed for {upload_id}: {storage_error}")

                results["success"].append(upload_id)

            except Exception as e:
                logging.error(f"Failed to delete upload {upload_id}: {e}")
                results["failed"].append(upload_id)

        # Delete database records for successful deletions
        if results["success"]:
            supabase.table("uploads").delete().in_(
                "id", results["success"]
            ).eq("org_id", org_id).execute()

        return results

    except Exception as e:
        logging.error(f"Batch delete failed: {e}")
        return {"success": [], "failed": upload_ids, "error": str(e)}


@router.post("/{upload_id}/verify-deletion")
async def verify_upload_deletion(upload_id: str, user=Depends(verify_jwt_token)):
    """Verify that upload vectors were completely deleted"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]
        namespace = f"org-{org_id}"

        with LogContext(upload_id=upload_id, org_id=org_id):
            logger.info(f"Verifying deletion for upload {upload_id}")

            verification_result = await vector_management_service.verify_deletion(
                upload_id, namespace
            )

            return {
                "upload_id": upload_id,
                "namespace": namespace,
                "verification": verification_result,
                "timestamp": datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Deletion verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed"
        ) from e


@router.post("/cleanup-orphaned")
async def cleanup_orphaned_vectors(
    days_old: int = 7,
    user=Depends(verify_jwt_token)
):
    """Clean up orphaned vectors that no longer have corresponding uploads"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        with LogContext(org_id=org_id, days_old=days_old):
            logger.info(f"Starting orphaned vector cleanup for org {org_id}")

            cleanup_stats = await vector_management_service.cleanup_orphaned_vectors(
                org_id, days_old
            )

            return {
                "success": True,
                "org_id": org_id,
                "cleanup_stats": cleanup_stats,
                "parameters": {"days_old": days_old},
                "timestamp": datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Orphaned vector cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cleanup failed"
        ) from e

# Helper functions


async def _batch_delete_storage_files(upload_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Delete files from storage for multiple uploads"""
    storage_results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }

    for upload_record in upload_records:
        if upload_record["type"] in ["pdf", "json"] and upload_record["source"]:
            try:
                supabase.storage.from_("uploads").remove(
                    [upload_record["source"]])
                storage_results["successful"].append(upload_record["id"])
                logger.info(
                    f"Deleted storage file for upload {upload_record['id']}")
            except Exception as e:
                storage_results["failed"].append({
                    "upload_id": upload_record["id"],
                    "error": str(e)
                })
                logger.warning(
                    f"Failed to delete storage file for upload {upload_record['id']}: {e}")
        else:
            storage_results["skipped"].append(upload_record["id"])

    return storage_results


async def _delete_single_storage_file(upload_record: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a single file from storage"""
    try:
        supabase.storage.from_("uploads").remove([upload_record["source"]])
        logger.info(f"Deleted storage file: {upload_record['source']}")
        return {"success": True, "file_path": upload_record["source"]}
    except Exception as e:
        logger.warning(f"Failed to delete storage file: {e}")
        return {"success": False, "error": str(e), "file_path": upload_record["source"]}


@router.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
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
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Generate unique file path
        file_id = str(uuid.uuid4())
        supabase_path = f"org-{org_id}/{file_id}.json"

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
