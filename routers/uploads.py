import os
import uuid
from supabase import create_client
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org

router = APIRouter()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)


class URLIngestRequest(BaseModel):
    url: str


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

        return {
            "message": "JSON uploaded successfully",
            "upload_id": db_result.data[0]["id"],
            "path": supabase_path
        }

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
