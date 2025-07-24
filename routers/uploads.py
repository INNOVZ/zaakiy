
import os
import uuid
from supabase import create_client
from fastapi import APIRouter, Depends, UploadFile, File
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.supabase_client import client


router = APIRouter()


@router.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(verify_jwt_token)
):
    user_data = await get_user_with_org(user["user_id"])
    org_id = user_data["org_id"]
    pinecone_namespace = f"org-{org_id}"

    file_id = str(uuid.uuid4())
    supabase_path = f"org-{org_id}/{file_id}.pdf"

    # Upload to Supabase Storage
    file_content = await file.read()

    # Upload manually using REST API
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    supabase.storage.from_("uploads").upload(supabase_path, file_content)

    # Insert into uploads table
    await client.post("/uploads", json={
        "org_id": org_id,
        "type": "pdf",
        "source": supabase_path,
        "pinecone_namespace": pinecone_namespace,
        "status": "pending"
    })

    return {"message": "File uploaded successfully", "path": supabase_path}
