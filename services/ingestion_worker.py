import os
import asyncio
import io
import json
import requests
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from supabase import create_client
from services.web_scraper import scrape_url_text

# Load environment variables
load_dotenv()

# Initialize clients
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)


def get_supabase_storage_url(file_path: str) -> str:
    """Convert Supabase storage path to full URL"""
    return f"{supabase_url}/storage/v1/object/public/uploads/{file_path}"


def extract_text_from_pdf_url(url: str) -> str:
    try:
        # Convert Supabase path to full URL if needed
        if not url.startswith('http'):
            url = get_supabase_storage_url(url)

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        pdf_reader = PdfReader(io.BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:
        print(f"[Error] Failed to extract text from PDF {url}: {e}")
        return ""


def extract_text_from_json_url(url: str) -> str:
    try:
        # Convert Supabase path to full URL if needed
        if not url.startswith('http'):
            url = get_supabase_storage_url(url)

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract text from various possible JSON structures
        if isinstance(data, dict):
            if 'content' in data:
                return str(data['content'])
            elif 'text' in data:
                return str(data['text'])
            elif 'body' in data:
                return str(data['body'])
            else:
                # Convert entire dict to text
                return json.dumps(data, indent=2)
        elif isinstance(data, list):
            # Join list items
            return '\n'.join(str(item) for item in data)
        else:
            return str(data)

    except Exception as e:
        print(f"[Error] Failed to extract text from JSON {url}: {e}")
        return ""


def split_into_chunks(text: str) -> list:
    if not text.strip():
        return []

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]


def get_embeddings_for_chunks(chunks: list) -> list:
    if not chunks:
        return []

    embedder = OpenAIEmbeddings()
    vectors = embedder.embed_documents(chunks)
    return vectors


def upload_to_pinecone(chunks: list, vectors: list, namespace: str, upload_id: str):
    if not chunks or not vectors:
        print(f"[Warning] No chunks or vectors to upload for {upload_id}")
        return

    to_upsert = [
        (f"{upload_id}-{i}", vec, {"chunk": chunk, "upload_id": upload_id})
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    print(
        f"[Info] Uploading {len(to_upsert)} vectors to Pinecone namespace '{namespace}'")
    try:
        result = index.upsert(vectors=to_upsert, namespace=namespace)
        print(f"[Info] Pinecone upsert result: {result}")
    except Exception as e:
        print(f"[Error] Pinecone upload failed: {e}")
        raise


async def process_pending_uploads():
    """Main function to process all pending uploads"""
    try:
        # Get pending uploads from Supabase
        result = supabase.table("uploads").select(
            "*").eq("status", "pending").execute()

        uploads = result.data
        print(f"[Info] Found {len(uploads)} pending uploads")

        for upload in uploads:
            try:
                upload_id = upload["id"]
                org_id = upload["org_id"]
                source = upload["source"]
                doc_type = upload["type"]
                namespace = upload["pinecone_namespace"]

                print(
                    f"[Info] Processing upload {upload_id} of type {doc_type}")

                # Extract text based on document type
                if doc_type == "pdf":
                    text = extract_text_from_pdf_url(source)
                elif doc_type == "url":
                    text = scrape_url_text(source)
                elif doc_type == "json":
                    text = extract_text_from_json_url(source)
                else:
                    raise ValueError(f"Unsupported document type: {doc_type}")

                # Validate extracted text
                if not text or len(text.strip()) < 10:
                    raise ValueError("No meaningful text content extracted")

                print(
                    f"[Info] Extracted {len(text)} characters from {doc_type}")

                # Process text into chunks and embeddings
                chunks = split_into_chunks(text)
                if not chunks:
                    raise ValueError("No text chunks generated")

                embeddings = get_embeddings_for_chunks(chunks)
                if not embeddings:
                    raise ValueError("No embeddings generated")

                print(
                    f"[Info] Generated {len(chunks)} chunks and {len(embeddings)} embeddings")

                # Upload to Pinecone
                upload_to_pinecone(chunks, embeddings, namespace, upload_id)

                # Update status to completed
                update_result = supabase.table("uploads").update({
                    "status": "completed",
                    "error_message": None
                }).eq("id", upload_id).execute()

                print(f"[Success] Completed processing upload {upload_id}")

            except Exception as e:
                error_msg = str(e)
                print(
                    f"[Error] Processing failed for upload {upload.get('id', 'unknown')}: {error_msg}")

                # Update status to failed
                try:
                    update_result = supabase.table("uploads").update({
                        "status": "failed",
                        "error_message": error_msg
                    }).eq("id", upload["id"]).execute()
                except Exception as update_error:
                    print(
                        f"[Error] Failed to update error status: {update_error}")

    except Exception as e:
        print(f"[Error] Failed to process pending uploads: {e}")

# For testing purposes
if __name__ == "__main__":
    import asyncio
    asyncio.run(process_pending_uploads())
