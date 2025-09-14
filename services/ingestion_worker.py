"# services/ingestion_worker.py\n\n"

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
    """Convert Supabase storage path to authenticated URL"""
    # For private buckets, we'll use the authenticated storage endpoint
    return f"{supabase_url}/storage/v1/object/uploads/{file_path}"


def get_authenticated_headers() -> dict:
    """Get headers for authenticated Supabase requests"""
    return {
        'Authorization': f'Bearer {supabase_key}',
        'apikey': supabase_key,
        'Content-Type': 'application/json'
    }


def extract_text_from_pdf_url(url: str) -> str:
    """Extract text from PDF with authenticated access"""
    try:
        # Convert Supabase path to authenticated URL if needed
        if not url.startswith('http'):
            url = get_supabase_storage_url(url)

        print(f"[Info] Attempting to fetch PDF from: {url}")

        # Use authenticated request for private buckets
        headers = get_authenticated_headers()

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        print(
            f"[Info] Successfully fetched PDF, size: {len(response.content)} bytes")

        if len(response.content) == 0:
            raise ValueError("Downloaded file is empty")

        # Check if response is actually a PDF
        if not response.content.startswith(b'%PDF'):
            print(
                f"[Warning] File doesn't appear to be a PDF. Content type: {response.headers.get('content-type')}")
            print(f"[Warning] First 100 bytes: {response.content[:100]}")
            raise ValueError("File is not a valid PDF")

        pdf_reader = PdfReader(io.BytesIO(response.content))
        print(f"[Info] PDF has {len(pdf_reader.pages)} pages")

        text = ""
        pages_with_text = 0

        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pages_with_text += 1
                    print(
                        f"[Info] Extracted text from page {i+1}: {len(page_text)} chars")
                else:
                    print(f"[Warning] No text found on page {i+1}")
            except (AttributeError, ValueError, TypeError) as page_error:
                print(
                    f"[Warning] Error extracting text from page {i+1}: {page_error}")
                continue

        print(
            f"[Info] Total text extracted: {len(text)} characters from {pages_with_text} pages")

        if len(text.strip()) < 10:
            raise ValueError(
                f"PDF contains insufficient text content. Only {len(text)} characters extracted from {pages_with_text} pages out of {len(pdf_reader.pages)} total pages. This might be a scanned/image-based PDF.")

        return text.strip()

    except requests.RequestException as e:
        print(f"[Error] HTTP error while fetching PDF {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[Error] Response status: {e.response.status_code}")
            print(f"[Error] Response headers: {e.response.headers}")
            print(f"[Error] Response content: {e.response.text[:500]}")
        raise ValueError(f"Failed to download PDF: {str(e)}") from e
    except Exception as e:
        print(f"[Error] PDF processing error: {e}")
        raise ValueError(f"PDF processing failed: {str(e)}") from e


def extract_text_from_json_url(url: str) -> str:
    """Extract text from JSON with authenticated access"""
    try:
        # Convert Supabase path to authenticated URL if needed
        if not url.startswith('http'):
            url = get_supabase_storage_url(url)

        print(f"[Info] Attempting to fetch JSON from: {url}")

        # Use authenticated request for private buckets
        headers = get_authenticated_headers()

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        print(
            f"[Info] Successfully fetched JSON, size: {len(response.content)} bytes")

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

    except requests.RequestException as e:
        print(f"[Error] HTTP error while extracting text from JSON {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[Error] Response status: {e.response.status_code}")
            print(f"[Error] Response content: {e.response.text[:200]}")
        raise ValueError(f"Failed to download JSON: {str(e)}") from e
    except json.JSONDecodeError as e:
        print(
            f"[Error] JSON decode error while extracting text from JSON {url}: {e}")
        raise ValueError(f"Invalid JSON format: {str(e)}") from e
    except Exception as e:
        print(f"[Error] JSON processing error: {e}")
        raise ValueError(f"JSON processing failed: {str(e)}") from e


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
    """Main function to process all pending uploads with authenticated storage access"""
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
                    text = scrape_url_text(source)  # URLs don't need auth
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
                supabase.table("uploads").update({
                    "status": "completed",
                    "error_message": None
                }).eq("id", upload_id).execute()

                print(f"[Success] Completed processing upload {upload_id}")

            except (ValueError, TypeError, requests.RequestException) as e:
                error_msg = str(e)
                print(
                    f"[Error] Processing failed for upload {upload.get('id', 'unknown')}: {error_msg}")

                # Update status to failed
                try:
                    supabase.table("uploads").update({
                        "status": "failed",
                        "error_message": error_msg
                    }).eq("id", upload["id"]).execute()
                except (requests.RequestException, ValueError) as update_error:
                    print(
                        f"[Error] Failed to update error status: {update_error}")

    except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"[Error] Failed to process pending uploads: {e}")

    # For testing purposes
    if __name__ == "__main__":
        asyncio.run(process_pending_uploads())
