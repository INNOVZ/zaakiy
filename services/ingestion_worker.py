import os
import asyncio
import io
import json
import httpx
from dotenv import load_dotenv
from pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client
from services.web_scraper import scrape_url_text
from services.client_manager import client_manager

# Load environment variables
load_dotenv()

# Use shared clients
supabase = client_manager.supabase
index = client_manager.pinecone_index


async def extract_text_from_pdf_url(url: str) -> str:
    """Extract text from PDF with async HTTP client"""
    try:
        if not url.startswith('http'):
            supabase_url = os.getenv("SUPABASE_URL")
            url = f"{supabase_url}/storage/v1/object/uploads/{url}"

        headers = {
            'Authorization': f'Bearer {os.getenv("SUPABASE_SERVICE_ROLE_KEY")}',
            'apikey': os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            'Content-Type': 'application/json'
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        if len(response.content) == 0:
            raise ValueError("Downloaded file is empty")

        if not response.content.startswith(b'%PDF'):
            raise ValueError("File is not a valid PDF")

        pdf_reader = PdfReader(io.BytesIO(response.content))

        text = ""
        pages_with_text = 0

        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pages_with_text += 1
            except (AttributeError, ValueError, TypeError):
                continue

        if len(text.strip()) < 10:
            raise ValueError(
                f"PDF contains insufficient text content. "
                f"Only {len(text)} characters extracted from {pages_with_text} pages.")

        return text.strip()

    except httpx.HTTPError as e:
        raise ValueError(f"Failed to download PDF: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"PDF processing failed: {str(e)}") from e


async def extract_text_from_json_url(url: str) -> str:
    """Extract text from JSON with async HTTP client"""
    try:
        if not url.startswith('http'):
            supabase_url = os.getenv("SUPABASE_URL")
            url = f"{supabase_url}/storage/v1/object/uploads/{url}"

        headers = {
            'Authorization': f'Bearer {os.getenv("SUPABASE_SERVICE_ROLE_KEY")}',
            'apikey': os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            'Content-Type': 'application/json'
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            if 'content' in data:
                return str(data['content'])
            elif 'text' in data:
                return str(data['text'])
            elif 'body' in data:
                return str(data['body'])
            else:
                return json.dumps(data, indent=2)
        elif isinstance(data, list):
            return '\n'.join(str(item) for item in data)
        else:
            return str(data)

    except httpx.HTTPError as e:
        raise ValueError(f"Failed to download JSON: {str(e)}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"JSON processing failed: {str(e)}") from e


def split_into_chunks(text: str) -> list:
    """Split text into chunks"""
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]


async def get_embeddings_for_chunks(chunks: list) -> list:
    """Generate embeddings using async OpenAI client"""
    if not chunks:
        return []

    try:
        openai_client = client_manager.openai

        # Process chunks in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            response = await openai_client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

            # Small delay to respect rate limits
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)

        return all_embeddings

    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {str(e)}") from e


def upload_to_pinecone(chunks: list, vectors: list, namespace: str, upload_id: str):
    """Upload vectors to Pinecone (this remains sync as Pinecone client is sync)"""
    if not chunks or not vectors:
        return

    to_upsert = [
        (f"{upload_id}-{i}", vec, {"chunk": chunk, "upload_id": upload_id})
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    try:
        result = index.upsert(vectors=to_upsert, namespace=namespace)
        print(f"Uploaded {len(to_upsert)} vectors to Pinecone: {result}")
    except Exception as e:
        raise ValueError(f"Pinecone upload failed: {str(e)}") from e


async def process_pending_uploads():
    """Main async function to process all pending uploads"""
    try:
        # Get pending uploads from Supabase
        result = supabase.table("uploads").select(
            "*").eq("status", "pending").execute()

        uploads = result.data
        print(f"Found {len(uploads)} pending uploads")

        for upload in uploads:
            try:
                upload_id = upload["id"]
                org_id = upload["org_id"]
                source = upload["source"]
                doc_type = upload["type"]
                namespace = upload["pinecone_namespace"]

                print(f"Processing upload {upload_id} of type {doc_type}")

                # Extract text based on document type
                if doc_type == "pdf":
                    text = await extract_text_from_pdf_url(source)
                elif doc_type == "url":
                    text = await scrape_url_text(source)
                elif doc_type == "json":
                    text = await extract_text_from_json_url(source)
                else:
                    raise ValueError(f"Unsupported document type: {doc_type}")

                if not text or len(text.strip()) < 10:
                    raise ValueError("No meaningful text content extracted")

                print(f"Extracted {len(text)} characters from {doc_type}")

                # Process text into chunks and embeddings
                chunks = split_into_chunks(text)
                if not chunks:
                    raise ValueError("No text chunks generated")

                embeddings = await get_embeddings_for_chunks(chunks)
                if not embeddings:
                    raise ValueError("No embeddings generated")

                print(
                    f"Generated {len(chunks)} chunks and {len(embeddings)} embeddings")

                # Upload to Pinecone
                upload_to_pinecone(chunks, embeddings, namespace, upload_id)

                # Update status to completed
                supabase.table("uploads").update({
                    "status": "completed",
                    "error_message": None
                }).eq("id", upload_id).execute()

                print(f"Completed processing upload {upload_id}")

            except (ValueError, TypeError, httpx.HTTPError) as e:
                error_msg = str(e)
                print(
                    f"Processing failed for upload {upload.get('id', 'unknown')}: {error_msg}")

                # Update status to failed
                try:
                    supabase.table("uploads").update({
                        "status": "failed",
                        "error_message": error_msg
                    }).eq("id", upload["id"]).execute()
                except Exception as update_error:
                    print(f"Failed to update error status: {update_error}")

    except Exception as e:
        print(f"Failed to process pending uploads: {e}")


if __name__ == "__main__":
    asyncio.run(process_pending_uploads())
