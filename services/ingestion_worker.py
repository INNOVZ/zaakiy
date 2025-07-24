import os
import io
import requests
import httpx
import openai
from pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from services.supabase_client import client
from services.web_scraper import scrape_url_text

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client (v3)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Index name like 'chatbot-index'
index = pc.Index(os.getenv("PINECONE_INDEX"))


# Extract text from PDF URL
def extract_text_from_pdf_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        pdf_reader = PdfReader(io.BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()

    except requests.RequestException as e:
        print(
            f"[Error] HTTP error while extracting text from PDF URL {url}: {e}")
        return ""

    except (OSError, IOError) as e:
        print(
            f"[Error] File error while extracting text from PDF URL {url}: {e}")
        return ""


# Extract text from JSON URL
def extract_text_from_json(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()  # Assuming JSON contains a text key
        text = data.get('content', '')  # Adjust based on your JSON structure
        return text.strip()

    except requests.RequestException as e:
        print(
            f"[Error] HTTP error while extracting text from JSON URL {url}: {e}")
        return ""
    except ValueError as e:
        print(
            f"[Error] JSON decoding error while extracting text from JSON URL {url}: {e}")
        return ""


# Split text into chunks
def split_into_chunks(text: str) -> list:
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks


# Get embeddings for text chunks
def get_embeddings_for_chunks(chunks: list) -> list:
    embedder = OpenAIEmbeddings()
    vectors = embedder.embed_documents(chunks)
    return vectors


# Upload vectors to Pinecone
def upload_to_pinecone(chunks: list, vectors: list, namespace: str):
    to_upsert = [
        (f"{namespace}-{i}", vec, {"chunk": chunk})
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    index.upsert(vectors=to_upsert, namespace=namespace)


# Main ingestion worker that processes pending uploads
async def process_pending_uploads():
    res = await client.get("/uploads", params={"status": "eq.pending"})
    uploads = res.json()

    for upload in uploads:
        try:
            org_ns = upload["pinecone_namespace"]
            upload_id = upload["id"]
            source = upload["source"]
            doc_type = upload["type"]

            # Handle different document types (PDF, URL, JSON)
            if doc_type == "pdf":
                text = extract_text_from_pdf_url(source)

            elif doc_type == "url":
                text = scrape_url_text(source)

            elif doc_type == "json":
                text = extract_text_from_json(source)

            else:
                raise ValueError("Unsupported type")

            # If no text is extracted, fail the upload
            if not text:
                class NoTextExtractedError(ValueError):
                    pass
                raise NoTextExtractedError("No text extracted")

            # Process the extracted text (split into chunks, generate embeddings)
            chunks = split_into_chunks(text)
            embeddings = get_embeddings_for_chunks(chunks)

            # Upload embeddings to Pinecone
            upload_to_pinecone(chunks, embeddings, org_ns)

            # Update the status of the upload to 'completed'
            await client.patch(f"/uploads?id=eq.{upload_id}", json={
                "status": "completed"
            })

        except (ValueError, RuntimeError, httpx.HTTPError, requests.RequestException) as e:
            print(f"[!] Ingestion failed: {e}")
            await client.patch(f"/uploads?id=eq.{upload['id']}", json={
                "status": "failed",
                "error_message": str(e)
            })
