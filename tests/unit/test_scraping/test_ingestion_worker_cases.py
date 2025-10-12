import asyncio
from unittest.mock import MagicMock, patch


def test_ingestion_worker_best_case(monkeypatch):
    """Best-case: URL upload, scrape returns text, embeddings and pinecone succeed"""

    # Prepare a fake pending upload
    fake_upload = {
        "id": "u1",
        "org_id": "org1",
        "type": "url",
        "source": "https://example.com/article",
        "pinecone_namespace": "ns1",
        "status": "pending",
    }

    # Patch supabase client used in the ingestion worker
    fake_table = MagicMock()
    fake_table.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[fake_upload]
    )

    fake_supabase = MagicMock()
    fake_supabase.table.return_value = fake_table

    async def fake_scrape(url: str):
        return "This is some example text content for testing." * 10

    # Patch OpenAIEmbeddings.embed_documents with a simple sync function wrapper
    with patch("app.services.scraping.ingestion_worker.supabase", fake_supabase), patch(
        "app.services.scraping.ingestion_worker.scrape_url_text", new=fake_scrape
    ), patch(
        "app.services.scraping.ingestion_worker.OpenAIEmbeddings.embed_documents"
    ) as mock_embed, patch(
        "app.services.scraping.ingestion_worker.index"
    ) as mock_index:
        # Make embeddings return a vector per chunk
        mock_embed.return_value = [[0.01] * 1536]

        # Make pinecone upsert return a success dict
        mock_index.upsert.return_value = {"upserted_count": 1}

        # Run the worker synchronously
        asyncio.run(app_services_worker_call())

        # Assertions: embeddings called and pinecone upsert called
        assert mock_embed.called
        assert mock_index.upsert.called


def test_ingestion_worker_worst_case_large_json(monkeypatch):
    """Worst-case: JSON file too large triggers ValueError and upload marked failed"""

    fake_upload = {
        "id": "u2",
        "org_id": "org2",
        "type": "json",
        "source": "private/path/large.json",
        "pinecone_namespace": "ns2",
        "status": "pending",
    }

    # Patch supabase table select to return the fake upload
    fake_table = MagicMock()
    fake_table.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[fake_upload]
    )

    fake_supabase = MagicMock()
    fake_supabase.table.return_value = fake_table

    # Patch requests.get to return a response with huge content-length
    class FakeResponse:
        def __init__(self):
            self.headers = {"content-length": str(60 * 1024 * 1024)}  # 60 MB

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192, decode_unicode=False):
            # yield some chunks but overall > 50MB will be enforced by the ingestion code
            for _ in range(10):
                yield b"x" * 8192

        def close(self):
            return None

    with patch("app.services.scraping.ingestion_worker.supabase", fake_supabase), patch(
        "app.services.scraping.ingestion_worker.requests.get",
        return_value=FakeResponse(),
    ):
        # Run the worker synchronously (worker handles errors internally)
        asyncio.run(app_services_worker_call())

        # Ensure supabase.update was called to mark failed
        assert fake_supabase.table.return_value.update.called


async def app_services_worker_call():
    # Import inside function to ensure patches apply
    from app.services.scraping.ingestion_worker import process_pending_uploads

    await process_pending_uploads()
