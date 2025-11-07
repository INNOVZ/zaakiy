import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

with patch("pinecone.Pinecone") as MockPinecone:
    MockPinecone.return_value = MagicMock()
    from app.services.storage import pinecone_client

pinecone_client._pc = MagicMock()
pinecone_client._index = MagicMock()


async def app_services_worker_call():
    # Import inside function so each test can patch module-level globals safely
    from app.services.scraping.ingestion_worker import process_pending_uploads

    await process_pending_uploads()


def _build_fake_supabase(upload):
    fake_table = MagicMock()
    fake_table.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[upload]
    )
    fake_table.update.return_value.eq.return_value.execute.return_value = MagicMock()

    fake_supabase = MagicMock()
    fake_supabase.table.return_value = fake_table
    return fake_supabase, fake_table


def test_ingestion_worker_best_case():
    """URL ingestion succeeds, embeddings generated, upload stored"""

    fake_upload = {
        "id": "u1",
        "org_id": "org1",
        "type": "url",
        "source": "https://example.com/article",
        "pinecone_namespace": "ns1",
        "status": "pending",
    }

    fake_supabase, _ = _build_fake_supabase(fake_upload)

    async def fake_scrape(_url: str):
        return "This is some example text content for testing." * 5

    with patch("app.services.scraping.ingestion_worker.supabase", fake_supabase), patch(
        "app.services.scraping.ingestion_worker.scrape_url_text", new=fake_scrape
    ), patch(
        "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
    ) as MockEmbeddings, patch(
        "app.services.scraping.ingestion_worker.upload_to_pinecone"
    ) as mock_upload:
        mock_embedder = MockEmbeddings.return_value
        mock_embedder.embed_documents.side_effect = lambda chunks: [
            [0.01] * 3 for _ in chunks
        ]

        asyncio.run(app_services_worker_call())

        assert mock_embedder.embed_documents.called
        mock_upload.assert_called_once()


def test_ingestion_worker_worst_case_large_json():
    """JSON ingestion fails when file is too large and upload marked failed"""

    fake_upload = {
        "id": "u2",
        "org_id": "org2",
        "type": "json",
        "source": "private/path/large.json",
        "pinecone_namespace": "ns2",
        "status": "pending",
    }

    fake_supabase, _ = _build_fake_supabase(fake_upload)

    class FakeResponse:
        def __init__(self):
            self.headers = {"content-length": str(60 * 1024 * 1024)}  # 60 MB

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192, decode_unicode=False):
            yield b"x" * chunk_size

        def close(self):
            return None

    with patch("app.services.scraping.ingestion_worker.supabase", fake_supabase), patch(
        "app.services.scraping.ingestion_worker.requests.get",
        return_value=FakeResponse(),
    ), patch(
        "app.services.scraping.ingestion_worker.upload_to_pinecone"
    ) as mock_upload:
        asyncio.run(app_services_worker_call())

        assert fake_supabase.table.return_value.update.called
        mock_upload.assert_not_called()


def test_recursive_url_ingestion():
    """Recursive URL configuration is honored and uploads complete"""

    recursive_config = {
        "url": "https://example.com/start",
        "max_pages": 3,
        "max_depth": 2,
    }

    fake_upload = {
        "id": "recursive1",
        "org_id": "org_recursive",
        "type": "url",
        "source": json.dumps(recursive_config),
        "pinecone_namespace": "ns_recursive",
        "status": "pending",
    }

    fake_supabase, fake_table = _build_fake_supabase(fake_upload)

    fake_recursive = AsyncMock(
        return_value="Recursive scrape content from multiple pages. " * 5
    )

    with patch("app.services.scraping.ingestion_worker.supabase", fake_supabase), patch(
        "app.services.scraping.ingestion_worker.recursive_scrape_website",
        new=fake_recursive,
    ), patch(
        "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
    ) as MockEmbeddings, patch(
        "app.services.scraping.ingestion_worker.upload_to_pinecone"
    ) as mock_upload:
        mock_embedder = MockEmbeddings.return_value
        mock_embedder.embed_documents.side_effect = lambda chunks: [
            [0.02] * 3 for _ in chunks
        ]

        asyncio.run(app_services_worker_call())

        fake_recursive.assert_awaited_once_with(
            start_url=recursive_config["url"],
            max_pages=recursive_config["max_pages"],
            max_depth=recursive_config["max_depth"],
        )
        assert mock_embedder.embed_documents.called
        mock_upload.assert_called_once()

    # Verify Supabase marked upload as completed
    update_calls = fake_table.update.call_args_list
    assert update_calls
    assert update_calls[0].args[0]["status"] == "completed"
