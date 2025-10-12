"""
Comprehensive test suite for ingestion worker and scraping system
Tests both best-case and worst-case scenarios with realistic data
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest
import requests


class TestIngestionWorkerBestCase:
    """Test best-case scenarios for the ingestion worker"""

    @pytest.mark.asyncio
    async def test_url_ingestion_best_case(self):
        """Best case: Fast URL with clean HTML content"""

        # Mock upload data
        mock_upload = {
            "id": "upload_001",
            "org_id": "org_123",
            "type": "url",
            "source": "https://example.com/article",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        # Mock clean HTML content
        mock_html_content = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <h1>Important Article</h1>
                <p>This is a well-structured article with meaningful content.</p>
                <p>It contains multiple paragraphs with valuable information.</p>
                <p>The content is clean and easy to process.</p>
            </body>
        </html>
        """

        expected_text = "Important Article This is a well-structured article with meaningful content. It contains multiple paragraphs with valuable information. The content is clean and easy to process."

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.scrape_url_text"
        ) as mock_scrape, patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings, patch(
            "app.services.scraping.ingestion_worker.index"
        ) as mock_index:
            # Setup mocks
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_scrape.return_value = expected_text
            mock_embeddings.return_value.embed_documents.return_value = [
                [0.1] * 1536,
                [0.2] * 1536,
            ]
            mock_index.upsert.return_value = {"upserted_count": 2}

            # Import and run the worker
            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify successful processing
            assert mock_scrape.called
            assert mock_embeddings.return_value.embed_documents.called
            assert mock_index.upsert.called

            # Verify status update to completed
            mock_supabase.table.return_value.update.assert_called_with(
                {"status": "completed", "error_message": None}
            )

    @pytest.mark.asyncio
    async def test_pdf_ingestion_best_case(self):
        """Best case: Small PDF with extractable text"""

        mock_upload = {
            "id": "upload_002",
            "org_id": "org_123",
            "type": "pdf",
            "source": "documents/sample.pdf",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        # Mock PDF response
        mock_pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        class MockResponse:
            def __init__(self):
                self.headers = {
                    "content-length": "1024",
                    "content-type": "application/pdf",
                }
                self.status_code = 200

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192, decode_unicode=False):
                yield mock_pdf_content

            def close(self):
                pass

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.requests.get"
        ) as mock_get, patch(
            "app.services.scraping.ingestion_worker.PdfReader"
        ) as mock_pdf_reader, patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings, patch(
            "app.services.scraping.ingestion_worker.index"
        ) as mock_index:
            # Setup mocks
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_get.return_value = MockResponse()

            # Mock PDF reader
            mock_page = MagicMock()
            mock_page.extract_text.return_value = (
                "This is extracted PDF text content with meaningful information."
            )
            mock_pdf_reader.return_value.pages = [mock_page]

            mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_index.upsert.return_value = {"upserted_count": 1}

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify successful processing
            assert mock_get.called
            assert mock_pdf_reader.called
            assert mock_embeddings.return_value.embed_documents.called
            assert mock_index.upsert.called

    @pytest.mark.asyncio
    async def test_json_ingestion_best_case(self):
        """Best case: Well-structured JSON with content field"""

        mock_upload = {
            "id": "upload_003",
            "org_id": "org_123",
            "type": "json",
            "source": "data/content.json",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        mock_json_data = {
            "content": "This is structured JSON content that should be easily processed and indexed.",
            "metadata": {"author": "Test Author", "date": "2024-01-01"},
        }

        class MockResponse:
            def __init__(self):
                self.headers = {
                    "content-length": "256",
                    "content-type": "application/json",
                }
                self.status_code = 200

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192, decode_unicode=True):
                yield json.dumps(mock_json_data)

            def close(self):
                pass

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.requests.get"
        ) as mock_get, patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings, patch(
            "app.services.scraping.ingestion_worker.index"
        ) as mock_index:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_get.return_value = MockResponse()
            mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_index.upsert.return_value = {"upserted_count": 1}

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify successful processing
            assert mock_embeddings.return_value.embed_documents.called
            assert mock_index.upsert.called


class TestIngestionWorkerWorstCase:
    """Test worst-case scenarios for the ingestion worker"""

    @pytest.mark.asyncio
    async def test_oversized_pdf_rejection(self):
        """Worst case: PDF file exceeds size limit"""

        mock_upload = {
            "id": "upload_004",
            "org_id": "org_123",
            "type": "pdf",
            "source": "documents/huge.pdf",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        class MockOversizedResponse:
            def __init__(self):
                self.headers = {"content-length": str(150 * 1024 * 1024)}  # 150MB
                self.status_code = 200

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192, decode_unicode=False):
                # Simulate large file
                for _ in range(100):
                    yield b"x" * chunk_size

            def close(self):
                pass

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.requests.get"
        ) as mock_get:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_get.return_value = MockOversizedResponse()

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify failure was recorded
            mock_supabase.table.return_value.update.assert_called()
            call_args = mock_supabase.table.return_value.update.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "too large" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_corrupted_pdf_handling(self):
        """Worst case: Corrupted PDF file"""

        mock_upload = {
            "id": "upload_005",
            "org_id": "org_123",
            "type": "pdf",
            "source": "documents/corrupted.pdf",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        class MockCorruptedResponse:
            def __init__(self):
                self.headers = {
                    "content-length": "1024",
                    "content-type": "application/pdf",
                }
                self.status_code = 200

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192, decode_unicode=False):
                yield b"NOT A PDF FILE - CORRUPTED DATA"

            def close(self):
                pass

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.requests.get"
        ) as mock_get:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_get.return_value = MockCorruptedResponse()

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify failure was recorded
            mock_supabase.table.return_value.update.assert_called()
            call_args = mock_supabase.table.return_value.update.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "not a valid PDF" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Worst case: Malformed JSON file"""

        mock_upload = {
            "id": "upload_006",
            "org_id": "org_123",
            "type": "json",
            "source": "data/malformed.json",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        class MockMalformedResponse:
            def __init__(self):
                self.headers = {
                    "content-length": "256",
                    "content-type": "application/json",
                }
                self.status_code = 200

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192, decode_unicode=True):
                yield '{"invalid": json, "missing": quotes}'  # Malformed JSON

            def close(self):
                pass

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.requests.get"
        ) as mock_get:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_get.return_value = MockMalformedResponse()

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify failure was recorded
            mock_supabase.table.return_value.update.assert_called()
            call_args = mock_supabase.table.return_value.update.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "Invalid JSON format" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Worst case: Network timeout during URL scraping"""

        mock_upload = {
            "id": "upload_007",
            "org_id": "org_123",
            "type": "url",
            "source": "https://slow-server.com/timeout",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.scrape_url_text"
        ) as mock_scrape:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_scrape.side_effect = ValueError("Request timeout after 30s")

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify failure was recorded
            mock_supabase.table.return_value.update.assert_called()
            call_args = mock_supabase.table.return_value.update.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "timeout" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_embedding_service_failure(self):
        """Worst case: OpenAI embeddings service failure"""

        mock_upload = {
            "id": "upload_008",
            "org_id": "org_123",
            "type": "url",
            "source": "https://example.com/article",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.scrape_url_text"
        ) as mock_scrape, patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_scrape.return_value = "Valid text content for processing"
            mock_embeddings.return_value.embed_documents.side_effect = Exception(
                "OpenAI API rate limit exceeded"
            )

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify failure was recorded
            mock_supabase.table.return_value.update.assert_called()
            call_args = mock_supabase.table.return_value.update.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "rate limit" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_pinecone_upsert_failure(self):
        """Worst case: Pinecone vector database failure"""

        mock_upload = {
            "id": "upload_009",
            "org_id": "org_123",
            "type": "url",
            "source": "https://example.com/article",
            "pinecone_namespace": "test_namespace",
            "status": "pending",
        }

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.scrape_url_text"
        ) as mock_scrape, patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings, patch(
            "app.services.scraping.ingestion_worker.index"
        ) as mock_index:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                mock_upload
            ]
            mock_scrape.return_value = "Valid text content for processing"
            mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_index.upsert.side_effect = Exception("Pinecone service unavailable")

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Verify failure was recorded
            mock_supabase.table.return_value.update.assert_called()
            call_args = mock_supabase.table.return_value.update.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "Pinecone" in call_args["error_message"]


class TestScrapingSystemBestCase:
    """Test best-case scenarios for the scraping system"""

    @pytest.mark.asyncio
    async def test_secure_scraper_best_case(self):
        """Best case: Fast, secure URL with clean content"""

        from app.services.scraping.web_scraper import ScrapingConfig, SecureWebScraper

        config = ScrapingConfig(timeout=10, max_retries=1)
        scraper = SecureWebScraper(config)

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html", "content-length": "1024"}
        mock_response.text = """
        <html>
            <body>
                <h1>Clean Article</h1>
                <p>This is well-formatted content.</p>
                <p>Easy to extract and process.</p>
            </body>
        </html>
        """
        mock_response.content = mock_response.text.encode()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await scraper.scrape_url_text("https://example.com/article")

            assert "Clean Article" in result
            assert "well-formatted content" in result
            assert len(result) > 10

    @pytest.mark.asyncio
    async def test_adaptive_scraper_best_case(self):
        """Best case: Multiple fast URLs with adaptive scaling"""

        from app.services.scraping.adaptive_scraper import create_adaptive_scraper

        scraper = create_adaptive_scraper(min_workers=2, max_workers=5)

        test_urls = [
            "https://fast-site.com/page1",
            "https://fast-site.com/page2",
            "https://fast-site.com/page3",
        ]

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html", "content-length": "512"}
        mock_response.text = "<html><body><p>Fast loading content</p></body></html>"
        mock_response.content = mock_response.text.encode()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            results = await scraper.scrape_urls_adaptive(test_urls)

            assert len(results) == 3
            for url in test_urls:
                assert url in results
                assert "Fast loading content" in results[url]

    @pytest.mark.asyncio
    async def test_concurrent_scraping_performance(self):
        """Best case: High-performance concurrent scraping"""

        from app.services.scraping.web_scraper import SecureWebScraper

        scraper = SecureWebScraper()

        # Simulate 10 fast URLs
        test_urls = [f"https://example.com/page{i}" for i in range(10)]

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html", "content-length": "256"}
        mock_response.text = "<html><body><p>Content {}</p></body></html>"
        mock_response.content = mock_response.text.encode()

        async def mock_scrape(url):
            # Simulate fast response
            await asyncio.sleep(0.1)
            return f"Scraped content from {url}"

        with patch.object(scraper, "scrape_url_text", side_effect=mock_scrape):
            start_time = time.time()

            # Run concurrent scraping
            tasks = [scraper.scrape_url_text(url) for url in test_urls]
            results = await asyncio.gather(*tasks)

            end_time = time.time()

            # Should complete much faster than sequential (< 2 seconds vs 10+ seconds)
            assert end_time - start_time < 2.0
            assert len(results) == 10
            for i, result in enumerate(results):
                assert f"page{i}" in result


class TestScrapingSystemWorstCase:
    """Test worst-case scenarios for the scraping system"""

    @pytest.mark.asyncio
    async def test_ssrf_protection_worst_case(self):
        """Worst case: SSRF attack attempts"""

        from app.services.scraping.web_scraper import SecureWebScraper

        scraper = SecureWebScraper()

        malicious_urls = [
            "http://localhost:8080/admin",
            "http://127.0.0.1/secrets",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "ftp://internal.server/data",
        ]

        for url in malicious_urls:
            with pytest.raises(ValueError) as exc_info:
                await scraper.scrape_url_text(url)
            assert "security validation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_oversized_content_rejection(self):
        """Worst case: Extremely large web page"""

        from app.services.scraping.web_scraper import ScrapingConfig, SecureWebScraper

        config = ScrapingConfig(max_content_size=1024)  # 1KB limit
        scraper = SecureWebScraper(config)

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/html",
            "content-length": str(10 * 1024 * 1024),  # 10MB
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            with pytest.raises(ValueError) as exc_info:
                await scraper.scrape_url_text("https://example.com/huge-page")
            assert "too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_malicious_content_type(self):
        """Worst case: Malicious content type"""

        from app.services.scraping.web_scraper import SecureWebScraper

        scraper = SecureWebScraper()

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/x-executable"}
        mock_response.content = b"MALICIOUS BINARY DATA"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            with pytest.raises(ValueError) as exc_info:
                await scraper.scrape_url_text("https://malicious.com/virus.exe")
            assert "Unsupported content type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_network_failure_cascade(self):
        """Worst case: Network failures affecting multiple URLs"""

        from app.services.scraping.adaptive_scraper import create_adaptive_scraper

        scraper = create_adaptive_scraper(min_workers=1, max_workers=3)

        failing_urls = [
            "https://timeout.com/page1",
            "https://error.com/page2",
            "https://down.com/page3",
        ]

        async def mock_failing_scrape(url):
            if "timeout" in url:
                raise ValueError("Request timeout after 30s")
            elif "error" in url:
                raise ValueError("HTTP 500: Internal Server Error")
            else:
                raise ValueError("Connection refused")

        with patch.object(scraper, "scrape_url_text", side_effect=mock_failing_scrape):
            results = await scraper.scrape_urls_adaptive(failing_urls)

            # Should handle failures gracefully
            assert len(results) == 0  # No successful results

            # Check performance stats show the failures
            stats = scraper.get_performance_stats()
            assert stats["total_scraped"] == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self):
        """Worst case: Rate limiting with aggressive scraping"""

        from app.services.scraping.web_scraper import ScrapingConfig, SecureWebScraper

        # Aggressive rate limiting
        config = ScrapingConfig(min_delay=2.0, max_delay=3.0)
        scraper = SecureWebScraper(config)

        # Same domain URLs (should trigger rate limiting)
        same_domain_urls = [f"https://example.com/page{i}" for i in range(5)]

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html", "content-length": "256"}
        mock_response.text = "<html><body><p>Rate limited content</p></body></html>"
        mock_response.content = mock_response.text.encode()

        with patch("httpx.AsyncClient") as mock_client, patch(
            "asyncio.sleep"
        ) as mock_sleep:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            start_time = time.time()

            # Sequential scraping of same domain should be rate limited
            for url in same_domain_urls:
                await scraper.scrape_url_text(url)

            # Verify rate limiting was applied
            assert mock_sleep.call_count >= 4  # Should sleep between requests


class TestStressScenarios:
    """Test system behavior under stress conditions"""

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""

        from app.services.scraping.ingestion_worker import clean_text, split_into_chunks

        # Simulate very large text content
        large_text = "This is a test sentence. " * 10000  # ~250KB of text

        # Should handle large text without memory issues
        cleaned = clean_text(large_text)
        chunks = split_into_chunks(cleaned)

        assert len(chunks) > 0
        # Reasonable chunk sizes
        assert all(len(chunk) <= 1000 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_concurrent_upload_processing(self):
        """Test processing multiple uploads concurrently"""

        # Simulate multiple pending uploads
        mock_uploads = [
            {
                "id": f"upload_{i:03d}",
                "org_id": "org_123",
                "type": "url",
                "source": f"https://example.com/page{i}",
                "pinecone_namespace": "test_namespace",
                "status": "pending",
            }
            for i in range(10)
        ]

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.scrape_url_text"
        ) as mock_scrape, patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings, patch(
            "app.services.scraping.ingestion_worker.index"
        ) as mock_index:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = (
                mock_uploads
            )
            mock_scrape.return_value = "Test content for processing"
            mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_index.upsert.return_value = {"upserted_count": 1}

            from app.services.scraping.ingestion_worker import process_pending_uploads

            start_time = time.time()
            await process_pending_uploads()
            end_time = time.time()

            # Should process all uploads
            assert mock_scrape.call_count == 10
            assert mock_index.upsert.call_count == 10

            # Should complete in reasonable time (not sequential)
            assert end_time - start_time < 30  # Should be much faster than 10 * timeout

    @pytest.mark.asyncio
    async def test_error_recovery_resilience(self):
        """Test system resilience with mixed success/failure scenarios"""

        mock_uploads = [
            {
                "id": "good_001",
                "type": "url",
                "source": "https://good.com/page1",
                "status": "pending",
            },
            {
                "id": "bad_002",
                "type": "url",
                "source": "https://bad.com/page2",
                "status": "pending",
            },
            {
                "id": "good_003",
                "type": "url",
                "source": "https://good.com/page3",
                "status": "pending",
            },
        ]

        def mock_scrape_mixed(url):
            if "bad.com" in url:
                raise ValueError("Simulated network error")
            return "Good content from reliable source"

        with patch(
            "app.services.scraping.ingestion_worker.supabase"
        ) as mock_supabase, patch(
            "app.services.scraping.ingestion_worker.scrape_url_text",
            side_effect=mock_scrape_mixed,
        ), patch(
            "app.services.scraping.ingestion_worker.OpenAIEmbeddings"
        ) as mock_embeddings, patch(
            "app.services.scraping.ingestion_worker.index"
        ) as mock_index:
            mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = (
                mock_uploads
            )
            mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_index.upsert.return_value = {"upserted_count": 1}

            from app.services.scraping.ingestion_worker import process_pending_uploads

            await process_pending_uploads()

            # Should have processed successful uploads despite failures
            update_calls = mock_supabase.table.return_value.update.call_args_list

            # Should have 3 update calls (2 success, 1 failure)
            assert len(update_calls) == 3

            # Check that good uploads succeeded and bad one failed
            statuses = [call[0][0]["status"] for call in update_calls]
            assert statuses.count("completed") == 2
            assert statuses.count("failed") == 1


if __name__ == "__main__":
    """Run comprehensive tests manually"""

    async def run_comprehensive_tests():
        print("🧪 Running Comprehensive Ingestion & Scraping Tests")
        print("=" * 60)

        # Test categories
        test_classes = [
            ("Best Case - Ingestion Worker", TestIngestionWorkerBestCase()),
            ("Worst Case - Ingestion Worker", TestIngestionWorkerWorstCase()),
            ("Best Case - Scraping System", TestScrapingSystemBestCase()),
            ("Worst Case - Scraping System", TestScrapingSystemWorstCase()),
            ("Stress Scenarios", TestStressScenarios()),
        ]

        total_tests = 0
        passed_tests = 0

        for category_name, test_instance in test_classes:
            print(f"\n📋 {category_name}")
            print("-" * 40)

            # Get all test methods
            test_methods = [
                method for method in dir(test_instance) if method.startswith("test_")
            ]

            for test_method_name in test_methods:
                test_method = getattr(test_instance, test_method_name)
                total_tests += 1

                try:
                    if asyncio.iscoroutinefunction(test_method):
                        await test_method()
                    else:
                        test_method()

                    print(f"  ✅ {test_method_name}")
                    passed_tests += 1

                except Exception as e:
                    print(f"  ❌ {test_method_name}: {type(e).__name__}: {e}")

        print(f"\n📊 Test Results Summary")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        if passed_tests == total_tests:
            print("\n🎉 All tests passed! System is robust and handles edge cases well.")
        else:
            print(
                f"\n⚠️  {total_tests - passed_tests} tests failed. Review implementation for edge cases."
            )

    # Run the comprehensive test suite
    asyncio.run(run_comprehensive_tests())
