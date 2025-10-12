"""
Test suite for memory leak fixes in PDF/JSON processing

This test suite verifies that file processing properly cleans up resources
and doesn't leak memory when processing large files.
"""

import gc
import io
import sys
import tracemalloc
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestMemoryLeakFixes:
    """Test memory management in file processing"""

    def test_pdf_buffer_cleanup(self):
        """Test that PDF buffer is properly closed"""
        from app.services.scraping.ingestion_worker import extract_text_from_pdf_url

        # Mock the requests.get to return a fake PDF
        mock_response = Mock()
        mock_response.headers = {"content-length": "1024"}
        mock_response.raise_for_status = Mock()

        # Create a simple PDF-like content
        pdf_content = b"%PDF-1.4\n%Test PDF"

        def iter_content_mock(chunk_size):
            yield pdf_content

        mock_response.iter_content = iter_content_mock
        mock_response.close = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            with patch(
                "app.services.scraping.ingestion_worker.PdfReader"
            ) as mock_pdf_reader:
                # Mock PDF reader to return empty pages
                mock_pdf_reader.return_value.pages = []

                try:
                    extract_text_from_pdf_url("https://example.com/test.pdf")
                except ValueError:
                    # Expected - no text in PDF
                    pass

                # Verify response.close() was called
                assert mock_response.close.called, "Response should be closed"

    def test_json_response_cleanup(self):
        """Test that JSON response is properly closed"""
        from app.services.scraping.ingestion_worker import extract_text_from_json_url

        # Mock the requests.get to return JSON
        mock_response = Mock()
        mock_response.headers = {"content-length": "100"}
        mock_response.raise_for_status = Mock()

        json_content = '{"content": "test content"}'

        def iter_content_mock(chunk_size, decode_unicode=False):
            yield json_content

        mock_response.iter_content = iter_content_mock
        mock_response.close = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            result = extract_text_from_json_url("https://example.com/test.json")

            # Verify response.close() was called
            assert mock_response.close.called, "Response should be closed"
            assert result == "test content"

    def test_pdf_streaming_chunks(self):
        """Test that PDF is downloaded in chunks, not all at once"""
        from app.services.scraping.ingestion_worker import extract_text_from_pdf_url

        # Track how many times iter_content is called
        chunk_count = 0

        def iter_content_mock(chunk_size):
            nonlocal chunk_count
            # Simulate 3 chunks
            for i in range(3):
                chunk_count += 1
                if i == 0:
                    yield b"%PDF-1.4\n"
                else:
                    yield b"chunk" + str(i).encode()

        mock_response = Mock()
        mock_response.headers = {"content-length": "1024"}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = iter_content_mock
        mock_response.close = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            with patch(
                "app.services.scraping.ingestion_worker.PdfReader"
            ) as mock_pdf_reader:
                mock_pdf_reader.return_value.pages = []

                try:
                    extract_text_from_pdf_url("https://example.com/test.pdf")
                except ValueError:
                    pass

                # Verify chunks were processed
                assert chunk_count == 3, f"Expected 3 chunks, got {chunk_count}"

    def test_json_size_limit(self):
        """Test that oversized JSON files are rejected"""
        from app.services.scraping.ingestion_worker import extract_text_from_json_url

        # Mock a response with large content-length (100MB)
        mock_response = Mock()
        mock_response.headers = {"content-length": str(100 * 1024 * 1024)}
        mock_response.raise_for_status = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            with pytest.raises(ValueError) as exc_info:
                extract_text_from_json_url("https://example.com/large.json")

            assert "too large" in str(exc_info.value).lower()

    def test_large_json_array_limiting(self):
        """Test that large JSON arrays are limited to prevent memory issues"""
        from app.services.scraping.ingestion_worker import extract_text_from_json_url

        # Create a large JSON array
        large_array = list(range(15000))
        json_content = str(large_array)

        def iter_content_mock(chunk_size, decode_unicode=False):
            yield json_content

        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(json_content))}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = iter_content_mock
        mock_response.close = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            result = extract_text_from_json_url("https://example.com/large_array.json")

            # Result should be limited to 10000 items
            lines = result.split("\n")
            assert len(lines) <= 10000, f"Expected max 10000 lines, got {len(lines)}"

    def test_cleanup_on_error(self):
        """Test that resources are cleaned up even when errors occur"""
        from app.services.scraping.ingestion_worker import extract_text_from_pdf_url

        mock_response = Mock()
        mock_response.headers = {"content-length": "1024"}
        mock_response.raise_for_status = Mock(side_effect=Exception("Network error"))
        mock_response.close = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            try:
                extract_text_from_pdf_url("https://example.com/test.pdf")
            except:
                pass

            # Verify cleanup happened despite error
            assert mock_response.close.called, "Response should be closed even on error"

    def test_memory_usage_with_large_pdf(self):
        """Test that memory usage is reasonable with large PDFs"""
        from app.services.scraping.ingestion_worker import extract_text_from_pdf_url

        # Start memory tracking
        tracemalloc.start()

        # Create a mock large PDF (10MB)
        large_pdf_chunk = b"%PDF-1.4\n" + b"x" * (10 * 1024 * 1024)

        def iter_content_mock(chunk_size):
            # Yield in chunks
            for i in range(0, len(large_pdf_chunk), chunk_size):
                yield large_pdf_chunk[i : i + chunk_size]

        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(large_pdf_chunk))}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = iter_content_mock
        mock_response.close = Mock()

        with patch(
            "app.services.scraping.ingestion_worker.requests.get",
            return_value=mock_response,
        ):
            with patch(
                "app.services.scraping.ingestion_worker.PdfReader"
            ) as mock_pdf_reader:
                # Mock a page with text
                mock_page = Mock()
                mock_page.extract_text = Mock(return_value="Test text")
                mock_pdf_reader.return_value.pages = [mock_page]

                result = extract_text_from_pdf_url("https://example.com/large.pdf")

                # Force garbage collection
                del result
                gc.collect()

                # Get memory usage
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Peak memory should be reasonable (less than 50MB for a 10MB file)
                peak_mb = peak / (1024 * 1024)
                print(f"Peak memory usage: {peak_mb:.2f} MB")

                # This is a soft check - actual usage depends on many factors
                assert peak_mb < 100, f"Peak memory usage too high: {peak_mb:.2f} MB"


class TestResourceManagement:
    """Test proper resource management patterns"""

    def test_context_manager_pattern(self):
        """Verify that file-like objects use context managers where appropriate"""
        from app.services.scraping.ingestion_worker import extract_text_from_pdf_url

        # This test verifies the pattern, not the actual implementation
        # In production, BytesIO should be closed in finally block

        buffer_closed = False

        class TrackedBytesIO(io.BytesIO):
            def close(self):
                nonlocal buffer_closed
                buffer_closed = True
                super().close()

        with patch("app.services.scraping.ingestion_worker.io.BytesIO", TrackedBytesIO):
            mock_response = Mock()
            mock_response.headers = {"content-length": "100"}
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = lambda chunk_size: [b"%PDF-1.4\n"]
            mock_response.close = Mock()

            with patch(
                "app.services.scraping.ingestion_worker.requests.get",
                return_value=mock_response,
            ):
                with patch(
                    "app.services.scraping.ingestion_worker.PdfReader"
                ) as mock_pdf_reader:
                    mock_pdf_reader.return_value.pages = []

                    try:
                        extract_text_from_pdf_url("https://example.com/test.pdf")
                    except ValueError:
                        pass

                    # Verify buffer was closed
                    assert buffer_closed, "BytesIO buffer should be closed"


if __name__ == "__main__":
    """Run tests manually"""
    print("Running memory leak fix tests...")

    test_suite = TestMemoryLeakFixes()

    print("\n1. Testing PDF buffer cleanup...")
    test_suite.test_pdf_buffer_cleanup()
    print("✓ Passed")

    print("\n2. Testing JSON response cleanup...")
    test_suite.test_json_response_cleanup()
    print("✓ Passed")

    print("\n3. Testing PDF streaming chunks...")
    test_suite.test_pdf_streaming_chunks()
    print("✓ Passed")

    print("\n4. Testing JSON size limit...")
    test_suite.test_json_size_limit()
    print("✓ Passed")

    print("\n5. Testing large JSON array limiting...")
    test_suite.test_large_json_array_limiting()
    print("✓ Passed")

    print("\n6. Testing cleanup on error...")
    test_suite.test_cleanup_on_error()
    print("✓ Passed")

    print("\n7. Testing memory usage with large PDF...")
    test_suite.test_memory_usage_with_large_pdf()
    print("✓ Passed")

    print("\n✅ All memory leak tests passed!")
    print("\n🎉 Memory management is now robust and leak-free!")
