"""
Tests for pagination utilities

Ensures pagination works correctly and prevents memory issues
"""
import pytest
from fastapi import HTTPException

from app.utils.pagination import (CursorPaginationParams, PaginationMeta,
                                  PaginationParams, create_pagination_meta,
                                  decode_cursor, encode_cursor,
                                  validate_pagination_params)


class TestPaginationParams:
    """Test pagination parameter validation"""

    def test_valid_params(self):
        """Valid parameters should work"""
        params = PaginationParams(page=1, page_size=20)
        assert params.page == 1
        assert params.page_size == 20
        assert params.offset == 0
        assert params.limit == 20

    def test_page_2_offset(self):
        """Page 2 should have correct offset"""
        params = PaginationParams(page=2, page_size=20)
        assert params.offset == 20

    def test_page_3_offset(self):
        """Page 3 should have correct offset"""
        params = PaginationParams(page=3, page_size=50)
        assert params.offset == 100

    def test_invalid_page_zero(self):
        """Page 0 should be rejected"""
        with pytest.raises(ValueError):
            PaginationParams(page=0, page_size=20)

    def test_invalid_page_negative(self):
        """Negative page should be rejected"""
        with pytest.raises(ValueError):
            PaginationParams(page=-1, page_size=20)

    def test_invalid_page_size_zero(self):
        """Page size 0 should be rejected"""
        with pytest.raises(ValueError):
            PaginationParams(page=1, page_size=0)

    def test_invalid_page_size_too_large(self):
        """Page size > 100 should be rejected"""
        with pytest.raises(ValueError):
            PaginationParams(page=1, page_size=101)

    def test_max_page_size_allowed(self):
        """Page size of 100 should be allowed"""
        params = PaginationParams(page=1, page_size=100)
        assert params.page_size == 100


class TestPaginationMeta:
    """Test pagination metadata generation"""

    def test_first_page_with_more(self):
        """First page with more items"""
        params = PaginationParams(page=1, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=100)

        assert meta.page == 1
        assert meta.page_size == 20
        assert meta.total_items == 100
        assert meta.total_pages == 5
        assert meta.has_next is True
        assert meta.has_prev is False

    def test_middle_page(self):
        """Middle page should have both next and prev"""
        params = PaginationParams(page=3, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=100)

        assert meta.page == 3
        assert meta.total_pages == 5
        assert meta.has_next is True
        assert meta.has_prev is True

    def test_last_page(self):
        """Last page should not have next"""
        params = PaginationParams(page=5, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=100)

        assert meta.page == 5
        assert meta.total_pages == 5
        assert meta.has_next is False
        assert meta.has_prev is True

    def test_single_page(self):
        """Single page should have no next or prev"""
        params = PaginationParams(page=1, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=10)

        assert meta.page == 1
        assert meta.total_pages == 1
        assert meta.has_next is False
        assert meta.has_prev is False

    def test_empty_results(self):
        """Empty results should still have 1 page"""
        params = PaginationParams(page=1, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=0)

        assert meta.total_pages == 1
        assert meta.has_next is False
        assert meta.has_prev is False

    def test_partial_last_page(self):
        """Partial last page should calculate correctly"""
        params = PaginationParams(page=1, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=45)

        assert meta.total_pages == 3  # 20 + 20 + 5
        assert meta.has_next is True


class TestValidatePaginationParams:
    """Test pagination parameter validation function"""

    def test_valid_params(self):
        """Valid parameters should return PaginationParams"""
        params = validate_pagination_params(page=1, page_size=20)
        assert isinstance(params, PaginationParams)
        assert params.page == 1
        assert params.page_size == 20

    def test_default_params(self):
        """Default parameters should work"""
        params = validate_pagination_params()
        assert params.page == 1
        assert params.page_size == 20

    def test_invalid_page_raises_http_exception(self):
        """Invalid page should raise HTTPException"""
        with pytest.raises(HTTPException) as exc_info:
            validate_pagination_params(page=0, page_size=20)
        assert exc_info.value.status_code == 400

    def test_exceeds_max_page_size(self):
        """Exceeding max page size should raise HTTPException"""
        with pytest.raises(HTTPException) as exc_info:
            validate_pagination_params(page=1, page_size=150, max_page_size=100)
        assert exc_info.value.status_code == 400


class TestCreatePaginationMeta:
    """Test pagination metadata creation function"""

    def test_creates_correct_meta(self):
        """Should create correct metadata dictionary"""
        meta = create_pagination_meta(page=2, page_size=20, total_items=100)

        assert meta["page"] == 2
        assert meta["page_size"] == 20
        assert meta["total_items"] == 100
        assert meta["total_pages"] == 5
        assert meta["has_next"] is True
        assert meta["has_prev"] is True

    def test_zero_items(self):
        """Zero items should still have 1 page"""
        meta = create_pagination_meta(page=1, page_size=20, total_items=0)
        assert meta["total_pages"] == 1


class TestCursorPagination:
    """Test cursor-based pagination"""

    def test_encode_decode_cursor(self):
        """Cursor should encode and decode correctly"""
        original = "2024-01-01T00:00:00Z"
        encoded = encode_cursor(original)
        decoded = decode_cursor(encoded)

        assert decoded == original

    def test_encode_decode_integer(self):
        """Integer cursor should work"""
        original = 12345
        encoded = encode_cursor(original)
        decoded = decode_cursor(encoded)

        assert decoded == str(original)  # Decoded as string

    def test_invalid_cursor(self):
        """Invalid cursor should raise ValueError"""
        with pytest.raises(ValueError):
            decode_cursor("invalid_base64!")

    def test_cursor_params_valid(self):
        """Valid cursor params should work"""
        params = CursorPaginationParams(cursor="abc123", limit=20)
        assert params.cursor == "abc123"
        assert params.limit == 20

    def test_cursor_params_no_cursor(self):
        """Cursor params without cursor should work"""
        params = CursorPaginationParams(limit=20)
        assert params.cursor is None
        assert params.limit == 20

    def test_cursor_params_invalid_limit(self):
        """Invalid limit should be rejected"""
        with pytest.raises(ValueError):
            CursorPaginationParams(limit=0)

    def test_cursor_params_limit_too_large(self):
        """Limit > 100 should be rejected"""
        with pytest.raises(ValueError):
            CursorPaginationParams(limit=101)


class TestPaginationEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_large_page_number(self):
        """Large page number should work"""
        params = PaginationParams(page=1000, page_size=20)
        assert params.offset == 19980

    def test_page_beyond_total(self):
        """Page beyond total should show no next"""
        params = PaginationParams(page=10, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=50)

        assert meta.has_next is False

    def test_exact_page_boundary(self):
        """Exact page boundary should calculate correctly"""
        params = PaginationParams(page=1, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=20)

        assert meta.total_pages == 1
        assert meta.has_next is False

    def test_one_over_boundary(self):
        """One item over boundary should create new page"""
        params = PaginationParams(page=1, page_size=20)
        meta = PaginationMeta.from_params(params, total_items=21)

        assert meta.total_pages == 2
        assert meta.has_next is True


class TestPaginationPerformance:
    """Test pagination performance characteristics"""

    def test_offset_calculation_performance(self):
        """Offset calculation should be fast"""
        import time

        start = time.time()
        for i in range(10000):
            params = PaginationParams(page=i + 1, page_size=20)
            _ = params.offset
        duration = time.time() - start

        # Should complete in less than 1 second
        assert duration < 1.0

    def test_meta_creation_performance(self):
        """Meta creation should be fast"""
        import time

        start = time.time()
        for i in range(10000):
            _ = create_pagination_meta(page=1, page_size=20, total_items=1000)
        duration = time.time() - start

        # Should complete in less than 1 second
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
