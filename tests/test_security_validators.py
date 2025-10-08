"""
Security validation tests for input validators

Tests SSRF protection, injection prevention, and input sanitization
"""
import pytest
from app.utils.validators import (
    validate_url,
    validate_upload_id,
    validate_namespace,
    validate_org_id,
    validate_metadata_filter,
    validate_json_safe,
    ValidationError
)


class TestURLValidation:
    """Test URL validation and SSRF protection"""

    def test_valid_https_url(self):
        """Valid HTTPS URLs should pass"""
        url = validate_url("https://example.com/path")
        assert url == "https://example.com/path"

    def test_valid_http_url(self):
        """Valid HTTP URLs should pass"""
        url = validate_url("http://example.com")
        assert url == "http://example.com"

    def test_localhost_blocked(self):
        """Localhost URLs should be blocked by default"""
        with pytest.raises(ValidationError, match="Localhost URLs are not allowed"):
            validate_url("http://localhost:8000")

    def test_localhost_allowed_when_enabled(self):
        """Localhost should be allowed when explicitly enabled"""
        url = validate_url("http://localhost:8000", allow_localhost=True)
        assert url == "http://localhost:8000"

    def test_private_ip_blocked(self):
        """Private IP addresses should be blocked"""
        private_ips = [
            "http://192.168.1.1",
            "http://10.0.0.1",
            "http://172.16.0.1",
        ]
        for ip in private_ips:
            with pytest.raises(ValidationError, match="private"):
                validate_url(ip)

    def test_loopback_blocked(self):
        """Loopback addresses should be blocked"""
        with pytest.raises(ValidationError, match="Localhost"):
            validate_url("http://127.0.0.1")

    def test_ipv6_loopback_blocked(self):
        """IPv6 loopback should be blocked"""
        with pytest.raises(ValidationError, match="Localhost"):
            validate_url("http://[::1]")

    def test_metadata_endpoint_blocked(self):
        """Cloud metadata endpoints should be blocked"""
        with pytest.raises(ValidationError, match="metadata"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_auth_in_url_blocked(self):
        """URLs with authentication should be blocked"""
        with pytest.raises(ValidationError, match="authentication"):
            validate_url("http://user:pass@example.com")

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked"""
        with pytest.raises(ValidationError, match="path traversal"):
            validate_url("http://example.com/../../../etc/passwd")

    def test_encoded_attacks_blocked(self):
        """Encoded attack patterns should be blocked"""
        attacks = [
            "http://example.com/%2e%2e/",
            "http://example.com/%00",
            "http://example.com/file://",
        ]
        for attack in attacks:
            with pytest.raises(ValidationError):
                validate_url(attack)

    def test_dangerous_schemes_blocked(self):
        """Non-HTTP schemes should be blocked"""
        schemes = [
            "file:///etc/passwd",
            "ftp://example.com",
            "javascript:alert(1)",
        ]
        for scheme in schemes:
            with pytest.raises(ValidationError):
                validate_url(scheme)

    def test_internal_tld_blocked(self):
        """Internal TLDs should be blocked"""
        with pytest.raises(ValidationError, match="TLD"):
            validate_url("http://server.local")

    def test_dangerous_ports_blocked(self):
        """Dangerous ports should be blocked"""
        with pytest.raises(ValidationError, match="Port"):
            validate_url("http://example.com:22")  # SSH

    def test_url_too_long(self):
        """Very long URLs should be rejected"""
        long_url = "http://example.com/" + "a" * 3000
        with pytest.raises(ValidationError, match="too long"):
            validate_url(long_url)


class TestUploadIDValidation:
    """Test upload ID validation"""

    def test_valid_uuid(self):
        """Valid UUIDs should pass"""
        upload_id = validate_upload_id("550e8400-e29b-41d4-a716-446655440000")
        assert upload_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_alphanumeric_with_hyphens(self):
        """Alphanumeric with hyphens should pass"""
        upload_id = validate_upload_id("upload-123-abc")
        assert upload_id == "upload-123-abc"

    def test_sql_injection_blocked(self):
        """SQL injection attempts should be blocked"""
        with pytest.raises(ValidationError):
            validate_upload_id("123'; DROP TABLE uploads; --")

    def test_special_chars_blocked(self):
        """Special characters should be blocked"""
        with pytest.raises(ValidationError):
            validate_upload_id("upload$123")

    def test_empty_string_blocked(self):
        """Empty strings should be blocked"""
        with pytest.raises(ValidationError):
            validate_upload_id("")

    def test_too_long_blocked(self):
        """Very long IDs should be blocked"""
        with pytest.raises(ValidationError):
            validate_upload_id("a" * 200)


class TestNamespaceValidation:
    """Test namespace validation"""

    def test_valid_namespace(self):
        """Valid namespaces should pass"""
        namespace = validate_namespace("org-123")
        assert namespace == "org-123"

    def test_path_traversal_blocked(self):
        """Path traversal in namespace should be blocked"""
        with pytest.raises(ValidationError):
            validate_namespace("org-123/../admin")

    def test_leading_dot_blocked(self):
        """Leading dots should be blocked"""
        with pytest.raises(ValidationError):
            validate_namespace(".hidden")

    def test_trailing_dot_blocked(self):
        """Trailing dots should be blocked"""
        with pytest.raises(ValidationError):
            validate_namespace("namespace.")


class TestMetadataFilterValidation:
    """Test metadata filter validation"""

    def test_valid_filter(self):
        """Valid filters should pass"""
        filter_dict = {"upload_id": "123", "org_id": "org-456"}
        validated = validate_metadata_filter(filter_dict)
        assert validated == filter_dict

    def test_empty_filter_blocked(self):
        """Empty filters should be blocked"""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_metadata_filter({})

    def test_too_many_keys_blocked(self):
        """Filters with too many keys should be blocked"""
        large_filter = {f"key{i}": f"value{i}" for i in range(20)}
        with pytest.raises(ValidationError, match="too many keys"):
            validate_metadata_filter(large_filter)

    def test_disallowed_key_blocked(self):
        """Disallowed keys should be blocked"""
        with pytest.raises(ValidationError, match="not allowed"):
            validate_metadata_filter({"malicious_key": "value"})

    def test_injection_in_value_blocked(self):
        """Injection patterns in values should be blocked"""
        with pytest.raises(ValidationError, match="dangerous pattern"):
            validate_metadata_filter({"upload_id": "123$ne"})

    def test_path_traversal_in_value_blocked(self):
        """Path traversal in values should be blocked"""
        with pytest.raises(ValidationError, match="dangerous pattern"):
            validate_metadata_filter({"upload_id": "../admin"})

    def test_null_byte_blocked(self):
        """Null bytes should be blocked"""
        with pytest.raises(ValidationError, match="dangerous pattern"):
            validate_metadata_filter({"upload_id": "123\x00"})

    def test_integer_validation(self):
        """Integer values should be validated"""
        validated = validate_metadata_filter({"chunk_index": 5})
        assert validated["chunk_index"] == 5

    def test_integer_out_of_range_blocked(self):
        """Out of range integers should be blocked"""
        with pytest.raises(ValidationError, match="out of range"):
            validate_metadata_filter({"chunk_index": 9999999})

    def test_boolean_validation(self):
        """Boolean values should be validated"""
        validated = validate_metadata_filter({"has_products": True})
        assert validated["has_products"] is True


class TestJSONSafeValidation:
    """Test JSON safety validation"""

    def test_simple_types_allowed(self):
        """Simple types should be allowed"""
        assert validate_json_safe("string")
        assert validate_json_safe(123)
        assert validate_json_safe(45.67)
        assert validate_json_safe(True)
        assert validate_json_safe(None)

    def test_simple_dict_allowed(self):
        """Simple dictionaries should be allowed"""
        data = {"key": "value", "number": 123}
        assert validate_json_safe(data)

    def test_simple_list_allowed(self):
        """Simple lists should be allowed"""
        data = [1, 2, 3, "four", True]
        assert validate_json_safe(data)

    def test_nested_structure_allowed(self):
        """Nested structures within depth limit should be allowed"""
        data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                }
            }
        }
        assert validate_json_safe(data)

    def test_too_deep_blocked(self):
        """Structures exceeding depth limit should be blocked"""
        # Create deeply nested structure
        data = {"level": 1}
        current = data
        for i in range(15):
            current["nested"] = {"level": i + 2}
            current = current["nested"]

        with pytest.raises(ValidationError, match="exceeds maximum depth"):
            validate_json_safe(data, max_depth=10)

    def test_too_many_dict_keys_blocked(self):
        """Dictionaries with too many keys should be blocked"""
        large_dict = {f"key{i}": i for i in range(2000)}
        with pytest.raises(ValidationError, match="too many keys"):
            validate_json_safe(large_dict)

    def test_too_many_list_items_blocked(self):
        """Lists with too many items should be blocked"""
        large_list = list(range(20000))
        with pytest.raises(ValidationError, match="too many items"):
            validate_json_safe(large_list)

    def test_string_too_long_blocked(self):
        """Very long strings should be blocked"""
        long_string = "a" * 200000
        with pytest.raises(ValidationError, match="too long"):
            validate_json_safe(long_string)

    def test_non_string_dict_keys_blocked(self):
        """Non-string dictionary keys should be blocked"""
        # This would fail in actual dict creation, but testing the validator
        with pytest.raises(ValidationError, match="keys must be strings"):
            validate_json_safe({123: "value"})


class TestOrgIDValidation:
    """Test organization ID validation"""

    def test_valid_org_id(self):
        """Valid org IDs should pass"""
        org_id = validate_org_id("org-123-abc")
        assert org_id == "org-123-abc"

    def test_injection_blocked(self):
        """Injection attempts should be blocked"""
        with pytest.raises(ValidationError):
            validate_org_id("org'; DROP TABLE organizations; --")

    def test_special_chars_blocked(self):
        """Special characters should be blocked"""
        with pytest.raises(ValidationError):
            validate_org_id("org@123")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
