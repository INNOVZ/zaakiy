"""
Input validation utilities for the ZaaKy AI Platform
Provides reusable validators for common input types
"""
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from pydantic import validator


class ValidationError(Exception):
    """Custom validation error"""

    pass


def validate_message_length(
    message: str, min_length: int = 1, max_length: int = 4000
) -> str:
    """
    Validate message length to prevent token overflow and empty messages

    Args:
        message: The message to validate
        min_length: Minimum allowed length (default: 1)
        max_length: Maximum allowed length (default: 4000)

    Returns:
        Stripped message if valid

    Raises:
        ValidationError: If message is invalid
    """
    if not message or not isinstance(message, str):
        raise ValidationError("Message must be a non-empty string")

    stripped = message.strip()

    if len(stripped) < min_length:
        raise ValidationError(f"Message must be at least {min_length} characters")

    if len(stripped) > max_length:
        raise ValidationError(f"Message too long (max {max_length} characters)")

    return stripped


def validate_url(url: str, allow_localhost: bool = False) -> str:
    """
    Validate URL format and security with comprehensive SSRF protection

    Args:
        url: The URL to validate
        allow_localhost: Whether to allow localhost URLs (default: False)

    Returns:
        Validated URL

    Raises:
        ValidationError: If URL is invalid or unsafe
    """
    import ipaddress
    import socket

    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")

    url = url.strip()

    # Check length
    if len(url) > 2048:
        raise ValidationError("URL too long (max 2048 characters)")

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {str(e)}")

    # Check scheme - only allow http/https
    if parsed.scheme not in ["http", "https"]:
        raise ValidationError("URL must use http or https protocol")

    # Get hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValidationError("URL must have a valid hostname")

    # Check for suspicious patterns BEFORE DNS resolution
    if "@" in url:
        raise ValidationError("URLs with authentication are not allowed")

    if ".." in url:
        raise ValidationError("URLs with path traversal are not allowed")

    # Block URLs with encoded characters that could bypass filters
    suspicious_patterns = [
        "%00",
        "%0a",
        "%0d",  # Null bytes and newlines
        "%2e%2e",  # Encoded ..
        "%2f%2f",  # Encoded //
        "file://",
        "ftp://",
        "gopher://",
        "data://",  # Dangerous schemes
        "javascript:",
        "vbscript:",  # Script injection
    ]
    url_lower = url.lower()
    for pattern in suspicious_patterns:
        if pattern in url_lower:
            raise ValidationError(f"URL contains suspicious pattern: {pattern}")

    # Check for localhost/private IPs (unless explicitly allowed)
    if not allow_localhost:
        # Block localhost names
        localhost_names = [
            "localhost",
            "localhost.localdomain",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "0:0:0:0:0:0:0:1",
            "[::1]",
            "[0:0:0:0:0:0:0:1]",
        ]
        if hostname.lower() in localhost_names:
            raise ValidationError("Localhost URLs are not allowed")

        # Try to resolve hostname to IP and check if it's private
        try:
            # Get all IP addresses for the hostname
            addr_info = socket.getaddrinfo(hostname, None)

            for info in addr_info:
                ip_str = info[4][0]

                try:
                    ip_obj = ipaddress.ip_address(ip_str)

                    # Check if IP is private, loopback, link-local, or reserved
                    if ip_obj.is_private:
                        raise ValidationError(
                            f"URL resolves to private IP address: {ip_str}"
                        )

                    if ip_obj.is_loopback:
                        raise ValidationError(
                            f"URL resolves to loopback address: {ip_str}"
                        )

                    if ip_obj.is_link_local:
                        raise ValidationError(
                            f"URL resolves to link-local address: {ip_str}"
                        )

                    if ip_obj.is_reserved:
                        raise ValidationError(
                            f"URL resolves to reserved IP address: {ip_str}"
                        )

                    # Block cloud metadata endpoints
                    metadata_ips = [
                        "169.254.169.254",  # AWS, Azure, GCP metadata
                        "fd00:ec2::254",  # AWS IPv6 metadata
                    ]
                    if ip_str in metadata_ips:
                        raise ValidationError(
                            f"URL resolves to cloud metadata endpoint: {ip_str}"
                        )

                except ValueError:
                    # Not a valid IP address, skip
                    continue

        except socket.gaierror:
            # DNS resolution failed - this is actually safer, allow it
            pass
        except Exception as e:
            # Other DNS errors - be conservative and block
            raise ValidationError(f"Unable to validate URL safety: {str(e)}")

        # Additional hostname checks
        # Block common internal/private TLDs
        blocked_tlds = [".local", ".internal", ".private", ".corp", ".home", ".lan"]
        hostname_lower = hostname.lower()
        for tld in blocked_tlds:
            if hostname_lower.endswith(tld):
                raise ValidationError(f"URLs with {tld} TLD are not allowed")

        # Block IP addresses in hostname (both IPv4 and IPv6)
        try:
            ip_obj = ipaddress.ip_address(hostname.strip("[]"))
            # If we get here, hostname is an IP address
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                raise ValidationError(
                    "Direct IP addresses to private networks are not allowed"
                )
        except ValueError:
            # Not an IP address, which is fine
            pass

    # Check port if specified
    if parsed.port:
        # Block commonly dangerous ports
        blocked_ports = [
            22,  # SSH
            23,  # Telnet
            25,  # SMTP
            3306,  # MySQL
            5432,  # PostgreSQL
            6379,  # Redis
            27017,  # MongoDB
            9200,  # Elasticsearch
        ]
        if parsed.port in blocked_ports:
            raise ValidationError(f"Port {parsed.port} is not allowed")

    return url


def validate_chatbot_name(name: str, min_length: int = 2, max_length: int = 100) -> str:
    """
    Validate chatbot name

    Args:
        name: The chatbot name to validate
        min_length: Minimum allowed length (default: 2)
        max_length: Maximum allowed length (default: 100)

    Returns:
        Stripped name if valid

    Raises:
        ValidationError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Chatbot name must be a non-empty string")

    stripped = name.strip()

    if len(stripped) < min_length:
        raise ValidationError(f"Chatbot name must be at least {min_length} characters")

    if len(stripped) > max_length:
        raise ValidationError(f"Chatbot name too long (max {max_length} characters)")

    # Check for valid characters (alphanumeric, spaces, and common punctuation)
    if not re.match(r"^[a-zA-Z0-9\s\-_.,!?]+$", stripped):
        raise ValidationError("Chatbot name contains invalid characters")

    return stripped


def validate_hex_color(color: str) -> str:
    """
    Validate hex color code

    Args:
        color: The hex color to validate (e.g., #FF0000)

    Returns:
        Validated color code

    Raises:
        ValidationError: If color is invalid
    """
    if not color or not isinstance(color, str):
        raise ValidationError("Color must be a non-empty string")

    color = color.strip()

    # Check format
    if not re.match(r"^#[0-9A-Fa-f]{6}$", color):
        raise ValidationError("Color must be a valid hex code (e.g., #FF0000)")

    return color.upper()


def validate_file_size(file_size: int, max_size_mb: int = 50) -> int:
    """
    Validate file size

    Args:
        file_size: File size in bytes
        max_size_mb: Maximum allowed size in MB (default: 50)

    Returns:
        File size if valid

    Raises:
        ValidationError: If file is too large
    """
    max_bytes = max_size_mb * 1024 * 1024

    if file_size > max_bytes:
        raise ValidationError(f"File too large (max {max_size_mb}MB)")

    if file_size <= 0:
        raise ValidationError("File is empty")

    return file_size


def validate_file_type(filename: str, allowed_types: list) -> str:
    """
    Validate file type by extension

    Args:
        filename: The filename to validate
        allowed_types: List of allowed extensions (e.g., ['.pdf', '.json'])

    Returns:
        Filename if valid

    Raises:
        ValidationError: If file type is not allowed
    """
    if not filename or not isinstance(filename, str):
        raise ValidationError("Filename must be a non-empty string")

    filename = filename.strip()

    # Get extension
    extension = filename.lower().split(".")[-1] if "." in filename else ""

    if not extension:
        raise ValidationError("File must have an extension")

    # Normalize allowed types
    normalized_allowed = [t.lower().lstrip(".") for t in allowed_types]

    if extension not in normalized_allowed:
        raise ValidationError(
            f"File type '.{extension}' not allowed. "
            f"Allowed types: {', '.join(allowed_types)}"
        )

    return filename


def validate_rating(rating: int) -> int:
    """
    Validate feedback rating

    Args:
        rating: The rating value (should be 1 or -1)

    Returns:
        Rating if valid

    Raises:
        ValidationError: If rating is invalid
    """
    if rating not in [1, -1]:
        raise ValidationError("Rating must be 1 (thumbs up) or -1 (thumbs down)")

    return rating


def validate_top_k(top_k: int, min_value: int = 1, max_value: int = 100) -> int:
    """
    Validate top_k parameter for search queries

    Args:
        top_k: Number of results to return
        min_value: Minimum allowed value (default: 1)
        max_value: Maximum allowed value (default: 100)

    Returns:
        top_k if valid

    Raises:
        ValidationError: If top_k is out of range
    """
    if not isinstance(top_k, int):
        raise ValidationError("top_k must be an integer")

    if top_k < min_value:
        raise ValidationError(f"top_k must be at least {min_value}")

    if top_k > max_value:
        raise ValidationError(f"top_k cannot exceed {max_value}")

    return top_k


def validate_temperature(temperature: float) -> float:
    """
    Validate AI model temperature parameter

    Args:
        temperature: Temperature value (0.0 to 2.0)

    Returns:
        Temperature if valid

    Raises:
        ValidationError: If temperature is out of range
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError("Temperature must be a number")

    if temperature < 0.0 or temperature > 2.0:
        raise ValidationError("Temperature must be between 0.0 and 2.0")

    return float(temperature)


def validate_max_tokens(max_tokens: int) -> int:
    """
    Validate max_tokens parameter

    Args:
        max_tokens: Maximum tokens to generate

    Returns:
        max_tokens if valid

    Raises:
        ValidationError: If max_tokens is invalid
    """
    if not isinstance(max_tokens, int):
        raise ValidationError("max_tokens must be an integer")

    if max_tokens < 1:
        raise ValidationError("max_tokens must be at least 1")

    if max_tokens > 8000:
        raise ValidationError("max_tokens cannot exceed 8000")

    return max_tokens


def sanitize_text_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input by removing potentially dangerous characters

    Args:
        text: Text to sanitize
        max_length: Optional maximum length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters except newlines and tabs
    text = "".join(
        char for char in text if char == "\n" or char == "\t" or ord(char) >= 32
    )

    # Trim if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text.strip()


def validate_upload_id(upload_id: str) -> str:
    """
    Validate upload ID to prevent injection attacks

    Args:
        upload_id: The upload ID to validate

    Returns:
        Validated upload ID

    Raises:
        ValidationError: If upload ID is invalid
    """
    if not upload_id or not isinstance(upload_id, str):
        raise ValidationError("Upload ID must be a non-empty string")

    upload_id = upload_id.strip()

    # Check length
    if len(upload_id) < 1 or len(upload_id) > 100:
        raise ValidationError("Upload ID must be between 1 and 100 characters")

    # Only allow alphanumeric, hyphens, and underscores (UUID-safe)
    if not re.match(r"^[a-zA-Z0-9\-_]+$", upload_id):
        raise ValidationError(
            "Upload ID can only contain letters, numbers, hyphens, and underscores"
        )

    return upload_id


def validate_namespace(namespace: str) -> str:
    """
    Validate Pinecone namespace to prevent injection

    Args:
        namespace: The namespace to validate

    Returns:
        Validated namespace

    Raises:
        ValidationError: If namespace is invalid
    """
    if not namespace or not isinstance(namespace, str):
        raise ValidationError("Namespace must be a non-empty string")

    namespace = namespace.strip()

    # Check length
    if len(namespace) < 1 or len(namespace) > 200:
        raise ValidationError("Namespace must be between 1 and 200 characters")

    # Only allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r"^[a-zA-Z0-9\-_.]+$", namespace):
        raise ValidationError(
            "Namespace can only contain letters, numbers, hyphens, underscores, and dots"
        )

    # Prevent path traversal
    if ".." in namespace or namespace.startswith(".") or namespace.endswith("."):
        raise ValidationError("Invalid namespace format")

    return namespace


def validate_org_id(org_id: str) -> str:
    """
    Validate organization ID to prevent injection

    Args:
        org_id: The organization ID to validate

    Returns:
        Validated org ID

    Raises:
        ValidationError: If org ID is invalid
    """
    if not org_id or not isinstance(org_id, str):
        raise ValidationError("Organization ID must be a non-empty string")

    org_id = org_id.strip()

    # Check length
    if len(org_id) < 1 or len(org_id) > 100:
        raise ValidationError("Organization ID must be between 1 and 100 characters")

    # Only allow alphanumeric, hyphens, and underscores
    if not re.match(r"^[a-zA-Z0-9\-_]+$", org_id):
        raise ValidationError(
            "Organization ID can only contain letters, numbers, hyphens, and underscores"
        )

    return org_id


def validate_metadata_filter(filter_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate metadata filter dictionary to prevent injection attacks

    Args:
        filter_dict: The filter dictionary to validate

    Returns:
        Validated filter dictionary

    Raises:
        ValidationError: If filter is invalid or potentially dangerous
    """
    if not isinstance(filter_dict, dict):
        raise ValidationError("Filter must be a dictionary")

    if len(filter_dict) == 0:
        raise ValidationError("Filter cannot be empty")

    if len(filter_dict) > 10:
        raise ValidationError("Filter has too many keys (max 10)")

    validated = {}

    # Allowed filter keys (whitelist approach)
    allowed_keys = {
        "upload_id",
        "org_id",
        "source",
        "type",
        "chunk_index",
        "has_products",
        "created_at",
        "updated_at",
    }

    for key, value in filter_dict.items():
        # Validate key
        if not isinstance(key, str):
            raise ValidationError(f"Filter key must be string, got {type(key)}")

        if key not in allowed_keys:
            raise ValidationError(f"Filter key '{key}' is not allowed")

        # Validate value based on key
        if key in ["upload_id", "org_id", "source", "type"]:
            if not isinstance(value, str):
                raise ValidationError(f"Filter value for '{key}' must be string")

            # Sanitize string values
            if len(value) > 500:
                raise ValidationError(f"Filter value for '{key}' is too long")

            # Check for injection patterns
            dangerous_patterns = ["$", "{", "}", "..", "/", "\\", "\x00"]
            for pattern in dangerous_patterns:
                if pattern in value:
                    raise ValidationError(
                        f"Filter value for '{key}' contains dangerous pattern: {pattern}"
                    )

        elif key == "chunk_index":
            if not isinstance(value, int):
                raise ValidationError(f"Filter value for '{key}' must be integer")

            if value < 0 or value > 1000000:
                raise ValidationError(f"Filter value for '{key}' is out of range")

        elif key == "has_products":
            if not isinstance(value, bool):
                raise ValidationError(f"Filter value for '{key}' must be boolean")

        validated[key] = value

    return validated


def validate_json_safe(data: Any, max_depth: int = 10, current_depth: int = 0) -> bool:
    """
    Validate that data structure is safe for JSON serialization

    Args:
        data: Data to validate
        max_depth: Maximum nesting depth allowed
        current_depth: Current recursion depth

    Returns:
        True if safe

    Raises:
        ValidationError: If data structure is unsafe
    """
    if current_depth > max_depth:
        raise ValidationError(f"Data structure exceeds maximum depth of {max_depth}")

    if isinstance(data, dict):
        if len(data) > 1000:
            raise ValidationError("Dictionary has too many keys (max 1000)")

        for key, value in data.items():
            if not isinstance(key, str):
                raise ValidationError("Dictionary keys must be strings")

            if len(key) > 500:
                raise ValidationError("Dictionary key is too long (max 500)")

            validate_json_safe(value, max_depth, current_depth + 1)

    elif isinstance(data, list):
        if len(data) > 10000:
            raise ValidationError("List has too many items (max 10000)")

        for item in data:
            validate_json_safe(item, max_depth, current_depth + 1)

    elif isinstance(data, str):
        if len(data) > 100000:
            raise ValidationError("String is too long (max 100000)")

    elif isinstance(data, (int, float, bool, type(None))):
        pass  # These are safe

    else:
        raise ValidationError(f"Unsupported data type: {type(data)}")

    return True
