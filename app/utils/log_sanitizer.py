"""
Logging sanitization utilities to prevent PII and secret leakage

This module provides utilities to sanitize sensitive data before logging.
"""
import re
from typing import Any, Dict, Set


# Patterns that might contain sensitive data
SENSITIVE_PATTERNS = {
    # API Keys and Tokens
    "api_key": r"(api[_-]?key|apikey)",
    "access_token": r"(access[_-]?token|bearer)",
    "secret": r"(secret|password|passwd|pwd)",
    "auth": r"(authorization|auth[_-]?token)",
    
    # Personal Information
    "email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
    "phone": r"(\+?[\d\s\-\(\)]{10,})",
    "ssn": r"(\d{3}-?\d{2}-?\d{4})",
    "credit_card": r"(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})",
    
    # System Information
    "ip_address": r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
}

# Field names that should always be redacted
SENSITIVE_FIELD_NAMES: Set[str] = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "bearer",
    "authorization",
    "auth_token",
    "private_key",
    "secret_key",
    "service_key",
    "jwt_secret",
    "session_key",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "pin",
}


def sanitize_string(value: str, redact_text: str = "[REDACTED]") -> str:
    """
    Sanitize a string by removing sensitive patterns
    
    Args:
        value: String to sanitize
        redact_text: Replacement text for sensitive data
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return value
    
    # Redact based on patterns
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        if pattern_name in ["api_key", "access_token", "secret", "auth"]:
            value = re.sub(pattern, redact_text, value, flags=re.IGNORECASE)
    
    return value


def sanitize_dict(data: Dict[str, Any], redact_text: str = "[REDACTED]") -> Dict[str, Any]:
    """
    Sanitize a dictionary by removing sensitive fields
    
    Args:
        data: Dictionary to sanitize
        redact_text: Replacement text for sensitive data
        
    Returns:
        Sanitized dictionary
    """
    if not isinstance(data, dict):
        return data
    
    sanitized = {}
    for key, value in data.items():
        # Check if field name is sensitive
        if key.lower() in SENSITIVE_FIELD_NAMES:
            sanitized[key] = redact_text
        # Recursively sanitize nested dictionaries
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, redact_text)
        # Sanitize list items
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_dict(item, redact_text) if isinstance(item, dict)
                else sanitize_string(str(item), redact_text) if isinstance(item, str)
                else item
                for item in value
            ]
        # Sanitize string values
        elif isinstance(value, str):
            sanitized[key] = sanitize_string(value, redact_text)
        else:
            sanitized[key] = value
    
    return sanitized


def sanitize_log_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize extra fields for logging
    
    Args:
        extra: Extra fields dictionary
        
    Returns:
        Sanitized extra fields
    """
    return sanitize_dict(extra)


def truncate_long_string(value: str, max_length: int = 500) -> str:
    """
    Truncate long strings to prevent log bloat
    
    Args:
        value: String to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string
    """
    if not isinstance(value, str):
        return value
    
    if len(value) <= max_length:
        return value
    
    return value[:max_length] + "... [truncated]"


def safe_log_extra(**kwargs) -> Dict[str, Any]:
    """
    Create safe extra fields for logging
    
    Usage:
        logger.info("User action", extra=safe_log_extra(
            user_id=user_id,
            action="login",
            email=email  # Will be sanitized
        ))
    
    Args:
        **kwargs: Key-value pairs to include in log extra
        
    Returns:
        Sanitized extra fields dictionary
    """
    # Sanitize sensitive data
    sanitized = sanitize_dict(kwargs)
    
    # Truncate long strings
    for key, value in sanitized.items():
        if isinstance(value, str):
            sanitized[key] = truncate_long_string(value)
    
    return sanitized

