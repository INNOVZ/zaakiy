"""
URL sanitization utilities for secure logging
Prevents exposure of sensitive information in logs
"""

from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

class URLSanitizer:
    """Sanitize URLs for safe logging without exposing sensitive information"""
    
    # Common sensitive parameter names to redact
    SENSITIVE_PARAMS = {
        # Authentication
        'api_key', 'apikey', 'api-key', 'key', 'token', 'access_token', 
        'refresh_token', 'auth', 'authorization', 'bearer', 'jwt',
        
        # Session management
        'session', 'sessionid', 'session_id', 'sid', 'cookie',
        
        # Security
        'password', 'pwd', 'pass', 'secret', 'private_key', 'public_key',
        'signature', 'sig', 'hash', 'nonce', 'csrf', 'csrf_token',
        
        # Cloud services
        'aws_access_key', 'aws_secret_key', 'azure_key', 'gcp_key',
        'x-api-key', 'x-auth-token', 'x-access-token',
        
        # Custom application tokens
        'user_token', 'app_token', 'client_secret', 'client_key'
    }
    
    # Sensitive domains that should be completely redacted
    SENSITIVE_DOMAINS = {
        'localhost', '127.0.0.1', '0.0.0.0',
        '169.254.169.254',  # AWS metadata
        'metadata.google.internal',  # GCP metadata
        'metadata.azure.com'  # Azure metadata
    }
    
    @classmethod
    def sanitize_url_for_logging(cls, url: str, preserve_domain: bool = True) -> str:
        """
        Sanitize URL for safe logging
        
        Args:
            url: Original URL to sanitize
            preserve_domain: Whether to preserve the domain name
            
        Returns:
            Sanitized URL safe for logging
        """
        try:
            parsed = urlparse(url)
            
            # Check if domain should be completely redacted
            if parsed.hostname and parsed.hostname.lower() in cls.SENSITIVE_DOMAINS:
                return f"[REDACTED_INTERNAL]/{parsed.path.split('/')[-1] if parsed.path else ''}"
            
            # Sanitize query parameters
            sanitized_query = ""
            if parsed.query:
                query_params = parse_qs(parsed.query)
                safe_params = {}
                
                for key, values in query_params.items():
                    key_lower = key.lower().replace('-', '_').replace(' ', '_')
                    
                    if any(sensitive in key_lower for sensitive in cls.SENSITIVE_PARAMS):
                        # Redact sensitive parameters
                        safe_params[key] = ['[REDACTED]'] * len(values)
                    else:
                        # Keep non-sensitive parameters
                        safe_params[key] = values
                
                sanitized_query = urlencode(safe_params, doseq=True)
            
            # Build sanitized URL
            if preserve_domain:
                # Keep domain but sanitize sensitive parts
                sanitized_netloc = parsed.netloc
                if '@' in sanitized_netloc:
                    # Remove user credentials from URL
                    sanitized_netloc = sanitized_netloc.split('@')[-1]
                
                sanitized_url = urlunparse((
                    parsed.scheme,
                    sanitized_netloc,
                    parsed.path,
                    parsed.params,
                    sanitized_query,
                    ""  # Remove fragment for security
                ))
                
                return sanitized_url
            else:
                # Only show path and sanitized parameters
                path_part = parsed.path if parsed.path else "/"
                if sanitized_query:
                    return f"[DOMAIN_REDACTED]{path_part}?{sanitized_query}"
                else:
                    return f"[DOMAIN_REDACTED]{path_part}"
                    
        except Exception:
            # If URL parsing fails, return a safe fallback
            return "[INVALID_URL]"
    
    @classmethod
    def get_safe_domain(cls, url: str) -> str:
        """
        Extract domain safely for logging
        
        Returns:
            Safe domain name or redacted placeholder
        """
        try:
            parsed = urlparse(url)
            if parsed.hostname and parsed.hostname.lower() in cls.SENSITIVE_DOMAINS:
                return "[INTERNAL_DOMAIN]"
            return parsed.hostname or "[UNKNOWN_DOMAIN]"
        except Exception:
            return "[INVALID_DOMAIN]"
    
    @classmethod
    def get_safe_path(cls, url: str) -> str:
        """
        Extract path safely for logging
        
        Returns:
            Safe path without sensitive parameters
        """
        try:
            parsed = urlparse(url)
            return parsed.path or "/"
        except Exception:
            return "[INVALID_PATH]"
    
    @classmethod
    def create_safe_log_message(cls, action: str, url: str, extra_info: str = "") -> str:
        """
        Create a safe log message with sanitized URL
        
        Args:
            action: Action being performed (e.g., "fetching", "scraping")
            url: Original URL
            extra_info: Additional information to include
            
        Returns:
            Safe log message
        """
        safe_domain = cls.get_safe_domain(url)
        safe_path = cls.get_safe_path(url)
        
        base_message = f"{action} from domain: {safe_domain}, path: {safe_path}"
        
        if extra_info:
            return f"{base_message} - {extra_info}"
        
        return base_message


# Convenience functions for common logging scenarios
def log_url_safely(url: str) -> str:
    """Quick function to sanitize URL for logging"""
    return URLSanitizer.sanitize_url_for_logging(url)

def log_domain_safely(url: str) -> str:
    """Quick function to get safe domain for logging"""
    return URLSanitizer.get_safe_domain(url)

def create_safe_fetch_message(url: str, content_size: Optional[int] = None) -> str:
    """Create safe message for fetch operations"""
    extra = f"size: {content_size} bytes" if content_size else ""
    return URLSanitizer.create_safe_log_message("Fetching content", url, extra)

def create_safe_success_message(url: str, result_info: str) -> str:
    """Create safe message for successful operations"""
    return URLSanitizer.create_safe_log_message("Successfully processed", url, result_info)

def create_safe_error_message(url: str, error_type: str) -> str:
    """Create safe message for error operations"""
    return URLSanitizer.create_safe_log_message("Error processing", url, f"error: {error_type}")
