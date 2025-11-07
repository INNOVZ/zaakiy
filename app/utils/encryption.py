"""Encryption utilities for sensitive data storage."""

import base64
import logging
import os
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Global encryption key (initialized on first use)
_encryption_key: Optional[bytes] = None


def _get_encryption_key() -> bytes:
    """
    Get or generate encryption key from environment variable.

    Uses ENCRYPTION_KEY environment variable if set, otherwise generates
    a key from ENCRYPTION_SALT (for deterministic key generation).

    CRITICAL: In production, ENCRYPTION_KEY must be set. Default credentials
    are only for development and will fail in production environments.

    Returns:
        Encryption key bytes

    Raises:
        ValueError: If ENCRYPTION_KEY is not set in production environment
    """
    global _encryption_key

    if _encryption_key is not None:
        return _encryption_key

    # Try to get key from environment
    key_str = os.getenv("ENCRYPTION_KEY")
    if key_str:
        try:
            # Verify ENCRYPTION_KEY format before use
            # Fernet keys must be 44 bytes base64-encoded (32 bytes raw)
            key_bytes = key_str.encode()

            # Validate length: Fernet keys are exactly 44 base64 characters
            if len(key_str) != 44:
                raise ValueError(
                    f"ENCRYPTION_KEY must be 44 characters (Fernet key format), got {len(key_str)}"
                )

            # Validate it's valid base64
            try:
                decoded = base64.urlsafe_b64decode(key_bytes)
                if len(decoded) != 32:
                    raise ValueError(
                        f"ENCRYPTION_KEY decodes to {len(decoded)} bytes, expected 32 bytes"
                    )
            except Exception as e:
                raise ValueError(f"ENCRYPTION_KEY is not valid base64: {str(e)}")

            # Validate it's a valid Fernet key
            Fernet(key_bytes)
            _encryption_key = key_bytes
            logger.info("Using encryption key from ENCRYPTION_KEY environment variable")
            return _encryption_key
        except Exception as e:
            logger.error(
                "Invalid ENCRYPTION_KEY format: %s. Key must be a valid 44-character Fernet key.",
                str(e),
            )
            raise ValueError(
                f"Invalid ENCRYPTION_KEY format: {str(e)}. "
                "Generate a key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            ) from e

    # Check if we're in production - fail if ENCRYPTION_KEY is not set
    environment = os.getenv("ENVIRONMENT", "").lower()
    is_production = environment in ("production", "prod")

    if is_production:
        # CRITICAL: Fail in production if ENCRYPTION_KEY is not set
        # Hardcoded default credentials create severe security risk
        logger.error(
            "ENCRYPTION_KEY not set in production environment. "
            "This is a critical security risk. Set ENCRYPTION_KEY environment variable."
        )
        raise ValueError(
            "ENCRYPTION_KEY must be set in production. "
            "Generate a key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )

    # Development fallback: Generate key from salt (deterministic)
    # WARNING: This is only for development - never use in production
    salt = os.getenv("ENCRYPTION_SALT")
    password = os.getenv("ENCRYPTION_PASSWORD")

    if not salt or not password:
        logger.error(
            "ENCRYPTION_KEY not set and ENCRYPTION_SALT/ENCRYPTION_PASSWORD not provided. "
            "Cannot generate encryption key. Set ENCRYPTION_KEY environment variable."
        )
        raise ValueError(
            "ENCRYPTION_KEY must be set, or ENCRYPTION_SALT and ENCRYPTION_PASSWORD for development. "
            "Generate a key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )

    salt_bytes = salt.encode()

    # Derive key using PBKDF2 with increased iterations for security
    # Current security standard: 600,000+ iterations for PBKDF2-HMAC-SHA256
    # Increased from 100,000 to meet OWASP and NIST recommendations
    iterations = int(os.getenv("PBKDF2_ITERATIONS", "600000"))
    if iterations < 600000:
        logger.warning(
            "PBKDF2 iterations (%d) below recommended minimum (600,000). "
            "Set PBKDF2_ITERATIONS environment variable to at least 600000.",
            iterations,
        )
        iterations = 600000

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_bytes,
        iterations=iterations,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

    _encryption_key = key
    logger.warning(
        "Using generated encryption key from salt (DEVELOPMENT ONLY). "
        "Set ENCRYPTION_KEY environment variable for production. "
        "PBKDF2 iterations: %d",
        iterations,
    )
    return _encryption_key


def encrypt_value(value: str) -> str:
    """
    Encrypt a string value using Fernet symmetric encryption.

    Args:
        value: Plaintext string to encrypt

    Returns:
        Base64-encoded encrypted string
    """
    if not value:
        return value

    try:
        key = _get_encryption_key()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        logger.error("Failed to encrypt value: %s", str(e), exc_info=True)
        raise ValueError(f"Encryption failed: {str(e)}") from e


def decrypt_value(encrypted_value: str) -> str:
    """
    Decrypt a string value that was encrypted with encrypt_value.

    Args:
        encrypted_value: Base64-encoded encrypted string

    Returns:
        Decrypted plaintext string
    """
    if not encrypted_value:
        return encrypted_value

    try:
        key = _get_encryption_key()
        fernet = Fernet(key)
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        logger.error("Failed to decrypt value: %s", str(e), exc_info=True)
        raise ValueError(f"Decryption failed: {str(e)}") from e


def is_encrypted(value: str) -> bool:
    """
    Check if a value appears to be encrypted.

    This is a heuristic check - encrypted values are base64-encoded
    and typically longer than the original.

    Args:
        value: String to check

    Returns:
        True if value appears to be encrypted
    """
    if not value:
        return False

    # Encrypted values are base64-encoded, so they should:
    # 1. Be longer than typical plaintext
    # 2. Contain only base64 characters
    # 3. Have length that's a multiple of 4 (base64 padding)
    try:
        # Try to decode as base64
        base64.urlsafe_b64decode(value.encode())
        # If it decodes successfully and is reasonably long, likely encrypted
        return len(value) > 20
    except Exception:
        return False
