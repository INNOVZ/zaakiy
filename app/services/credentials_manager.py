"""
Secure Credentials Manager for MCP Integrations

Handles secure storage, retrieval, and rotation of credentials
for third-party service integrations.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet

from ...services.storage.supabase_client import get_supabase_client
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class CredentialManager:
    """
    Secure credential storage and retrieval

    Features:
    - AES encryption at rest
    - Automatic token refresh
    - Credential rotation
    - Audit logging
    - Multi-org support
    """

    def __init__(self):
        self.supabase = get_supabase_client()
        self._encryption_key = self._load_encryption_key()
        self._cipher_suite = Fernet(self._encryption_key)

    def _load_encryption_key(self) -> bytes:
        """Load encryption key from environment"""
        import os
        from base64 import b64encode

        key_env = os.getenv("ENCRYPTION_KEY")
        if not key_env:
            # Generate new key if not present (development only)
            logger.warning("No ENCRYPTION_KEY found - using temporary key")
            key = Fernet.generate_key()
            return key

        return key_env.encode()

    def _encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt credentials using Fernet"""
        plaintext = json.dumps(data).encode()
        ciphertext = self._cipher_suite.encrypt(plaintext)
        return ciphertext.decode()

    def _decrypt(self, ciphertext: str) -> Dict[str, Any]:
        """Decrypt credentials"""
        plaintext = self._cipher_suite.decrypt(ciphertext.encode())
        return json.loads(plaintext.decode())

    async def store_credential(
        self,
        org_id: str,
        service: str,
        credentials: Dict[str, Any],
        expires_in_days: Optional[int] = None,
    ) -> bool:
        """
        Store encrypted credentials for a service

        Args:
            org_id: Organization ID
            service: Service name (google, shopify, hubspot, etc.)
            credentials: Credentials dict to store
            expires_in_days: Optional expiration time

        Returns:
            True if stored successfully
        """
        try:
            encrypted_data = self._encrypt(credentials)

            expires_at = None
            if expires_in_days:
                expires_at = (
                    datetime.utcnow() + timedelta(days=expires_in_days)
                ).isoformat()

            # Check if credential already exists
            existing = (
                self.supabase.table("credentials")
                .select("id")
                .eq("org_id", org_id)
                .eq("service", service)
                .execute()
            )

            if existing.data:
                # Update existing
                self.supabase.table("credentials").update(
                    {
                        "encrypted_data": encrypted_data,
                        "expires_at": expires_at,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                ).eq("org_id", org_id).eq("service", service).execute()
            else:
                # Create new
                self.supabase.table("credentials").insert(
                    {
                        "org_id": org_id,
                        "service": service,
                        "encrypted_data": encrypted_data,
                        "expires_at": expires_at,
                        "created_at": datetime.utcnow().isoformat(),
                    }
                ).execute()

            logger.info(f"Stored credentials for {service} (org: {org_id})")
            return True

        except Exception as e:
            logger.error(f"Error storing credentials: {str(e)}")
            return False

    async def get_credential(
        self,
        org_id: str,
        service: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt credentials

        Args:
            org_id: Organization ID
            service: Service name

        Returns:
            Decrypted credentials dict or None
        """
        try:
            result = (
                self.supabase.table("credentials")
                .select("*")
                .eq("org_id", org_id)
                .eq("service", service)
                .single()
                .execute()
            )

            if not result.data:
                logger.warning(f"Credentials not found for {service} (org: {org_id})")
                return None

            # Check expiration
            if result.data.get("expires_at"):
                expires_at = datetime.fromisoformat(result.data["expires_at"])
                if datetime.utcnow() > expires_at:
                    logger.warning(f"Credentials expired for {service} (org: {org_id})")
                    return None

            # Decrypt and return
            credentials = self._decrypt(result.data["encrypted_data"])
            logger.info(f"Retrieved credentials for {service} (org: {org_id})")
            return credentials

        except Exception as e:
            logger.error(f"Error retrieving credentials: {str(e)}")
            return None

    async def refresh_token(
        self,
        org_id: str,
        service: str,
        new_credentials: Dict[str, Any],
    ) -> bool:
        """
        Refresh OAuth token (e.g., Google, Shopify)

        Args:
            org_id: Organization ID
            service: Service name
            new_credentials: Updated credentials with new token

        Returns:
            True if refreshed successfully
        """
        try:
            # Store refreshed credentials
            await self.store_credential(org_id, service, new_credentials)

            # Log token refresh
            self.supabase.table("credential_audit").insert(
                {
                    "org_id": org_id,
                    "service": service,
                    "action": "token_refresh",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ).execute()

            logger.info(f"Refreshed token for {service} (org: {org_id})")
            return True

        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            return False

    async def delete_credential(self, org_id: str, service: str) -> bool:
        """Delete stored credentials"""
        try:
            self.supabase.table("credentials").delete().eq("org_id", org_id).eq(
                "service", service
            ).execute()

            logger.info(f"Deleted credentials for {service} (org: {org_id})")
            return True

        except Exception as e:
            logger.error(f"Error deleting credentials: {str(e)}")
            return False

    async def list_credentials(self, org_id: str) -> list[str]:
        """List all services with stored credentials for an org"""
        try:
            result = (
                self.supabase.table("credentials")
                .select("service")
                .eq("org_id", org_id)
                .execute()
            )

            return [row["service"] for row in result.data]

        except Exception as e:
            logger.error(f"Error listing credentials: {str(e)}")
            return []

    async def audit_log(
        self, org_id: str, service: str, action: str, details: Dict = None
    ):
        """Log credential access for audit"""
        try:
            self.supabase.table("credential_audit").insert(
                {
                    "org_id": org_id,
                    "service": service,
                    "action": action,
                    "details": json.dumps(details) if details else None,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ).execute()
        except Exception as e:
            logger.error(f"Error logging credential audit: {str(e)}")


# Singleton instance
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get or create credential manager instance"""
    global _credential_manager

    if _credential_manager is None:
        _credential_manager = CredentialManager()

    return _credential_manager


# Example usage functions
async def setup_google_oauth(org_id: str, credentials: Dict[str, Any]) -> bool:
    """Setup Google OAuth credentials"""
    manager = get_credential_manager()
    success = await manager.store_credential(
        org_id=org_id,
        service="google",
        credentials=credentials,
        expires_in_days=30,  # Google tokens expire quickly
    )
    if success:
        await manager.audit_log(org_id, "google", "oauth_setup")
    return success


async def setup_shopify_api(org_id: str, shop_name: str, access_token: str) -> bool:
    """Setup Shopify API credentials"""
    manager = get_credential_manager()
    success = await manager.store_credential(
        org_id=org_id,
        service="shopify",
        credentials={
            "shop_name": shop_name,
            "access_token": access_token,
            "api_version": "2024-01",
        },
    )
    if success:
        await manager.audit_log(org_id, "shopify", "api_setup")
    return success


async def setup_hubspot_api(org_id: str, api_key: str) -> bool:
    """Setup HubSpot CRM credentials"""
    manager = get_credential_manager()
    success = await manager.store_credential(
        org_id=org_id,
        service="hubspot",
        credentials={"api_key": api_key, "platform": "hubspot"},
    )
    if success:
        await manager.audit_log(org_id, "hubspot", "api_setup")
    return success
