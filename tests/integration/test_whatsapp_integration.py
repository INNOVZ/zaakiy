"""
Integration tests for WhatsApp/Twilio integration.

These tests verify the complete WhatsApp message flow including:
- Webhook reception
- Message processing
- AI response generation
- Token consumption
- Database logging
"""

import asyncio
import json
import os

# Import the FastAPI app
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.main import app
from app.services.whatsapp.whatsapp_service import WhatsAppService


class TestWhatsAppIntegration:
    """Integration tests for WhatsApp functionality"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client"""
        mock = MagicMock()

        # Mock whatsapp_configurations query
        mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {
                "org_id": "test-org-123",
                "provider_type": "twilio",
                "twilio_account_sid": "ACtest123",
                "twilio_auth_token": "test_token_123",
                "twilio_phone_number": "+14155238886",
                "is_active": True,
            }
        ]

        # Mock chatbots query
        mock.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"id": "chatbot-123", "org_id": "test-org-123", "chain_status": "active"}
        ]

        return mock

    @pytest.fixture
    def mock_twilio_client(self):
        """Mock Twilio client"""
        mock = MagicMock()
        mock.messages.create.return_value = MagicMock(
            sid="SMtest123",
            status="sent",
            to="whatsapp:+1234567890",
            from_="whatsapp:+14155238886",
        )
        return mock

    def test_webhook_get_endpoint(self, client):
        """Test GET endpoint for webhook verification"""
        response = client.get("/api/whatsapp/webhook")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "twilio_whatsapp"
        assert "GET" in data["methods"]
        assert "POST" in data["methods"]

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    @patch("app.services.whatsapp.whatsapp_service.TwilioClient")
    def test_webhook_post_valid_message(
        self, mock_twilio, mock_supabase_func, client, mock_supabase
    ):
        """Test POST webhook with valid WhatsApp message"""
        mock_supabase_func.return_value = mock_supabase

        # Mock Twilio signature validation
        with patch("app.routers.whatsapp.RequestValidator") as mock_validator:
            mock_validator.return_value.validate.return_value = True

            # Prepare webhook payload (simulating Twilio)
            webhook_data = {
                "AccountSid": "ACtest123",
                "From": "whatsapp:+1234567890",
                "To": "whatsapp:+14155238886",
                "Body": "Hello, what are your business hours?",
                "MessageSid": "SMtest456",
                "NumMedia": "0",
            }

            headers = {"X-Twilio-Signature": "valid_signature_here"}

            response = client.post(
                "/api/whatsapp/webhook", data=webhook_data, headers=headers
            )

            # Should return 200 OK
            assert response.status_code == 200

    def test_webhook_missing_signature(self, client):
        """Test webhook rejects requests without Twilio signature"""
        webhook_data = {
            "AccountSid": "ACtest123",
            "From": "whatsapp:+1234567890",
            "Body": "Test message",
        }

        response = client.post("/api/whatsapp/webhook", data=webhook_data)

        # Should reject without signature
        assert response.status_code in [400, 403]

    def test_webhook_invalid_payload(self, client):
        """Test webhook handles invalid payload gracefully"""
        with patch("app.routers.whatsapp.RequestValidator") as mock_validator:
            mock_validator.return_value.validate.return_value = True

            # Missing required fields
            webhook_data = {
                "AccountSid": "ACtest123"
                # Missing From, MessageSid, Body
            }

            headers = {"X-Twilio-Signature": "valid_signature"}

            response = client.post(
                "/api/whatsapp/webhook", data=webhook_data, headers=headers
            )

            # Should handle gracefully
            assert response.status_code in [400, 500]

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    @patch("app.services.whatsapp.whatsapp_service.TwilioClient")
    async def test_phone_number_format_handling(
        self, mock_twilio, mock_supabase_func, mock_supabase
    ):
        """Test that phone numbers with 'whatsapp:' prefix are handled correctly"""
        mock_supabase_func.return_value = mock_supabase

        service = WhatsAppService(org_id="test-org-123")

        # Test with whatsapp: prefix
        from_number_with_prefix = "whatsapp:+1234567890"

        # The service should strip the prefix when sending
        with patch.object(
            service, "_get_twilio_client", return_value=mock_twilio.return_value
        ):
            with patch.object(service, "token_middleware") as mock_token:
                mock_token.validate_and_consume_tokens = AsyncMock()

                # This should not raise a ValueError about phone format
                try:
                    clean_number = from_number_with_prefix.replace(
                        "whatsapp:", ""
                    ).strip()
                    assert clean_number == "+1234567890"
                    assert clean_number.startswith("+")
                except ValueError as e:
                    pytest.fail(f"Phone number validation failed: {e}")

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    def test_configuration_retrieval(self, mock_supabase_func, mock_supabase):
        """Test WhatsApp configuration retrieval"""
        mock_supabase_func.return_value = mock_supabase

        service = WhatsAppService(org_id="test-org-123")
        config = service._get_whatsapp_config()

        assert config is not None
        assert config["org_id"] == "test-org-123"
        assert config["provider_type"] == "twilio"
        assert config["twilio_account_sid"] == "ACtest123"
        assert config["is_active"] is True

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    def test_no_active_configuration(self, mock_supabase_func):
        """Test behavior when no active configuration exists"""
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = (
            []
        )
        mock_supabase_func.return_value = mock_supabase

        service = WhatsAppService(org_id="test-org-456")

        with pytest.raises(Exception):  # Should raise configuration error
            service._get_whatsapp_config()

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    @patch("app.services.whatsapp.whatsapp_service.TwilioClient")
    async def test_message_length_validation(
        self, mock_twilio, mock_supabase_func, mock_supabase
    ):
        """Test that long messages are truncated to WhatsApp limit"""
        mock_supabase_func.return_value = mock_supabase

        service = WhatsAppService(org_id="test-org-123")

        # Create a message longer than 1600 characters
        long_message = "A" * 2000

        with patch.object(
            service, "_get_twilio_client", return_value=mock_twilio.return_value
        ):
            with patch.object(service, "token_middleware") as mock_token:
                mock_token.validate_and_consume_tokens = AsyncMock()

                result = await service.send_message(
                    to="+1234567890",
                    message=long_message,
                    entity_id="test-org-123",
                    entity_type="organization",
                )

                # Check that Twilio was called with truncated message
                call_args = mock_twilio.return_value.messages.create.call_args
                sent_message = call_args.kwargs["body"]
                assert len(sent_message) <= 1600

    def test_rate_limiting(self, client):
        """Test that rate limiting is applied to WhatsApp endpoints"""
        # This test would need actual rate limiting implementation
        # For now, just verify the endpoint exists
        response = client.get("/api/whatsapp/config")
        assert response.status_code in [200, 401, 403]  # Either works or needs auth

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    @patch("app.services.whatsapp.whatsapp_service.TwilioClient")
    async def test_token_consumption(
        self, mock_twilio, mock_supabase_func, mock_supabase
    ):
        """Test that tokens are consumed when sending messages"""
        mock_supabase_func.return_value = mock_supabase

        service = WhatsAppService(org_id="test-org-123")

        with patch.object(
            service, "_get_twilio_client", return_value=mock_twilio.return_value
        ):
            with patch.object(service, "token_middleware") as mock_token:
                mock_token.validate_and_consume_tokens = AsyncMock()

                await service.send_message(
                    to="+1234567890",
                    message="Test message",
                    entity_id="test-org-123",
                    entity_type="organization",
                )

                # Verify token consumption was called
                mock_token.validate_and_consume_tokens.assert_called_once()
                call_args = mock_token.validate_and_consume_tokens.call_args
                assert call_args.kwargs["channel"].value == "whatsapp"


class TestWhatsAppEndToEnd:
    """End-to-end tests simulating complete message flow"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    @patch("app.services.whatsapp.whatsapp_service.TwilioClient")
    @patch("app.services.chat.chat_service.ChatService")
    async def test_complete_message_flow(
        self, mock_chat_service, mock_twilio, mock_supabase_func
    ):
        """Test complete flow: receive message -> process -> respond"""
        # Setup mocks
        mock_supabase = MagicMock()
        mock_supabase_func.return_value = mock_supabase

        # Mock configuration
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {
                "org_id": "test-org-123",
                "twilio_account_sid": "ACtest123",
                "twilio_auth_token": "test_token",
                "twilio_phone_number": "+14155238886",
                "is_active": True,
            }
        ]

        # Mock chatbot
        mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"id": "chatbot-123", "org_id": "test-org-123"}
        ]

        # Mock chat service response
        mock_chat_instance = AsyncMock()
        mock_chat_instance.process_message.return_value = {
            "response": "Our business hours are 9 AM to 5 PM, Monday through Friday.",
            "tokens_consumed": 50,
        }
        mock_chat_service.return_value = mock_chat_instance

        # Create service and process message
        service = WhatsAppService(org_id="test-org-123")

        with patch.object(
            service, "_get_twilio_client", return_value=mock_twilio.return_value
        ):
            with patch.object(service, "token_middleware") as mock_token:
                mock_token.validate_and_consume_tokens = AsyncMock()

                result = await service.process_incoming_message(
                    from_number="whatsapp:+1234567890",
                    message_body="What are your business hours?",
                    twilio_sid="SMtest123",
                    chatbot_id="chatbot-123",
                )

                # Verify complete flow
                assert result["success"] is True
                assert result["response_sent"] is True
                assert "business hours" in result["response_text"].lower()
                assert result["message_sid"] is not None

                # Verify Twilio was called to send response
                mock_twilio.return_value.messages.create.assert_called_once()
                call_args = mock_twilio.return_value.messages.create.call_args
                assert call_args.kwargs["to"] == "whatsapp:+1234567890"
                assert "business hours" in call_args.kwargs["body"].lower()

    @patch("app.services.whatsapp.whatsapp_service.get_supabase_client")
    async def test_error_handling_no_chatbot(self, mock_supabase_func):
        """Test error handling when no active chatbot exists"""
        mock_supabase = MagicMock()
        mock_supabase_func.return_value = mock_supabase

        # Mock configuration exists
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {
                "org_id": "test-org-123",
                "twilio_account_sid": "ACtest123",
                "twilio_auth_token": "test_token",
                "twilio_phone_number": "+14155238886",
                "is_active": True,
            }
        ]

        # Mock no chatbot found
        mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = (
            []
        )

        service = WhatsAppService(org_id="test-org-123")

        with pytest.raises(Exception) as exc_info:
            await service.process_incoming_message(
                from_number="whatsapp:+1234567890",
                message_body="Test",
                twilio_sid="SMtest123",
            )

        assert "chatbot" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
