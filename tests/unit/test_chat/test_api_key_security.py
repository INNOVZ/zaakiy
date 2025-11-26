"""
Test API Key Handling Security
Tests that the ChatService properly handles missing API keys without exposing system internals
"""
import os
from unittest.mock import patch

import pytest

from app.services.chat.chat_service import ChatService, ChatServiceError


class TestAPIKeyHandling:
    """Test secure API key handling"""

    def test_missing_api_key_in_development(self):
        """Test that missing API key in development provides helpful error"""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            # Remove OPENAI_API_KEY temporarily
            original_key = os.environ.get("OPENAI_API_KEY")
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            try:
                with pytest.raises(ChatServiceError) as exc_info:
                    ChatService(
                        org_id="test-org",
                        chatbot_config={
                            "id": "test-bot",
                            "model": "gpt-3.5-turbo",
                        },
                    )

                # Verify error message is helpful for developers
                error_message = str(exc_info.value)
                assert "OPENAI_API_KEY" in error_message
                assert ".env" in error_message
                assert "not set" in error_message

            finally:
                # Restore original key
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key

    @patch("app.services.chat.chat_service.error_monitor")
    def test_missing_api_key_in_production(self, mock_error_monitor):
        """Test that missing API key in production doesn't expose system internals"""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            # Remove OPENAI_API_KEY temporarily
            original_key = os.environ.get("OPENAI_API_KEY")
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            try:
                with pytest.raises(ChatServiceError) as exc_info:
                    ChatService(
                        org_id="test-org",
                        chatbot_config={
                            "id": "test-bot",
                            "model": "gpt-3.5-turbo",
                        },
                    )

                # Verify error message is generic and doesn't expose internals
                error_message = str(exc_info.value)
                assert "configuration error" in error_message.lower()
                assert "contact support" in error_message.lower()

                # Should NOT contain these sensitive details
                assert "OPENAI_API_KEY" not in error_message
                assert ".env" not in error_message
                assert "environment variable" not in error_message

                # Verify error was recorded in monitoring system
                mock_error_monitor.record_error.assert_called_once()
                call_args = mock_error_monitor.record_error.call_args
                assert call_args[1]["error_type"] == "MissingAPIKeyError"
                assert call_args[1]["severity"] == "critical"

            finally:
                # Restore original key
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key

    def test_valid_api_key_initializes_successfully(self):
        """Test that valid API key initializes ChatService successfully"""
        # This test assumes OPENAI_API_KEY is set in test environment
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set in test environment")

        try:
            chat_service = ChatService(
                org_id="test-org",
                chatbot_config={
                    "id": "test-bot",
                    "model": "gpt-3.5-turbo",
                },
            )

            # Verify OpenAI client is initialized
            assert chat_service.openai_client is not None
            assert hasattr(chat_service.openai_client, "chat")

        except Exception as e:
            pytest.fail(f"ChatService initialization failed with valid API key: {e}")

    def test_api_key_not_logged_on_success(self, caplog):
        """Test that API key is never logged, even on success"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set in test environment")

        with caplog.at_level("DEBUG"):
            ChatService(
                org_id="test-org",
                chatbot_config={
                    "id": "test-bot",
                    "model": "gpt-3.5-turbo",
                },
            )

        # Check that API key is not in any log messages
        api_key = os.getenv("OPENAI_API_KEY")
        for record in caplog.records:
            assert api_key not in record.message, "API key found in logs!"

    def test_api_key_warning_not_logged_on_missing(self, caplog):
        """Test that no warning is logged when API key is missing (fail fast instead)"""
        with patch.dict(os.environ, {}, clear=False):
            # Remove OPENAI_API_KEY temporarily
            original_key = os.environ.get("OPENAI_API_KEY")
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            try:
                with caplog.at_level("WARNING"):
                    with pytest.raises(ChatServiceError):
                        ChatService(
                            org_id="test-org",
                            chatbot_config={
                                "id": "test-bot",
                                "model": "gpt-3.5-turbo",
                            },
                        )

                # Verify no warning was logged about missing API key
                warning_messages = [
                    record.message
                    for record in caplog.records
                    if record.levelname == "WARNING"
                ]

                for msg in warning_messages:
                    assert "API key not found" not in msg
                    assert "OPENAI_API_KEY" not in msg

            finally:
                # Restore original key
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
