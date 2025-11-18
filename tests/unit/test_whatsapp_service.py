import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub vector_management to avoid heavy Pinecone initialization during tests
fake_vector_module = ModuleType("app.services.storage.vector_management")
fake_vector_module.VectorManagementService = object
fake_vector_module.VectorDeletionStrategy = object
fake_vector_module.QueryBatchDeletion = object
fake_vector_module.vector_management_service = None
sys.modules.setdefault("app.services.storage.vector_management", fake_vector_module)

from app.models.subscription import Channel
from app.services.whatsapp.whatsapp_service import WhatsAppService


class FakeQuery:
    def __init__(self, data):
        self._data = data

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class FakeSupabase:
    def __init__(self, mapping=None):
        self._mapping = mapping or {}
        self.auth = SimpleNamespace()

    def table(self, name):
        data = self._mapping.get(name, [])
        return FakeQuery(data)


@pytest.mark.asyncio
async def test_send_message_consumes_tokens(monkeypatch):
    fake_supabase = FakeSupabase()
    monkeypatch.setattr(
        "app.services.whatsapp.whatsapp_service.get_supabase_client",
        lambda: fake_supabase,
    )

    service = WhatsAppService(org_id="org-1")
    service._get_whatsapp_config = MagicMock(
        return_value={"twilio_phone_number": "+15550000000"}
    )

    mock_twilio_client = MagicMock()
    mock_message = SimpleNamespace(sid="SM123", status="sent")
    mock_twilio_client.messages.create.return_value = mock_message
    service._get_twilio_client = MagicMock(return_value=mock_twilio_client)

    service.token_middleware.validate_and_consume_tokens = AsyncMock()
    service._log_message = AsyncMock()

    await service.send_message(
        to="+15551234567",
        message="Hello from tests",
        chatbot_id="chatbot-1",
        session_id="session-1",
        entity_id="org-1",
        entity_type="organization",
        requesting_user_id="user-9",
    )

    service.token_middleware.validate_and_consume_tokens.assert_awaited_once()
    _args, kwargs = service.token_middleware.validate_and_consume_tokens.await_args
    assert kwargs["channel"] == Channel.WHATSAPP
    assert kwargs["requesting_user_id"] == "user-9"

    service._log_message.assert_awaited_once()
    _log_args, log_kwargs = service._log_message.await_args
    assert log_kwargs["direction"] == "outbound"
    assert log_kwargs["customer_number"] == "+15551234567"
    assert log_kwargs["tokens_consumed"] > 0


@pytest.mark.asyncio
async def test_process_incoming_message_routes_with_whatsapp_channel(monkeypatch):
    fake_supabase = FakeSupabase({"chatbots": [{"id": "chatbot-123"}]})
    monkeypatch.setattr(
        "app.services.whatsapp.whatsapp_service.get_supabase_client",
        lambda: fake_supabase,
    )

    service = WhatsAppService(org_id="org-xyz")
    service.supabase = fake_supabase
    service._get_whatsapp_config = MagicMock(
        return_value={"twilio_phone_number": "+15550000000"}
    )
    service._log_message = AsyncMock()

    captured = {}

    class DummyChatService:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        async def process_message(
            self,
            message,
            session_id,
            channel=None,
            end_user_identifier=None,
            requesting_user_id=None,
        ):
            captured["process"] = {
                "message": message,
                "session_id": session_id,
                "channel": channel,
                "end_user_identifier": end_user_identifier,
                "requesting_user_id": requesting_user_id,
            }
            return {"response": "Bot reply"}

    monkeypatch.setattr("app.services.chat.chat_service.ChatService", DummyChatService)

    service.send_message = AsyncMock(return_value={"message_sid": "SM-response"})

    result = await service.process_incoming_message(
        from_number="+15557654321",
        message_body="Hi bot",
        twilio_sid="SM-inbound",
    )

    assert result["success"]
    assert captured["process"]["channel"] == Channel.WHATSAPP
    assert captured["process"]["end_user_identifier"] == "+15557654321"
    assert captured["process"]["session_id"].startswith("whatsapp_")
    assert captured["process"]["requesting_user_id"] == service.org_id

    service.send_message.assert_awaited_once()
    _send_args, send_kwargs = service.send_message.await_args
    assert send_kwargs["requesting_user_id"] == service.org_id

    service._log_message.assert_awaited()
    _log_args, log_kwargs = service._log_message.await_args
    assert log_kwargs["direction"] == "inbound"
    assert log_kwargs["customer_number"] == "+15557654321"
