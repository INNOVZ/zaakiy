"""
Test Context Engineering Alignment
Tests that backend properly uses context engineering settings from frontend/database
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Stub external dependencies
if "pinecone" not in sys.modules:
    pinecone_stub = types.ModuleType("pinecone")

    class _StubPinecone:
        def __init__(self, *args, **kwargs):
            pass

        def Index(self, *args, **kwargs):
            class _Index:
                def describe_index_stats(self):
                    return {}

            return _Index()

    pinecone_stub.Pinecone = _StubPinecone
    sys.modules["pinecone"] = pinecone_stub

from app.services.chat.response_generation_service import ResponseGenerationService


class MockContextConfig:
    """Mock context config object"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_service(context_config=None, chatbot_config=None):
    """Create ResponseGenerationService with optional context_config"""
    if chatbot_config is None:
        chatbot_config = {
            "name": "Test Assistant",
            "tone": "friendly",
            "system_prompt": "You are a helpful assistant.",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 300,
        }

    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    mock_openai.chat.completions.create = MagicMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))],
            usage=MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50),
        )
    )

    return ResponseGenerationService(
        org_id="test-org",
        openai_client=mock_openai,
        context_config=context_config,
        chatbot_config=chatbot_config,
    )


class TestMaxContextLength:
    """Test max_context_length from context_config"""

    def test_uses_context_config_max_context_length(self):
        """Test that max_context_length is taken from context_config"""
        context_config = MockContextConfig(max_context_length=2000)
        service = _build_service(context_config=context_config)

        assert (
            service.max_context_length == 2000
        ), "Should use context_config.max_context_length"

    def test_uses_dict_context_config_max_context_length(self):
        """Test that max_context_length works with dict context_config"""
        context_config = {"max_context_length": 3000}
        service = _build_service(context_config=context_config)

        assert (
            service.max_context_length == 3000
        ), "Should use dict context_config max_context_length"

    def test_defaults_to_4000_when_no_context_config(self):
        """Test that defaults to 4000 when context_config is None"""
        service = _build_service(context_config=None)

        assert service.max_context_length == 4000, "Should default to 4000"

    def test_defaults_to_4000_when_no_max_context_length(self):
        """Test that defaults to 4000 when context_config has no max_context_length"""
        context_config = MockContextConfig(some_other_field="value")
        service = _build_service(context_config=context_config)

        assert service.max_context_length == 4000, "Should default to 4000"


class TestFinalContextChunks:
    """Test final_context_chunks limit"""

    def test_applies_final_context_chunks_limit(self):
        """Test that final_context_chunks limits the number of chunks"""
        context_config = MockContextConfig(final_context_chunks=6)
        service = _build_service(context_config=context_config)

        # Create mock documents
        documents = [
            {"chunk": f"Chunk {i}", "source": f"source_{i}", "score": 0.9 - i * 0.1}
            for i in range(10)
        ]

        context_data = service._build_context(documents)

        # Count chunks in context_text (approximate by checking separators)
        context_text = context_data.get("context_text", "")
        chunk_count = context_text.count("---") + 1 if context_text else 0

        # Should be limited to 6 chunks (or less if compressed)
        assert chunk_count <= 6, f"Should limit to 6 chunks, got {chunk_count}"

    def test_respects_final_context_chunks_from_dict(self):
        """Test that final_context_chunks works with dict context_config"""
        context_config = {"final_context_chunks": 3}
        service = _build_service(context_config=context_config)

        documents = [
            {"chunk": f"Chunk {i}", "source": f"source_{i}", "score": 0.9 - i * 0.1}
            for i in range(10)
        ]

        context_data = service._build_context(documents)
        context_text = context_data.get("context_text", "")
        chunk_count = context_text.count("---") + 1 if context_text else 0

        assert chunk_count <= 3, f"Should limit to 3 chunks, got {chunk_count}"

    def test_no_limit_when_final_context_chunks_not_set(self):
        """Test that all chunks are used when final_context_chunks is not set"""
        context_config = MockContextConfig()  # No final_context_chunks
        service = _build_service(context_config=context_config)

        # Use longer chunks that won't be filtered out (must be > 10 chars)
        documents = [
            {
                "chunk": f"This is chunk number {i} with sufficient content to pass filtering.",
                "source": f"source_{i}",
                "score": 0.9 - i * 0.1,
            }
            for i in range(5)
        ]

        context_data = service._build_context(documents)
        context_text = context_data.get("context_text", "")

        # Should use all chunks (or be limited by max_context_length)
        assert len(context_text) > 0, "Should have context text"
        # Should contain multiple chunks (indicated by separators or content)
        assert "chunk number" in context_text, "Should contain chunk content"

    def test_prioritizes_contact_chunks(self):
        """Test that contact chunks are prioritized and counted separately"""
        context_config = MockContextConfig(final_context_chunks=4)
        service = _build_service(context_config=context_config)

        # Create documents with some contact info
        documents = [
            {"chunk": "Phone: 123-456-7890", "source": "contact", "score": 0.95},
            {"chunk": "Email: test@example.com", "source": "contact", "score": 0.94},
            {"chunk": "Regular chunk 1", "source": "regular", "score": 0.8},
            {"chunk": "Regular chunk 2", "source": "regular", "score": 0.7},
            {"chunk": "Regular chunk 3", "source": "regular", "score": 0.6},
            {"chunk": "Regular chunk 4", "source": "regular", "score": 0.5},
        ]

        context_data = service._build_context(documents)

        # Should include contact info in contact_info dict
        contact_info = context_data.get("contact_info", {})
        assert (
            len(contact_info.get("phones", [])) > 0
            or len(contact_info.get("emails", [])) > 0
        ), "Should extract contact information"


class TestBusinessContext:
    """Test business_context integration"""

    def test_integrates_business_context_into_system_prompt(self):
        """Test that business_context is added to system prompt"""
        context_config = MockContextConfig(
            business_context="We are a SaaS platform providing AI solutions."
        )
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        assert (
            "BUSINESS CONTEXT" in system_prompt
        ), "Should include BUSINESS CONTEXT section"
        assert (
            "We are a SaaS platform" in system_prompt
        ), "Should include business context text"

    def test_business_context_from_dict(self):
        """Test that business_context works with dict context_config"""
        context_config = {"business_context": "We are a tech startup."}
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        assert "BUSINESS CONTEXT" in system_prompt, "Should include BUSINESS CONTEXT"
        assert "tech startup" in system_prompt, "Should include business context text"

    def test_no_business_context_when_not_set(self):
        """Test that no BUSINESS CONTEXT section when not set"""
        context_config = MockContextConfig()  # No business_context
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        # Should not have BUSINESS CONTEXT section
        assert (
            "BUSINESS CONTEXT:" not in system_prompt
            or system_prompt.count("BUSINESS CONTEXT:") == 0
        ), "Should not have BUSINESS CONTEXT when not set"

    def test_empty_business_context_not_added(self):
        """Test that empty business_context is not added"""
        context_config = MockContextConfig(business_context="")
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        # Should not have BUSINESS CONTEXT section with empty string
        assert (
            "\n\nBUSINESS CONTEXT:\n" not in system_prompt
        ), "Should not add empty BUSINESS CONTEXT section"


class TestSpecializedInstructions:
    """Test specialized_instructions integration"""

    def test_integrates_specialized_instructions_into_system_prompt(self):
        """Test that specialized_instructions is added to system prompt"""
        context_config = MockContextConfig(
            specialized_instructions="Always be concise and professional."
        )
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        assert (
            "SPECIALIZED INSTRUCTIONS" in system_prompt
        ), "Should include SPECIALIZED INSTRUCTIONS section"
        assert (
            "Always be concise" in system_prompt
        ), "Should include specialized instructions text"

    def test_specialized_instructions_from_dict(self):
        """Test that specialized_instructions works with dict context_config"""
        context_config = {"specialized_instructions": "Use emojis sparingly."}
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        assert (
            "SPECIALIZED INSTRUCTIONS" in system_prompt
        ), "Should include SPECIALIZED INSTRUCTIONS"
        assert (
            "emojis sparingly" in system_prompt
        ), "Should include specialized instructions text"

    def test_both_business_context_and_specialized_instructions(self):
        """Test that both business_context and specialized_instructions can be used together"""
        context_config = MockContextConfig(
            business_context="We are a SaaS platform.",
            specialized_instructions="Always be helpful.",
        )
        service = _build_service(context_config=context_config)

        context_data = {"context_text": "Some context"}
        system_prompt = service._create_system_prompt(context_data)

        assert "BUSINESS CONTEXT" in system_prompt, "Should have BUSINESS CONTEXT"
        assert (
            "SPECIALIZED INSTRUCTIONS" in system_prompt
        ), "Should have SPECIALIZED INSTRUCTIONS"
        assert "SaaS platform" in system_prompt, "Should have business context"
        assert (
            "Always be helpful" in system_prompt
        ), "Should have specialized instructions"


class TestContextChunksLimit:
    """Test that final_context_chunks properly limits chunks"""

    def test_limits_to_specified_number(self):
        """Test that chunks are limited to final_context_chunks"""
        context_config = MockContextConfig(final_context_chunks=2)
        service = _build_service(context_config=context_config)

        # Create 5 documents
        documents = [
            {
                "chunk": f"This is chunk number {i} with some content.",
                "source": f"source_{i}",
                "score": 0.9 - i * 0.1,
            }
            for i in range(5)
        ]

        context_data = service._build_context(documents)
        context_text = context_data.get("context_text", "")

        # Count actual chunks by checking for chunk content
        # With limit of 2, we should see at most 2 chunks
        chunk_indicators = [f"chunk number {i}" for i in range(5)]
        found_chunks = sum(
            1 for indicator in chunk_indicators if indicator in context_text
        )

        assert (
            found_chunks <= 2
        ), f"Should limit to 2 chunks, but found {found_chunks} chunks in context"


class TestMaxContextLengthUsage:
    """Test that max_context_length is used in context building"""

    def test_respects_max_context_length(self):
        """Test that context text respects max_context_length"""
        context_config = MockContextConfig(max_context_length=1000)
        service = _build_service(context_config=context_config)

        # Create long chunks
        documents = [
            {"chunk": "A" * 500, "source": f"source_{i}", "score": 0.9}
            for i in range(10)
        ]

        context_data = service._build_context(documents)
        context_text = context_data.get("context_text", "")

        # Context should be limited by max_context_length (with some buffer)
        assert (
            len(context_text) <= context_config.max_context_length + 100
        ), f"Context length {len(context_text)} should respect max_context_length {context_config.max_context_length}"


class TestContextConfigIntegration:
    """Test full integration of context_config settings"""

    def test_multiple_settings_together(self):
        """Test that multiple context_config settings work together"""
        context_config = MockContextConfig(
            max_context_length=3000,
            final_context_chunks=5,
            business_context="Test business",
            specialized_instructions="Test instructions",
        )
        service = _build_service(context_config=context_config)

        # Verify all settings are applied
        assert service.max_context_length == 3000, "Should use max_context_length"

        # Test system prompt integration
        context_data = {"context_text": "Test context"}
        system_prompt = service._create_system_prompt(context_data)

        assert "BUSINESS CONTEXT" in system_prompt, "Should have business context"
        assert (
            "SPECIALIZED INSTRUCTIONS" in system_prompt
        ), "Should have specialized instructions"

        # Test chunk limiting
        documents = [
            {"chunk": f"Chunk {i}", "source": f"source_{i}", "score": 0.9 - i * 0.1}
            for i in range(10)
        ]
        context_data = service._build_context(documents)
        context_text = context_data.get("context_text", "")
        chunk_count = context_text.count("---") + 1 if context_text else 0

        assert chunk_count <= 5, f"Should limit to 5 chunks, got {chunk_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
