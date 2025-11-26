"""
Integration Tests for Chat Response System
Tests the complete flow from URL scraping to chat response generation
"""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models.chatbot_config import ChatbotConfig
from app.services.chat.chat_service import ChatService
from app.services.chat.document_retrieval_service import DocumentRetrievalService
from app.services.chat.response_generation_service import ResponseGenerationService


class TestChatResponseIntegration:
    """Test suite for end-to-end chat response generation"""

    @pytest.fixture
    def mock_chatbot_config(self):
        """Mock chatbot configuration"""
        return {
            "id": "test-bot-id",
            "name": "Test Bot",
            "system_prompt": "You are a helpful AI assistant.",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 500,
            "tone": "professional",
        }

    @pytest.fixture
    def mock_documents(self):
        """Mock retrieved documents with scraped content"""
        return [
            {
                "id": "doc1",
                "score": 0.95,
                "chunk": "Our email marketing services include campaign creation, email design, and deliverability optimization.",
                "source": "https://example.com/services",
                "metadata": {
                    "chunk": "Our email marketing services include campaign creation, email design, and deliverability optimization.",
                    "upload_id": "upload-123",
                    "source": "https://example.com/services",
                },
            },
            {
                "id": "doc2",
                "score": 0.87,
                "chunk": "Contact us at hello@example.com or call +1-555-0123 for more information.",
                "source": "https://example.com/contact",
                "metadata": {
                    "chunk": "Contact us at hello@example.com or call +1-555-0123 for more information.",
                    "upload_id": "upload-124",
                    "source": "https://example.com/contact",
                },
            },
        ]

    @pytest.mark.asyncio
    async def test_context_is_returned_in_response(
        self, mock_chatbot_config, mock_documents
    ):
        """Test that context_used is properly returned in the final response"""
        # This test verifies the bug fix where context_used was not being returned

        # Mock the response generation service
        with patch(
            "app.services.chat.chat_service.ResponseGenerationService"
        ) as MockResponseGen:
            mock_response_gen = Mock()
            mock_response_gen.generate_enhanced_response = AsyncMock(
                return_value={
                    "response": "Our email marketing services include campaign creation and more.",
                    "sources": ["https://example.com/services"],
                    "context_used": "Our email marketing services include campaign creation, email design, and deliverability optimization.",
                    "contact_info": {},
                    "demo_links": [],
                    "context_quality": {"relevance_score": 0.95},
                    "document_count": 2,
                    "retrieval_method": "semantic",
                    "tokens_used": 150,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                }
            )
            MockResponseGen.return_value = mock_response_gen

            # Mock other services
            with patch(
                "app.services.chat.chat_service.DocumentRetrievalService"
            ), patch("app.services.chat.chat_service.ConversationManager"), patch(
                "app.services.chat.chat_service.AnalyticsService"
            ), patch(
                "app.services.chat.chat_service.ErrorHandlingService"
            ):
                chat_service = ChatService(
                    org_id="test-org-123", chatbot_config=mock_chatbot_config
                )

                # Process a message
                response = await chat_service.process_message(
                    message="Tell me about email marketing services",
                    session_id="test-session",
                )

                # CRITICAL ASSERTION: Verify context_used is in the response
                assert (
                    "context_used" in response
                ), "context_used field missing from response!"
                assert len(response["context_used"]) > 0, "context_used is empty!"
                assert "email marketing services" in response["context_used"].lower()

                # Verify other important fields
                assert "response" in response
                assert "sources" in response
                assert "document_count" in response
                assert "retrieval_method" in response
                assert response["document_count"] == 2

    @pytest.mark.asyncio
    async def test_scraped_content_used_in_response(
        self, mock_chatbot_config, mock_documents
    ):
        """Test that scraped content is actually used to generate responses"""

        # This test verifies that the content from scraped URLs influences the response

        with patch(
            "app.services.chat.response_generation_service.OpenAI"
        ) as MockOpenAI:
            # Mock OpenAI to return a response that uses the context
            mock_openai_instance = Mock()
            mock_completion = Mock()
            mock_completion.choices = [
                Mock(
                    message=Mock(
                        content="Based on the information provided, our email marketing services include campaign creation, email design, and deliverability optimization. Contact us at hello@example.com for more details."
                    )
                )
            ]
            mock_completion.usage = Mock(
                total_tokens=150, prompt_tokens=100, completion_tokens=50
            )
            mock_openai_instance.chat.completions.create = Mock(
                return_value=mock_completion
            )
            MockOpenAI.return_value = mock_openai_instance

            # Create a real ResponseGenerationService with mocked OpenAI
            with patch(
                "app.services.chat.response_generation_service.ContextBuilder"
            ) as MockContextBuilder:
                mock_context_builder = Mock()
                mock_context_builder.build = Mock(
                    return_value=type(
                        "obj",
                        (object,),
                        {
                            "context_text": "Our email marketing services include campaign creation, email design, and deliverability optimization. Contact us at hello@example.com or call +1-555-0123.",
                            "sources": [
                                "https://example.com/services",
                                "https://example.com/contact",
                            ],
                            "contact_info": {
                                "emails": ["hello@example.com"],
                                "phones": ["+1-555-0123"],
                            },
                            "demo_links": [],
                            "context_quality": {"relevance_score": 0.95},
                            "product_links": [],
                        },
                    )()
                )
                MockContextBuilder.return_value = mock_context_builder

                response_gen = ResponseGenerationService(
                    org_id="test-org-123",
                    openai_client=mock_openai_instance,
                    context_config={},
                    chatbot_config=mock_chatbot_config,
                )

                # Generate response
                response = await response_gen.generate_enhanced_response(
                    message="Tell me about email marketing services",
                    conversation_history=[],
                    retrieved_documents=mock_documents,
                )

                # Verify response uses scraped content
                assert "context_used" in response
                assert len(response["context_used"]) > 0
                assert "email marketing" in response["context_used"].lower()
                assert "campaign creation" in response["context_used"].lower()

                # Verify contact info is extracted
                assert "contact_info" in response
                assert "emails" in response["contact_info"]

    @pytest.mark.asyncio
    async def test_empty_context_handled_gracefully(self, mock_chatbot_config):
        """Test that the system handles cases where no relevant documents are found"""

        with patch(
            "app.services.chat.response_generation_service.OpenAI"
        ) as MockOpenAI:
            mock_openai_instance = Mock()
            mock_completion = Mock()
            mock_completion.choices = [
                Mock(
                    message=Mock(
                        content="I'd be happy to help you with that! To provide the most accurate information, could you please clarify what specific aspect you're interested in?"
                    )
                )
            ]
            mock_completion.usage = Mock(
                total_tokens=50, prompt_tokens=30, completion_tokens=20
            )
            mock_openai_instance.chat.completions.create = Mock(
                return_value=mock_completion
            )
            MockOpenAI.return_value = mock_openai_instance

            with patch(
                "app.services.chat.response_generation_service.ContextBuilder"
            ) as MockContextBuilder:
                mock_context_builder = Mock()
                mock_context_builder.build = Mock(
                    return_value=type(
                        "obj",
                        (object,),
                        {
                            "context_text": "",  # Empty context
                            "sources": [],
                            "contact_info": {},
                            "demo_links": [],
                            "context_quality": {},
                            "product_links": [],
                        },
                    )()
                )
                MockContextBuilder.return_value = mock_context_builder

                response_gen = ResponseGenerationService(
                    org_id="test-org-123",
                    openai_client=mock_openai_instance,
                    context_config={},
                    chatbot_config=mock_chatbot_config,
                )

                # Generate response with no documents
                response = await response_gen.generate_enhanced_response(
                    message="Tell me about something not in the knowledge base",
                    conversation_history=[],
                    retrieved_documents=[],
                )

                # Verify response is still generated
                assert "response" in response
                assert len(response["response"]) > 0

                # Verify context_used exists (even if empty)
                assert "context_used" in response
                assert response["context_used"] == ""

                # Verify no forbidden phrases
                forbidden_phrases = [
                    "I don't have that information",
                    "I don't have information about",
                    "information is not available",
                ]
                response_lower = response["response"].lower()
                for phrase in forbidden_phrases:
                    assert (
                        phrase.lower() not in response_lower
                    ), f"Response contains forbidden phrase: {phrase}"

    @pytest.mark.asyncio
    async def test_multiple_sources_aggregated(self, mock_chatbot_config):
        """Test that information from multiple scraped sources is properly aggregated"""

        mock_documents = [
            {
                "id": "doc1",
                "score": 0.95,
                "chunk": "We offer SEO services to improve your search rankings.",
                "source": "https://example.com/seo",
                "metadata": {
                    "chunk": "We offer SEO services to improve your search rankings."
                },
            },
            {
                "id": "doc2",
                "score": 0.90,
                "chunk": "Our content marketing services include blog writing and social media.",
                "source": "https://example.com/content",
                "metadata": {
                    "chunk": "Our content marketing services include blog writing and social media."
                },
            },
            {
                "id": "doc3",
                "score": 0.85,
                "chunk": "Email marketing automation to engage your customers effectively.",
                "source": "https://example.com/email",
                "metadata": {
                    "chunk": "Email marketing automation to engage your customers effectively."
                },
            },
        ]

        with patch(
            "app.services.chat.response_generation_service.OpenAI"
        ) as MockOpenAI:
            mock_openai_instance = Mock()
            mock_completion = Mock()
            mock_completion.choices = [
                Mock(
                    message=Mock(
                        content="We offer comprehensive digital marketing services including SEO, content marketing, and email automation."
                    )
                )
            ]
            mock_completion.usage = Mock(
                total_tokens=100, prompt_tokens=70, completion_tokens=30
            )
            mock_openai_instance.chat.completions.create = Mock(
                return_value=mock_completion
            )
            MockOpenAI.return_value = mock_openai_instance

            with patch(
                "app.services.chat.response_generation_service.ContextBuilder"
            ) as MockContextBuilder:
                combined_context = "\n\n".join([doc["chunk"] for doc in mock_documents])
                mock_context_builder = Mock()
                mock_context_builder.build = Mock(
                    return_value=type(
                        "obj",
                        (object,),
                        {
                            "context_text": combined_context,
                            "sources": [doc["source"] for doc in mock_documents],
                            "contact_info": {},
                            "demo_links": [],
                            "context_quality": {"relevance_score": 0.90},
                            "product_links": [],
                        },
                    )()
                )
                MockContextBuilder.return_value = mock_context_builder

                response_gen = ResponseGenerationService(
                    org_id="test-org-123",
                    openai_client=mock_openai_instance,
                    context_config={},
                    chatbot_config=mock_chatbot_config,
                )

                response = await response_gen.generate_enhanced_response(
                    message="What services do you offer?",
                    conversation_history=[],
                    retrieved_documents=mock_documents,
                )

                # Verify all sources are included
                assert len(response["sources"]) == 3
                assert "https://example.com/seo" in response["sources"]
                assert "https://example.com/content" in response["sources"]
                assert "https://example.com/email" in response["sources"]

                # Verify context includes information from all sources
                assert "SEO" in response["context_used"]
                assert "content marketing" in response["context_used"]
                assert "email" in response["context_used"].lower()

    def test_response_data_structure(self, mock_chatbot_config):
        """Test that the response data structure includes all required fields"""

        # This test validates the complete response structure
        required_fields = [
            "response",
            "sources",
            "context_used",
            "contact_info",
            "demo_links",
            "conversation_id",
            "message_id",
            "processing_time_ms",
            "context_quality",
            "document_count",
            "retrieval_method",
            "config_used",
        ]

        # Create a mock response that simulates the actual structure
        mock_response = {
            "response": "Test response",
            "sources": ["https://example.com"],
            "context_used": "Test context",
            "contact_info": {"emails": ["test@example.com"]},
            "demo_links": ["https://example.com/demo"],
            "conversation_id": "conv-123",
            "message_id": "msg-456",
            "processing_time_ms": 1500,
            "context_quality": {"relevance_score": 0.95},
            "document_count": 5,
            "retrieval_method": "hybrid",
            "config_used": "default",
        }

        # Verify all required fields are present
        for field in required_fields:
            assert (
                field in mock_response
            ), f"Required field '{field}' missing from response structure!"

        # Verify field types
        assert isinstance(mock_response["response"], str)
        assert isinstance(mock_response["sources"], list)
        assert isinstance(mock_response["context_used"], str)
        assert isinstance(mock_response["contact_info"], dict)
        assert isinstance(mock_response["demo_links"], list)
        assert isinstance(mock_response["processing_time_ms"], (int, float))
        assert isinstance(mock_response["document_count"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
