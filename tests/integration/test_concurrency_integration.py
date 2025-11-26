"""
Integration Tests for Concurrency
Tests actual concurrent behavior with real components
"""
import asyncio
import time
import uuid
from typing import Dict, List

import pytest

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestRealConcurrentConversations:
    """Integration tests for concurrent conversation operations"""

    @pytest.mark.asyncio
    async def test_concurrent_conversation_creation_integration(self):
        """
        Integration test: Create conversations concurrently with real database
        This test requires actual database connection
        """
        # Skip if no database connection
        pytest.skip("Requires database connection - run in integration environment")

        from app.services.chat.conversation_manager import ConversationManager
        from app.services.storage.supabase_client import get_supabase_client

        supabase = get_supabase_client()
        manager = ConversationManager(
            org_id="test-org-integration", supabase_client=supabase
        )

        session_id = f"integration-test-{uuid.uuid4()}"

        # Create 10 concurrent requests for same session
        tasks = [
            manager.get_or_create_conversation(
                session_id=session_id, chatbot_id="test-bot"
            )
            for _ in range(10)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            pytest.fail(f"Got {len(errors)} errors: {errors[0]}")

        # All should return same conversation ID
        conversation_ids = [r["id"] for r in results]
        unique_ids = set(conversation_ids)

        assert len(unique_ids) == 1, (
            f"Race condition! Created {len(unique_ids)} conversations. "
            f"IDs: {unique_ids}"
        )

        # Verify in database
        response = (
            supabase.table("conversations")
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )

        assert len(response.data) == 1, (
            f"Database has {len(response.data)} conversations for session. "
            "Expected 1!"
        )

        print(f"✅ Created 1 conversation from 10 concurrent requests in {elapsed:.2f}s")

        # Cleanup
        supabase.table("conversations").delete().eq("session_id", session_id).execute()


class TestConcurrentChatRequests:
    """Integration tests for concurrent chat requests"""

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests_same_session(self):
        """
        Test multiple concurrent chat requests for same session
        Simulates real-world scenario of user clicking send multiple times
        """
        pytest.skip("Requires full chat service - run in integration environment")

        from app.services.chat.chat_service import ChatService

        chatbot_config = {
            "id": "test-bot",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        }

        chat_service = ChatService(org_id="test-org", chatbot_config=chatbot_config)

        session_id = f"concurrent-test-{uuid.uuid4()}"

        # Send 5 messages concurrently
        messages = [
            "What are your products?",
            "How do I contact you?",
            "What are your prices?",
            "Tell me about your services",
            "Where are you located?",
        ]

        tasks = [
            chat_service.process_message(message=msg, session_id=session_id)
            for msg in messages
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"⚠️  {len(errors)} requests failed: {errors[0]}")

        # Count successful responses
        successful = [r for r in results if isinstance(r, dict) and "response" in r]

        print(f"✅ Processed {len(successful)}/5 concurrent requests in {elapsed:.2f}s")
        print(f"   Average: {elapsed/len(successful):.2f}s per request")

        assert len(successful) >= 4, "At least 4 requests should succeed"

    @pytest.mark.asyncio
    async def test_concurrent_chat_different_sessions(self):
        """
        Test concurrent chat requests from different sessions
        Simulates multiple users chatting simultaneously
        """
        pytest.skip("Requires full chat service - run in integration environment")

        from app.services.chat.chat_service import ChatService

        chatbot_config = {
            "id": "test-bot",
            "model": "gpt-3.5-turbo",
        }

        chat_service = ChatService(org_id="test-org", chatbot_config=chatbot_config)

        # Simulate 10 different users
        tasks = [
            chat_service.process_message(
                message="Hello, what are your products?", session_id=f"user-{i}"
            )
            for i in range(10)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successful = [r for r in results if isinstance(r, dict)]

        print(f"✅ Handled {len(successful)}/10 concurrent users in {elapsed:.2f}s")
        print(f"   Average: {elapsed/len(successful):.2f}s per user")

        assert len(successful) >= 8, "At least 8 users should get responses"


class TestCacheConsistencyIntegration:
    """Integration tests for cache consistency with real Redis"""

    @pytest.mark.asyncio
    async def test_cache_consistency_under_load(self):
        """
        Test cache remains consistent under concurrent load
        Requires real Redis connection
        """
        pytest.skip("Requires Redis connection - run in integration environment")

        from app.services.shared import cache_service

        if not cache_service:
            pytest.skip("Cache service not available")

        cache_key = f"test-consistency-{uuid.uuid4()}"

        # Concurrent writes to same key
        async def write_value(value: int):
            await cache_service.set(cache_key, value, ttl=60)
            await asyncio.sleep(0.01)  # Small delay
            cached = await cache_service.get(cache_key)
            return cached

        # 20 concurrent writes
        tasks = [write_value(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # Final value should be one of the written values
        final_value = await cache_service.get(cache_key)
        assert final_value in range(20), "Cache value is corrupted"

        # Cleanup
        await cache_service.delete(cache_key)

        print(f"✅ Cache remained consistent under 20 concurrent writes")


class TestPerformanceUnderConcurrency:
    """Performance tests under concurrent load"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_response_time_under_load(self):
        """
        Measure response time degradation under concurrent load
        This is a load test - run separately
        """
        pytest.skip("Load test - run manually with: pytest -m slow")

        from app.services.chat.chat_service import ChatService

        chatbot_config = {"id": "test-bot", "model": "gpt-3.5-turbo"}
        chat_service = ChatService(org_id="test-org", chatbot_config=chatbot_config)

        async def measure_response_time(session_id: str) -> float:
            start = time.time()
            await chat_service.process_message(
                message="What are your products?", session_id=session_id
            )
            return time.time() - start

        # Test with increasing concurrency
        concurrency_levels = [1, 5, 10, 20, 50]
        results = {}

        for concurrency in concurrency_levels:
            tasks = [
                measure_response_time(f"load-test-{i}") for i in range(concurrency)
            ]

            times = await asyncio.gather(*tasks, return_exceptions=True)
            valid_times = [t for t in times if isinstance(t, float)]

            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                results[concurrency] = avg_time
                print(f"Concurrency {concurrency}: {avg_time:.2f}s average")

        # Response time shouldn't degrade too much
        if 1 in results and 20 in results:
            degradation = results[20] / results[1]
            assert degradation < 3.0, (
                f"Response time degraded {degradation:.1f}x under load. "
                "System may not scale well."
            )


class TestRaceConditionScenarios:
    """Test specific race condition scenarios"""

    @pytest.mark.asyncio
    async def test_message_ordering_race_condition(self):
        """
        Test that messages maintain correct order despite concurrent processing
        """
        pytest.skip("Requires database - run in integration environment")

        from app.services.chat.conversation_manager import ConversationManager
        from app.services.storage.supabase_client import get_supabase_client

        supabase = get_supabase_client()
        manager = ConversationManager(org_id="test-org", supabase_client=supabase)

        # Create conversation
        conversation = await manager.get_or_create_conversation(
            session_id=f"order-test-{uuid.uuid4()}", chatbot_id="test-bot"
        )

        # Add 20 messages concurrently
        tasks = [
            manager.add_message(
                conversation_id=conversation["id"],
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                metadata={"sequence": i},
            )
            for i in range(20)
        ]

        await asyncio.gather(*tasks)

        # Retrieve history
        history = await manager.get_conversation_history(
            conversation_id=conversation["id"], limit=20
        )

        # All messages should be present
        assert len(history) == 20, f"Expected 20 messages, got {len(history)}"

        # Messages should have unique IDs
        message_ids = [msg["id"] for msg in history]
        assert len(set(message_ids)) == 20, "Duplicate message IDs detected!"

        print("✅ All 20 concurrent messages saved with unique IDs")

        # Cleanup
        supabase.table("conversations").delete().eq("id", conversation["id"]).execute()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
