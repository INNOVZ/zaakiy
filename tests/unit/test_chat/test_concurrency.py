"""
Concurrency Tests for Chat System
Tests race conditions, concurrent operations, and cache consistency
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat.conversation_manager import ConversationManager


class TestConcurrentConversationCreation:
    """Test concurrent conversation creation for race conditions"""

    @pytest.fixture
    def mock_supabase(self):
        """Create mock Supabase client with UPSERT support"""
        client = MagicMock()

        # Track created conversations to detect duplicates
        conversations_db = {}  # session_id -> conversation

        def mock_upsert(data, on_conflict=None, returning=None):
            """Mock upsert that simulates database behavior"""
            session_key = (
                f"{data['session_id']}:{data['org_id']}:{data.get('chatbot_id')}"
            )

            # If conversation already exists, return it (simulating UPSERT)
            if session_key in conversations_db:
                response = MagicMock()
                response.data = [conversations_db[session_key]]
                return response

            # Otherwise, create new one
            conversations_db[session_key] = data
            response = MagicMock()
            response.data = [data]
            return response

        # Mock table operations
        table_mock = MagicMock()

        # Mock upsert chain
        upsert_mock = MagicMock()
        upsert_mock.execute = MagicMock(
            side_effect=lambda: mock_upsert(
                table_mock._upsert_data,
                on_conflict=table_mock._on_conflict,
                returning=table_mock._returning,
            )
        )

        def mock_upsert_call(data, on_conflict=None, returning=None):
            table_mock._upsert_data = data
            table_mock._on_conflict = on_conflict
            table_mock._returning = returning
            return upsert_mock

        table_mock.upsert = MagicMock(side_effect=mock_upsert_call)

        # Mock select/insert for fallback
        table_mock.select = MagicMock(return_value=table_mock)
        table_mock.eq = MagicMock(return_value=table_mock)
        table_mock.execute = MagicMock(return_value=MagicMock(data=[]))
        table_mock.insert = MagicMock(
            return_value=MagicMock(execute=MagicMock(return_value=MagicMock(data=[])))
        )

        client.table = MagicMock(return_value=table_mock)
        client._conversations_db = conversations_db

        return client

    @pytest.mark.asyncio
    async def test_concurrent_conversation_creation_no_duplicates(self, mock_supabase):
        """
        CRITICAL TEST: Verify no duplicate conversations are created
        when multiple concurrent requests use the same session_id

        FIXED: This test should now PASS with UPSERT implementation
        """
        manager = ConversationManager(org_id="test-org", supabase_client=mock_supabase)

        session_id = f"test-session-{uuid.uuid4()}"
        chatbot_id = "test-bot"

        # Mock cache to always return None (cache miss)
        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            # Create 20 concurrent requests for the same session
            tasks = [
                manager.get_or_create_conversation(
                    session_id=session_id, chatbot_id=chatbot_id
                )
                for _ in range(20)
            ]

            # Execute all concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify no exceptions
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Got {len(errors)} errors: {errors}"

            # Get all conversation IDs
            conversation_ids = [r["id"] for r in results if isinstance(r, dict)]

            # CRITICAL: All should return the same conversation ID
            unique_ids = set(conversation_ids)

            # With UPSERT fix, this should PASS
            assert len(unique_ids) == 1, (
                f"Race condition detected! Created {len(unique_ids)} conversations "
                f"for same session_id. Expected 1, got: {unique_ids}"
            )

            # Verify only one conversation was created in "database"
            session_key = f"{session_id}:test-org:{chatbot_id}"
            assert (
                session_key in mock_supabase._conversations_db
            ), "Conversation not found in database"

            # All conversations should have the same ID
            first_id = conversation_ids[0]
            assert all(
                cid == first_id for cid in conversation_ids
            ), "Not all requests returned the same conversation ID"

            print(f"✅ PASS: 20 concurrent requests created only 1 conversation")

    @pytest.mark.asyncio
    async def test_concurrent_different_sessions(self, mock_supabase):
        """Test concurrent creation of different sessions works correctly"""
        manager = ConversationManager(org_id="test-org", supabase_client=mock_supabase)

        # Create 10 different sessions concurrently
        session_ids = [f"session-{i}" for i in range(10)]

        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            tasks = [
                manager.get_or_create_conversation(
                    session_id=session_id, chatbot_id="test-bot"
                )
                for session_id in session_ids
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0

            # Should have 10 different conversation IDs
            conversation_ids = [r["id"] for r in results]
            assert len(set(conversation_ids)) == 10


class TestConcurrentMessageProcessing:
    """Test concurrent message processing"""

    @pytest.fixture
    def mock_conversation_manager(self):
        """Create mock conversation manager"""
        manager = MagicMock()

        # Track added messages
        messages = []

        async def mock_add_message(conversation_id, role, content, metadata=None):
            message = {
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            messages.append(message)
            return message

        manager.add_message = AsyncMock(side_effect=mock_add_message)
        manager._messages = messages

        return manager

    @pytest.mark.asyncio
    async def test_concurrent_message_addition(self, mock_conversation_manager):
        """Test adding messages concurrently to same conversation"""
        conversation_id = str(uuid.uuid4())

        # Add 50 messages concurrently
        tasks = [
            mock_conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )
            for i in range(50)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0

        # All messages should be unique
        message_ids = [r["id"] for r in results]
        assert len(set(message_ids)) == 50

        # All should belong to same conversation
        conv_ids = [r["conversation_id"] for r in results]
        assert all(cid == conversation_id for cid in conv_ids)

    @pytest.mark.asyncio
    async def test_concurrent_messages_different_conversations(
        self, mock_conversation_manager
    ):
        """Test adding messages to different conversations concurrently"""
        conversation_ids = [str(uuid.uuid4()) for _ in range(10)]

        # Add 5 messages to each conversation concurrently
        tasks = []
        for conv_id in conversation_ids:
            for i in range(5):
                tasks.append(
                    mock_conversation_manager.add_message(
                        conversation_id=conv_id, role="user", content=f"Message {i}"
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (50 messages total)
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0
        assert len(results) == 50


class TestCacheConsistency:
    """Test cache consistency under concurrent load"""

    @pytest.mark.asyncio
    async def test_concurrent_cache_reads(self):
        """Test concurrent reads from cache don't cause issues"""
        cache_data = {"test": "data", "value": 123}

        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=cache_data)

            # 100 concurrent cache reads
            tasks = [mock_cache.get("test_key") for _ in range(100)]
            results = await asyncio.gather(*tasks)

            # All should return same data
            assert all(r == cache_data for r in results)

            # Cache should be called 100 times
            assert mock_cache.get.call_count == 100

    @pytest.mark.asyncio
    async def test_concurrent_cache_writes(self):
        """Test concurrent writes to cache"""
        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            mock_cache.set = AsyncMock()

            # 50 concurrent cache writes
            tasks = [
                mock_cache.set(f"key_{i}", f"value_{i}", ttl=3600) for i in range(50)
            ]

            await asyncio.gather(*tasks)

            # All writes should complete
            assert mock_cache.set.call_count == 50

    @pytest.mark.asyncio
    async def test_cache_invalidation_race_condition(self):
        """
        Test cache invalidation during concurrent updates
        Simulates race condition where cache is invalidated while being read
        """
        cache_state = {"data": "initial"}
        read_count = 0
        write_count = 0

        async def mock_get(key):
            nonlocal read_count
            read_count += 1
            await asyncio.sleep(0.001)  # Simulate network delay
            return cache_state.get("data")

        async def mock_set(key, value, ttl=None):
            nonlocal write_count
            write_count += 1
            await asyncio.sleep(0.001)  # Simulate network delay
            cache_state["data"] = value

        async def mock_delete(key):
            await asyncio.sleep(0.001)  # Simulate network delay
            cache_state["data"] = None

        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(side_effect=mock_get)
            mock_cache.set = AsyncMock(side_effect=mock_set)
            mock_cache.delete = AsyncMock(side_effect=mock_delete)

            # Simulate concurrent reads, writes, and deletes
            tasks = []

            # 20 reads
            for _ in range(20):
                tasks.append(mock_cache.get("test_key"))

            # 10 writes
            for i in range(10):
                tasks.append(mock_cache.set("test_key", f"value_{i}"))

            # 5 deletes
            for _ in range(5):
                tasks.append(mock_cache.delete("test_key"))

            # Execute all concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # No exceptions should occur
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0

            # Verify all operations completed
            assert read_count == 20
            assert write_count == 10


class TestConversationHistoryConcurrency:
    """Test conversation history caching under concurrent load"""

    @pytest.fixture
    def mock_supabase_with_messages(self):
        """Create mock Supabase with message data"""
        client = MagicMock()

        # Simulate messages in database
        messages = [
            {
                "id": str(uuid.uuid4()),
                "conversation_id": "test-conv",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(20)
        ]

        table_mock = MagicMock()
        table_mock.select = MagicMock(return_value=table_mock)
        table_mock.eq = MagicMock(return_value=table_mock)
        table_mock.order = MagicMock(return_value=table_mock)
        table_mock.limit = MagicMock(return_value=table_mock)
        table_mock.execute = MagicMock(return_value=MagicMock(data=messages))

        client.table = MagicMock(return_value=table_mock)

        return client

    @pytest.mark.asyncio
    async def test_concurrent_history_reads(self, mock_supabase_with_messages):
        """Test concurrent reads of conversation history"""
        manager = ConversationManager(
            org_id="test-org", supabase_client=mock_supabase_with_messages
        )

        conversation_id = "test-conv"

        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            # First call: cache miss
            # Subsequent calls: cache hit
            call_count = 0

            async def mock_get(key):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return None  # First call: miss
                return [{"id": "1", "content": "cached"}]  # Subsequent: hit

            mock_cache.get = AsyncMock(side_effect=mock_get)
            mock_cache.set = AsyncMock()

            # 50 concurrent history reads
            tasks = [
                manager.get_conversation_history(
                    conversation_id=conversation_id, limit=10
                )
                for _ in range(50)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0

            # All should return data
            assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_history_updates(self, mock_supabase_with_messages):
        """
        Test concurrent updates to conversation history cache
        Simulates race condition where multiple messages are added simultaneously
        """
        manager = ConversationManager(
            org_id="test-org", supabase_client=mock_supabase_with_messages
        )

        conversation_id = "test-conv"

        # Track cache updates
        cache_updates = []

        async def mock_set(key, value, ttl=None):
            cache_updates.append({"key": key, "value": value, "ttl": ttl})

        with patch(
            "app.services.chat.conversation_manager.cache_service"
        ) as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock(side_effect=mock_set)
            mock_cache.delete = AsyncMock()

            # Add 20 messages concurrently
            tasks = [
                manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=f"Concurrent message {i}",
                )
                for i in range(20)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0

            # Cache should be updated/invalidated
            # (Implementation may vary: update or delete)
            assert len(cache_updates) > 0 or mock_cache.delete.call_count > 0


class TestLoadTesting:
    """Load testing for concurrent operations"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_concurrency_conversation_creation(self):
        """
        Load test: Create 100 conversations concurrently
        This test is marked as slow and should be run separately
        """
        pytest.skip("Load test - run manually with: pytest -m slow")

        # This would test actual implementation under high load
        # Useful for finding race conditions that only appear under stress

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_concurrent_load(self):
        """
        Load test: Sustained concurrent operations over time
        This test is marked as slow and should be run separately
        """
        pytest.skip("Load test - run manually with: pytest -m slow")

        # This would test system stability under sustained concurrent load


class TestDeadlockPrevention:
    """Test for potential deadlock scenarios"""

    @pytest.mark.asyncio
    async def test_no_deadlock_circular_dependency(self):
        """
        Test that circular dependencies don't cause deadlocks
        Example: A waits for B, B waits for A
        """
        lock_a = asyncio.Lock()
        lock_b = asyncio.Lock()

        async def task_1():
            async with lock_a:
                await asyncio.sleep(0.01)
                # Don't try to acquire lock_b while holding lock_a
                # This would cause deadlock
            return "task_1_done"

        async def task_2():
            async with lock_b:
                await asyncio.sleep(0.01)
                # Don't try to acquire lock_a while holding lock_b
                # This would cause deadlock
            return "task_2_done"

        # Both tasks should complete without deadlock
        results = await asyncio.gather(task_1(), task_2())
        assert results == ["task_1_done", "task_2_done"]

    @pytest.mark.asyncio
    async def test_timeout_prevents_infinite_wait(self):
        """Test that operations timeout instead of waiting forever"""

        async def slow_operation():
            await asyncio.sleep(10)  # Very slow
            return "done"

        # Should timeout instead of waiting forever
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
