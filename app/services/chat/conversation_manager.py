"""
Conversation Management Service
Handles conversation creation, message storage, and conversation history
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.services.shared import cache_service
from app.services.storage.supabase_client import run_supabase

logger = logging.getLogger(__name__)


class ConversationManagerError(Exception):
    """Exception for conversation management errors"""


class ConversationManager:
    """Handles conversation lifecycle and message persistence"""

    def __init__(self, org_id: str, supabase_client):
        self.org_id = org_id
        self.supabase = supabase_client

    async def get_or_create_conversation(
        self,
        session_id: str,
        chatbot_id: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get existing conversation or create new one with write-through caching"""
        # Cache key for this conversation
        cache_key = f"conversation:session:{self.org_id}:{session_id}"

        try:
            # Check cache first (Cache-Aside pattern) with error handling
            cached_conversation = None
            try:
                if cache_service:
                    cached_conversation = await cache_service.get(cache_key)
                    if cached_conversation:
                        logger.debug(
                            "Retrieved conversation from cache for session %s",
                            session_id,
                        )
                        return cached_conversation
            except Exception as cache_error:
                logger.warning(
                    "Cache retrieval failed, continuing without cache: %s", cache_error
                )

            # Check database for existing conversation
            def _query_conversation():
                query = (
                    self.supabase.table("conversations")
                    .select("*")
                    .eq("session_id", session_id)
                    .eq("org_id", self.org_id)
                )

                # If chatbot_id is provided, ensure we lookup by it as well
                if chatbot_id:
                    query = query.eq("chatbot_id", chatbot_id)

                return query.execute()

            response = await run_supabase(_query_conversation)

            if response.data:
                conversation = response.data[0]
                # Cache the found conversation
                try:
                    if cache_service:
                        await cache_service.set(cache_key, conversation)
                except Exception as cache_error:
                    logger.warning("Cache set failed: %s", cache_error)
                logger.debug("Found existing conversation for session %s", session_id)
                return conversation

            # Create new conversation with consistent metadata handling
            conversation_data = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "org_id": self.org_id,
                "chatbot_id": chatbot_id,
                "channel": channel or "website",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "message_count": 0,
                "status": "active",
            }

            # Handle metadata consistently: merge channel into metadata if provided
            if metadata:
                # If metadata is provided, ensure channel is included if channel was specified
                if channel and "channel" not in metadata:
                    metadata = {**metadata, "channel": channel}
                conversation_data["metadata"] = metadata
            elif channel:
                # If only channel is provided, store it in metadata for consistency
                conversation_data["metadata"] = {"channel": channel}
            else:
                # No metadata or channel, use empty dict for consistency
                conversation_data["metadata"] = {}

            # Insert into database (Write-Through caching)
            response = await run_supabase(
                lambda: self.supabase.table("conversations")
                .insert(conversation_data)
                .execute()
            )

            if response.data:
                new_conversation = response.data[0]
                # Cache the new conversation
                try:
                    if cache_service:
                        await cache_service.set(cache_key, new_conversation)
                except Exception as cache_error:
                    logger.warning("Cache set failed: %s", cache_error)
                logger.info("Created new conversation for session %s", session_id)
                return new_conversation
            else:
                raise ConversationManagerError("Failed to create conversation")

        except Exception as e:
            logger.error("Error in get_or_create_conversation: %s", e)
            raise ConversationManagerError(
                f"Conversation management failed: {e}"
            ) from e

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add message to conversation with write-through caching"""
        try:
            message_data = {
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Insert message into database
            response = await run_supabase(
                lambda: self.supabase.table("messages").insert(message_data).execute()
            )

            if response.data:
                message = response.data[0]

                # Update conversation's updated_at
                # Note: We skip message_count update for now to avoid RPC issues
                await run_supabase(
                    lambda: self.supabase.table("conversations")
                    .update({"updated_at": datetime.now(timezone.utc).isoformat()})
                    .eq("id", conversation_id)
                    .execute()
                )

                # Update caches
                await self._update_message_caches(conversation_id, message)

                logger.debug(
                    "Added %s message to conversation %s", role, conversation_id
                )
                return message
            else:
                raise ConversationManagerError("Failed to insert message")

        except Exception as e:
            logger.error("Error adding message: %s", e)
            raise ConversationManagerError(f"Failed to add message: {e}") from e

    async def _update_message_caches(
        self, conversation_id: str, message_data: Dict[str, Any]
    ):
        """Update conversation-related caches after message addition"""
        try:
            # Update conversation history cache
            history_cache_key = f"conversation_history:{self.org_id}:{conversation_id}"

            # Get current cached history
            cached_history = None
            try:
                if cache_service:
                    cached_history = await cache_service.get(history_cache_key)
            except Exception as cache_error:
                logger.warning("Cache get failed: %s", cache_error)
            if cached_history:
                # Add new message to cached history
                cached_history.append(message_data)
                # Keep only last 50 messages in cache
                if len(cached_history) > 50:
                    cached_history = cached_history[-50:]
                try:
                    if cache_service:
                        # Remove ttl parameter
                        await cache_service.set(history_cache_key, cached_history)
                except Exception as cache_error:
                    logger.warning("Cache set failed: %s", cache_error)

            # Invalidate conversation cache to force refresh
            conversation_cache_key = f"conversation:id:{self.org_id}:{conversation_id}"
            try:
                if cache_service:
                    await cache_service.delete(conversation_cache_key)
            except Exception as cache_error:
                logger.warning("Cache delete failed: %s", cache_error)

        except Exception as e:
            logger.warning("Failed to update message caches: %s", e)

    async def get_conversation_history(
        self, conversation_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversation history with multi-level caching"""
        cache_key = f"conversation_history:{self.org_id}:{conversation_id}"

        try:
            # Check cache first
            cached_history = None
            try:
                if cache_service:
                    cached_history = await cache_service.get(cache_key)
            except Exception as cache_error:
                logger.warning("Cache get failed: %s", cache_error)
            if cached_history:
                # Return limited history from cache
                limited_history = (
                    cached_history[-limit:]
                    if len(cached_history) > limit
                    else cached_history
                )
                logger.debug(
                    "Retrieved conversation history from cache (conversation %s)",
                    conversation_id,
                )
                return limited_history

            # Fetch from database
            response = await run_supabase(
                lambda: (
                    self.supabase.table("messages")
                    .select("*")
                    .eq("conversation_id", conversation_id)
                    .order("created_at", desc=False)
                    .limit(limit)
                    .execute()
                )
            )

            history = response.data or []

            # Cache the history
            try:
                if cache_service:
                    # Remove ttl parameter
                    await cache_service.set(cache_key, history)
            except Exception as cache_error:
                logger.warning("Cache set failed: %s", cache_error)

            logger.debug(
                "Fetched conversation history from database (conversation %s)",
                conversation_id,
            )
            return history

        except Exception as e:
            logger.error("Error getting conversation history: %s", e)
            return []

    async def update_conversation_metadata(
        self, conversation_id: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update conversation metadata"""
        try:
            response = await run_supabase(
                lambda: (
                    self.supabase.table("conversations")
                    .update(
                        {
                            "metadata": metadata,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    .eq("id", conversation_id)
                    .eq("org_id", self.org_id)
                    .execute()
                )
            )

            if response.data:
                # Invalidate related caches
                try:
                    if cache_service:
                        await cache_service.delete(
                            f"conversation:id:{self.org_id}:{conversation_id}"
                        )
                except Exception as cache_error:
                    logger.warning("Cache delete failed: %s", cache_error)
                logger.debug("Updated metadata for conversation %s", conversation_id)
                return True
            return False

        except Exception as e:
            logger.error("Error updating conversation metadata: %s", e)
            return False

    async def close_conversation(self, conversation_id: str) -> bool:
        """Mark conversation as closed"""
        try:
            response = await run_supabase(
                lambda: (
                    self.supabase.table("conversations")
                    .update(
                        {
                            "status": "closed",
                            "closed_at": datetime.now(timezone.utc).isoformat(),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    .eq("id", conversation_id)
                    .eq("org_id", self.org_id)
                    .execute()
                )
            )

            if response.data:
                # Invalidate caches
                try:
                    if cache_service:
                        await cache_service.delete(
                            f"conversation:id:{self.org_id}:{conversation_id}"
                        )
                        await cache_service.delete(
                            f"conversation_history:{self.org_id}:{conversation_id}"
                        )
                except Exception as cache_error:
                    logger.warning("Cache delete failed: %s", cache_error)
                logger.info("Closed conversation %s", conversation_id)
                return True
            return False

        except Exception as e:
            logger.error("Error closing conversation: %s", e)
            return False

    async def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation"""
        try:
            # Get message count by role
            response = await run_supabase(
                lambda: (
                    self.supabase.table("messages")
                    .select("role")
                    .eq("conversation_id", conversation_id)
                    .execute()
                )
            )

            messages = response.data or []

            user_messages = len([m for m in messages if m["role"] == "user"])
            assistant_messages = len([m for m in messages if m["role"] == "assistant"])

            stats = {
                "total_messages": len(messages),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "conversation_id": conversation_id,
            }

            return stats

        except Exception as e:
            logger.error("Error getting conversation stats: %s", e)
            return {}

    async def get_recent_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversations for this organization"""
        try:
            response = await run_supabase(
                lambda: (
                    self.supabase.table("conversations")
                    .select("*, messages!inner(content)")
                    .eq("org_id", self.org_id)
                    .order("updated_at", desc=True)
                    .limit(limit)
                    .execute()
                )
            )

            return response.data or []

        except Exception as e:
            logger.error("Error getting recent conversations: %s", e)
            return []
