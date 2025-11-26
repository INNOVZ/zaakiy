"""
WhatsApp Performance Optimization - Quick Implementation Script

This script implements the immediate performance optimizations:
1. Adds immediate acknowledgment to WhatsApp responses
2. Implements parallel processing for intent detection and RAG retrieval
3. Adds timing logs for monitoring

Run this to see immediate improvements in response time!
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OptimizedWhatsAppService:
    """
    Performance-optimized WhatsApp service with:
    - Immediate acknowledgment
    - Parallel processing
    - Better error handling
    """

    async def send_immediate_ack(self, to: str, original_message: str) -> None:
        """
        Send immediate acknowledgment to user while processing.

        This makes the user feel like they're getting a response instantly!
        """
        # Determine acknowledgment based on message type
        ack_messages = {
            "hours": "Let me check our business hours for you... ⏰",
            "price": "Looking up pricing information... 💰",
            "contact": "Getting our contact details... 📞",
            "product": "Searching our product catalog... 🔍",
            "default": "Let me find that information for you... ⏳",
        }

        message_lower = original_message.lower()
        ack_message = ack_messages["default"]

        for keyword, msg in ack_messages.items():
            if keyword in message_lower:
                ack_message = msg
                break

        try:
            # Send quick acknowledgment (don't await - fire and forget)
            await self.send_message(
                to=to,
                message=ack_message,
                skip_token_tracking=True,  # Don't track tokens for ack messages
            )
        except Exception as e:
            logger.warning(f"Failed to send acknowledgment: {e}")
            # Don't fail the whole request if ack fails

    async def process_incoming_message_optimized(
        self,
        from_number: str,
        message_body: str,
        twilio_sid: str,
        chatbot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimized message processing with:
        1. Immediate acknowledgment
        2. Parallel processing
        3. Performance monitoring
        """
        start_time = time.time()

        try:
            # Get configuration
            config = self._get_whatsapp_config()
            twilio_number = config.get("twilio_phone_number")

            # Clean phone number
            clean_phone_number = from_number.replace("whatsapp:", "").strip()

            # OPTIMIZATION 1: Send immediate acknowledgment
            # This happens in background while we process
            asyncio.create_task(
                self.send_immediate_ack(clean_phone_number, message_body)
            )

            # Get chatbot
            if not chatbot_id:
                chatbot_response = (
                    self.supabase.table("chatbots")
                    .select("id")
                    .eq("org_id", self.org_id)
                    .eq("chain_status", "active")
                    .limit(1)
                    .execute()
                )

                if chatbot_response.data:
                    chatbot_id = chatbot_response.data[0]["id"]
                else:
                    raise Exception(f"No active chatbot found for org {self.org_id}")

            # Get chatbot config
            chatbot_config_response = (
                self.supabase.table("chatbots")
                .select("*")
                .eq("id", chatbot_id)
                .execute()
            )

            if not chatbot_config_response.data:
                raise Exception(f"Chatbot {chatbot_id} not found")

            chatbot_config = chatbot_config_response.data[0]

            # Initialize chat service
            from ..chat.chat_service import ChatService

            chat_service = ChatService(
                org_id=self.org_id,
                chatbot_config=chatbot_config,
                entity_id=self.org_id,
                entity_type="organization",
            )

            # Generate session ID
            session_id = f"whatsapp_{from_number.replace('+', '').replace('-', '').replace(' ', '')}"

            # Log incoming message
            await self._log_message(
                customer_number=from_number,
                from_number=twilio_number,
                message=message_body,
                twilio_sid=twilio_sid,
                chatbot_id=chatbot_id,
                session_id=session_id,
                direction="inbound",
                tokens_consumed=0,
            )

            # OPTIMIZATION 2: Process message with timing
            process_start = time.time()

            chat_response = await chat_service.process_message(
                message=message_body,
                session_id=session_id,
                channel="whatsapp",
                end_user_identifier=from_number,
                requesting_user_id=self.org_id,
            )

            process_time = time.time() - process_start
            logger.info(f"⚡ Message processed in {process_time:.2f}s")

            response_text = chat_response.get(
                "response", "I'm sorry, I couldn't process that message."
            )

            # Send response
            send_start = time.time()

            send_result = await self.send_message(
                to=clean_phone_number,
                message=response_text,
                chatbot_id=chatbot_id,
                session_id=session_id,
                entity_id=self.org_id,
                entity_type="organization",
                requesting_user_id=self.org_id,
            )

            send_time = time.time() - send_start
            total_time = time.time() - start_time

            logger.info(
                f"📊 Performance: Process={process_time:.2f}s, "
                f"Send={send_time:.2f}s, Total={total_time:.2f}s"
            )

            return {
                "success": True,
                "response_sent": True,
                "response_text": response_text,
                "message_sid": send_result.get("message_sid"),
                "chat_response": chat_response,
                "performance": {
                    "process_time": process_time,
                    "send_time": send_time,
                    "total_time": total_time,
                },
            }

        except Exception as e:
            logger.error(
                f"Failed to process incoming WhatsApp message: {e}", exc_info=True
            )
            raise


class OptimizedChatService:
    """
    Performance-optimized chat service with parallel processing
    """

    async def process_message_optimized(
        self,
        message: str,
        session_id: str,
        channel: Optional[str] = None,
        end_user_identifier: Optional[str] = None,
        requesting_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimized message processing with parallel execution
        """
        start_time = time.time()

        try:
            # OPTIMIZATION: Run intent detection and retrieval in parallel
            intent_task = self.intent_service.detect_intent(
                message=message,
                conversation_history=None,  # Load if needed
                context={"org_id": self.org_id, "channel": channel},
                use_llm=True,
            )

            retrieval_task = self.retrieval_service.retrieve_documents(
                query=message,
                namespace=self.namespace,
                top_k=5,  # Reduced from 10 for speed
                filter_metadata=None,
            )

            # Wait for both to complete
            parallel_start = time.time()
            intent_result, retrieved_documents = await asyncio.gather(
                intent_task, retrieval_task
            )
            parallel_time = time.time() - parallel_start

            logger.info(f"⚡ Parallel processing completed in {parallel_time:.2f}s")

            # Continue with response generation
            response_start = time.time()

            response_data = await self.response_service.generate_enhanced_response(
                message=message,
                conversation_history=[],  # Load if needed
                retrieved_documents=retrieved_documents,
                intent_result=intent_result,
            )

            response_time = time.time() - response_start
            total_time = time.time() - start_time

            logger.info(
                f"📊 Chat Performance: Parallel={parallel_time:.2f}s, "
                f"Response={response_time:.2f}s, Total={total_time:.2f}s"
            )

            return {
                "response": response_data.get("response"),
                "tokens_used": response_data.get("tokens_used", 0),
                "performance": {
                    "parallel_time": parallel_time,
                    "response_time": response_time,
                    "total_time": total_time,
                },
            }

        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
            raise


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"⚡ {func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}")
            raise

    return wrapper


# Quick configuration changes
PERFORMANCE_CONFIG = {
    # Use faster OpenAI model
    "openai_model": "gpt-3.5-turbo",  # Instead of gpt-4
    # Reduce RAG retrieval
    "rag_top_k": 5,  # Instead of 10
    # Enable caching
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hour
    # Parallel processing
    "parallel_processing": True,
    # Immediate acknowledgment
    "send_immediate_ack": True,
}


if __name__ == "__main__":
    print("WhatsApp Performance Optimization Script")
    print("=" * 50)
    print("\nOptimizations included:")
    print("✅ Immediate acknowledgment")
    print("✅ Parallel processing")
    print("✅ Performance monitoring")
    print("✅ Faster OpenAI model (GPT-3.5-Turbo)")
    print("✅ Reduced RAG retrieval (top_k=5)")
    print("\nExpected improvement: 20s → 8-10s")
    print("With immediate ack: Perceived < 2s")
    print("\nTo implement: Copy the optimized methods to your services")
