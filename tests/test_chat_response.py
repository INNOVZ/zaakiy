#!/usr/bin/env python3
"""
Test script to debug chat response issues
"""
from services.chat.chat_service import ChatService
import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_chat_response():
    """Test the chat service directly"""

    # Test configuration
    org_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID format
    chatbot_config = {
        "id": "test-chatbot",
        "name": "Test Assistant",
        "tone": "helpful",
        "behavior": "Be helpful and informative",
        "greeting_message": "Hello! How can I help you?",
        "fallback_message": "I'm sorry, I don't have information about that."
    }

    print("🔧 Initializing ChatService...")
    try:
        chat_service = ChatService(
            org_id=org_id, chatbot_config=chatbot_config)
        print("✅ ChatService initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ChatService: {e}")
        return

    # Test message
    test_message = "Hello, can you help me with information about your services?"
    session_id = "550e8400-e29b-41d4-a716-446655440001"  # Valid UUID format

    print(f"\n💬 Testing message: '{test_message}'")
    print("⏳ Generating response...")

    try:
        result = await chat_service.chat(
            message=test_message,
            session_id=session_id,
            chatbot_id=chatbot_config["id"]
        )

        print("\n✅ Response generated successfully!")
        print(f"📝 Response: {result['response']}")
        print(f"📊 Sources: {result.get('sources', [])}")
        print(f"⏱️  Processing time: {result.get('processing_time_ms', 0)}ms")
        print(f"🎯 Context quality: {result.get('context_quality', {})}")
        print(f"🔧 Config used: {result.get('config_used', 'unknown')}")

        # Check if response is meaningful
        if len(result['response']) < 10:
            print("⚠️  WARNING: Response seems too short")

        if "I apologize" in result['response'] or "I don't have" in result['response']:
            print("⚠️  WARNING: Response seems like a fallback message")

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        import traceback
        traceback.print_exc()


async def test_context_config():
    """Test context configuration"""
    print("\n🔧 Testing context configuration...")

    try:
        from services.analytics.context_config import context_config_manager

        org_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID format
        config = await context_config_manager.get_config(org_id)

        print("✅ Context config loaded successfully")
        print(f"📋 Config name: {config.config_name}")
        print(f"🎯 Model tier: {config.model_tier}")
        print(
            f"📊 Retrieval counts: {config.initial_retrieval_count} -> {config.final_context_chunks}")
        print(f"🔍 Query rewriting: {config.enable_query_rewriting}")
        print(f"🛡️  Hallucination check: {config.enable_hallucination_check}")

    except Exception as e:
        print(f"❌ Error loading context config: {e}")
        import traceback
        traceback.print_exc()


async def test_client_connections():
    """Test API client connections"""
    print("\n🔧 Testing API client connections...")

    try:
        from services.shared.client_manager import client_manager

        health = client_manager.health_check()

        print("📊 Client Health Status:")
        for client, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {client}: {'Healthy' if status else 'Failed'}")

        if not all(health.values()):
            print("⚠️  Some clients are not healthy - this may affect responses")

    except Exception as e:
        print(f"❌ Error checking client health: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function"""
    print("🚀 Starting Chat Service Debug Test")
    print("=" * 50)

    # Test client connections first
    await test_client_connections()

    # Test context configuration
    await test_context_config()

    # Test chat response
    await test_chat_response()

    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
