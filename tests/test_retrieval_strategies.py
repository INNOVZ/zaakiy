#!/usr/bin/env python3
"""
Test script to demonstrate different retrieval strategies
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

from app.services.chat.chat_service import ChatService

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_retrieval_strategies():
    """Test different retrieval strategies"""

    # Test configuration
    org_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID format
    chatbot_config = {
        "id": "test-chatbot",
        "name": "Test Assistant",
        "tone": "helpful",
        "behavior": "Be helpful and informative",
        "greeting_message": "Hello! How can I help you?",
        "fallback_message": "I'm sorry, I don't have information about that.",
    }

    print("🚀 Testing Retrieval Strategies")
    print("=" * 50)

    # Test different strategies
    strategies = [
        ("semantic_only", "Pure semantic similarity search"),
        ("hybrid", "Combined semantic + keyword matching"),
        ("keyword_boost", "Keyword-boosted retrieval"),
        ("domain_specific", "Domain-specific retrieval"),
    ]

    for strategy_name, description in strategies:
        print(f"\n🔍 Testing Strategy: {strategy_name}")
        print(f"📝 Description: {description}")
        print("-" * 40)

        try:
            # Initialize ChatService
            chat_service = ChatService(org_id=org_id, chatbot_config=chatbot_config)

            # Test message
            test_message = "What services do you offer for automation?"
            session_id = f"test-session-{strategy_name}"

            print(f"💬 Query: '{test_message}'")
            print("⏳ Generating response...")

            # Get response
            result = await chat_service.chat(
                message=test_message,
                session_id=session_id,
                chatbot_id=chatbot_config["id"],
            )

            print(f"✅ Response: {result['response'][:100]}...")
            print(f"📊 Sources: {len(result.get('sources', []))}")
            print(f"⏱️  Processing time: {result.get('processing_time_ms', 0)}ms")

            # Show retrieval method used
            if "retrieved_documents" in result:
                methods = set()
                for doc in result["retrieved_documents"]:
                    if "retrieval_method" in doc:
                        methods.add(doc["retrieval_method"])
                print(
                    f"🔧 Retrieval methods used: {', '.join(methods) if methods else 'None'}"
                )

        except Exception as e:
            print(f"❌ Error testing {strategy_name}: {e}")

    print("\n" + "=" * 50)
    print("🏁 Retrieval strategy testing completed")


async def test_retrieval_configuration():
    """Test retrieval configuration details"""
    print("\n🔧 Testing Retrieval Configuration")
    print("=" * 50)

    try:
        from app.services.analytics.context_config import context_config_manager

        org_id = "550e8400-e29b-41d4-a716-446655440000"
        config = await context_config_manager.get_config(org_id)

        print("✅ Context config loaded successfully")
        print(f"📋 Config name: {config.config_name}")
        print(f"🎯 Model tier: {config.model_tier}")
        print(f"🔍 Retrieval strategy: {config.retrieval_strategy}")
        print(f"⚖️  Semantic weight: {config.semantic_weight}")
        print(f"🔑 Keyword weight: {config.keyword_weight}")
        print(
            f"📊 Retrieval counts: {config.initial_retrieval_count} -> {config.final_context_chunks}"
        )
        print(f"🔍 Query rewriting: {config.enable_query_rewriting}")
        print(f"🛡️  Hallucination check: {config.enable_hallucination_check}")

        # Show available strategies
        from app.services.analytics.context_config import RetrievalStrategy

        print("\n📋 Available strategies:")
        for strategy in RetrievalStrategy:
            print(f"  - {strategy.value}: {strategy.name}")

    except Exception as e:
        print(f"❌ Error loading context config: {e}")


async def main():
    """Main test function"""
    print("🚀 Starting Retrieval Strategy Tests")
    print("=" * 50)

    # Test configuration
    await test_retrieval_configuration()

    # Test strategies
    await test_retrieval_strategies()

    print("\n" + "=" * 50)
    print("🏁 All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
