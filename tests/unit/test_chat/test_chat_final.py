#!/usr/bin/env python3

import asyncio
import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, "/Users/jithinjacob/Documents/personal/backend")


async def test_chat_final():
    try:
        from app.services.chat.chat_service import ChatService

        print("🚀 Creating ChatService...")
        chat_service = ChatService(
            org_id="21bcca33-a10c-442c-9bc5-7a0208b5928f",
            chatbot_config={
                "id": "test-bot",
                "name": "Test Bot",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 500,
            },
        )
        print("✅ ChatService created")

        print("📤 Processing message...")
        response = await chat_service.process_message(
            message="What is this project about? Please provide details.",
            session_id="final-test-session",
        )

        print("📥 Response received!")
        if isinstance(response, dict):
            content = response.get("response", "")
            is_fallback = response.get("is_fallback", False)
            sources = response.get("sources", [])

            print(f"   Content length: {len(content)}")
            print(f"   Is fallback: {is_fallback}")
            print(f"   Sources: {len(sources)}")
            print(f"   Processing time: {response.get('processing_time_ms', 0)}ms")

            print(f"\n📖 Response preview:")
            print(content[:300] + "..." if len(content) > 300 else content)

            if not is_fallback:
                print("\n🎉 SUCCESS: Real AI response with Pinecone integration!")
                return True
            else:
                print("\n⚠️  Still getting fallback response")
                return False

        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_chat_final())
    print(f"\n🏁 Final result: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)
