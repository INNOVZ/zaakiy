"""Quick manual test for intent detection service without full app boot."""
import asyncio
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Create lightweight module placeholders so intent_detection_service can import cache_service
app_module = types.ModuleType("app")
services_module = types.ModuleType("app.services")
shared_module = types.ModuleType("app.services.shared")
chat_package = types.ModuleType("app.services.chat")


# Minimal no-op cache service (disables caching for the test)
class _DummyCacheService:
    async def get(self, *args, **kwargs):
        return None

    async def set(self, *args, **kwargs):
        return False


shared_module.cache_service = _DummyCacheService()

app_module.services = services_module
services_module.shared = shared_module
services_module.chat = chat_package
chat_package.shared = shared_module

sys.modules["app"] = app_module
sys.modules["app.services"] = services_module
sys.modules["app.services.shared"] = shared_module
sys.modules["app.services.chat"] = chat_package

# Load intent detection module directly
intent_path = ROOT / "app" / "services" / "chat" / "intent_detection_service.py"
intent_spec = importlib.util.spec_from_file_location(
    "app.services.chat.intent_detection_service",
    intent_path,
)
intent_module = importlib.util.module_from_spec(intent_spec)
sys.modules["app.services.chat.intent_detection_service"] = intent_module
intent_spec.loader.exec_module(intent_module)

IntentDetectionService = intent_module.IntentDetectionService

TEST_MESSAGES = {
    "greeting": "Hey there!",
    "contact": "How can I talk to someone on your team?",
    "booking": "I'd like to schedule a demo for next week",
    "pricing": "What does the enterprise plan cost?",
    "product": "Show me the products you offer",
    "support": "I'm having trouble logging into my account",
    "recommendation": "What do you recommend for a small startup?",
    "unknown": "Purple clouds dance sideways",  # should fall back to unknown/information
}


def format_result(intent_result):
    primary = intent_result.primary_intent.value
    confidence = f"{intent_result.confidence:.2f}"
    method = intent_result.detection_method
    secondary = [intent.value for intent in intent_result.secondary_intents]
    metadata = intent_result.intent_metadata
    return {
        "primary_intent": primary,
        "confidence": confidence,
        "detection_method": method,
        "secondary_intents": secondary,
        "metadata": metadata,
    }


async def main():
    detector = IntentDetectionService(openai_client=None, org_id="test-org")

    print("Testing intent detection (rule-based only)\n" + "-" * 60)
    for label, message in TEST_MESSAGES.items():
        result = await detector.detect_intent(message, use_llm=False)
        formatted = format_result(result)
        print(f"Message label: {label}")
        print(f"Input: {message}")
        print(
            f"Detected intent: {formatted['primary_intent']} (confidence {formatted['confidence']})"
        )
        print(f"Detection method: {formatted['detection_method']}")
        if formatted["secondary_intents"]:
            print(f"Secondary intents: {formatted['secondary_intents']}")
        print(f"Metadata: {formatted['metadata']}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
