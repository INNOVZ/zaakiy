"""Manual test for pricing context and retrieval flow."""
import asyncio
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAT_PATH = ROOT / "app" / "services" / "chat"

sys_modules = {}


def register_module(name: str, module: types.ModuleType):
    sys_modules[name] = module
    sys.modules[name] = module


app_mod = types.ModuleType("app")
services_mod = types.ModuleType("app.services")
chat_mod = types.ModuleType("app.services.chat")
shared_mod = types.ModuleType("app.services.shared")
shared_pkg_mod = types.ModuleType("app.services.chat.shared")
register_module("app", app_mod)
register_module("app.services", services_mod)
register_module("app.services.chat", chat_mod)
register_module("app.services.shared", shared_mod)
register_module("app.services.chat.shared", shared_pkg_mod)


class DummyCacheService:
    async def get(self, *args, **kwargs):
        return None

    async def set(self, *args, **kwargs):
        return False


shared_mod.cache_service = DummyCacheService()

imports = {
    "shared/constants": CHAT_PATH / "shared" / "constants.py",
    "shared/keyword_extractor": CHAT_PATH / "shared" / "keyword_extractor.py",
    "contact_extractor": CHAT_PATH / "contact_extractor.py",
    "chat_utilities": CHAT_PATH / "chat_utilities.py",
    "context_builder": CHAT_PATH / "context_builder.py",
    "response_generation_service": CHAT_PATH / "response_generation_service.py",
}


def load_module(key: str, path: pathlib.Path):
    name = f"app.services.chat.{key.replace('/', '.')}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loaded = {k: load_module(k, p) for k, p in imports.items()}

shared_pkg_mod.constants = loaded["shared/constants"]
shared_pkg_mod.keyword_extractor = loaded["shared/keyword_extractor"]

chat_mod.contact_extractor = loaded["contact_extractor"]
chat_mod.chat_utilities = loaded["chat_utilities"]
chat_mod.context_builder = loaded["context_builder"]
chat_mod.response_generation_service = loaded["response_generation_service"]

intent_spec = importlib.util.spec_from_file_location(
    "app.services.chat.intent_detection_service",
    CHAT_PATH / "intent_detection_service.py",
)
intent_module = importlib.util.module_from_spec(intent_spec)
sys.modules["app.services.chat.intent_detection_service"] = intent_module
intent_spec.loader.exec_module(intent_module)

IntentResult = intent_module.IntentResult
IntentType = intent_module.IntentType

pricing_chunk = """
Pricing plans/cost of zaakiy.
There are three pricing plans available.
Essential - $ 40/month
Essential features for individuals and small teams:
- 2000 Messages
- Basic customization
- 1GB Training Data
- 1 Zaakiy Assistant
- Basic analytics
- Email support

Pro - $100/month
- 5000 Messages
- Advanced customization
- 3GB Training Data
- 1 Zaakiy assistant
- Advanced analytics
- Email support
- Custom domains

Enterprise - $250/month
- 10000 Messages
- Advanced customization
- Unlimited Training Data
- 3 Zaakiy assistants
- Advanced analytics
- Email support
- Priority support
"""

pricing_doc = {
    "id": "pricing-doc-1",
    "chunk": pricing_chunk,
    "score": 0.95,
    "source": "https://zaakiy.io/pricing",
}

ContextBuilder = loaded["context_builder"].ContextBuilder
context_builder = ContextBuilder(loaded["chat_utilities"].ChatUtilities())
context_result = context_builder.build(
    [pricing_doc], max_context_length=4000, context_config=None
)

print("Context text snippet:\n", context_result.context_text[:500])

ResponseGenerationService = loaded[
    "response_generation_service"
].ResponseGenerationService
intent_result = IntentResult(IntentType.PRICING, 0.8, [], {}, "rule_based")


def run_enhance_query():
    service = ResponseGenerationService(
        org_id="test-org",
        openai_client=None,
        context_config=None,
        chatbot_config={},
    )
    qs = asyncio.run(service.enhance_query("what's the pricing?", [], intent_result))
    print("\nGenerated query variants:", qs)


run_enhance_query()
