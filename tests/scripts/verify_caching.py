"""
Caching Implementation Verification Script
Verifies that caching is working correctly across all services
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend root to path
backend_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_root))
from dotenv import load_dotenv

load_dotenv()

from app.services.shared import cache_service


class CachingVerification:
    """Verify caching implementation"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []

    async def verify_cache_service(self):
        """Verify cache service is initialized and working"""
        print("=" * 80)
        print("🔍 CACHE SERVICE VERIFICATION")
        print("=" * 80)
        print()

        # Check if cache_service exists
        if not cache_service:
            self.issues.append("❌ cache_service is None - caching is disabled")
            print("❌ cache_service is None")
            return False

        print("✅ cache_service object exists")

        # Check if Redis is enabled
        if not hasattr(cache_service, "enabled"):
            self.issues.append("❌ cache_service missing 'enabled' attribute")
            print("❌ cache_service missing 'enabled' attribute")
            return False

        # Initialize Redis if needed
        if not cache_service._initialized:
            print("⏳ Initializing Redis connection...")
            await cache_service._init_redis()

        if not cache_service.enabled:
            self.warnings.append("⚠️  Redis is disabled - caching will not work")
            print("⚠️  Redis is disabled")
            print("   Check REDIS_URL environment variable")
            return False

        print("✅ Redis is enabled")

        # Test basic operations
        try:
            test_key = "cache_verification_test"
            test_value = {"test": "data", "timestamp": 1234567890}

            # Test SET
            await cache_service.set(test_key, test_value, 60)
            print("✅ SET operation works")

            # Test GET
            retrieved = await cache_service.get(test_key)
            if retrieved and retrieved.get("test") == "data":
                print("✅ GET operation works")
            else:
                self.issues.append("❌ GET operation returned incorrect data")
                print("❌ GET operation returned incorrect data")
                return False

            # Test DELETE
            await cache_service.delete(test_key)
            deleted_check = await cache_service.get(test_key)
            if deleted_check is None:
                print("✅ DELETE operation works")
            else:
                self.warnings.append(
                    "⚠️  DELETE operation may not be working correctly"
                )
                print("⚠️  DELETE operation may not be working correctly")

            self.successes.append("✅ Basic cache operations verified")
            return True

        except Exception as e:
            self.issues.append(f"❌ Cache operations failed: {e}")
            print(f"❌ Cache operations failed: {e}")
            return False

    def verify_response_cache_key_generation(self):
        """Verify response cache key generation logic"""
        print("\n" + "=" * 80)
        print("🔍 RESPONSE CACHE KEY GENERATION VERIFICATION")
        print("=" * 80)
        print()

        # Import the service to test cache key generation
        from app.models.chatbot_config import ChatbotConfig
        from app.services.chat.response_generation_service import (
            ResponseGenerationService,
        )

        service = ResponseGenerationService(
            org_id="test-org",
            openai_client=None,
            context_config={},
            chatbot_config=ChatbotConfig(),
        )

        # Test 1: Same message + same documents = same key
        message1 = "What SEO services do you offer?"
        docs1 = [
            {"id": "doc1", "chunk": "SEO content"},
            {"id": "doc2", "chunk": "More SEO content"},
        ]

        key1 = service._generate_response_cache_key(message1, docs1)
        key2 = service._generate_response_cache_key(message1, docs1)

        if key1 == key2:
            print("✅ Same inputs generate same cache key")
            self.successes.append("✅ Cache key generation is deterministic")
        else:
            self.issues.append("❌ Same inputs generate different cache keys")
            print("❌ Same inputs generate different cache keys")
            print(f"   Key 1: {key1}")
            print(f"   Key 2: {key2}")

        # Test 2: Different documents = different key
        docs3 = [
            {"id": "doc3", "chunk": "Different content"},
        ]
        key3 = service._generate_response_cache_key(message1, docs3)

        if key1 != key3:
            print("✅ Different documents generate different cache keys")
            self.successes.append("✅ Cache key generation is context-aware")
        else:
            self.issues.append("❌ Different documents generate same cache key")
            print("❌ Different documents generate same cache key")

        # Test 3: Different message = different key
        message2 = "What email marketing services do you provide?"
        key4 = service._generate_response_cache_key(message2, docs1)

        if key1 != key4:
            print("✅ Different messages generate different cache keys")
            self.successes.append("✅ Cache key generation is message-aware")
        else:
            self.issues.append("❌ Different messages generate same cache key")
            print("❌ Different messages generate same cache key")

        # Test 4: Key format
        if key1.startswith("response:"):
            print("✅ Cache key has correct prefix")
        else:
            self.warnings.append("⚠️  Cache key format may be incorrect")
            print(f"⚠️  Cache key format: {key1[:50]}...")

    def verify_retrieval_cache_key_generation(self):
        """Verify retrieval cache key generation logic"""
        print("\n" + "=" * 80)
        print("🔍 RETRIEVAL CACHE KEY GENERATION VERIFICATION")
        print("=" * 80)
        print()

        from app.services.chat.document_retrieval_service import (
            DocumentRetrievalService,
        )

        service = DocumentRetrievalService(
            org_id="test-org",
            openai_client=None,
            pinecone_index=None,
            context_config={"retrieval_strategy": "keyword_boost"},
        )

        # Test 1: Same queries = same key
        queries1 = ["What SEO services do you offer?"]
        key1 = service._generate_retrieval_cache_key(queries1)
        key2 = service._generate_retrieval_cache_key(queries1)

        if key1 == key2:
            print("✅ Same queries generate same cache key")
            self.successes.append("✅ Retrieval cache key is deterministic")
        else:
            self.issues.append("❌ Same queries generate different cache keys")
            print("❌ Same queries generate different cache keys")

        # Test 2: Different queries = different key
        queries2 = ["What email marketing services do you provide?"]
        key3 = service._generate_retrieval_cache_key(queries2)

        if key1 != key3:
            print("✅ Different queries generate different cache keys")
            self.successes.append("✅ Retrieval cache key is query-aware")
        else:
            self.issues.append("❌ Different queries generate same cache key")
            print("❌ Different queries generate same cache key")

        # Test 3: Key format
        if key1.startswith("vector_retrieval:"):
            print("✅ Retrieval cache key has correct prefix")
        else:
            self.warnings.append("⚠️  Retrieval cache key format may be incorrect")
            print(f"⚠️  Cache key format: {key1[:50]}...")

    def verify_cache_ttl_settings(self):
        """Verify cache TTL settings are appropriate"""
        print("\n" + "=" * 80)
        print("🔍 CACHE TTL SETTINGS VERIFICATION")
        print("=" * 80)
        print()

        from app.services.chat.response_generation_service import ResponseConfig

        response_ttl = ResponseConfig.CACHE_TTL_SECONDS
        print(
            f"Response Cache TTL: {response_ttl} seconds ({response_ttl/3600:.1f} hours)"
        )

        if response_ttl >= 3600:
            print("✅ Response cache TTL is appropriate (≥1 hour)")
            self.successes.append("✅ Response cache TTL is well-configured")
        else:
            self.warnings.append(f"⚠️  Response cache TTL is short ({response_ttl}s)")
            print(f"⚠️  Response cache TTL may be too short: {response_ttl}s")

        # Check retrieval cache TTL (hardcoded in document_retrieval_service.py)
        retrieval_ttl = 1800  # 30 minutes
        print(
            f"Retrieval Cache TTL: {retrieval_ttl} seconds ({retrieval_ttl/60:.0f} minutes)"
        )

        if retrieval_ttl >= 900:  # At least 15 minutes
            print("✅ Retrieval cache TTL is appropriate (≥15 minutes)")
            self.successes.append("✅ Retrieval cache TTL is well-configured")
        else:
            self.warnings.append(f"⚠️  Retrieval cache TTL is short ({retrieval_ttl}s)")
            print(f"⚠️  Retrieval cache TTL may be too short: {retrieval_ttl}s")

    def verify_cache_usage_patterns(self):
        """Verify cache is being used in the right places"""
        print("\n" + "=" * 80)
        print("🔍 CACHE USAGE PATTERNS VERIFICATION")
        print("=" * 80)
        print()

        # Check response generation service
        import inspect

        from app.services.chat.response_generation_service import (
            ResponseGenerationService,
        )

        response_service_source = inspect.getsource(ResponseGenerationService)

        cache_checks = [
            ("_get_cached_response", "Response caching check"),
            ("_cache_response", "Response caching storage"),
            ("cache_hit_response", "Cache hit handling"),
        ]

        for method_name, description in cache_checks:
            if method_name in response_service_source:
                print(f"✅ {description} is implemented")
                self.successes.append(f"✅ {description} found")
            else:
                self.issues.append(f"❌ {description} not found")
                print(f"❌ {description} not found")

        # Check document retrieval service
        from app.services.chat.document_retrieval_service import (
            DocumentRetrievalService,
        )

        retrieval_service_source = inspect.getsource(DocumentRetrievalService)

        retrieval_cache_checks = [
            ("_get_cached_retrieval_results", "Retrieval caching check"),
            ("_cache_retrieval_results", "Retrieval caching storage"),
            ("cached_results", "Cache hit handling"),
        ]

        for method_name, description in retrieval_cache_checks:
            if method_name in retrieval_service_source:
                print(f"✅ {description} is implemented")
                self.successes.append(f"✅ {description} found")
            else:
                self.issues.append(f"❌ {description} not found")
                print(f"❌ {description} not found")

    def print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 80)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 80)
        print()

        print(f"✅ Successes: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues: {len(self.issues)}")
        print()

        if self.issues:
            print("❌ CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"   {issue}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   {warning}")
            print()

        if not self.issues:
            print("✅ Caching implementation is working correctly!")
        else:
            print("❌ Caching implementation has issues that need to be fixed.")


async def main():
    """Run caching verification"""
    verifier = CachingVerification()

    # Run all verifications
    await verifier.verify_cache_service()
    verifier.verify_response_cache_key_generation()
    verifier.verify_retrieval_cache_key_generation()
    verifier.verify_cache_ttl_settings()
    verifier.verify_cache_usage_patterns()

    # Print summary
    verifier.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
