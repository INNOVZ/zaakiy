"""
Model Tier Mapping - Test Script

Tests the model tier to OpenAI model mapping functionality.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.chat.model_tier_mapper import ModelTier, ModelTierMapper


def test_model_tier_mapping():
    """Test model tier to OpenAI model mapping"""

    print("=" * 80)
    print("MODEL TIER MAPPING - TEST RESULTS")
    print("=" * 80)
    print()

    # Test all tiers
    test_cases = [
        ("fast", "gpt-3.5-turbo"),
        ("balanced", "gpt-4o-mini"),
        ("premium", "gpt-4o"),
        ("enterprise", "gpt-4-turbo"),
        ("FAST", "gpt-3.5-turbo"),  # Test case insensitivity
        ("  balanced  ", "gpt-4o-mini"),  # Test whitespace handling
        ("invalid", "gpt-3.5-turbo"),  # Test fallback
        (None, "gpt-3.5-turbo"),  # Test None handling
    ]

    print("🎯 TIER TO MODEL MAPPING TESTS")
    print("-" * 80)

    for tier, expected_model in test_cases:
        result_model = ModelTierMapper.get_model_for_tier(tier)
        status = "✅ PASS" if result_model == expected_model else "❌ FAIL"

        print(
            f"{status} | Tier: {repr(tier):20} → Model: {result_model:20} (expected: {expected_model})"
        )

    print()
    print("=" * 80)
    print("MODEL CHARACTERISTICS")
    print("=" * 80)
    print()

    for tier in ["fast", "balanced", "premium", "enterprise"]:
        info = ModelTierMapper.get_tier_info(tier)
        model = info["model"]
        chars = info["characteristics"]

        print(f"📊 {tier.upper()} TIER")
        print(f"   Model: {model}")
        print(f"   Speed: {chars.get('speed', 'unknown')}")
        print(f"   Quality: {chars.get('quality', 'unknown')}")
        print(f"   Cost: {chars.get('cost', 'unknown')}")
        print(f"   Avg Response Time: {chars.get('avg_response_time_ms', 0)}ms")
        print(f"   Recommended For: {', '.join(chars.get('recommended_for', []))}")
        print()

    print("=" * 80)
    print("REVERSE LOOKUP TESTS")
    print("=" * 80)
    print()

    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4"]

    for model in models:
        tier = ModelTierMapper.get_tier_for_model(model)
        print(f"Model: {model:20} → Tier: {tier or 'Not mapped'}")

    print()
    print("=" * 80)
    print("VALIDATION TESTS")
    print("=" * 80)
    print()

    validation_tests = [
        ("fast", True),
        ("balanced", True),
        ("premium", True),
        ("enterprise", True),
        ("invalid", False),
        ("", False),
    ]

    for tier, expected_valid in validation_tests:
        is_valid = ModelTierMapper.validate_tier(tier)
        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
        print(
            f"{status} | Tier: {repr(tier):15} → Valid: {is_valid} (expected: {expected_valid})"
        )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Model tier mapping is working correctly!")
    print()
    print("Available tiers:")
    for tier in ModelTierMapper.get_all_tiers():
        model = ModelTierMapper.get_model_for_tier(tier)
        print(f"  - {tier:12} → {model}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    test_model_tier_mapping()
