"""
Model Tier Mapping Service

Maps user-friendly model tiers (fast, balanced, premium, enterprise)
to actual OpenAI model names with performance and cost optimization.
"""

import logging
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier options available to users"""

    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ModelTierMapper:
    """
    Maps model tiers to actual OpenAI models with intelligent selection.

    This provides a clean abstraction layer between user-facing tier names
    and the actual OpenAI models, making it easy to update models as new
    ones become available.
    """

    # Model tier to OpenAI model mapping
    # Updated for latest OpenAI models (as of 2024)
    TIER_TO_MODEL: Dict[str, str] = {
        # Fast: Optimized for speed and cost
        ModelTier.FAST: "gpt-3.5-turbo",
        # Balanced: Good balance of speed, quality, and cost
        # Using gpt-4o-mini for better quality than 3.5 at similar speed
        ModelTier.BALANCED: "gpt-4o-mini",
        # Premium: High quality, slower but better responses
        # Using gpt-4o for best quality/speed ratio
        ModelTier.PREMIUM: "gpt-4o",
        # Enterprise: Maximum quality and capabilities
        # Using gpt-4-turbo for comprehensive responses
        ModelTier.ENTERPRISE: "gpt-4-turbo",
    }

    # Fallback model if tier is not recognized
    DEFAULT_MODEL = "gpt-3.5-turbo"

    # Model characteristics for logging and monitoring
    MODEL_CHARACTERISTICS: Dict[str, Dict[str, any]] = {
        "gpt-3.5-turbo": {
            "speed": "very_fast",
            "quality": "good",
            "cost": "low",
            "avg_response_time_ms": 1500,
            "recommended_for": ["simple_queries", "high_volume", "cost_sensitive"],
        },
        "gpt-4o-mini": {
            "speed": "fast",
            "quality": "very_good",
            "cost": "medium",
            "avg_response_time_ms": 2500,
            "recommended_for": [
                "standard_queries",
                "balanced_performance",
                "most_use_cases",
            ],
        },
        "gpt-4o": {
            "speed": "moderate",
            "quality": "excellent",
            "cost": "high",
            "avg_response_time_ms": 4000,
            "recommended_for": [
                "complex_queries",
                "high_quality_needed",
                "detailed_responses",
            ],
        },
        "gpt-4-turbo": {
            "speed": "moderate",
            "quality": "excellent",
            "cost": "very_high",
            "avg_response_time_ms": 5000,
            "recommended_for": [
                "enterprise",
                "comprehensive_analysis",
                "maximum_quality",
            ],
        },
    }

    @classmethod
    def get_model_for_tier(cls, model_tier: Optional[str]) -> str:
        """
        Get the OpenAI model name for a given tier.

        Args:
            model_tier: The tier name (fast, balanced, premium, enterprise)

        Returns:
            OpenAI model name (e.g., "gpt-3.5-turbo")
        """
        if not model_tier:
            logger.warning("No model tier provided, using default model")
            return cls.DEFAULT_MODEL

        # Normalize tier name
        tier_normalized = model_tier.lower().strip()

        # Get model from mapping
        model = cls.TIER_TO_MODEL.get(tier_normalized)

        if not model:
            logger.warning(
                f"Unknown model tier '{model_tier}', using default model '{cls.DEFAULT_MODEL}'"
            )
            return cls.DEFAULT_MODEL

        logger.info(f"🎯 Model tier '{model_tier}' mapped to '{model}'")
        return model

    @classmethod
    def get_tier_for_model(cls, model: str) -> Optional[str]:
        """
        Reverse lookup: Get the tier for a given model.

        Args:
            model: OpenAI model name

        Returns:
            Tier name or None if not found
        """
        for tier, tier_model in cls.TIER_TO_MODEL.items():
            if tier_model == model:
                return tier
        return None

    @classmethod
    def get_model_characteristics(cls, model: str) -> Dict[str, any]:
        """
        Get characteristics for a model.

        Args:
            model: OpenAI model name

        Returns:
            Dictionary of model characteristics
        """
        return cls.MODEL_CHARACTERISTICS.get(
            model,
            {
                "speed": "unknown",
                "quality": "unknown",
                "cost": "unknown",
                "avg_response_time_ms": 3000,
                "recommended_for": [],
            },
        )

    @classmethod
    def validate_tier(cls, model_tier: str) -> bool:
        """
        Check if a tier is valid.

        Args:
            model_tier: Tier name to validate

        Returns:
            True if valid, False otherwise
        """
        return model_tier.lower() in cls.TIER_TO_MODEL

    @classmethod
    def get_all_tiers(cls) -> list[str]:
        """Get list of all available tiers"""
        return list(cls.TIER_TO_MODEL.keys())

    @classmethod
    def get_tier_info(cls, model_tier: str) -> Dict[str, any]:
        """
        Get comprehensive information about a tier.

        Args:
            model_tier: Tier name

        Returns:
            Dictionary with tier information
        """
        model = cls.get_model_for_tier(model_tier)
        characteristics = cls.get_model_characteristics(model)

        return {
            "tier": model_tier,
            "model": model,
            "characteristics": characteristics,
            "is_valid": cls.validate_tier(model_tier),
        }


# Convenience function for quick access
def get_model_from_tier(model_tier: Optional[str]) -> str:
    """
    Quick function to get model from tier.

    Args:
        model_tier: Model tier name

    Returns:
        OpenAI model name
    """
    return ModelTierMapper.get_model_for_tier(model_tier)


# Export for easy imports
__all__ = ["ModelTier", "ModelTierMapper", "get_model_from_tier"]
