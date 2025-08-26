from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import os
import json
from datetime import datetime, timedelta
from supabase import create_client, Client
import logging


class RetrievalStrategy(str, Enum):
    """Retrieval strategy options"""
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    KEYWORD_BOOST = "keyword_boost"
    DOMAIN_SPECIFIC = "domain_specific"


class ModelTier(str, Enum):
    """AI model tiers for different use cases"""
    FAST = "fast"          # gpt-3.5-turbo
    BALANCED = "balanced"  # gpt-4
    PREMIUM = "premium"    # gpt-4-turbo, claude-3
    ENTERPRISE = "enterprise"  # Custom fine-tuned models


class ContextEngineeringConfig(BaseModel):
    """Configuration model for context engineering parameters"""

    # Organization identification
    org_id: str
    config_name: str = "default"

    # Retrieval Configuration
    initial_retrieval_count: int = Field(
        default=20, ge=5, le=50, description="Initial document retrieval count")
    semantic_rerank_count: int = Field(
        default=10, ge=3, le=25, description="Documents to semantically re-rank")
    final_context_chunks: int = Field(
        default=5, ge=1, le=10, description="Final chunks to include in context")
    max_context_length: int = Field(
        default=4000, ge=1000, le=8000, description="Maximum context length in tokens")

    # Query Enhancement
    enable_query_rewriting: bool = Field(
        default=True, description="Enable intelligent query rewriting")
    max_query_variants: int = Field(
        default=3, ge=1, le=5, description="Maximum query variants to generate")
    conversation_context_turns: int = Field(
        default=3, ge=1, le=10, description="Previous conversation turns to consider")

    # Retrieval Strategy
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID, description="Retrieval strategy to use")
    semantic_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight for semantic similarity")
    keyword_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for keyword matching")

    # Model Configuration
    model_tier: ModelTier = Field(
        default=ModelTier.BALANCED, description="AI model tier to use")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model")
    rerank_model: str = Field(default="gpt-3.5-turbo",
                              description="Model for semantic re-ranking")

    # Quality Controls
    enable_semantic_rerank: bool = Field(
        default=True, description="Enable semantic re-ranking")
    enable_hallucination_check: bool = Field(
        default=True, description="Enable hallucination detection")
    enable_source_verification: bool = Field(
        default=True, description="Enable source verification")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for responses")

    # Performance Settings
    max_response_time_ms: int = Field(
        default=5000, ge=1000, le=15000, description="Maximum response time in milliseconds")
    enable_caching: bool = Field(
        default=True, description="Enable response caching")
    cache_ttl_minutes: int = Field(
        default=60, ge=5, le=1440, description="Cache TTL in minutes")

    # Domain-Specific Settings
    domain_filters: Dict[str, Any] = Field(
        default_factory=dict, description="Domain-specific filters")
    business_context: str = Field(
        default="", description="Business context for this organization")
    specialized_instructions: str = Field(
        default="", description="Specialized instructions for this domain")

    # Monitoring & Analytics
    enable_detailed_logging: bool = Field(
        default=True, description="Enable detailed context logging")
    log_user_queries: bool = Field(
        default=True, description="Log user queries for analytics")
    collect_feedback: bool = Field(
        default=True, description="Collect user feedback")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextConfigManager:
    """Manager for context engineering configurations"""

    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Default configurations for different business types
        self.default_configs = {
            "saas": ContextEngineeringConfig(
                org_id="default",
                config_name="saas_optimized",
                retrieval_strategy=RetrievalStrategy.HYBRID,
                model_tier=ModelTier.BALANCED,
                final_context_chunks=5,
                business_context="SaaS platform with technical documentation and user guides"
            ),
            "ecommerce": ContextEngineeringConfig(
                org_id="default",
                config_name="ecommerce_optimized",
                retrieval_strategy=RetrievalStrategy.SEMANTIC_ONLY,
                model_tier=ModelTier.FAST,
                final_context_chunks=3,
                business_context="E-commerce platform with product information and customer support"
            ),
            "healthcare": ContextEngineeringConfig(
                org_id="default",
                config_name="healthcare_optimized",
                retrieval_strategy=RetrievalStrategy.DOMAIN_SPECIFIC,
                model_tier=ModelTier.PREMIUM,
                final_context_chunks=7,
                confidence_threshold=0.85,
                enable_hallucination_check=True,
                business_context="Healthcare organization with medical information and patient support"
            ),
            "finance": ContextEngineeringConfig(
                org_id="default",
                config_name="finance_optimized",
                retrieval_strategy=RetrievalStrategy.HYBRID,
                model_tier=ModelTier.PREMIUM,
                final_context_chunks=6,
                confidence_threshold=0.8,
                business_context="Financial services with regulatory information and customer support"
            )
        }

    async def get_config(self, org_id: str, config_name: str = "default") -> ContextEngineeringConfig:
        """Get context engineering configuration for organization"""
        try:
            # Try to get from database first
            response = self.supabase.table("context_engineering_configs").select("*").eq(
                "org_id", org_id
            ).eq("config_name", config_name).execute()

            if response.data and len(response.data) > 0:
                config_data = response.data[0]["config_data"]
                return ContextEngineeringConfig(**config_data)

            # Fall back to org's business type default
            org_response = self.supabase.table("organizations").select(
                "type").eq("id", org_id).execute()

            if org_response.data and len(org_response.data) > 0:
                org_type = org_response.data[0].get("type", "saas").lower()
                if org_type in self.default_configs:
                    config = self.default_configs[org_type].copy(
                        update={"org_id": org_id})
                    # Save default config to database
                    await self.save_config(config)
                    return config

            # Ultimate fallback - SaaS default
            config = self.default_configs["saas"].copy(
                update={"org_id": org_id})
            await self.save_config(config)
            return config

        except Exception as e:
            logging.error(
                f"Error getting context config for org {org_id}: {e}")
            # Return SaaS default as emergency fallback
            return self.default_configs["saas"].copy(update={"org_id": org_id})

    async def save_config(self, config: ContextEngineeringConfig) -> bool:
        """Save context engineering configuration"""
        try:
            config.updated_at = datetime.utcnow()

            config_data = {
                "org_id": config.org_id,
                "config_name": config.config_name,
                "config_data": config.dict(),
                "updated_at": config.updated_at.isoformat()
            }

            # Upsert configuration
            response = self.supabase.table("context_engineering_configs").upsert(
                config_data,
                on_conflict="org_id,config_name"
            ).execute()

            return bool(response.data)

        except Exception as e:
            logging.error(f"Error saving context config: {e}")
            return False

    async def update_config(
        self,
        org_id: str,
        updates: Dict[str, Any],
        config_name: str = "default"
    ) -> bool:
        """Update specific configuration parameters"""
        try:
            # Get current config
            current_config = await self.get_config(org_id, config_name)

            # Apply updates
            for key, value in updates.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)

            # Save updated config
            return await self.save_config(current_config)

        except Exception as e:
            logging.error(f"Error updating context config: {e}")
            return False

    async def create_custom_config(
        self,
        org_id: str,
        config_name: str,
        base_template: str = "saas",
        custom_settings: Dict[str, Any] = None
    ) -> ContextEngineeringConfig:
        """Create a custom configuration based on template"""
        try:
            # Start with base template
            base_config = self.default_configs.get(
                base_template, self.default_configs["saas"])
            config = base_config.copy(update={
                "org_id": org_id,
                "config_name": config_name
            })

            # Apply custom settings
            if custom_settings:
                for key, value in custom_settings.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            # Save the custom config
            await self.save_config(config)
            return config

        except Exception as e:
            logging.error(f"Error creating custom config: {e}")
            raise

    def get_model_config(self, tier: ModelTier) -> Dict[str, Any]:
        """Get model configuration based on tier"""
        model_configs = {
            ModelTier.FAST: {
                "model": "gpt-3.5-turbo",
                "max_tokens": 800,
                "temperature": 0.7,
                "timeout": 3000
            },
            ModelTier.BALANCED: {
                "model": "gpt-4",
                "max_tokens": 1000,
                "temperature": 0.7,
                "timeout": 5000
            },
            ModelTier.PREMIUM: {
                "model": "gpt-4-turbo",
                "max_tokens": 1500,
                "temperature": 0.6,
                "timeout": 8000
            },
            ModelTier.ENTERPRISE: {
                "model": "gpt-4-turbo",  # This could be customized per org
                "max_tokens": 2000,
                "temperature": 0.5,
                "timeout": 10000
            }
        }

        return model_configs.get(tier, model_configs[ModelTier.BALANCED])

    async def get_performance_recommendations(self, org_id: str) -> Dict[str, Any]:
        """Get performance-based configuration recommendations"""
        try:
            # Get recent performance metrics
            response = self.supabase.table("context_analytics").select("*").eq(
                "org_id", org_id
            ).gte(
                "created_at", (datetime.utcnow() -
                               timedelta(days=7)).isoformat()
            ).execute()

            if not response.data:
                return {"recommendations": [], "confidence": 0}

            # Analyze performance data
            metrics = response.data
            avg_response_time = sum(m.get("response_time_ms", 0)
                                    for m in metrics) / len(metrics)
            avg_satisfaction = sum(m.get("user_satisfaction", 0.5)
                                   for m in metrics) / len(metrics)
            avg_context_quality = sum(
                m.get("context_quality_score", 0.5) for m in metrics) / len(metrics)

            recommendations = []

            # Response time recommendations
            if avg_response_time > 5000:
                recommendations.append({
                    "type": "performance",
                    "issue": "slow_response_time",
                    "recommendation": "Consider switching to FAST model tier or reducing context chunks",
                    "suggested_changes": {
                        "model_tier": "fast",
                        "final_context_chunks": max(3, min(5, int(avg_response_time / 1000)))
                    }
                })

            # Quality recommendations
            if avg_context_quality < 0.7:
                recommendations.append({
                    "type": "quality",
                    "issue": "low_context_quality",
                    "recommendation": "Increase retrieval count and enable semantic re-ranking",
                    "suggested_changes": {
                        "initial_retrieval_count": 25,
                        "enable_semantic_rerank": True,
                        "semantic_rerank_count": 12
                    }
                })

            # Satisfaction recommendations
            if avg_satisfaction < 0.6:
                recommendations.append({
                    "type": "satisfaction",
                    "issue": "low_user_satisfaction",
                    "recommendation": "Enable hallucination checking and increase confidence threshold",
                    "suggested_changes": {
                        "enable_hallucination_check": True,
                        "confidence_threshold": 0.8,
                        "model_tier": "premium"
                    }
                })

            return {
                "recommendations": recommendations,
                # More data = higher confidence
                "confidence": min(1.0, len(metrics) / 100),
                "metrics_summary": {
                    "avg_response_time_ms": avg_response_time,
                    "avg_satisfaction": avg_satisfaction,
                    "avg_context_quality": avg_context_quality,
                    "sample_size": len(metrics)
                }
            }

        except Exception as e:
            logging.error(f"Error getting performance recommendations: {e}")
            return {"recommendations": [], "confidence": 0}


# Global instance for easy access
context_config_manager = ContextConfigManager()
