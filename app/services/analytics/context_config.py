import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field

from ..shared.cache_service import cache_service
from ..storage.supabase_client import get_supabase_client


class RetrievalStrategy(str, Enum):
    """Retrieval strategy options"""

    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    KEYWORD_BOOST = "keyword_boost"
    DOMAIN_SPECIFIC = "domain_specific"


class ModelTier(str, Enum):
    """AI model tiers for different use cases"""

    FAST = "fast"  # gpt-3.5-turbo
    BALANCED = "balanced"  # gpt-4
    PREMIUM = "premium"  # gpt-4-turbo, claude-3
    ENTERPRISE = "enterprise"  # Custom fine-tuned models


# Update the ContextEngineeringConfig class (around line 30)

# Update the ContextEngineeringConfig class (around line 30)


class ContextEngineeringConfig(BaseModel):
    """Configuration model for context engineering parameters"""

    # Chatbot identification
    # chatbot_id: str
    org_id: str
    config_name: str = "default"

    # Retrieval Configuration
    initial_retrieval_count: int = Field(
        default=30, ge=5, le=50, description="Initial document retrieval count"
    )
    semantic_rerank_count: int = Field(
        default=25, ge=3, le=30, description="Documents to semantically re-rank"
    )
    final_context_chunks: int = Field(
        default=10, ge=1, le=12, description="Final chunks to include in context"
    )
    max_context_length: int = Field(
        default=4000, ge=1000, le=8000, description="Maximum context length in tokens"
    )

    # Query Enhancement
    enable_query_rewriting: bool = Field(
        default=True, description="Enable intelligent query rewriting"
    )
    max_query_variants: int = Field(
        default=3, ge=1, le=5, description="Maximum query variants to generate"
    )
    conversation_context_turns: int = Field(
        default=3, ge=1, le=10, description="Previous conversation turns to consider"
    )

    # Retrieval Strategy
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID, description="Retrieval strategy to use"
    )
    semantic_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight for semantic similarity"
    )
    keyword_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for keyword matching"
    )

    # Model Configuration
    model_tier: ModelTier = Field(
        default=ModelTier.BALANCED, description="AI model tier to use"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model"
    )
    rerank_model: str = Field(
        default="gpt-3.5-turbo", description="Model for semantic re-ranking"
    )

    # Quality Controls
    enable_semantic_rerank: bool = Field(
        default=True, description="Enable semantic re-ranking"
    )
    enable_hallucination_check: bool = Field(
        default=True, description="Enable hallucination detection"
    )
    enable_source_verification: bool = Field(
        default=True, description="Enable source verification"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for responses"
    )

    # Performance Settings
    max_response_time_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Maximum response time in milliseconds",
    )
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_minutes: int = Field(
        default=60, ge=5, le=1440, description="Cache TTL in minutes"
    )

    # Monitoring & Analytics
    enable_detailed_logging: bool = Field(
        default=True, description="Enable detailed context logging"
    )
    log_user_queries: bool = Field(
        default=True, description="Log user queries for analytics"
    )
    collect_feedback: bool = Field(default=True, description="Collect user feedback")

    # Context & Behavior Fields - ADDED
    business_context: str = Field(
        default="",
        description="Organization-specific business context and industry information",
    )
    domain_knowledge: str = Field(
        default="", description="Domain-specific knowledge and terminology"
    )
    response_style: str = Field(
        default="professional", description="Preferred response style"
    )
    fallback_behavior: str = Field(
        default="apologetic", description="Behavior when no relevant context is found"
    )
    specialized_instructions: str = Field(
        default="", description="Specialized instructions for the AI assistant"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else v
        }


class ContextConfigManager:
    """Manager for context engineering configurations"""

    def __init__(self):
        self.supabase = get_supabase_client()

    async def get_config(
        self, org_id: str, config_name: str = "default"
    ) -> ContextEngineeringConfig:
        """Get context engineering configuration for an organization with Redis caching"""
        try:
            logging.info("Getting context config for org %s", org_id)

            # Check cache first
            cached_config = await self._get_cached_config(org_id, config_name)
            if cached_config:
                return cached_config

            # Get existing config from database
            config_row = await self._fetch_config_from_db(org_id, config_name)

            if config_row:
                return await self._process_existing_config(
                    config_row, org_id, config_name
                )
            else:
                return await self._create_default_config(org_id, config_name)

        except Exception as e:
            logging.error("Error getting config for org %s: %s", org_id, e)
            # Return basic default config as fallback
            return ContextEngineeringConfig(org_id=org_id, config_name=config_name)

    async def _get_cached_config(
        self, org_id: str, config_name: str
    ) -> Optional[ContextEngineeringConfig]:
        """Get configuration from cache if available"""
        cache_key = f"context_config:{org_id}:{config_name}"
        cached_config = await cache_service.get(cache_key)

        if cached_config:
            logging.info("Cache hit for context config: %s", cache_key)
            return ContextEngineeringConfig(**cached_config)

        return None

    async def _fetch_config_from_db(
        self, org_id: str, config_name: str
    ) -> Optional[dict]:
        """Fetch configuration from database"""
        response = (
            self.supabase.table("context_engineering_configs")
            .select("*")
            .eq("org_id", org_id)
            .eq("config_name", config_name)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    async def _process_existing_config(
        self, config_row: dict, org_id: str, config_name: str
    ) -> ContextEngineeringConfig:
        """Process existing configuration from database"""
        # FIXED: Always use config_data JSON blob for configuration
        if "config_data" in config_row and isinstance(config_row["config_data"], dict):
            return await self._process_config_data(config_row, org_id, config_name)
        else:
            return await self._process_fallback_config(config_row, org_id, config_name)

    async def _process_config_data(
        self, config_row: dict, org_id: str, config_name: str
    ) -> ContextEngineeringConfig:
        """Process configuration using config_data field"""
        config_data = config_row["config_data"]

        # Ensure required fields are present (fallback to top-level if needed)
        config_data.setdefault("org_id", config_row.get("org_id", org_id))
        config_data.setdefault(
            "config_name", config_row.get("config_name", config_name)
        )

        config_obj = ContextEngineeringConfig(**config_data)
        await self._cache_config(config_obj, config_data, org_id, config_name)

        return config_obj

    async def _process_fallback_config(
        self, config_row: dict, org_id: str, config_name: str
    ) -> ContextEngineeringConfig:
        """Process configuration using fallback top-level fields"""
        logging.warning(
            "No config_data found for org %s, using top-level fields", org_id
        )
        config_obj = ContextEngineeringConfig(**config_row)
        await self._cache_config(config_obj, config_row, org_id, config_name)

        return config_obj

    async def _cache_config(
        self,
        config_obj: ContextEngineeringConfig,
        config_data: dict,
        org_id: str,
        config_name: str,
    ) -> None:
        """Cache configuration with appropriate TTL"""
        cache_key = f"context_config:{org_id}:{config_name}"
        ttl_seconds = (
            config_obj.cache_ttl_minutes * 60 if config_obj.enable_caching else 300
        )
        await cache_service.set(cache_key, config_data, ttl_seconds)
        logging.info("Cached context config: %s (TTL: %ds)", cache_key, ttl_seconds)

    async def _create_default_config(
        self, org_id: str, config_name: str
    ) -> ContextEngineeringConfig:
        """Create and save default configuration"""
        logging.info("Creating default config for org %s", org_id)
        default_config = ContextEngineeringConfig(
            org_id=org_id, config_name=config_name
        )

        # Save to database
        await self.save_config(default_config)

        # Cache the new default config
        cache_key = f"context_config:{org_id}:{config_name}"
        ttl_seconds = (
            default_config.cache_ttl_minutes * 60
            if default_config.enable_caching
            else 300
        )
        await cache_service.set(cache_key, default_config.dict(), ttl_seconds)

        return default_config

    async def save_config(self, config: ContextEngineeringConfig) -> bool:
        """Save context engineering configuration with proper datetime handling"""
        try:
            logging.info("Saving config for org %s", config.org_id)

            # Ensure timestamps are datetime objects
            now = datetime.now(timezone.utc)
            config.updated_at = now

            # If created_at is a string, convert it to datetime
            if isinstance(config.created_at, str):
                try:
                    config.created_at = datetime.fromisoformat(
                        config.created_at.replace("Z", "+00:00")
                    )
                except ValueError:
                    config.created_at = now

            # Convert config to dict with proper datetime serialization
            config_dict = config.model_dump()

            # Manually handle datetime serialization
            for key, value in config_dict.items():
                if isinstance(value, datetime):
                    config_dict[key] = value.isoformat()

            config_data = {
                # Top-level metadata for queries/indexing
                "org_id": config.org_id,
                "config_name": config.config_name,
                "created_at": config.created_at.isoformat()
                if isinstance(config.created_at, datetime)
                else config.created_at,
                "updated_at": now.isoformat(),
                # All configuration details in JSON blob
                "config_data": config_dict,
            }

            # Upsert configuration
            response = (
                self.supabase.table("context_engineering_configs")
                .upsert(config_data, on_conflict="org_id,config_name")
                .execute()
            )

            success = bool(response.data)

            if success:
                # Invalidate cache after successful save
                cache_key = f"context_config:{config.org_id}:{config.config_name}"
                await cache_service.delete(cache_key)
                logging.info("Invalidated cache for config: %s", cache_key)

            logging.info("Config save result for org %s: %s", config.org_id, success)
            return success

        except Exception as e:
            logging.error(
                "Error saving context config for org %s: %s", config.org_id, e
            )
            return False

    def _get_valid_fields(self) -> Set[str]:
        """Get valid field names for ContextEngineeringConfig - Pylint safe"""
        # Use a hardcoded set of valid fields to avoid Pylint issues
        # This is the most reliable approach for static analysis
        valid_fields = {
            "org_id",
            "config_name",
            "initial_retrieval_count",
            "semantic_rerank_count",
            "final_context_chunks",
            "max_context_length",
            "enable_query_rewriting",
            "max_query_variants",
            "conversation_context_turns",
            "retrieval_strategy",
            "semantic_weight",
            "keyword_weight",
            "model_tier",
            "embedding_model",
            "rerank_model",
            "enable_semantic_rerank",
            "enable_hallucination_check",
            "enable_source_verification",
            "confidence_threshold",
            "max_response_time_ms",
            "enable_caching",
            "cache_ttl_minutes",
            "enable_detailed_logging",
            "log_user_queries",
            "collect_feedback",
            "created_at",
            "updated_at",
            "business_context",
            "domain_knowledge",
            "response_style",
            "fallback_behavior",
        }

        # Try to get dynamic fields as backup, but fallback to hardcoded set
        try:
            model_fields = getattr(ContextEngineeringConfig, "model_fields", None)
            if model_fields and hasattr(model_fields, "keys"):
                dynamic_fields = set(model_fields.keys())
                return dynamic_fields if dynamic_fields else valid_fields
        except (AttributeError, TypeError):
            pass

        try:
            fields = getattr(ContextEngineeringConfig, "__fields__", None)
            if fields and hasattr(fields, "keys"):
                dynamic_fields = set(fields.keys())
                return dynamic_fields if dynamic_fields else valid_fields
        except (AttributeError, TypeError):
            pass

        try:
            annotations = getattr(ContextEngineeringConfig, "__annotations__", {})
            if annotations and hasattr(annotations, "keys"):
                dynamic_fields = set(annotations.keys())
                return dynamic_fields if dynamic_fields else valid_fields
        except (AttributeError, TypeError):
            pass

        return valid_fields

    def _get_field_type(self, field_name: str) -> Optional[type]:
        """Get field type for a given field name - Pylint safe"""
        # Try different approaches to get field type
        try:
            model_fields = getattr(ContextEngineeringConfig, "model_fields", None)
            if model_fields and hasattr(model_fields, "get"):
                field_info = model_fields.get(field_name)
                if field_info and hasattr(field_info, "annotation"):
                    return field_info.annotation
        except (AttributeError, TypeError):
            pass

        try:
            fields = getattr(ContextEngineeringConfig, "__fields__", None)
            if fields and hasattr(fields, "get"):
                field_info = fields.get(field_name)
                if field_info and hasattr(field_info, "type_"):
                    return field_info.type_
        except (AttributeError, TypeError):
            pass

        try:
            annotations = getattr(ContextEngineeringConfig, "__annotations__", {})
            if annotations and hasattr(annotations, "get"):
                return annotations.get(field_name)
        except (AttributeError, TypeError):
            pass

        return None

    def _is_enum_type(self, field_type: Any) -> bool:
        """Check if a field type is an enum - Pylint safe"""
        if not field_type:
            return False

        try:
            # Handle Union types (like Optional[Enum])
            if hasattr(field_type, "__origin__"):
                if hasattr(field_type, "__args__"):
                    args = getattr(field_type, "__args__", ())
                    for arg in args:
                        if arg != type(None) and self._is_enum_type(arg):
                            return True
                return False

            # Check if it's a direct enum type
            if isinstance(field_type, type):
                return issubclass(field_type, Enum)
        except (TypeError, AttributeError):
            pass

        return False

    def _set_field_value(
        self, config: ContextEngineeringConfig, key: str, value: Any
    ) -> bool:
        """Set field value with proper type handling - Pylint safe"""
        try:
            field_type = self._get_field_type(key)

            if self._is_enum_type(field_type):
                # Handle enum types
                if isinstance(value, str):
                    setattr(config, key, value)
                elif field_type and isinstance(field_type, type):
                    setattr(config, key, field_type(value))
                else:
                    setattr(config, key, value)
            else:
                # Handle regular types
                setattr(config, key, value)

            return True
        except (ValueError, TypeError) as e:
            logging.error("Error setting field %s to %s: %s", key, value, e)
            return False

    async def update_config(
        self, org_id: str, updates: Dict[str, Any], config_name: str = "default"
    ) -> bool:
        """Update specific configuration parameters"""
        try:
            logging.info("Updating config for org %s with updates: %s", org_id, updates)

            # Get current config
            current_config = await self.get_config(org_id, config_name)

            # Get valid fields using safe method
            valid_fields = self._get_valid_fields()

            # Filter out invalid fields
            valid_updates = {}
            invalid_fields = []

            for key, value in updates.items():
                if key in valid_fields:
                    valid_updates[key] = value
                else:
                    invalid_fields.append(key)

            if invalid_fields:
                logging.warning(
                    "Ignoring invalid fields in updates: %s", invalid_fields
                )

            if not valid_updates:
                logging.warning("No valid updates provided")
                return False

            # Apply valid updates
            success_count = 0
            for key, value in valid_updates.items():
                if hasattr(current_config, key):
                    if self._set_field_value(current_config, key, value):
                        success_count += 1

            if success_count == 0:
                logging.error("Failed to apply any updates")
                return False

            # Save updated config
            success = await self.save_config(current_config)
            logging.info(
                "Config update result for org %s: %s (%s/%s fields updated)",
                org_id,
                success,
                success_count,
                len(valid_updates),
            )
            return success

        except Exception as e:
            logging.error("Error updating context config for org %s: %s", org_id, e)
            return False

    def get_model_config(self, tier: ModelTier) -> Dict[str, Any]:
        """Get model configuration based on tier"""
        model_configs: Dict[ModelTier, Dict[str, Any]] = {
            ModelTier.FAST: {
                "model": "gpt-3.5-turbo",
                "max_tokens": 800,
                "temperature": 0.7,
                "timeout": 3000,
            },
            ModelTier.BALANCED: {
                "model": "gpt-4",
                "max_tokens": 1000,
                "temperature": 0.7,
                "timeout": 5000,
            },
            ModelTier.PREMIUM: {
                "model": "gpt-4-turbo",
                "max_tokens": 1500,
                "temperature": 0.6,
                "timeout": 8000,
            },
            ModelTier.ENTERPRISE: {
                "model": "gpt-4-turbo",
                "max_tokens": 2000,
                "temperature": 0.5,
                "timeout": 10000,
            },
        }

        return model_configs.get(tier, model_configs[ModelTier.BALANCED])

    async def get_performance_recommendations(self, org_id: str) -> Dict[str, Any]:
        """Get performance-based configuration recommendations"""
        try:
            # Get recent performance metrics
            response = (
                self.supabase.table("context_analytics")
                .select("*")
                .eq("org_id", org_id)
                .gte(
                    "created_at",
                    (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                )
                .execute()
            )

            if not response.data:
                return {"recommendations": [], "confidence": 0}

            # Analyze performance data
            metrics = response.data
            avg_response_time = sum(
                m.get("response_time_ms", 0) for m in metrics
            ) / len(metrics)
            avg_satisfaction = sum(
                m.get("user_satisfaction", 0.5) for m in metrics
            ) / len(metrics)
            avg_context_quality = sum(
                m.get("context_quality_score", 0.5) for m in metrics
            ) / len(metrics)

            recommendations = []

            # Response time recommendations
            if avg_response_time > 5000:
                recommendations.append(
                    {
                        "type": "performance",
                        "issue": "slow_response_time",
                        "recommendation": "Consider switching to FAST model tier or reducing context chunks",
                        "suggested_changes": {
                            "model_tier": "fast",
                            "final_context_chunks": max(
                                3, min(5, int(avg_response_time / 1000))
                            ),
                        },
                    }
                )

            # Quality recommendations
            if avg_context_quality < 0.7:
                recommendations.append(
                    {
                        "type": "quality",
                        "issue": "low_context_quality",
                        "recommendation": "Increase retrieval count and enable semantic re-ranking",
                        "suggested_changes": {
                            "initial_retrieval_count": 25,
                            "enable_semantic_rerank": True,
                            "semantic_rerank_count": 12,
                        },
                    }
                )

            # Satisfaction recommendations
            if avg_satisfaction < 0.6:
                recommendations.append(
                    {
                        "type": "satisfaction",
                        "issue": "low_user_satisfaction",
                        "recommendation": "Enable hallucination checking and increase confidence threshold",
                        "suggested_changes": {
                            "enable_hallucination_check": True,
                            "confidence_threshold": 0.8,
                            "model_tier": "premium",
                        },
                    }
                )

            return {
                "recommendations": recommendations,
                "confidence": min(1.0, len(metrics) / 100),
                "metrics_summary": {
                    "avg_response_time_ms": avg_response_time,
                    "avg_satisfaction": avg_satisfaction,
                    "avg_context_quality": avg_context_quality,
                    "sample_size": len(metrics),
                },
            }

        except Exception as e:
            logging.error("Error getting performance recommendations: %s", e)
            return {"recommendations": [], "confidence": 0}


# Global instance for easy access
context_config_manager = ContextConfigManager()
