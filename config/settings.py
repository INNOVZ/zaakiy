import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class DatabaseSettings(BaseModel):
    """Database configuration settings"""
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(...,
                                      description="Supabase service role key")
    supabase_jwt_secret: str = Field(..., description="Supabase JWT secret")
    supabase_project_id: str = Field(..., description="Supabase project ID")

    def __init__(self, **data):
        # Auto-populate from environment if not provided
        if 'supabase_url' not in data:
            data['supabase_url'] = os.getenv("SUPABASE_URL", "")
        if 'supabase_service_key' not in data:
            data['supabase_service_key'] = os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY", "")
        if 'supabase_jwt_secret' not in data:
            data['supabase_jwt_secret'] = os.getenv("SUPABASE_JWT_SECRET", "")
        if 'supabase_project_id' not in data:
            data['supabase_project_id'] = os.getenv("SUPABASE_PROJECT_ID", "")
        super().__init__(**data)


class AISettings(BaseModel):
    """AI service configuration settings"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index: str = Field(..., description="Pinecone index name")

    # Model defaults
    default_embedding_model: str = Field(default="text-embedding-3-small")
    default_chat_model: str = Field(default="gpt-4")
    max_tokens_default: int = Field(default=1000)
    temperature_default: float = Field(default=0.7)

    def __init__(self, **data):
        if 'openai_api_key' not in data:
            data['openai_api_key'] = os.getenv("OPENAI_API_KEY", "")
        if 'pinecone_api_key' not in data:
            data['pinecone_api_key'] = os.getenv("PINECONE_API_KEY", "")
        if 'pinecone_index' not in data:
            data['pinecone_index'] = os.getenv("PINECONE_INDEX", "")
        super().__init__(**data)


class SecuritySettings(BaseModel):
    """Security configuration settings"""
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)  # seconds
    max_upload_size: int = Field(default=10485760)  # 10MB
    allowed_file_types: List[str] = Field(default=["pdf", "json", "txt"])

    def __init__(self, **data):
        if 'rate_limit_requests' not in data:
            data['rate_limit_requests'] = int(
                os.getenv("RATE_LIMIT_REQUESTS", "100"))
        if 'rate_limit_window' not in data:
            data['rate_limit_window'] = int(
                os.getenv("RATE_LIMIT_WINDOW", "60"))
        if 'max_upload_size' not in data:
            data['max_upload_size'] = int(
                os.getenv("MAX_UPLOAD_SIZE", "10485760"))
        super().__init__(**data)


class PerformanceSettings(BaseModel):
    """Performance and optimization settings"""
    max_concurrent_requests: int = Field(default=50)
    request_timeout: int = Field(default=30)
    cache_ttl: int = Field(default=3600)  # 1 hour
    background_worker_interval: int = Field(default=30)

    # Context engineering defaults
    max_context_length: int = Field(default=4000)
    max_retrieval_count: int = Field(default=20)
    max_conversation_history: int = Field(default=10)

    def __init__(self, **data):
        if 'max_concurrent_requests' not in data:
            data['max_concurrent_requests'] = int(
                os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
        if 'request_timeout' not in data:
            data['request_timeout'] = int(os.getenv("REQUEST_TIMEOUT", "30"))
        if 'cache_ttl' not in data:
            data['cache_ttl'] = int(os.getenv("CACHE_TTL", "3600"))
        if 'background_worker_interval' not in data:
            data['background_worker_interval'] = int(
                os.getenv("WORKER_INTERVAL", "30"))
        super().__init__(**data)


class ApplicationSettings(BaseModel):
    """Main application settings"""
    app_name: str = Field(default="ZaaKy AI Platform")
    app_version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    api_base_url: str = Field(default="http://localhost:8001")

    # Feature flags
    enable_analytics: bool = Field(default=True)
    enable_caching: bool = Field(default=True)
    enable_rate_limiting: bool = Field(default=True)

    def __init__(self, **data):
        if 'environment' not in data:
            data['environment'] = os.getenv("ENVIRONMENT", "development")
        if 'debug' not in data:
            data['debug'] = os.getenv("DEBUG", "false").lower() == "true"
        if 'api_base_url' not in data:
            data['api_base_url'] = os.getenv(
                "API_BASE_URL", "http://localhost:8001")
        if 'enable_analytics' not in data:
            data['enable_analytics'] = os.getenv(
                "ENABLE_ANALYTICS", "true").lower() == "true"
        if 'enable_caching' not in data:
            data['enable_caching'] = os.getenv(
                "ENABLE_CACHING", "true").lower() == "true"
        if 'enable_rate_limiting' not in data:
            data['enable_rate_limiting'] = os.getenv(
                "ENABLE_RATE_LIMITING", "true").lower() == "true"
        super().__init__(**data)


class Settings:
    """Main settings class combining all configuration"""

    def __init__(self):
        self.database = DatabaseSettings()
        self.ai = AISettings()
        self.security = SecuritySettings()
        self.performance = PerformanceSettings()
        self.app = ApplicationSettings()

    def validate_all(self) -> Dict[str, Any]:
        """Validate all settings and return validation results"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check required environment variables
        required_vars = [
            "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_JWT_SECRET",
            "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Missing required environment variables: {missing_vars}")

        # Validate URL formats
        if not self.database.supabase_url.startswith(("http://", "https://")):
            validation_results["valid"] = False
            validation_results["errors"].append(
                "SUPABASE_URL must be a valid URL")

        # Check performance settings
        if self.performance.max_context_length > 8000:
            validation_results["warnings"].append(
                "Max context length is very high, may impact performance")

        if self.performance.max_concurrent_requests > 100:
            validation_results["warnings"].append(
                "High concurrent request limit may impact stability")

        return validation_results

    def get_model_config(self, tier: str) -> Dict[str, Any]:
        """Get model configuration based on tier"""
        configs = {
            "fast": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 800,
                "temperature": 0.7,
                "timeout": 3000
            },
            "balanced": {
                "model": "gpt-4",
                "max_tokens": self.ai.max_tokens_default,
                "temperature": self.ai.temperature_default,
                "timeout": 5000
            },
            "premium": {
                "model": "gpt-4-turbo",
                "max_tokens": 1500,
                "temperature": 0.6,
                "timeout": 8000
            },
            "enterprise": {
                "model": "gpt-4-turbo",
                "max_tokens": 2000,
                "temperature": 0.5,
                "timeout": 10000
            }
        }
        return configs.get(tier, configs["balanced"])


# Global settings instance
settings = Settings()


# Environment validation function
def validate_environment() -> None:
    """Validate that all required environment variables are set"""
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_JWT_SECRET",
        "SUPABASE_PROJECT_ID",
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}")

    print("✅ All required environment variables are configured")


# Convenience functions for getting specific settings
def get_database_url() -> str:
    """Get database URL"""
    return settings.database.supabase_url


def get_api_keys() -> Dict[str, str]:
    """Get all API keys"""
    return {
        "openai": settings.ai.openai_api_key,
        "pinecone": settings.ai.pinecone_api_key,
        "supabase": settings.database.supabase_service_key
    }


def is_production() -> bool:
    """Check if running in production environment"""
    return settings.app.environment.lower() == "production"


def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return settings.app.debug
