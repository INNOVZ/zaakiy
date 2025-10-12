"""
Centralized configuration management with validation and type safety
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Constants
HTTPS_PREFIX = "https://"
HTTP_PREFIX = "http://"

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration with validation"""

    supabase_url: str
    supabase_service_key: str
    supabase_jwt_secret: str
    supabase_project_id: str

    def validate(self) -> Dict[str, Any]:
        """Validate database configuration with strict requirements"""
        errors = []
        warnings = []

        # CRITICAL: These are required for basic functionality
        if not self.supabase_url:
            errors.append(
                "SUPABASE_URL is REQUIRED - server cannot function without it"
            )
        elif not self.supabase_url.startswith(HTTPS_PREFIX):
            warnings.append("SUPABASE_URL should use HTTPS")

        if not self.supabase_service_key:
            errors.append(
                "SUPABASE_SERVICE_ROLE_KEY is REQUIRED - "
                "database operations will fail"
            )
        elif len(self.supabase_service_key) < 50:
            warnings.append("SUPABASE_SERVICE_ROLE_KEY seems too short")

        if not self.supabase_jwt_secret:
            errors.append("SUPABASE_JWT_SECRET is REQUIRED - authentication will fail")

        # Project ID is less critical, but recommended
        if not self.supabase_project_id:
            warnings.append(
                "SUPABASE_PROJECT_ID is missing - some features may not work"
            )

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@dataclass
class AIServiceConfig:
    """AI service configuration with validation"""

    openai_api_key: str
    pinecone_api_key: str
    pinecone_index: str
    default_model: str = "gpt-4"
    default_temperature: float = 0.7
    max_tokens: int = 1000
    embedding_model: str = "text-embedding-3-small"

    def validate(self) -> Dict[str, Any]:
        """Validate AI service configuration with strict requirements"""
        errors = []
        warnings = []

        # CRITICAL: These are required for AI functionality
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is REQUIRED - AI features will not work")
        elif not self.openai_api_key.startswith("sk-"):
            errors.append("OPENAI_API_KEY format is invalid - must start with 'sk-'")

        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is REQUIRED - vector operations will fail")

        if not self.pinecone_index:
            errors.append("PINECONE_INDEX is REQUIRED - document search will not work")

        # These are warnings, not errors
        if not 0.0 <= self.default_temperature <= 2.0:
            warnings.append("Temperature should be between 0.0 and 2.0")

        if self.max_tokens < 100 or self.max_tokens > 4000:
            warnings.append("max_tokens should be between 100 and 4000")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@dataclass
class SecurityConfig:
    """Security configuration with validation"""

    cors_origins: List[str]
    allowed_hosts: List[str]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_auth_validation: bool = True

    def validate(self) -> Dict[str, Any]:
        """Validate security configuration"""
        errors = []
        warnings = []

        if "*" in self.cors_origins:
            warnings.append(
                "CORS allows all origins - " "consider restricting in production"
            )

        if not self.allowed_hosts:
            warnings.append("No allowed hosts specified")

        if self.rate_limit_requests > 1000:
            warnings.append("Rate limit seems very high")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@dataclass
class AppConfig:
    """Application configuration with validation"""

    app_name: str = "ZaaKy AI Platform"
    app_version: str = "2.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    enable_analytics: bool = True
    enable_caching: bool = True
    api_base_url: str = f"{HTTP_PREFIX}localhost:8001"

    def validate(self) -> Dict[str, Any]:
        """Validate application configuration"""
        errors = []
        warnings = []

        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            errors.append(f"Environment must be one of: {valid_environments}")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")

        if self.environment == "production" and self.debug:
            warnings.append("Debug mode enabled in production")

        if not self.api_base_url.startswith((HTTP_PREFIX, HTTPS_PREFIX)):
            warnings.append("API_BASE_URL should include protocol (http/https)")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@dataclass
class PerformanceConfig:
    """Performance configuration with validation"""

    max_concurrent_requests: int = 100
    request_timeout: int = 30
    connection_pool_size: int = 20
    enable_compression: bool = True
    worker_interval_seconds: int = 30

    def validate(self) -> Dict[str, Any]:
        """Validate performance configuration"""
        errors = []
        warnings = []

        if self.max_concurrent_requests < 10:
            warnings.append("max_concurrent_requests seems low")
        elif self.max_concurrent_requests > 1000:
            warnings.append("max_concurrent_requests seems very high")

        if self.request_timeout < 5:
            warnings.append("request_timeout seems very low")
        elif self.request_timeout > 120:
            warnings.append("request_timeout seems very high")

        if self.worker_interval_seconds < 10:
            warnings.append("worker_interval_seconds might be too frequent")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@dataclass
class WebScrapingConfig:
    """Web scraping configuration with security and performance settings"""

    timeout: int = 30
    max_content_size: int = 50 * 1024 * 1024  # 50MB
    min_delay_between_requests: float = 1.0
    max_delay_between_requests: float = 3.0
    max_retries: int = 3
    concurrent_requests: int = 3
    respect_robots_txt: bool = True
    enable_ssrf_protection: bool = True
    max_pages_per_site: int = 100
    max_depth: int = 5
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None

    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = []
        if self.blocked_domains is None:
            self.blocked_domains = [
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
                "169.254.169.254",  # AWS metadata
                "metadata.google.internal",  # GCP metadata
            ]

    def validate(self) -> Dict[str, Any]:
        """Validate web scraping configuration"""
        errors = []
        warnings = []

        if self.timeout < 5:
            warnings.append("scraping timeout seems very low")
        elif self.timeout > 120:
            warnings.append("scraping timeout seems very high")

        if self.max_content_size > 100 * 1024 * 1024:  # 100MB
            warnings.append("max_content_size is very high (>100MB)")

        if self.min_delay_between_requests < 0.5:
            warnings.append("min_delay_between_requests might be too aggressive")

        if self.concurrent_requests > 10:
            warnings.append("concurrent_requests might overwhelm target servers")

        if not self.enable_ssrf_protection:
            errors.append("SSRF protection should be enabled in production")

        if self.max_pages_per_site > 1000:
            warnings.append("max_pages_per_site is very high")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


class ConfigurationManager:
    """Centralized configuration manager with comprehensive validation"""

    def __init__(self):
        self.database = DatabaseConfig(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_service_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            supabase_jwt_secret=os.getenv("SUPABASE_JWT_SECRET", ""),
            supabase_project_id=os.getenv("SUPABASE_PROJECT_ID", ""),
        )

        self.ai_service = AIServiceConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index=os.getenv("PINECONE_INDEX", ""),
            default_model=os.getenv("DEFAULT_AI_MODEL", "gpt-4"),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "1000")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )

        cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        allowed_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")

        self.security = SecurityConfig(
            cors_origins=[origin.strip() for origin in cors_origins],
            allowed_hosts=[host.strip() for host in allowed_hosts if host.strip()],
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            enable_auth_validation=os.getenv("ENABLE_AUTH_VALIDATION", "true").lower()
            == "true",
        )

        self.app = AppConfig(
            app_name=os.getenv("APP_NAME", "ZaaKy AI Platform"),
            app_version=os.getenv("APP_VERSION", "2.1.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_analytics=os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            api_base_url=os.getenv("API_BASE_URL", f"{HTTP_PREFIX}localhost:8001"),
        )

        self.performance = PerformanceConfig(
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            connection_pool_size=int(os.getenv("CONNECTION_POOL_SIZE", "20")),
            enable_compression=os.getenv("ENABLE_COMPRESSION", "true").lower()
            == "true",
            worker_interval_seconds=int(os.getenv("WORKER_INTERVAL_SECONDS", "30")),
        )

        # Web scraping configuration
        allowed_domains = os.getenv("SCRAPING_ALLOWED_DOMAINS", "").split(",")
        blocked_domains = os.getenv("SCRAPING_BLOCKED_DOMAINS", "").split(",")

        self.web_scraping = WebScrapingConfig(
            timeout=int(os.getenv("SCRAPING_TIMEOUT", "30")),
            max_content_size=int(
                os.getenv("SCRAPING_MAX_CONTENT_SIZE", str(50 * 1024 * 1024))
            ),
            min_delay_between_requests=float(os.getenv("SCRAPING_MIN_DELAY", "1.0")),
            max_delay_between_requests=float(os.getenv("SCRAPING_MAX_DELAY", "3.0")),
            max_retries=int(os.getenv("SCRAPING_MAX_RETRIES", "3")),
            concurrent_requests=int(os.getenv("SCRAPING_CONCURRENT_REQUESTS", "3")),
            respect_robots_txt=os.getenv("SCRAPING_RESPECT_ROBOTS", "true").lower()
            == "true",
            enable_ssrf_protection=os.getenv(
                "SCRAPING_ENABLE_SSRF_PROTECTION", "true"
            ).lower()
            == "true",
            max_pages_per_site=int(os.getenv("SCRAPING_MAX_PAGES", "100")),
            max_depth=int(os.getenv("SCRAPING_MAX_DEPTH", "5")),
            allowed_domains=[d.strip() for d in allowed_domains if d.strip()],
            blocked_domains=[d.strip() for d in blocked_domains if d.strip()],
        )

    def validate_all(self) -> Dict[str, Any]:
        """Validate all configuration sections"""
        all_errors = []
        all_warnings = []

        configs = [
            ("database", self.database),
            ("ai_service", self.ai_service),
            ("security", self.security),
            ("app", self.app),
            ("performance", self.performance),
            ("web_scraping", self.web_scraping),
        ]

        for name, config in configs:
            validation = config.validate()

            if validation["errors"]:
                all_errors.extend([f"{name}.{error}" for error in validation["errors"]])

            if validation["warnings"]:
                all_warnings.extend(
                    [f"{name}.{warning}" for warning in validation["warnings"]]
                )

        return {
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "warnings": all_warnings,
            "sections_validated": len(configs),
        }

    def get_env_var(self, key: str, default: Any = None, required: bool = False) -> Any:
        """Get environment variable with validation"""
        value = os.getenv(key, default)

        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set")

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            "app": {
                "name": self.app.app_name,
                "version": self.app.app_version,
                "environment": self.app.environment,
                "debug": self.app.debug,
                "log_level": self.app.log_level,
            },
            "features": {
                "analytics_enabled": self.app.enable_analytics,
                "caching_enabled": self.app.enable_caching,
                "compression_enabled": self.performance.enable_compression,
            },
            "limits": {
                "max_concurrent_requests": (self.performance.max_concurrent_requests),
                "request_timeout": self.performance.request_timeout,
                "rate_limit_requests": self.security.rate_limit_requests,
            },
        }


# Global configuration instance
settings = ConfigurationManager()


def validate_environment() -> None:
    """Validate environment configuration on startup with proper guardrails"""
    validation = settings.validate_all()

    # Critical API keys that MUST be present
    critical_keys = [
        ("SUPABASE_URL", settings.database.supabase_url),
        ("SUPABASE_SERVICE_ROLE_KEY", settings.database.supabase_service_key),
        ("SUPABASE_JWT_SECRET", settings.database.supabase_jwt_secret),
        ("OPENAI_API_KEY", settings.ai_service.openai_api_key),
        ("PINECONE_API_KEY", settings.ai_service.pinecone_api_key),
        ("PINECONE_INDEX", settings.ai_service.pinecone_index),
    ]

    # Check for missing critical keys
    missing_critical = []
    for key_name, key_value in critical_keys:
        if not key_value or key_value.strip() == "":
            missing_critical.append(key_name)

    # STOP SERVER if critical keys are missing
    if missing_critical:
        missing_list = chr(10).join(f"  - {key}" for key in missing_critical)
        env_examples = chr(10).join(
            f"{key}=your_value_here" for key in missing_critical
        )

        error_msg = f"""
❌ CRITICAL ERROR: Missing required environment variables!

The following variables are REQUIRED for the server to function:
{missing_list}

Please add these to your .env file:
{env_examples}

SERVER CANNOT START WITHOUT THESE KEYS!
"""
        logger.error(error_msg)
        print(error_msg)  # Also print to console for visibility
        raise SystemExit(1)  # Exit immediately, don't let server start

    # Check other validation errors
    if not validation["valid"]:
        error_list = "\n".join(validation["errors"])
        error_msg = f"Configuration validation failed:\n{error_list}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log warnings but allow server to continue
    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning("Configuration warning: %s", warning)

    logger.info(
        "✅ All critical environment variables validated successfully " "(%s sections)",
        validation["sections_validated"],
    )


def validate_api_key_formats() -> Dict[str, bool]:
    """Validate API key formats without making API calls"""
    results = {"openai": False, "pinecone": False, "supabase": False}

    # Validate OpenAI API key format
    if settings.ai_service.openai_api_key.startswith("sk-"):
        results["openai"] = True
        logger.info("✅ OpenAI API key format validated")
    else:
        logger.error("❌ OpenAI API key format invalid")

    # Validate Supabase URL and key exist
    if (
        settings.database.supabase_url.startswith(HTTPS_PREFIX)
        and len(settings.database.supabase_service_key) > 50
    ):
        results["supabase"] = True
        logger.info("✅ Supabase configuration validated")
    else:
        logger.error("❌ Supabase configuration invalid")

    # Validate Pinecone API key exists
    if len(settings.ai_service.pinecone_api_key) > 10:
        results["pinecone"] = True
        logger.info("✅ Pinecone API key validated")
    else:
        logger.error("❌ Pinecone API key invalid")

    return results


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return settings.database


def get_ai_config() -> AIServiceConfig:
    """Get AI service configuration"""
    return settings.ai_service


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return settings.security


def get_app_config() -> AppConfig:
    """Get application configuration"""
    return settings.app


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return settings.performance


def get_web_scraping_config() -> WebScrapingConfig:
    """Get web scraping configuration"""
    return settings.web_scraping


def is_production() -> bool:
    """Check if running in production environment"""
    environment_value = getattr(settings.app, "environment", "development")
    return environment_value.lower() == "production"


def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return getattr(settings.app, "debug", False)


def get_cors_origins() -> List[str]:
    """Get CORS origins list"""
    return getattr(settings.security, "cors_origins", [f"{HTTP_PREFIX}localhost:3000"])


def get_performance_limits() -> Dict[str, int]:
    """Get performance configuration"""
    return {
        "max_concurrent_requests": getattr(
            settings.performance, "max_concurrent_requests", 50
        ),
        "request_timeout": getattr(settings.performance, "request_timeout", 30),
        "connection_pool_size": getattr(
            settings.performance, "connection_pool_size", 20
        ),
        "rate_limit_requests": getattr(settings.security, "rate_limit_requests", 100),
    }
