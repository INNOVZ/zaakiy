#!/usr/bin/env python3
"""
Environment Configuration Checker
Validates that all required environment variables are set
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.env_loader import safe_load_dotenv

# Load environment variables
safe_load_dotenv()


class Colors:
    """ANSI color codes"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def check_env_var(
    name: str, required: bool = True, sensitive: bool = False
) -> tuple[bool, str]:
    """
    Check if environment variable is set

    Returns:
        (is_set, value_display)
    """
    value = os.getenv(name)

    if not value:
        return False, f"{Colors.RED}❌ NOT SET{Colors.END}"

    # Mask sensitive values
    if sensitive:
        if len(value) > 10:
            display = f"{value[:4]}...{value[-4:]}"
        else:
            display = "****"
        return True, f"{Colors.GREEN}✅ SET{Colors.END} ({display})"
    else:
        return True, f"{Colors.GREEN}✅ SET{Colors.END} ({value})"


def main():
    """Main environment checker"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}🔍 Environment Configuration Checker{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

    # Track issues
    errors = []
    warnings = []

    # Critical environment variables
    print(f"{Colors.BOLD}📌 Critical Variables (REQUIRED):{Colors.END}\n")

    critical_vars = [
        ("SUPABASE_URL", False),
        ("SUPABASE_SERVICE_ROLE_KEY", True),
        ("SUPABASE_JWT_SECRET", True),
        ("OPENAI_API_KEY", True),
        ("PINECONE_API_KEY", True),
        ("PINECONE_INDEX", False),
    ]

    for var_name, is_sensitive in critical_vars:
        is_set, display = check_env_var(var_name, required=True, sensitive=is_sensitive)
        print(f"  {var_name:30} {display}")
        if not is_set:
            errors.append(f"Missing critical variable: {var_name}")

    # Redis configuration (NEW - required for caching)
    print(f"\n{Colors.BOLD}🔴 Redis Configuration (REQUIRED for caching):{Colors.END}\n")

    redis_vars = [
        ("REDIS_URL", False),
        ("REDIS_PASSWORD", True),
    ]

    for var_name, is_sensitive in redis_vars:
        is_set, display = check_env_var(
            var_name, required=False, sensitive=is_sensitive
        )
        print(f"  {var_name:30} {display}")
        if not is_set and var_name == "REDIS_URL":
            warnings.append(f"Redis URL not set - caching will be degraded")

    # Optional configuration
    print(f"\n{Colors.BOLD}⚙️  Optional Configuration:{Colors.END}\n")

    optional_vars = [
        ("ENVIRONMENT", False),
        ("DEBUG", False),
        ("LOG_LEVEL", False),
        ("ENABLE_CACHING", False),
        ("MAX_CONCURRENT_REQUESTS", False),
    ]

    for var_name, is_sensitive in optional_vars:
        is_set, display = check_env_var(
            var_name, required=False, sensitive=is_sensitive
        )
        print(f"  {var_name:30} {display}")

    # Test Redis connection
    print(f"\n{Colors.BOLD}🧪 Testing Redis Connection:{Colors.END}\n")
    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_password = os.getenv("REDIS_PASSWORD")

        r = redis.from_url(
            redis_url,
            password=redis_password if redis_password else None,
            socket_connect_timeout=5,
            decode_responses=True,
        )

        # Test ping
        r.ping()
        print(f"  {Colors.GREEN}✅ Redis connection successful{Colors.END}")
        print(f"     Server: {redis_url}")

        # Get server info
        info = r.info("server")
        print(f"     Version: {info.get('redis_version', 'unknown')}")
        print(f"     Uptime: {info.get('uptime_in_seconds', 0)} seconds")

    except ImportError:
        warnings.append("Redis Python client not installed (pip install redis)")
        print(f"  {Colors.YELLOW}⚠️  Redis client not installed{Colors.END}")
        print(f"     Run: pip install redis")
    except redis.ConnectionError as e:
        warnings.append(f"Cannot connect to Redis: {e}")
        print(f"  {Colors.YELLOW}⚠️  Cannot connect to Redis{Colors.END}")
        print(f"     Error: {str(e)}")
        print(f"     Run: redis-cli ping")
    except Exception as e:
        warnings.append(f"Redis test failed: {e}")
        print(f"  {Colors.RED}❌ Redis test failed{Colors.END}")
        print(f"     Error: {str(e)}")

    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}📊 Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

    if not errors and not warnings:
        print(
            f"{Colors.GREEN}✅ All checks passed! Environment is properly configured.{Colors.END}\n"
        )
        return 0

    if errors:
        print(f"{Colors.RED}❌ ERRORS ({len(errors)}):{Colors.END}")
        for error in errors:
            print(f"  • {error}")
        print()

    if warnings:
        print(f"{Colors.YELLOW}⚠️  WARNINGS ({len(warnings)}):{Colors.END}")
        for warning in warnings:
            print(f"  • {warning}")
        print()

    if errors:
        print(
            f"{Colors.RED}🚨 Critical errors found. Please fix before deploying.{Colors.END}\n"
        )
        return 1
    else:
        print(
            f"{Colors.YELLOW}⚠️  Warnings found. Review before deploying to production.{Colors.END}\n"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
