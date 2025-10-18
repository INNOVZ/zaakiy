"""
Safe environment loader utility

Handles dotenv loading gracefully for both production and test environments.
"""
import os
import sys
from pathlib import Path
from typing import Optional


def is_test_environment() -> bool:
    """Check if we're running in a test environment"""
    # Check for pytest
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    
    # Check for explicit test flag
    if os.getenv("TESTING", "false").lower() == "true":
        return True
    
    # Check if running from tests directory
    if "tests" in sys.argv[0]:
        return True
    
    return False


def safe_load_dotenv(verbose: bool = False) -> bool:
    """
    Safely load environment variables from .env file.
    
    Returns:
        bool: True if .env was loaded successfully, False otherwise
    """
    try:
        from dotenv import find_dotenv, load_dotenv
        
        # Try to find .env file starting from current directory
        # Use find_dotenv with usecwd=True to start from current directory
        # If it fails, try from the project root
        dotenv_path = None
        
        try:
            dotenv_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
        except (OSError, RuntimeError):
            # If find_dotenv fails, try explicit paths
            pass
        
        # Try alternative paths if find_dotenv didn't work
        if not dotenv_path:
            # Try current directory
            cwd_env = Path.cwd() / ".env"
            if cwd_env.exists():
                dotenv_path = str(cwd_env)
            else:
                # Try project root (assuming we're in app/utils)
                try:
                    project_root = Path(__file__).parent.parent.parent
                    root_env = project_root / ".env"
                    if root_env.exists():
                        dotenv_path = str(root_env)
                except Exception:
                    pass
        
        # Load the .env file if found
        if dotenv_path:
            load_dotenv(dotenv_path, verbose=verbose)
            if verbose:
                print(f"✓ Loaded environment from: {dotenv_path}")
            return True
        else:
            # No .env file found - this is OK in test/CI environments
            if verbose:
                print("ℹ No .env file found - using system environment variables")
            return False
            
    except Exception as e:
        # Gracefully handle any dotenv errors
        # In test environments, this is expected and OK
        if verbose and not is_test_environment():
            print(f"⚠ Warning: Could not load .env file: {e}")
        return False


def get_env_or_default(key: str, default: str = "") -> str:
    """
    Get environment variable with a default value.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        str: Environment variable value or default
    """
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """
    Get required environment variable.
    
    Raises:
        ValueError: If the environment variable is not set
        
    Returns:
        str: Environment variable value
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


# Load environment variables once when this module is imported
# This is safe because it handles errors gracefully
_DOTENV_LOADED = safe_load_dotenv(verbose=False)

