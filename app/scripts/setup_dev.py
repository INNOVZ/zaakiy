#!/usr/bin/env python3
"""
Development environment setup script for Zentria backend

This script helps set up the development environment by:
- Checking Python version
- Installing dependencies
- Setting up environment variables
- Running initial health checks
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def check_python_version() -> bool:
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_virtual_environment() -> bool:
    """Check if virtual environment is active"""
    print("🔧 Checking virtual environment...")

    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("   Consider creating one: python -m venv .venv")
        return False


def install_dependencies() -> bool:
    """Install project dependencies"""
    print("📦 Installing dependencies...")

    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found")
            return False

        # Install dependencies
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def setup_environment_file() -> bool:
    """Set up environment file from template"""
    print("🔐 Setting up environment file...")

    env_file = Path(".env")
    env_example = Path("env_example")

    if env_file.exists():
        print("✅ .env file already exists")
        return True

    if not env_example.exists():
        print("❌ env_example file not found")
        return False

    try:
        # Copy env_example to .env
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("   Please edit .env with your actual configuration values")
        return True

    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def check_required_tools() -> Dict[str, bool]:
    """Check if required development tools are available"""
    print("🛠️  Checking development tools...")

    tools = {
        "git": shutil.which("git") is not None,
        "make": shutil.which("make") is not None,
        "docker": shutil.which("docker") is not None,
        "docker-compose": shutil.which("docker-compose") is not None,
    }

    for tool, available in tools.items():
        status = "✅" if available else "❌"
        print(f"   {status} {tool}")

    return tools


def run_health_check() -> bool:
    """Run health check to verify setup"""
    print("🏥 Running health check...")

    try:
        # Import and run health check
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import asyncio

        from health_check import HealthChecker

        async def run_check():
            checker = HealthChecker()
            results = await checker.run_all_checks()
            return all(r["status"] in ["healthy", "warning"] for r in results.values())

        success = asyncio.run(run_check())

        if success:
            print("✅ Health check passed")
        else:
            print("⚠️  Health check completed with warnings/errors")
            print("   Check the output above for details")

        return success

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def print_next_steps():
    """Print next steps for development setup"""
    print("\n" + "=" * 60)
    print("🚀 DEVELOPMENT SETUP COMPLETE")
    print("=" * 60)

    print("\n📋 Next steps:")
    print("1. Edit .env file with your actual configuration values")
    print("2. Run 'make dev' to start the development server")
    print("3. Visit http://localhost:8001/docs for API documentation")
    print("4. Run 'make test' to run the test suite")
    print("5. Run 'make format' to format your code")

    print("\n🔧 Useful commands:")
    print("   make dev          - Start development server")
    print("   make test         - Run tests")
    print("   make format       - Format code")
    print("   make lint         - Run linters")
    print("   make health       - Run health check")

    print("\n📚 Documentation:")
    print("   README.md         - Project overview")
    print("   http://localhost:8001/docs - API documentation")


def main():
    """Main setup function"""
    print("🚀 Zentria Backend Development Setup")
    print("=" * 50)

    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print(f"📁 Working directory: {project_root}")

    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Environment File", setup_environment_file),
    ]

    results = {}
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            results[step_name] = False

    # Check development tools
    print(f"\nDevelopment Tools:")
    tools = check_required_tools()

    # Add tools check to results
    tools_available = all(tools.values())
    results["Development Tools"] = tools_available

    # Run health check if basic setup succeeded
    if all(results.values()):
        print(f"\nHealth Check:")
        health_ok = run_health_check()
        results["Health Check"] = health_ok
    else:
        print(f"\n⚠️  Skipping health check due to setup issues")
        results["Health Check"] = False

    # Print summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)

    for step_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")

    all_success = all(results.values())

    if all_success:
        print_next_steps()
        sys.exit(0)
    else:
        print(f"\n❌ Setup incomplete. Please fix the issues above and run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
