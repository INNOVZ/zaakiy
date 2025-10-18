"""
Pytest configuration and fixtures

Sets up test environment and common fixtures for all tests.
"""
import os
import sys

import pytest

# Set test environment flag before any imports
os.environ["TESTING"] = "true"

# Ensure the app module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_env():
    """Fixture to ensure test environment is set"""
    os.environ["TESTING"] = "true"
    yield
    # Cleanup not needed as the process will exit


@pytest.fixture(autouse=True)
def ensure_test_env():
    """Automatically ensure test environment for all tests"""
    os.environ["TESTING"] = "true"

