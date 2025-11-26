# Tests Directory Structure

This directory contains all test files organized by type and purpose.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and fixtures
├── integration/                  # Integration tests
│   ├── test_scraping/            # Scraping service integration tests
│   │   ├── __init__.py
│   │   ├── test_recursive_integration.py
│   │   ├── test_recursive_topics_simple.py
│   │   ├── test_recursive_url_keyword_boost.py
│   │   ├── test_recursive_url_topics.py
│   │   ├── test_recursive_url_topics_unit.py
│   │   ├── test_refactored_scraping.py
│   │   └── test_scraping_services.py
│   └── test_whatsapp_integration.py
├── unit/                         # Unit tests
│   ├── test_chat/                # Chat service unit tests
│   ├── test_scraping/            # Scraping service unit tests
│   ├── test_auth/                # Auth service unit tests
│   └── test_whatsapp_service.py
├── e2e/                          # End-to-end tests
│   └── test_whatsapp_e2e.py
├── manual/                       # Manual test scripts
│   ├── test_contact_query.py
│   ├── test_collection_direct.py
│   ├── test_contact_extractor_standalone.py
│   ├── test_contact_retrieval.py
│   └── test_intent_analytics_endpoint.py
├── scripts/                       # Test utility scripts
│   ├── __init__.py
│   └── verify_caching.py
├── load/                         # Load testing scripts
└── performance/                  # Performance tests
```

## Test Categories

### Integration Tests (`integration/`)
Tests that verify the interaction between multiple components or services.

- **Scraping Integration Tests** (`integration/test_scraping/`): Test scraping services end-to-end
  - Recursive URL scraping
  - Topic extraction
  - URL-to-chunks mapping
  - Refactored scraping modules

### Unit Tests (`unit/`)
Tests for individual functions, classes, or modules in isolation.

- **Chat Tests** (`unit/test_chat/`): Chat service unit tests
- **Scraping Tests** (`unit/test_scraping/`): Scraping service unit tests
- **Auth Tests** (`unit/test_auth/`): Authentication unit tests

### End-to-End Tests (`e2e/`)
Tests that verify the complete flow from API request to response.

### Manual Tests (`manual/`)
Test scripts that require manual execution or verification.

- Contact query tests
- Collection scraping tests
- Intent analytics tests

### Test Scripts (`scripts/`)
Utility scripts for testing and verification.

- Caching verification
- Performance monitoring
- Test data generation

### Load Tests (`load/`)
Load testing scripts using Locust.

### Performance Tests (`performance/`)
Performance benchmarking and profiling tests.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test category
```bash
# Integration tests
pytest tests/integration/

# Unit tests
pytest tests/unit/

# E2E tests
pytest tests/e2e/
```

### Run specific test file
```bash
pytest tests/integration/test_scraping/test_refactored_scraping.py
```

### Run with coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

## Test Organization Principles

1. **By Type**: Tests are organized by test type (unit, integration, e2e)
2. **By Service**: Within each type, tests are grouped by service/feature
3. **Clear Naming**: Test files follow `test_*.py` naming convention
4. **Isolated**: Each test should be independent and not rely on other tests
5. **Fast Unit Tests**: Unit tests should be fast and not require external services
6. **Integration Tests**: Integration tests may require database/external services

## Adding New Tests

When adding new tests:

1. **Place in correct directory**: Choose the appropriate test type directory
2. **Follow naming convention**: Use `test_*.py` for test files
3. **Use fixtures**: Leverage `conftest.py` for shared test fixtures
4. **Update this README**: Document new test categories or patterns

## Notes

- All test files moved from backend root have been updated with correct import paths
- Test scripts in `scripts/` are utility scripts, not automated tests
- Manual tests in `manual/` require manual execution or verification
