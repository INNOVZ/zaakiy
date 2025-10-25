# ZaaKy Backend Makefile
# Common development commands

.PHONY: help install dev test lint format clean docker-build docker-run setup-dev setup health check-storage migrate migrate-execute load-test load-test-quick load-test-normal load-test-stress load-test-monitor

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-dev    - Set up development environment"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Run development server"
	@echo "  test         - Run test suite"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean up build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  security     - Run security checks"
	@echo "  setup        - Set up development environment (interactive)"
	@echo "  health       - Run health check"
	@echo "  check-storage - Check storage configuration"
	@echo "  migrate      - Run data migration (dry run)"
	@echo "  migrate-execute - Execute data migration"
	@echo ""
	@echo "Load Testing:"
	@echo "  load-test-quick   - Quick smoke test (10 users, 1 min)"
	@echo "  load-test-normal  - Normal load test (50 users, 5 min)"
	@echo "  load-test-stress  - Stress test (500 users, 10 min)"
	@echo "  load-test-web     - Start Locust web UI"
	@echo "  load-test-monitor - Monitor system during load test"

# Development environment setup
setup-dev: install
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp env_example .env; echo "Created .env file from template"; fi
	@echo "Development environment ready!"
	@echo "Don't forget to configure your .env file with actual credentials"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run development server
dev:
	@echo "Starting development server..."
	python start_server.py

# Run production server
prod:
	@echo "Starting production server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8001

# Run test suite
test:
	@echo "Running tests..."
	pytest

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=app --cov=services --cov=utils --cov-report=html --cov-report=term-missing

# Run specific test categories
test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-security:
	pytest -m security -v

# Code quality checks
lint:
	@echo "Running linting checks..."
	flake8 .
	mypy .

# Format code
format:
	@echo "Formatting code..."
	black .
	isort .

# Security checks
security:
	@echo "Running security checks..."
	python test_secure_web_scraper.py
	@echo "Security tests completed"

# Script commands
setup:
	@echo "🔧 Setting up development environment..."
	python scripts/setup_dev.py

health:
	@echo "🏥 Running health check..."
	python scripts/health_check.py

check-storage:
	@echo "📁 Checking storage configuration..."
	python scripts/check_storage.py

migrate:
	@echo "🔄 Running data migration (dry run)..."
	python scripts/migrate_data.py --dry-run

migrate-execute:
	@echo "🔄 Executing data migration..."
	python scripts/migrate_data.py --execute

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.pyd" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	@echo "Cleanup completed"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t zaaky-backend .

docker-run:
	@echo "Running with Docker Compose..."
	docker-compose up -d

docker-dev:
	@echo "Running development environment with Docker..."
	docker-compose -f docker-compose.dev.yml up

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f

# Database commands
db-migrate:
	@echo "Running database migrations..."
	python scripts/migrate_db.py

db-seed:
	@echo "Seeding database..."
	python scripts/seed_data.py

# Health checks (legacy - use scripts/health_check.py instead)
health-legacy:
	@echo "Checking application health..."
	curl -f http://localhost:8001/health || echo "Health check failed"

# Pre-commit setup
pre-commit:
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	pre-commit run --all-files

# Generate requirements.txt from pyproject.toml
requirements:
	@echo "Generating requirements.txt..."
	pip-compile pyproject.toml

# Check environment configuration
check-env:
	@echo "Validating environment configuration..."
	python -c "from config.settings import validate_environment; validate_environment()"

# Development utilities
logs:
	@echo "Showing recent logs..."
	tail -f logs/*.log

monitor:
	@echo "Monitoring system resources..."
	python scripts/monitor_performance.py

# All-in-one development setup
dev-setup: clean setup-dev check-env
	@echo "Development setup completed!"
	@echo "Run 'make dev' to start the development server"

# All-in-one testing
test-all: test-unit test-integration test-security
	@echo "All tests completed!"

# Deployment preparation
deploy-prep: clean format lint test security
	@echo "Deployment preparation completed!"
	@echo "Ready for deployment"

# Load Testing commands
load-test-quick:
	@echo "🚀 Running quick smoke test..."
	./tests/load/run_load_test.sh quick

load-test-normal:
	@echo "🚀 Running normal load test..."
	./tests/load/run_load_test.sh normal

load-test-peak:
	@echo "🚀 Running peak load test..."
	./tests/load/run_load_test.sh peak

load-test-stress:
	@echo "🚀 Running stress test..."
	@echo "⚠️  WARNING: This will put significant load on your system"
	./tests/load/run_load_test.sh stress

load-test-spike:
	@echo "🚀 Running spike test..."
	./tests/load/run_load_test.sh spike

load-test-web:
	@echo "🚀 Starting Locust web UI..."
	@echo "Open your browser to: http://localhost:8089"
	./tests/load/run_load_test.sh web

load-test-monitor:
	@echo "📊 Starting system monitor..."
	@echo "Run this in a separate terminal while load testing"
	./tests/load/monitor_during_test.sh

load-test-compare:
	@echo "📊 Comparing load test results..."
	@if [ -z "$(FILES)" ]; then \
		python tests/load/compare_results.py reports/*_stats.csv; \
	else \
		python tests/load/compare_results.py $(FILES); \
	fi

load-test-check:
	@echo "✓ Checking performance thresholds..."
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make load-test-check FILE=reports/your_stats.csv"; \
	else \
		python tests/load/check_thresholds.py $(FILE); \
	fi

# Clean load test reports
clean-reports:
	@echo "🗑️  Cleaning load test reports..."
	rm -rf reports/*.html reports/*.csv reports/*.log
	@echo "Reports cleaned"
