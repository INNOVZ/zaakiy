#!/bin/bash

# ZaaKiy AI Platform - Production Deployment Script
# This script handles the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="zaaky-backend"
DOCKER_IMAGE="zaaky-backend"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if .env.production exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Production environment file $ENV_FILE not found!"
        log_info "Please copy .env.production and configure it with your production values."
        exit 1
    fi

    log_success "All prerequisites met"
}

# Validate environment configuration
validate_environment() {
    log_info "Validating environment configuration..."

    # Check for required environment variables
    required_vars=(
        "SUPABASE_URL"
        "SUPABASE_SERVICE_ROLE_KEY"
        "SUPABASE_JWT_SECRET"
        "OPENAI_API_KEY"
        "PINECONE_API_KEY"
        "PINECONE_INDEX"
    )

    missing_vars=()
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$ENV_FILE" || grep -q "^${var}=$" "$ENV_FILE" || grep -q "^${var}=your_.*_here" "$ENV_FILE"; then
            missing_vars+=("$var")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing or incomplete environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        log_info "Please update $ENV_FILE with your production values"
        exit 1
    fi

    log_success "Environment configuration validated"
}

# Clean Python bytecode cache
clean_python_cache() {
    log_info "Cleaning Python bytecode cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    log_success "Python cache cleaned"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t "$DOCKER_IMAGE:latest" .
    log_success "Docker image built successfully"
}

# Build Docker image without cache (force rebuild)
build_image_no_cache() {
    log_info "Building Docker image WITHOUT cache (force rebuild)..."
    docker build --no-cache -t "$DOCKER_IMAGE:latest" .
    log_success "Docker image built successfully (no cache)"
}

# Stop existing containers
stop_containers() {
    log_info "Stopping existing containers..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true
    log_success "Existing containers stopped"
}

# Force remove containers and images
force_cleanup() {
    log_info "Force cleaning containers and images..."
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down --rmi all --volumes --remove-orphans || true
    # Remove the specific image if it exists
    docker rmi "$DOCKER_IMAGE:latest" 2>/dev/null || true
    log_success "Force cleanup completed"
}

# Start production containers
start_containers() {
    log_info "Starting production containers..."
    docker-compose -f "$COMPOSE_FILE" up -d
    log_success "Production containers started"
}

# Wait for health check
wait_for_health() {
    log_info "Waiting for application to be healthy..."

    max_attempts=30
    attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8001/health &> /dev/null; then
            log_success "Application is healthy and ready!"
            return 0
        fi

        log_info "Health check attempt $attempt/$max_attempts - waiting..."
        sleep 10
        ((attempt++))
    done

    log_error "Application failed to become healthy after $max_attempts attempts"
    log_info "Checking container logs..."
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo ""
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    log_info "Application URLs:"
    echo "  - API: http://localhost:8001"
    echo "  - Health: http://localhost:8001/health"
    echo "  - Docs: http://localhost:8001/docs"
    echo ""
    log_info "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
    log_info "To stop: docker-compose -f $COMPOSE_FILE down"
}

# Cleanup old images
cleanup_images() {
    log_info "Cleaning up old Docker images..."
    docker image prune -f || true
    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting ZaaKy AI Platform deployment..."
    echo ""

    check_root
    check_prerequisites
    validate_environment
    clean_python_cache
    build_image
    stop_containers
    start_containers
    wait_for_health
    show_status
    cleanup_images

    echo ""
    log_success "🎉 Deployment completed successfully!"
    log_info "Your ZaaKy AI Platform is now running in production mode"
}

# Force rebuild deployment (no cache, complete cleanup)
force_deploy() {
    log_info "Starting FORCE rebuild deployment (no cache)..."
    echo ""

    check_root
    check_prerequisites
    validate_environment
    clean_python_cache
    force_cleanup
    build_image_no_cache
    start_containers
    wait_for_health
    show_status
    cleanup_images

    echo ""
    log_success "🎉 Force deployment completed successfully!"
    log_info "Your ZaaKy AI Platform is now running in production mode (fresh build)"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "force"|"force-deploy"|"rebuild")
        force_deploy
        ;;
    "stop")
        log_info "Stopping production containers..."
        docker-compose -f "$COMPOSE_FILE" down
        log_success "Containers stopped"
        ;;
    "restart")
        log_info "Restarting production containers..."
        docker-compose -f "$COMPOSE_FILE" restart
        log_success "Containers restarted"
        ;;
    "clean")
        log_info "Cleaning Python cache..."
        clean_python_cache
        log_success "Cache cleaned"
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    "status")
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "health")
        curl -f http://localhost:8001/health || log_error "Health check failed"
        ;;
    *)
        echo "Usage: $0 {deploy|force|stop|restart|logs|status|health|clean}"
        echo ""
        echo "Commands:"
        echo "  deploy       - Deploy the application (default)"
        echo "  force        - Force rebuild without cache (use when code reverted)"
        echo "  stop         - Stop all containers"
        echo "  restart      - Restart all containers"
        echo "  clean        - Clean Python bytecode cache"
        echo "  logs         - Show container logs"
        echo "  status       - Show container status"
        echo "  health       - Check application health"
        exit 1
        ;;
esac
