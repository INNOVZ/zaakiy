#!/bin/bash
# ================================
# Staging Deployment Script
# ================================
# Deploys the application with Redis caching to staging environment

set -e  # Exit on error

echo "🚀 ZaaKy Platform - Staging Deployment"
echo "======================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="staging"
COMPOSE_FILE="docker-compose.yml"

# Step 1: Pre-flight checks
echo "📋 Step 1/6: Pre-flight Checks"
echo "-------------------------------"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo "Please create .env from ENV_TEMPLATE.md"
    exit 1
fi
echo -e "${GREEN}✅ .env file found${NC}"

# Check environment variables
echo "Checking critical environment variables..."
python3 scripts/check_env.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Environment check failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Environment variables validated${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

echo ""

# Step 2: Build containers
echo "🏗️  Step 2/6: Building Docker Images"
echo "------------------------------------"
docker-compose build --no-cache
echo -e "${GREEN}✅ Docker images built${NC}"
echo ""

# Step 3: Start Redis first
echo "🔴 Step 3/6: Starting Redis"
echo "---------------------------"
docker-compose up -d redis

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
sleep 5

# Test Redis connection
if docker exec zaakiy-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is ready${NC}"
else
    echo -e "${RED}❌ Redis failed to start${NC}"
    docker logs zaakiy-redis
    exit 1
fi
echo ""

# Step 4: Start application
echo "🚀 Step 4/6: Starting Application"
echo "---------------------------------"
docker-compose up -d zaakiy-backend

# Wait for application to be ready
echo "Waiting for application to be ready..."
sleep 10

# Health check
MAX_RETRIES=30
RETRY_COUNT=0
echo "Performing health check..."

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Application is healthy${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "  Attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Application health check failed${NC}"
    echo "Showing last 50 lines of logs:"
    docker logs --tail 50 zaakiy-backend
    exit 1
fi
echo ""

# Step 5: Verify Redis caching
echo "🧪 Step 5/6: Verifying Redis Caching"
echo "------------------------------------"

# Check if caching is enabled
CACHE_ENABLED=$(docker exec zaakiy-backend env | grep ENABLE_CACHING || echo "")
if [[ $CACHE_ENABLED == *"true"* ]]; then
    echo -e "${GREEN}✅ Caching is enabled${NC}"
else
    echo -e "${YELLOW}⚠️  Caching is disabled${NC}"
fi

# Check Redis keys
REDIS_KEYS=$(docker exec zaakiy-redis redis-cli DBSIZE | awk '{print $2}')
echo "Redis database size: $REDIS_KEYS keys"

# Monitor cache for 10 seconds
echo "Monitoring cache for 10 seconds..."
timeout 10 docker exec zaakiy-redis redis-cli MONITOR > /tmp/redis_monitor.log 2>&1 || true
if [ -s /tmp/redis_monitor.log ]; then
    echo -e "${GREEN}✅ Redis is receiving commands${NC}"
else
    echo -e "${YELLOW}⚠️  No Redis activity detected yet${NC}"
fi
echo ""

# Step 6: Display status
echo "📊 Step 6/6: Deployment Summary"
echo "-------------------------------"
echo ""

# Show running containers
docker-compose ps

echo ""
echo "📝 Service URLs:"
echo "  Backend:  http://localhost:8001"
echo "  Health:   http://localhost:8001/health"
echo "  Docs:     http://localhost:8001/docs"
echo "  Redis:    localhost:6379"

echo ""
echo "📊 Monitoring Commands:"
echo "  View logs:           docker logs zaakiy-backend -f"
echo "  View Redis logs:     docker logs zaakiy-redis -f"
echo "  Monitor cache:       python scripts/monitor_cache.py"
echo "  Check environment:   python scripts/check_env.py"

echo ""
echo "🧪 Test Cache:"
echo "  1. Upload a document (URL/PDF/JSON)"
echo "  2. Upload the same document again"
echo "  3. Second request should be faster (cache hit)"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ Staging Deployment Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Optional: Start monitoring
read -p "Do you want to start cache monitoring? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 scripts/monitor_cache.py
fi
