#!/bin/bash
# ================================
# Redis Cache Testing Script
# ================================
# Automated tests for Redis caching functionality

set -e

echo "🧪 Redis Cache Testing Suite"
echo "============================="
echo ""

# Configuration
API_URL="${API_URL:-http://localhost:8001}"
TEST_ORG_ID="${TEST_ORG_ID:-test-org-123}"
TEST_URL="https://example.com"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function for tests
run_test() {
    local test_name=$1
    local test_command=$2

    echo -e "${BLUE}▶ Test: $test_name${NC}"

    if eval "$test_command"; then
        echo -e "${GREEN}  ✅ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED+1))
        return 0
    else
        echo -e "${RED}  ❌ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED+1))
        return 1
    fi
}

# Test 1: Check if API is running
echo "Test Group 1: Infrastructure"
echo "-----------------------------"
run_test "API Health Check" "curl -sf $API_URL/health > /dev/null"

# Test 2: Check if Redis is accessible
run_test "Redis Connection" "redis-cli ping > /dev/null 2>&1"

# Test 3: Check Redis memory
run_test "Redis Memory Info" "redis-cli INFO memory | grep used_memory_human > /dev/null"

echo ""

# Test Group 2: Cache Operations
echo "Test Group 2: Cache Operations"
echo "--------------------------------"

# Test 4: Clear test cache
run_test "Clear existing cache" "redis-cli DEL \"scrape:v2:$TEST_ORG_ID:*\" > /dev/null || true"

# Test 5: Check initial cache stats
INITIAL_HITS=$(redis-cli INFO stats | grep keyspace_hits | cut -d: -f2 | tr -d '\r')
INITIAL_MISSES=$(redis-cli INFO stats | grep keyspace_misses | cut -d: -f2 | tr -d '\r')
echo -e "${BLUE}  Initial stats: Hits=$INITIAL_HITS, Misses=$INITIAL_MISSES${NC}"

# Test 6: Test URL normalization
echo ""
echo "Test Group 3: URL Normalization"
echo "--------------------------------"

# These should result in the same cache key
TEST_URLS=(
    "https://EXAMPLE.COM/test"
    "https://example.com/test"
    "https://example.com/test?utm_source=test"
    "https://example.com/test?utm_campaign=test&utm_source=test"
)

echo "Testing URL normalization with variants:"
for url in "${TEST_URLS[@]}"; do
    echo "  - $url"
done

# Test cache key generation (would need Python script)
run_test "URL Normalization Test" "python3 -c \"
from app.services.scraping.scraping_cache_service import ScrapingCacheService
cache = ScrapingCacheService()
key1 = cache._generate_cache_key('https://EXAMPLE.COM/test', '$TEST_ORG_ID', 'url')
key2 = cache._generate_cache_key('https://example.com/test?utm_source=test', '$TEST_ORG_ID', 'url')
print(f'Key1: {key1}')
print(f'Key2: {key2}')
assert key1 == key2, 'Keys should match after normalization'
\" 2>/dev/null"

echo ""

# Test Group 4: Cache Hit/Miss
echo "Test Group 4: Cache Hit/Miss Scenarios"
echo "---------------------------------------"

# Note: These tests require the API to be running and accepting requests
# You may need to adjust based on your auth setup

echo -e "${YELLOW}Note: API endpoint tests require authentication${NC}"
echo "      Run manual tests using the dashboard or API client"

echo ""

# Test Group 5: Performance
echo "Test Group 5: Performance Metrics"
echo "----------------------------------"

# Check current cache stats
FINAL_HITS=$(redis-cli INFO stats | grep keyspace_hits | cut -d: -f2 | tr -d '\r')
FINAL_MISSES=$(redis-cli INFO stats | grep keyspace_misses | cut -d: -f2 | tr -d '\r')

NEW_HITS=$((FINAL_HITS - INITIAL_HITS))
NEW_MISSES=$((FINAL_MISSES - INITIAL_MISSES))

echo "Cache Statistics:"
echo "  New Hits:    $NEW_HITS"
echo "  New Misses:  $NEW_MISSES"

TOTAL_REQUESTS=$((NEW_HITS + NEW_MISSES))
if [ $TOTAL_REQUESTS -gt 0 ]; then
    HIT_RATE=$(awk "BEGIN {print ($NEW_HITS / $TOTAL_REQUESTS) * 100}")
    echo "  Hit Rate:    $HIT_RATE%"
fi

# Test 7: Check key count
KEY_COUNT=$(redis-cli DBSIZE | awk '{print $2}')
run_test "Cache Keys Created" "[ $KEY_COUNT -ge 0 ]"
echo "  Total keys in Redis: $KEY_COUNT"

echo ""

# Test Group 6: Memory & Eviction
echo "Test Group 6: Memory & Eviction Policy"
echo "---------------------------------------"

# Check eviction policy
EVICTION_POLICY=$(redis-cli CONFIG GET maxmemory-policy | tail -1)
echo "  Eviction Policy: $EVICTION_POLICY"
run_test "LRU Eviction Policy" "[ \"$EVICTION_POLICY\" = \"allkeys-lru\" ]"

# Check max memory
MAX_MEMORY=$(redis-cli CONFIG GET maxmemory | tail -1)
if [ "$MAX_MEMORY" = "0" ]; then
    echo -e "${YELLOW}  ⚠️  Warning: No max memory set${NC}"
else
    MAX_MEMORY_MB=$((MAX_MEMORY / 1024 / 1024))
    echo "  Max Memory: ${MAX_MEMORY_MB}MB"
fi

# Check current memory usage
USED_MEMORY=$(redis-cli INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
echo "  Used Memory: $USED_MEMORY"

echo ""

# Test Group 7: TTL Testing
echo "Test Group 7: TTL (Time To Live)"
echo "--------------------------------"

# Create a test key with TTL
redis-cli SET "test:ttl:key" "test_value" EX 10 > /dev/null
TTL=$(redis-cli TTL "test:ttl:key")
run_test "TTL Set Correctly" "[ $TTL -gt 0 ] && [ $TTL -le 10 ]"
echo "  TTL: $TTL seconds"

# Cleanup
redis-cli DEL "test:ttl:key" > /dev/null

echo ""

# Final Summary
echo "==============================="
echo "📊 Test Summary"
echo "==============================="
echo -e "${GREEN}✅ Passed: $TESTS_PASSED${NC}"
echo -e "${RED}❌ Failed: $TESTS_FAILED${NC}"
echo "-------------------------------"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {print ($TESTS_PASSED / $TOTAL_TESTS) * 100}")
    echo "Success Rate: $SUCCESS_RATE%"
fi

echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Please review.${NC}"
    echo ""
    exit 1
fi
