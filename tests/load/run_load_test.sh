#!/bin/bash

###############################################################################
# Load Testing Runner Script for ZaaKy AI Platform
#
# Usage:
#   ./tests/load/run_load_test.sh [scenario] [host]
#
# Scenarios:
#   quick       - Quick smoke test (10 users, 1 minute)
#   normal      - Normal load test (50 users, 5 minutes)
#   peak        - Peak load test (200 users, 10 minutes)
#   stress      - Stress test (500 users, 10 minutes)
#   spike       - Spike test (500 users, fast ramp)
#   endurance   - Long-running test (100 users, 2 hours)
#
# Examples:
#   ./tests/load/run_load_test.sh quick
#   ./tests/load/run_load_test.sh normal http://localhost:8001
#   ./tests/load/run_load_test.sh stress https://api.production.com
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCENARIO=${1:-normal}
HOST=${2:-http://localhost:8001}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORTS_DIR="reports"
LOCUST_FILE="tests/load/locustfile.py"

# Create reports directory
mkdir -p "$REPORTS_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ZaaKy AI Platform - Load Testing Runner         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if locust is installed
if ! command -v locust &> /dev/null; then
    echo -e "${RED}❌ Locust is not installed${NC}"
    echo -e "${YELLOW}Install with: pip install locust${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Locust is installed${NC}"
echo ""

# Check if app is running
echo -e "${BLUE}🔍 Checking if application is running...${NC}"
if curl -s -f -o /dev/null "$HOST/health"; then
    echo -e "${GREEN}✅ Application is running at $HOST${NC}"
else
    echo -e "${RED}❌ Application is not reachable at $HOST${NC}"
    echo -e "${YELLOW}Please start your application first:${NC}"
    echo -e "${YELLOW}  cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8001${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📊 Test Configuration${NC}"
echo -e "  Scenario: ${GREEN}$SCENARIO${NC}"
echo -e "  Target:   ${GREEN}$HOST${NC}"
echo -e "  Time:     ${GREEN}$(date)${NC}"
echo ""

# Run load test based on scenario
case $SCENARIO in
    quick|smoke)
        echo -e "${YELLOW}🚀 Running Quick Smoke Test (10 users, 1 minute)${NC}"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users 10 \
            --spawn-rate 2 \
            --run-time 1m \
            --headless \
            --html "$REPORTS_DIR/quick_test_${TIMESTAMP}.html" \
            --csv "$REPORTS_DIR/quick_test_${TIMESTAMP}"
        ;;

    normal|baseline)
        echo -e "${YELLOW}🚀 Running Normal Load Test (50 users, 5 minutes)${NC}"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users 50 \
            --spawn-rate 5 \
            --run-time 5m \
            --headless \
            --html "$REPORTS_DIR/normal_load_${TIMESTAMP}.html" \
            --csv "$REPORTS_DIR/normal_load_${TIMESTAMP}"
        ;;

    peak)
        echo -e "${YELLOW}🚀 Running Peak Load Test (200 users, 10 minutes)${NC}"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users 200 \
            --spawn-rate 20 \
            --run-time 10m \
            --headless \
            --html "$REPORTS_DIR/peak_load_${TIMESTAMP}.html" \
            --csv "$REPORTS_DIR/peak_load_${TIMESTAMP}"
        ;;

    stress)
        echo -e "${YELLOW}🚀 Running Stress Test (500 users, 10 minutes)${NC}"
        echo -e "${RED}⚠️  WARNING: This will put significant load on your system${NC}"
        sleep 2
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users 500 \
            --spawn-rate 50 \
            --run-time 10m \
            --headless \
            --html "$REPORTS_DIR/stress_test_${TIMESTAMP}.html" \
            --csv "$REPORTS_DIR/stress_test_${TIMESTAMP}" \
            StressTestUser
        ;;

    spike)
        echo -e "${YELLOW}🚀 Running Spike Test (0→500 users quickly)${NC}"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users 500 \
            --spawn-rate 100 \
            --run-time 3m \
            --headless \
            --html "$REPORTS_DIR/spike_test_${TIMESTAMP}.html" \
            --csv "$REPORTS_DIR/spike_test_${TIMESTAMP}" \
            SpikeTester
        ;;

    endurance|soak)
        echo -e "${YELLOW}🚀 Running Endurance Test (100 users, 2 hours)${NC}"
        echo -e "${RED}⚠️  WARNING: This test will run for 2 hours${NC}"
        sleep 3
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users 100 \
            --spawn-rate 10 \
            --run-time 2h \
            --headless \
            --html "$REPORTS_DIR/endurance_test_${TIMESTAMP}.html" \
            --csv "$REPORTS_DIR/endurance_test_${TIMESTAMP}"
        ;;

    web|ui)
        echo -e "${YELLOW}🚀 Starting Locust Web UI${NC}"
        echo -e "${GREEN}Open your browser to: http://localhost:8089${NC}"
        echo ""
        locust -f "$LOCUST_FILE" --host="$HOST"
        exit 0
        ;;

    *)
        echo -e "${RED}❌ Unknown scenario: $SCENARIO${NC}"
        echo ""
        echo "Available scenarios:"
        echo "  quick       - Quick smoke test (10 users, 1 minute)"
        echo "  normal      - Normal load test (50 users, 5 minutes)"
        echo "  peak        - Peak load test (200 users, 10 minutes)"
        echo "  stress      - Stress test (500 users, 10 minutes)"
        echo "  spike       - Spike test (500 users, fast ramp)"
        echo "  endurance   - Long-running test (100 users, 2 hours)"
        echo "  web         - Start web UI"
        exit 1
        ;;
esac

# Show results
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Load test completed!${NC}"
echo ""
echo -e "${BLUE}📊 Reports generated:${NC}"
echo -e "  HTML: ${GREEN}$REPORTS_DIR/${SCENARIO}_test_${TIMESTAMP}.html${NC}"
echo -e "  CSV:  ${GREEN}$REPORTS_DIR/${SCENARIO}_test_${TIMESTAMP}_*.csv${NC}"
echo ""
echo -e "${YELLOW}💡 Quick tips:${NC}"
echo "  - Open the HTML report in your browser to see detailed results"
echo "  - Check CSV files for raw data analysis"
echo "  - Monitor system resources during the test"
echo "  - Compare results with previous runs"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
