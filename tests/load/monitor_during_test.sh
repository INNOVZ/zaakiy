#!/bin/bash

###############################################################################
# System Monitoring Script for Load Testing
#
# Run this in a separate terminal while load testing to monitor system health
#
# Usage:
#   ./tests/load/monitor_during_test.sh [interval]
#
# Example:
#   ./tests/load/monitor_during_test.sh 5  # Update every 5 seconds
###############################################################################

INTERVAL=${1:-2}  # Default 2 seconds
LOG_FILE="reports/system_monitoring_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create reports directory
mkdir -p reports

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      System Monitoring During Load Test            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Logging to: ${LOG_FILE}${NC}"
echo -e "${CYAN}Update interval: ${INTERVAL}s${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Initialize log file
echo "System Monitoring Log - $(date)" > "$LOG_FILE"
echo "Update interval: ${INTERVAL}s" >> "$LOG_FILE"
echo "=====================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to get CPU usage
get_cpu_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        ps -A -o %cpu | awk '{s+=$1} END {print s}'
    else
        # Linux
        top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'
    fi
}

# Function to get memory usage
get_memory_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f Mi\n", "$1:", $2 * $size / 1048576);' | grep "active:" | awk '{print $2}'
    else
        # Linux
        free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }'
    fi
}

# Function to get process info for uvicorn
get_uvicorn_stats() {
    if pgrep -f "uvicorn" > /dev/null; then
        local pid=$(pgrep -f "uvicorn" | head -1)
        local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null || echo "0")
        local mem=$(ps -p "$pid" -o %mem= 2>/dev/null || echo "0")
        local rss=$(ps -p "$pid" -o rss= 2>/dev/null || echo "0")

        # Convert RSS from KB to MB
        local mem_mb=$(echo "scale=2; $rss / 1024" | bc 2>/dev/null || echo "0")

        echo "$cpu|$mem|$mem_mb|$pid"
    else
        echo "0|0|0|N/A"
    fi
}

# Function to get network connections
get_connections() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        netstat -an | grep "8001" | grep "ESTABLISHED" | wc -l | tr -d ' '
    else
        netstat -an | grep ":8001" | grep "ESTABLISHED" | wc -l | tr -d ' '
    fi
}

# Function to check if Redis is running and get stats
get_redis_stats() {
    if command -v redis-cli &> /dev/null; then
        local connected=$(redis-cli ping 2>/dev/null)
        if [[ "$connected" == "PONG" ]]; then
            local clients=$(redis-cli CLIENT LIST 2>/dev/null | wc -l | tr -d ' ')
            local memory=$(redis-cli INFO memory 2>/dev/null | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
            echo "✓|$clients|$memory"
        else
            echo "✗|0|N/A"
        fi
    else
        echo "N/A|N/A|N/A"
    fi
}

# Main monitoring loop
iteration=0
while true; do
    clear

    echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║      System Monitoring - $(date +%H:%M:%S)                ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Get metrics
    cpu=$(get_cpu_usage)
    mem=$(get_memory_usage)
    uvicorn_stats=$(get_uvicorn_stats)
    connections=$(get_connections)
    redis_stats=$(get_redis_stats)

    # Parse uvicorn stats
    IFS='|' read -r uvicorn_cpu uvicorn_mem uvicorn_mem_mb uvicorn_pid <<< "$uvicorn_stats"

    # Parse Redis stats
    IFS='|' read -r redis_status redis_clients redis_memory <<< "$redis_stats"

    # Determine colors based on thresholds
    cpu_color=$GREEN
    [[ $(echo "$cpu > 70" | bc -l 2>/dev/null || echo 0) -eq 1 ]] && cpu_color=$YELLOW
    [[ $(echo "$cpu > 90" | bc -l 2>/dev/null || echo 0) -eq 1 ]] && cpu_color=$RED

    uvicorn_color=$GREEN
    [[ $(echo "$uvicorn_cpu > 70" | bc -l 2>/dev/null || echo 0) -eq 1 ]] && uvicorn_color=$YELLOW
    [[ $(echo "$uvicorn_cpu > 90" | bc -l 2>/dev/null || echo 0) -eq 1 ]] && uvicorn_color=$RED

    conn_color=$GREEN
    [[ $connections -gt 100 ]] && conn_color=$YELLOW
    [[ $connections -gt 500 ]] && conn_color=$RED

    # Display system metrics
    echo -e "${CYAN}System Resources:${NC}"
    echo -e "  CPU Usage:     ${cpu_color}${cpu}%${NC}"
    echo -e "  Memory Active: ${mem} MB"
    echo ""

    # Display application metrics
    echo -e "${CYAN}Application (Uvicorn):${NC}"
    if [[ "$uvicorn_pid" != "N/A" ]]; then
        echo -e "  Status:        ${GREEN}Running${NC} (PID: $uvicorn_pid)"
        echo -e "  CPU:           ${uvicorn_color}${uvicorn_cpu}%${NC}"
        echo -e "  Memory:        ${uvicorn_mem}% (${uvicorn_mem_mb} MB)"
        echo -e "  Connections:   ${conn_color}${connections}${NC}"
    else
        echo -e "  Status:        ${RED}Not Running${NC}"
    fi
    echo ""

    # Display Redis metrics
    echo -e "${CYAN}Redis Cache:${NC}"
    if [[ "$redis_status" == "✓" ]]; then
        echo -e "  Status:        ${GREEN}Connected${NC}"
        echo -e "  Clients:       ${redis_clients}"
        echo -e "  Memory:        ${redis_memory}"
    elif [[ "$redis_status" == "✗" ]]; then
        echo -e "  Status:        ${RED}Not Connected${NC}"
    else
        echo -e "  Status:        ${YELLOW}N/A (redis-cli not found)${NC}"
    fi
    echo ""

    # Health indicators
    echo -e "${CYAN}Health Indicators:${NC}"

    # CPU health
    if [[ $(echo "$cpu < 70" | bc -l 2>/dev/null || echo 1) -eq 1 ]]; then
        echo -e "  ${GREEN}✓${NC} CPU usage is healthy"
    elif [[ $(echo "$cpu < 90" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
        echo -e "  ${YELLOW}⚠${NC} CPU usage is high"
    else
        echo -e "  ${RED}✗${NC} CPU usage is critical"
    fi

    # Connection health
    if [[ $connections -lt 100 ]]; then
        echo -e "  ${GREEN}✓${NC} Connection count is normal"
    elif [[ $connections -lt 500 ]]; then
        echo -e "  ${YELLOW}⚠${NC} Connection count is high"
    else
        echo -e "  ${RED}✗${NC} Connection count is very high"
    fi

    # Application health
    if [[ "$uvicorn_pid" != "N/A" ]]; then
        echo -e "  ${GREEN}✓${NC} Application is running"
    else
        echo -e "  ${RED}✗${NC} Application is not running"
    fi

    echo ""
    echo -e "${YELLOW}Iteration: $iteration | Interval: ${INTERVAL}s | Press Ctrl+C to stop${NC}"

    # Log to file
    {
        echo "=== $(date) ==="
        echo "CPU: ${cpu}%"
        echo "Memory: ${mem} MB"
        echo "Uvicorn - CPU: ${uvicorn_cpu}%, Memory: ${uvicorn_mem}% (${uvicorn_mem_mb} MB), PID: ${uvicorn_pid}"
        echo "Connections: ${connections}"
        echo "Redis - Status: ${redis_status}, Clients: ${redis_clients}, Memory: ${redis_memory}"
        echo ""
    } >> "$LOG_FILE"

    ((iteration++))
    sleep "$INTERVAL"
done
