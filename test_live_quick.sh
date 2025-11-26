#!/bin/bash
# Quick Test Script for Your Live ngrok Setup
# ngrok URL: https://asomatous-meagerly-edmundo.ngrok-free.dev/
# Local: http://localhost:8001/

set -e

NGROK_URL="https://asomatous-meagerly-edmundo.ngrok-free.dev"
LOCAL_URL="http://localhost:8001"
CHATBOT_ID="${1:-your-chatbot-id}"  # Pass as first argument or set default

echo "🚀 Testing Your Live Application"
echo "=================================="
echo "ngrok URL: $NGROK_URL"
echo "Local URL: $LOCAL_URL"
echo "Chatbot ID: $CHATBOT_ID"
echo "=================================="

# Test 1: Health Check
echo ""
echo "📋 Test 1: Health Check"
echo "------------------------"
echo "Testing ngrok..."
if curl -s "$NGROK_URL/health" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    echo "✅ ngrok health check passed"
else
    echo "❌ ngrok health check failed"
    exit 1
fi

echo "Testing local..."
if curl -s "$LOCAL_URL/health" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    echo "✅ Local health check passed"
else
    echo "⚠️  Local health check failed (but ngrok works)"
fi

# Test 2: Cache Performance
echo ""
echo "📋 Test 2: Cache Performance Test"
echo "----------------------------------"

echo "First request (cache MISS expected)..."
TIME1=$(curl -s -w "%{time_total}" -o /tmp/response1.json \
    -X POST "$NGROK_URL/api/public/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"message\": \"What are your products?\",
        \"chatbot_id\": \"$CHATBOT_ID\",
        \"session_id\": \"cache-test-1\"
    }")

if [ -s /tmp/response1.json ]; then
    PROCESSING_TIME1=$(jq -r '.processing_time // "N/A"' /tmp/response1.json)
    echo "✅ Response time: ${TIME1}s (Processing: ${PROCESSING_TIME1}ms)"
else
    echo "❌ First request failed"
    exit 1
fi

sleep 2

echo "Second request (cache HIT expected)..."
TIME2=$(curl -s -w "%{time_total}" -o /tmp/response2.json \
    -X POST "$NGROK_URL/api/public/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"message\": \"What are your products?\",
        \"chatbot_id\": \"$CHATBOT_ID\",
        \"session_id\": \"cache-test-2\"
    }")

if [ -s /tmp/response2.json ]; then
    PROCESSING_TIME2=$(jq -r '.processing_time // "N/A"' /tmp/response2.json)
    echo "✅ Response time: ${TIME2}s (Processing: ${PROCESSING_TIME2}ms)"

    # Calculate improvement
    IMPROVEMENT=$(echo "scale=1; ($TIME1 - $TIME2) * 1000" | bc)
    PERCENT=$(echo "scale=1; ($IMPROVEMENT / ($TIME1 * 1000)) * 100" | bc)

    echo ""
    echo "📊 Cache Performance:"
    echo "   First:  ${TIME1}s"
    echo "   Second: ${TIME2}s"
    echo "   Improvement: ${IMPROVEMENT}ms (${PERCENT}%)"

    if (( $(echo "$IMPROVEMENT > 100" | bc -l) )); then
        echo "   ✅ Cache is working well!"
    elif (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
        echo "   ⚠️  Some improvement detected"
    else
        echo "   ⚠️  No improvement - cache may not be working"
    fi
else
    echo "❌ Second request failed"
fi

# Test 3: Cache Metrics
echo ""
echo "📋 Test 3: Cache Metrics"
echo "------------------------"

if curl -s "$NGROK_URL/api/cache/metrics" > /tmp/metrics.json 2>&1; then
    if [ -s /tmp/metrics.json ] && jq -e . /tmp/metrics.json > /dev/null 2>&1; then
        echo "✅ Cache metrics available:"
        jq '.' /tmp/metrics.json
    else
        echo "⚠️  Cache metrics endpoint not available (add it to your app)"
    fi
else
    echo "⚠️  Could not fetch cache metrics"
fi

# Test 4: Quick Load Test
echo ""
echo "📋 Test 4: Quick Load Test (5 requests)"
echo "----------------------------------------"

for i in {1..5}; do
    echo -n "Request $i... "
    TIME=$(curl -s -w "%{time_total}" -o /dev/null \
        -X POST "$NGROK_URL/api/public/chat" \
        -H "Content-Type: application/json" \
        -d "{
            \"message\": \"What are your products?\",
            \"chatbot_id\": \"$CHATBOT_ID\",
            \"session_id\": \"load-test-$i\"
        }")
    echo "${TIME}s"
done

echo ""
echo "=================================="
echo "✅ Testing Complete!"
echo "=================================="
echo ""
echo "💡 Next Steps:"
echo "   1. Check ngrok web interface: http://127.0.0.1:4040"
echo "   2. Run full test: python3 test_ngrok_live.py $NGROK_URL $CHATBOT_ID"
echo "   3. Monitor cache: watch -n 5 'curl -s $NGROK_URL/api/cache/metrics | jq'"
echo ""

# Cleanup
rm -f /tmp/response1.json /tmp/response2.json /tmp/metrics.json
