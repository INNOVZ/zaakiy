# Load Testing Guide for ZaaKy AI Platform

This guide provides comprehensive instructions for running load tests on your application.

## 🚀 Quick Start

### 1. Install Locust

```bash
# Add to requirements.txt or install directly
pip install locust
```

### 2. Run Basic Load Test

```bash
# From the backend directory
cd /Users/jithinjacob/Desktop/zaakiy\ core/backend

# Start Locust with web UI
locust -f tests/load/locustfile.py --host=http://localhost:8001

# Open browser to http://localhost:8089
# Configure: 100 users, 10 spawn rate, then click "Start"
```

### 3. Run Headless (CI/CD)

```bash
# Run without web UI (for automated testing)
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 60s \
    --headless \
    --html report.html
```

## 📊 Load Testing Scenarios

### Scenario 1: Normal Load (Baseline)

Simulates typical production traffic

```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless \
    --html reports/normal_load.html
```

**Expected Results:**

- Response time: < 500ms (p95)
- Success rate: > 99%
- RPS (requests per second): 20-50

### Scenario 2: Peak Load

Simulates peak business hours

```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --users 200 \
    --spawn-rate 20 \
    --run-time 10m \
    --headless \
    --html reports/peak_load.html
```

**Expected Results:**

- Response time: < 1000ms (p95)
- Success rate: > 98%
- RPS: 100-200

### Scenario 3: Stress Test

Find breaking point of your system

```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 10m \
    --headless \
    --html reports/stress_test.html \
    StressTestUser
```

**Watch for:**

- When response times exceed 3s
- When error rate exceeds 5%
- System resource utilization (CPU, memory, connections)

### Scenario 4: Spike Test

Sudden traffic surge

```bash
# Start with 10 users
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --users 500 \
    --spawn-rate 100 \
    --run-time 3m \
    --headless \
    --html reports/spike_test.html
```

**Expected Results:**

- System should recover within 30s
- No cascading failures
- Success rate should stabilize > 95%

### Scenario 5: Endurance Test

Long-running stability test

```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 2h \
    --headless \
    --html reports/endurance_test.html
```

**Watch for:**

- Memory leaks (gradually increasing memory)
- Connection pool exhaustion
- Database connection issues
- Cache degradation

## 🎯 Testing Specific Endpoints

### Test Only Chat Endpoints

```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --user-classes PublicChatUser \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m
```

### Test Only Authenticated Endpoints

```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8001 \
    --user-classes AuthenticatedUser \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m
```

## 📈 Alternative Load Testing Tools

### 1. K6 (Recommended for CI/CD)

Install:

```bash
brew install k6  # macOS
# or
# Download from https://k6.io/
```

Create `tests/load/k6_script.js`:

```javascript
import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 20 }, // Ramp up
    { duration: "1m", target: 20 }, // Stay at 20 users
    { duration: "30s", target: 0 }, // Ramp down
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"], // 95% of requests under 500ms
    http_req_failed: ["rate<0.01"], // Error rate under 1%
  },
};

export default function () {
  const url = "http://localhost:8001/api/public/chat";
  const payload = JSON.stringify({
    message: "What are your business hours?",
    chatbot_id: "chatbot_001",
    session_id: `session_${__VU}_${__ITER}`,
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "response time < 500ms": (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

Run:

```bash
k6 run tests/load/k6_script.js
```

### 2. Apache Bench (ab) - Quick & Simple

```bash
# Test health endpoint
ab -n 1000 -c 10 http://localhost:8001/health

# Test chat endpoint (POST)
ab -n 1000 -c 10 -p tests/load/chat_payload.json \
   -T application/json \
   http://localhost:8001/api/public/chat
```

Create `tests/load/chat_payload.json`:

```json
{
  "message": "What are your business hours?",
  "chatbot_id": "chatbot_001",
  "session_id": "test_session"
}
```

### 3. wrk - High Performance

```bash
# Install
brew install wrk  # macOS

# Simple GET test
wrk -t4 -c100 -d30s http://localhost:8001/health

# With Lua script for POST
wrk -t4 -c100 -d30s -s tests/load/chat.lua http://localhost:8001
```

Create `tests/load/chat.lua`:

```lua
wrk.method = "POST"
wrk.body   = '{"message":"Test","chatbot_id":"chatbot_001","session_id":"test"}'
wrk.headers["Content-Type"] = "application/json"
```

## 🔍 Monitoring During Load Tests

### 1. System Resources

```bash
# Terminal 1: Start your app
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Terminal 2: Monitor system resources
watch -n 1 'ps aux | grep python | grep -v grep'

# Or use htop
htop

# Monitor specific process
top -pid $(pgrep -f "uvicorn")
```

### 2. Application Logs

```bash
# Watch logs during load test
tail -f backend/app/logs/zaaky_*.log

# Watch error logs
tail -f backend/app/logs/zaaky_errors.log

# Filter for specific issues
tail -f backend/app/logs/zaaky_*.log | grep -i "error\|warning\|timeout"
```

### 3. Database Connections

```bash
# If using PostgreSQL locally
psql -U your_user -d your_db -c "SELECT count(*) FROM pg_stat_activity;"

# For Supabase, check dashboard
# Or use their REST API
```

### 4. Redis Cache

```bash
# Connect to Redis
redis-cli

# Monitor commands
MONITOR

# Check stats
INFO stats

# Check connected clients
CLIENT LIST

# Check memory usage
INFO memory
```

## 📊 Analyzing Results

### Key Metrics to Track

1. **Response Time**

   - P50 (median): < 200ms
   - P95: < 500ms
   - P99: < 1000ms

2. **Throughput**

   - Requests per second (RPS)
   - Total requests handled

3. **Error Rate**

   - Should be < 1% under normal load
   - Should be < 5% under stress

4. **Resource Utilization**
   - CPU: < 70% under normal load
   - Memory: Stable (no growth over time)
   - Database connections: Within pool limits

### Reading Locust Reports

After running with `--html report.html`, open the report to see:

- **Total requests**: Overall traffic handled
- **Failures**: Any errors encountered
- **Response time percentiles**: P50, P95, P99
- **RPS chart**: Requests per second over time
- **Response time chart**: How latency changed over time

### Red Flags to Watch For

🚨 **Performance Issues:**

- Response times increasing over time
- Error rates > 1%
- Timeouts occurring

🚨 **Resource Issues:**

- Memory continuously growing (memory leak)
- CPU at 100%
- Database connection pool exhausted

🚨 **System Instability:**

- Application crashes
- Connection refused errors
- Database deadlocks

## 🎯 Best Practices

1. **Start Small**: Begin with low load and gradually increase
2. **Test Isolated**: Test one component at a time before full system
3. **Monitor Everything**: Watch app, database, cache, and system resources
4. **Use Realistic Data**: Use production-like test data
5. **Test Regularly**: Make load testing part of your CI/CD
6. **Document Baselines**: Record normal performance for comparison
7. **Test Different Scenarios**: Normal, peak, stress, spike, endurance

## 🔧 Troubleshooting

### Issue: Connection Refused

```bash
# Check if app is running
curl http://localhost:8001/health

# Check port
lsof -i :8001
```

### Issue: High Error Rate

```bash
# Check application logs
tail -f backend/app/logs/zaaky_errors.log

# Check database connections
# Monitor Redis connections
```

### Issue: Slow Response Times

- Check database query performance
- Verify cache hit rates
- Monitor external API calls (OpenAI, Pinecone)
- Check network latency

## 📚 Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [K6 Documentation](https://k6.io/docs/)
- [Load Testing Best Practices](https://www.nginx.com/blog/load-testing-best-practices/)
- [Performance Testing Types](https://www.blazemeter.com/blog/performance-testing-vs-load-testing-vs-stress-testing)

## 🚦 Production Load Testing Checklist

Before testing production:

- [ ] Get approval from stakeholders
- [ ] Test in staging environment first
- [ ] Use production-like infrastructure
- [ ] Schedule during low-traffic periods
- [ ] Have rollback plan ready
- [ ] Monitor all systems
- [ ] Start with low load
- [ ] Have team on standby
- [ ] Document results
- [ ] Review and optimize based on findings
