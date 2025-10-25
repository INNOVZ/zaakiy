/**
 * K6 Load Testing Script for ZaaKy AI Platform
 *
 * Install K6:
 *   macOS: brew install k6
 *   Windows: choco install k6
 *   Linux: https://k6.io/docs/getting-started/installation/
 *
 * Usage:
 *   k6 run tests/load/k6_script.js
 *
 * Custom scenarios:
 *   k6 run tests/load/k6_script.js --vus 50 --duration 5m
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const chatResponseTime = new Trend('chat_response_time');
const successfulChats = new Counter('successful_chats');

// Test configuration
export const options = {
  scenarios: {
    // Scenario 1: Normal Load
    normal_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },   // Ramp up to 20 users
        { duration: '3m', target: 20 },   // Stay at 20 users
        { duration: '1m', target: 0 },    // Ramp down to 0
      ],
      gracefulRampDown: '30s',
    },

    // Scenario 2: Spike Test (uncomment to enable)
    // spike_test: {
    //   executor: 'ramping-vus',
    //   startVUs: 0,
    //   stages: [
    //     { duration: '10s', target: 100 },  // Quick spike
    //     { duration: '1m', target: 100 },   // Maintain
    //     { duration: '10s', target: 0 },    // Quick drop
    //   ],
    // },
  },

  // Thresholds - Test passes if these are met
  thresholds: {
    // 95% of requests should be below 500ms
    http_req_duration: ['p(95)<500'],

    // 99% of requests should be below 1000ms
    'http_req_duration{type:chat}': ['p(99)<1000'],

    // Error rate should be below 1%
    errors: ['rate<0.01'],

    // 95% of requests should succeed
    http_req_failed: ['rate<0.05'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8001';

// Sample test data
const QUESTIONS = [
  'What are your business hours?',
  'How can I contact support?',
  'What products do you offer?',
  'Tell me about your pricing',
  'Do you offer free shipping?',
  'What is your return policy?',
  'How do I track my order?',
  'Can I modify my order?',
  'What payment methods do you accept?',
  'Do you ship internationally?',
];

const CHATBOT_IDS = ['chatbot_001', 'chatbot_002', 'chatbot_003'];

function getRandomItem(array) {
  return array[Math.floor(Math.random() * array.length)];
}

function generateSessionId() {
  return `session_${__VU}_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
}

// Setup function - runs once per VU
export function setup() {
  // Test if the API is reachable
  const res = http.get(`${BASE_URL}/health`);

  check(res, {
    'API is reachable': (r) => r.status === 200,
  });

  if (res.status !== 200) {
    console.error('API health check failed. Aborting test.');
    throw new Error('API not reachable');
  }

  console.log('✅ API health check passed. Starting load test...');

  return {
    startTime: Date.now(),
  };
}

// Main test function
export default function (data) {
  const sessionId = generateSessionId();
  const chatbotId = getRandomItem(CHATBOT_IDS);

  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/health`);

    check(res, {
      'health status is 200': (r) => r.status === 200,
      'health response is valid': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'healthy' || body.status === 'degraded';
        } catch (e) {
          return false;
        }
      },
    });

    sleep(0.5);
  });

  group('Public Chat Interaction', function () {
    const question = getRandomItem(QUESTIONS);
    const startTime = Date.now();

    const payload = JSON.stringify({
      message: question,
      chatbot_id: chatbotId,
      session_id: sessionId,
    });

    const params = {
      headers: {
        'Content-Type': 'application/json',
      },
      tags: { type: 'chat' },
    };

    const res = http.post(`${BASE_URL}/api/public/chat`, payload, params);

    const duration = Date.now() - startTime;
    chatResponseTime.add(duration);

    const success = check(res, {
      'chat status is 200 or 201': (r) => r.status === 200 || r.status === 201,
      'chat response has body': (r) => r.body && r.body.length > 0,
      'chat response time < 2s': (r) => r.timings.duration < 2000,
    });

    if (success) {
      successfulChats.add(1);
    } else {
      errorRate.add(1);
      console.log(`❌ Chat failed: ${res.status} - ${res.body.substring(0, 100)}`);
    }

    errorRate.add(!success);

    sleep(2); // User thinks for 2 seconds
  });

  group('Chat History Retrieval', function () {
    const res = http.get(
      `${BASE_URL}/api/chat/history?session_id=${sessionId}`,
      {
        tags: { type: 'history' },
      }
    );

    check(res, {
      'history retrieval succeeds or requires auth': (r) =>
        r.status === 200 || r.status === 401,
    });

    sleep(1);
  });

  // Occasional health check
  if (Math.random() < 0.1) {
    group('Client Health Check', function () {
      const res = http.get(`${BASE_URL}/health/clients`);

      check(res, {
        'client health status is 200': (r) => r.status === 200,
      });
    });
  }

  sleep(1); // Wait between iterations
}

// Teardown function - runs once at the end
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`\n📊 Test completed in ${duration.toFixed(2)}s`);
  console.log('Check the summary above for detailed metrics.');
}

// Handle summary for custom reporting
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'reports/k6_summary.json': JSON.stringify(data),
    'reports/k6_summary.html': htmlReport(data),
  };
}

function textSummary(data, options) {
  // Simple text summary
  const { indent = '', enableColors = false } = options;
  let summary = '\n';

  summary += `${indent}✅ Test Summary:\n`;
  summary += `${indent}  Total Requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += `${indent}  Failed Requests: ${data.metrics.http_req_failed.values.passes}\n`;
  summary += `${indent}  Request Rate: ${data.metrics.http_reqs.values.rate.toFixed(2)}/s\n`;
  summary += `${indent}  Response Time (p95): ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += `${indent}  Response Time (p99): ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms\n`;

  return summary;
}

function htmlReport(data) {
  // Simple HTML report
  return `
<!DOCTYPE html>
<html>
<head>
  <title>K6 Load Test Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    h1 { color: #333; }
    .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
    .success { color: green; }
    .warning { color: orange; }
    .error { color: red; }
  </style>
</head>
<body>
  <h1>K6 Load Test Report</h1>
  <div class="metric">
    <strong>Total Requests:</strong> ${data.metrics.http_reqs.values.count}
  </div>
  <div class="metric">
    <strong>Request Rate:</strong> ${data.metrics.http_reqs.values.rate.toFixed(2)}/s
  </div>
  <div class="metric">
    <strong>Response Time (p95):</strong> ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
  </div>
  <div class="metric">
    <strong>Response Time (p99):</strong> ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms
  </div>
  <div class="metric">
    <strong>Error Rate:</strong> ${((data.metrics.http_req_failed.values.rate || 0) * 100).toFixed(2)}%
  </div>
</body>
</html>
  `;
}
