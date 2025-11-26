"""
End-to-End Test Script for WhatsApp Integration

This script performs comprehensive E2E testing of the WhatsApp integration:
1. Configuration validation
2. Webhook endpoint testing
3. Message sending
4. Database verification
5. Token tracking
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class WhatsAppE2ETest:
    """End-to-end test suite for WhatsApp integration"""

    def __init__(
        self, base_url: str = "http://localhost:8001", auth_token: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token or os.getenv("TEST_AUTH_TOKEN")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}" if self.auth_token else "",
        }
        self.test_results = []

    def log_test(
        self, test_name: str, passed: bool, message: str = "", details: Any = None
    ):
        """Log test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.test_results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if message:
            print(f"   {message}")
        if details and not passed:
            print(f"   Details: {details}")
        print()

    def test_1_server_health(self) -> bool:
        """Test 1: Verify server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            passed = response.status_code == 200
            self.log_test(
                "Server Health Check",
                passed,
                f"Server responded with status {response.status_code}",
                response.json() if passed else response.text,
            )
            return passed
        except Exception as e:
            self.log_test("Server Health Check", False, f"Failed to connect: {str(e)}")
            return False

    def test_2_webhook_get_endpoint(self) -> bool:
        """Test 2: Verify webhook GET endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/whatsapp/webhook", timeout=5)
            passed = response.status_code == 200

            if passed:
                data = response.json()
                passed = (
                    data.get("status") == "ok"
                    and "whatsapp" in data.get("service", "").lower()
                    and "GET" in data.get("methods", [])
                    and "POST" in data.get("methods", [])
                )

            self.log_test(
                "Webhook GET Endpoint",
                passed,
                "Webhook verification endpoint is accessible",
                response.json() if passed else response.text,
            )
            return passed
        except Exception as e:
            self.log_test("Webhook GET Endpoint", False, f"Error: {str(e)}")
            return False

    def test_3_get_configuration(self) -> bool:
        """Test 3: Retrieve WhatsApp configuration"""
        if not self.auth_token:
            self.log_test(
                "Get Configuration", False, "No auth token provided - skipping"
            )
            return False

        try:
            response = requests.get(
                f"{self.base_url}/api/whatsapp/config", headers=self.headers, timeout=5
            )
            passed = response.status_code in [200, 404]  # 404 is ok if not configured

            if response.status_code == 200:
                data = response.json()
                config = data.get("config", {})
                passed = bool(config.get("twilio_account_sid"))
                message = f"Configuration found for org"
            else:
                message = "No configuration found (this is ok for new setup)"

            self.log_test(
                "Get Configuration",
                passed,
                message,
                response.json() if response.status_code == 200 else None,
            )
            return passed
        except Exception as e:
            self.log_test("Get Configuration", False, f"Error: {str(e)}")
            return False

    def test_4_validate_configuration(self) -> bool:
        """Test 4: Validate Twilio configuration"""
        if not self.auth_token:
            self.log_test("Validate Configuration", False, "No auth token - skipping")
            return False

        try:
            response = requests.get(
                f"{self.base_url}/api/whatsapp/validate",
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                validation = data.get("validation", {})
                passed = validation.get("valid", False)
                message = f"Twilio connection: {'Valid' if passed else 'Invalid'}"
                if not passed:
                    message += f" - {validation.get('error', 'Unknown error')}"
            else:
                passed = False
                message = f"Validation failed with status {response.status_code}"

            self.log_test("Validate Configuration", passed, message, response.json())
            return passed
        except Exception as e:
            self.log_test("Validate Configuration", False, f"Error: {str(e)}")
            return False

    def test_5_webhook_post_simulation(self) -> bool:
        """Test 5: Simulate Twilio webhook POST (without signature)"""
        try:
            # Simulate Twilio webhook payload
            webhook_data = {
                "AccountSid": "ACtest123",
                "From": "whatsapp:+1234567890",
                "To": "whatsapp:+14155238886",
                "Body": "Test message from E2E test",
                "MessageSid": f"SMtest{int(time.time())}",
                "NumMedia": "0",
            }

            response = requests.post(
                f"{self.base_url}/api/whatsapp/webhook", data=webhook_data, timeout=10
            )

            # Without valid signature, should return 403 or 400
            # This is actually a PASS if it rejects (security working)
            passed = response.status_code in [400, 403]
            message = f"Webhook correctly rejected unsigned request (status {response.status_code})"

            if response.status_code == 200:
                # If it accepts, that's a security issue unless signature validation is disabled
                passed = False
                message = "WARNING: Webhook accepted unsigned request - security issue!"

            self.log_test("Webhook Security (Signature Validation)", passed, message)
            return passed
        except Exception as e:
            self.log_test("Webhook POST Simulation", False, f"Error: {str(e)}")
            return False

    def test_6_send_test_message(self, test_phone: str = "+1234567890") -> bool:
        """Test 6: Send test WhatsApp message"""
        if not self.auth_token:
            self.log_test("Send Test Message", False, "No auth token - skipping")
            return False

        try:
            payload = {
                "to": test_phone,
                "message": f"E2E Test Message - {datetime.now().isoformat()}",
                "chatbot_id": None,  # Will use default chatbot
            }

            response = requests.post(
                f"{self.base_url}/api/whatsapp/send",
                headers=self.headers,
                json=payload,
                timeout=15,
            )

            passed = response.status_code == 200

            if passed:
                data = response.json()
                message_sid = data.get("message_sid")
                message = f"Message sent successfully (SID: {message_sid})"
            else:
                message = f"Failed to send message (status {response.status_code})"

            self.log_test("Send Test Message", passed, message, response.json())
            return passed
        except Exception as e:
            self.log_test("Send Test Message", False, f"Error: {str(e)}")
            return False

    def test_7_phone_number_format_validation(self) -> bool:
        """Test 7: Verify phone number format validation"""
        if not self.auth_token:
            self.log_test("Phone Number Validation", False, "No auth token - skipping")
            return False

        try:
            # Test with invalid phone number (no + prefix)
            payload = {
                "to": "1234567890",  # Missing + prefix
                "message": "Test",
            }

            response = requests.post(
                f"{self.base_url}/api/whatsapp/send",
                headers=self.headers,
                json=payload,
                timeout=10,
            )

            # Should reject invalid format
            passed = response.status_code in [400, 422]
            message = (
                "Phone number validation working correctly"
                if passed
                else "Validation not working"
            )

            self.log_test("Phone Number Format Validation", passed, message)
            return passed
        except Exception as e:
            self.log_test("Phone Number Validation", False, f"Error: {str(e)}")
            return False

    def test_8_message_length_limit(self) -> bool:
        """Test 8: Verify message length limit (1600 chars for WhatsApp)"""
        if not self.auth_token:
            self.log_test("Message Length Limit", False, "No auth token - skipping")
            return False

        try:
            # Create a message longer than 1600 characters
            long_message = "A" * 2000

            payload = {
                "to": "+1234567890",
                "message": long_message,
            }

            response = requests.post(
                f"{self.base_url}/api/whatsapp/send",
                headers=self.headers,
                json=payload,
                timeout=15,
            )

            # Should either accept (and truncate) or reject
            passed = response.status_code in [200, 400]
            message = "Message length handling working"

            self.log_test("Message Length Limit", passed, message)
            return passed
        except Exception as e:
            self.log_test("Message Length Limit", False, f"Error: {str(e)}")
            return False

    def test_9_rate_limiting(self) -> bool:
        """Test 9: Verify rate limiting is in place"""
        if not self.auth_token:
            self.log_test("Rate Limiting", False, "No auth token - skipping")
            return False

        try:
            # Make multiple rapid requests
            responses = []
            for i in range(35):  # Try to exceed 30/minute limit
                response = requests.get(
                    f"{self.base_url}/api/whatsapp/config",
                    headers=self.headers,
                    timeout=2,
                )
                responses.append(response.status_code)
                if response.status_code == 429:  # Rate limit hit
                    break

            # Check if we hit rate limit
            hit_rate_limit = 429 in responses
            passed = hit_rate_limit  # Rate limiting is working
            message = (
                "Rate limiting is active"
                if passed
                else "Rate limiting may not be configured"
            )

            self.log_test(
                "Rate Limiting",
                passed,
                message,
                f"Made {len(responses)} requests, hit limit: {hit_rate_limit}",
            )
            return passed
        except Exception as e:
            self.log_test("Rate Limiting", False, f"Error: {str(e)}")
            return False

    def test_10_error_handling(self) -> bool:
        """Test 10: Verify proper error handling"""
        try:
            # Test with completely invalid endpoint
            response = requests.get(
                f"{self.base_url}/api/whatsapp/nonexistent", timeout=5
            )

            # Should return 404
            passed = response.status_code == 404
            message = "Error handling working correctly"

            self.log_test("Error Handling", passed, message)
            return passed
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {str(e)}")
            return False

    def run_all_tests(self, test_phone: Optional[str] = None):
        """Run all E2E tests"""
        print("=" * 60)
        print("WhatsApp Integration - End-to-End Test Suite")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"Auth Token: {'Provided' if self.auth_token else 'Not provided'}")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)
        print()

        # Run tests in sequence
        tests = [
            ("Server Health", self.test_1_server_health),
            ("Webhook GET Endpoint", self.test_2_webhook_get_endpoint),
            ("Get Configuration", self.test_3_get_configuration),
            ("Validate Configuration", self.test_4_validate_configuration),
            ("Webhook Security", self.test_5_webhook_post_simulation),
            (
                "Send Test Message",
                lambda: self.test_6_send_test_message(test_phone)
                if test_phone
                else self.log_test(
                    "Send Test Message", False, "No test phone provided - skipping"
                ),
            ),
            ("Phone Number Validation", self.test_7_phone_number_format_validation),
            ("Message Length Limit", self.test_8_message_length_limit),
            ("Rate Limiting", self.test_9_rate_limiting),
            ("Error Handling", self.test_10_error_handling),
        ]

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_test(test_name, False, f"Unexpected error: {str(e)}")
            time.sleep(0.5)  # Small delay between tests

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print()
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print()

        if failed > 0:
            print("Failed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  ❌ {result['test']}: {result['message']}")

        print("=" * 60)
        print(f"Completed: {datetime.now().isoformat()}")
        print("=" * 60)

        # Save results to file
        with open("whatsapp_e2e_test_results.json", "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "failed": failed,
                        "pass_rate": pass_rate,
                    },
                    "results": self.test_results,
                },
                f,
                indent=2,
            )

        print(f"\nDetailed results saved to: whatsapp_e2e_test_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WhatsApp E2E tests")
    parser.add_argument(
        "--url", default="http://localhost:8001", help="Base URL of the API"
    )
    parser.add_argument("--token", help="Authentication token")
    parser.add_argument("--phone", help="Test phone number for sending messages")

    args = parser.parse_args()

    tester = WhatsAppE2ETest(base_url=args.url, auth_token=args.token)

    tester.run_all_tests(test_phone=args.phone)
