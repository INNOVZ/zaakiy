"""
Locust load testing scenarios for ZaaKy AI Platform

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8001

Web UI: http://localhost:8089
"""

import json
import random
import time
from typing import Dict, List

from locust import HttpUser, TaskSet, between, task

# Test data
SAMPLE_QUESTIONS = [
    "What are your business hours?",
    "How can I contact customer support?",
    "What products do you offer?",
    "Tell me about your pricing",
    "Do you offer free shipping?",
    "What is your return policy?",
    "How do I track my order?",
    "Can I modify my order?",
    "What payment methods do you accept?",
    "Do you ship internationally?",
]

SAMPLE_CHATBOT_IDS = [
    "chatbot_001",
    "chatbot_002",
    "chatbot_003",
]

SAMPLE_ORG_IDS = [
    "org_001",
    "org_002",
    "org_003",
]


class ChatBotTasks(TaskSet):
    """Task set for chatbot interactions"""

    def on_start(self):
        """Initialize session data"""
        self.session_id = f"session_{random.randint(1000, 9999)}_{int(time.time())}"
        self.chatbot_id = random.choice(SAMPLE_CHATBOT_IDS)
        self.org_id = random.choice(SAMPLE_ORG_IDS)
        self.access_token = None

    @task(5)
    def send_chat_message(self):
        """Send a chat message (most common action)"""
        question = random.choice(SAMPLE_QUESTIONS)

        payload = {
            "message": question,
            "chatbot_id": self.chatbot_id,
            "session_id": self.session_id,
            "context": {},
        }

        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        with self.client.post(
            "/api/chat",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/api/chat [POST]",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                # Expected for unauthenticated requests
                response.success()
            else:
                response.failure(
                    f"Got unexpected status code {response.status_code}: {response.text}"
                )

    @task(3)
    def public_chat_message(self):
        """Send a public chat message (no auth required)"""
        question = random.choice(SAMPLE_QUESTIONS)

        payload = {
            "message": question,
            "chatbot_id": self.chatbot_id,
            "session_id": self.session_id,
        }

        with self.client.post(
            "/api/public/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="/api/public/chat [POST]",
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(
                    f"Got status code {response.status_code}: {response.text}"
                )

    @task(2)
    def get_chat_history(self):
        """Retrieve chat history"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        with self.client.get(
            f"/api/chat/history?session_id={self.session_id}",
            headers=headers,
            catch_response=True,
            name="/api/chat/history [GET]",
        ) as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(
                    f"Got status code {response.status_code}: {response.text}"
                )

    @task(1)
    def health_check(self):
        """Check system health (low frequency)"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health [GET]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class AuthenticatedUserTasks(TaskSet):
    """Task set for authenticated users"""

    def on_start(self):
        """Login and get access token"""
        # Note: Replace with actual test credentials
        login_payload = {
            "email": "test@example.com",
            "password": "testpassword123",
        }

        response = self.client.post(
            "/api/auth/login",
            json=login_payload,
            catch_response=True,
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get("access_token")
        else:
            # If login fails, use a mock token for testing
            self.access_token = "mock_token_for_testing"

        self.org_id = random.choice(SAMPLE_ORG_IDS)

    @task(3)
    def get_organization_info(self):
        """Get organization information"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get(
            f"/api/org/{self.org_id}",
            headers=headers,
            catch_response=True,
            name="/api/org/:id [GET]",
        ) as response:
            if response.status_code in [200, 401, 404]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(2)
    def list_chatbots(self):
        """List organization chatbots"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get(
            f"/api/org/{self.org_id}/chatbots",
            headers=headers,
            catch_response=True,
            name="/api/org/:id/chatbots [GET]",
        ) as response:
            if response.status_code in [200, 401, 404]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def check_user_profile(self):
        """Get user profile"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get(
            "/api/users/me",
            headers=headers,
            catch_response=True,
            name="/api/users/me [GET]",
        ) as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class UploadTasks(TaskSet):
    """Task set for upload operations"""

    def on_start(self):
        """Initialize with auth token"""
        self.access_token = "mock_token_for_testing"
        self.org_id = random.choice(SAMPLE_ORG_IDS)

    @task(2)
    def list_uploads(self):
        """List organization uploads"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get(
            f"/api/uploads?org_id={self.org_id}",
            headers=headers,
            catch_response=True,
            name="/api/uploads [GET]",
        ) as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def get_upload_status(self):
        """Check upload status"""
        upload_id = f"upload_{random.randint(1, 100)}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get(
            f"/api/uploads/{upload_id}",
            headers=headers,
            catch_response=True,
            name="/api/uploads/:id [GET]",
        ) as response:
            if response.status_code in [200, 401, 404]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


# User classes - These define different user behaviors
class PublicChatUser(HttpUser):
    """
    Simulates a public website visitor using the chatbot widget
    Most common user type (70% of traffic)
    """

    tasks = [ChatBotTasks]
    wait_time = between(2, 5)  # Wait 2-5 seconds between tasks
    weight = 7  # 70% of simulated users


class AuthenticatedUser(HttpUser):
    """
    Simulates an authenticated dashboard user
    Less common but more resource-intensive (20% of traffic)
    """

    tasks = [AuthenticatedUserTasks]
    wait_time = between(3, 8)
    weight = 2  # 20% of simulated users


class AdminUser(HttpUser):
    """
    Simulates admin users managing uploads and content
    Least common (10% of traffic)
    """

    tasks = [UploadTasks]
    wait_time = between(5, 10)
    weight = 1  # 10% of simulated users


# Specialized load test scenarios
class StressTestUser(HttpUser):
    """
    High-frequency user for stress testing
    Use this for peak load testing: locust -f locustfile.py --user-classes StressTestUser
    """

    tasks = [ChatBotTasks]
    wait_time = between(0.1, 0.5)  # Very short wait times
    weight = 1


class SpikeTester(HttpUser):
    """
    Simulates sudden traffic spikes
    Use with: locust -f locustfile.py --user-classes SpikeTester --spawn-rate 50
    """

    tasks = [ChatBotTasks]
    wait_time = between(0.5, 2)
    weight = 1
