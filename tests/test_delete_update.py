#!/usr/bin/env python3
"""
Test script for delete and update functionality
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword123"


def login():
    """Login and get JWT token"""
    login_data = {"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}

    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        token = response.json().get("access_token")
        print(f"✅ Login successful")
        return token
    else:
        print(f"❌ Login failed: {response.text}")
        return None


def test_upload_and_delete(token):
    """Test uploading a URL and then deleting it"""
    headers = {"Authorization": f"Bearer {token}"}

    # Upload a URL
    url_data = {"url": "https://example.com"}
    response = requests.post(f"{BASE_URL}/uploads/url", json=url_data, headers=headers)

    if response.status_code != 200:
        print(f"❌ URL upload failed: {response.text}")
        return None

    upload_id = response.json().get("upload_id")
    print(f"✅ URL uploaded successfully, ID: {upload_id}")

    # Wait a bit for potential processing
    time.sleep(2)

    # Check status
    response = requests.get(f"{BASE_URL}/uploads/{upload_id}/status", headers=headers)
    if response.status_code == 200:
        status = response.json().get("status")
        print(f"📊 Upload status: {status}")

    # Delete the upload
    response = requests.delete(f"{BASE_URL}/uploads/{upload_id}", headers=headers)

    if response.status_code == 200:
        print(f"✅ Upload deleted successfully")
        return True
    else:
        print(f"❌ Delete failed: {response.text}")
        return False


def test_upload_and_update(token):
    """Test uploading a URL and then updating it"""
    headers = {"Authorization": f"Bearer {token}"}

    # Upload initial URL
    url_data = {"url": "https://example.com"}
    response = requests.post(f"{BASE_URL}/uploads/url", json=url_data, headers=headers)

    if response.status_code != 200:
        print(f"❌ Initial URL upload failed: {response.text}")
        return None

    upload_id = response.json().get("upload_id")
    print(f"✅ Initial URL uploaded successfully, ID: {upload_id}")

    # Wait a bit
    time.sleep(2)

    # Update with new URL
    new_url_data = {"url": "https://httpbin.org/json"}
    response = requests.put(
        f"{BASE_URL}/uploads/{upload_id}/url", json=new_url_data, headers=headers
    )

    if response.status_code == 200:
        print(f"✅ URL updated successfully")
        new_url = response.json().get("new_url")
        print(f"🔄 New URL: {new_url}")

        # Check status after update
        response = requests.get(
            f"{BASE_URL}/uploads/{upload_id}/status", headers=headers
        )
        if response.status_code == 200:
            status = response.json().get("status")
            source = response.json().get("source")
            print(f"📊 Updated status: {status}, source: {source}")

        return upload_id
    else:
        print(f"❌ Update failed: {response.text}")
        return None


def test_search(token):
    """Test search functionality"""
    headers = {"Authorization": f"Bearer {token}"}

    search_data = {"query": "example test content", "top_k": 3}

    response = requests.post(
        f"{BASE_URL}/uploads/search", json=search_data, headers=headers
    )

    if response.status_code == 200:
        results = response.json()
        print(f"✅ Search successful")
        print(f"🔍 Found {results.get('total_results', 0)} results")

        # Show first 2 results
        for i, result in enumerate(results.get("results", [])[:2]):
            print(
                f"   Result {i+1}: Score={result.get('score', 0):.3f}, Source={result.get('source', 'N/A')}"
            )

        return True
    else:
        print(f"❌ Search failed: {response.text}")
        return False


def test_list_uploads(token):
    """Test listing uploads"""
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(f"{BASE_URL}/uploads/", headers=headers)

    if response.status_code == 200:
        uploads = response.json().get("uploads", [])
        print(f"✅ Listed {len(uploads)} uploads")

        for upload in uploads[:3]:  # Show first 3
            print(
                f"   Upload: {upload.get('id')} - {upload.get('type')} - {upload.get('status')}"
            )

        return uploads
    else:
        print(f"❌ List uploads failed: {response.text}")
        return []


def main():
    print("🧪 Testing Delete and Update Functionality")
    print("=" * 50)

    # Login
    token = login()
    if not token:
        return

    # Test list uploads (before)
    print("\n📋 Listing existing uploads...")
    initial_uploads = test_list_uploads(token)

    # Test upload and delete
    print("\n🗑️  Testing Upload and Delete...")
    test_upload_and_delete(token)

    # Test upload and update
    print("\n🔄 Testing Upload and Update...")
    updated_upload_id = test_upload_and_update(token)

    # Test search
    print("\n🔍 Testing Search...")
    test_search(token)

    # Test list uploads (after)
    print("\n📋 Listing uploads after tests...")
    final_uploads = test_list_uploads(token)

    # Clean up the updated upload
    if updated_upload_id:
        print(f"\n🧹 Cleaning up test upload...")
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.delete(
            f"{BASE_URL}/uploads/{updated_upload_id}", headers=headers
        )
        if response.status_code == 200:
            print("✅ Cleanup successful")
        else:
            print(f"❌ Cleanup failed: {response.text}")

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main()
