#!/usr/bin/env python3
"""
Health check script for the Zentria backend application

This script checks the health of various components:
- Database connectivity
- Vector database connectivity
- AI service connectivity
- Storage bucket accessibility
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import validate_environment
from services.shared.client_manager import client_manager
from services.storage.supabase_client import client

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check utility for system components"""

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now(timezone.utc)

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations"""
        try:
            # Test basic database connection
            response = await client.get("/users", params={"limit": "1"})

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "message": "Database connection successful",
                    "response_time_ms": 0,  # Could add timing here
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"Database returned status {response.status_code}",
                    "error": response.text,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Database connection failed",
                "error": str(e),
            }

    def check_vector_database(self) -> Dict[str, Any]:
        """Check Pinecone vector database connectivity"""
        try:
            # Test Pinecone connection
            index = client_manager.pinecone_index

            # Simple query to test connectivity
            test_query = [0.0] * 1536  # Dummy vector
            result = index.query(vector=test_query, top_k=1, include_metadata=False)

            return {
                "status": "healthy",
                "message": "Vector database connection successful",
                "index_stats": {
                    "total_vectors": getattr(result, "total_vector_count", "unknown")
                },
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Vector database connection failed",
                "error": str(e),
            }

    def check_ai_services(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity"""
        try:
            openai_client = client_manager.openai

            # Test with a simple completion
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )

            return {
                "status": "healthy",
                "message": "AI services connection successful",
                "model_used": response.model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "AI services connection failed",
                "error": str(e),
            }

    def check_storage(self) -> Dict[str, Any]:
        """Check Supabase storage bucket accessibility"""
        try:
            supabase = client_manager.supabase

            # List buckets to test storage access
            buckets = supabase.storage.list_buckets()

            uploads_bucket = None
            for bucket in buckets:
                if bucket.name == "uploads":
                    uploads_bucket = bucket
                    break

            if uploads_bucket:
                return {
                    "status": "healthy",
                    "message": "Storage bucket accessible",
                    "bucket_info": {
                        "name": uploads_bucket.name,
                        "public": uploads_bucket.public,
                        "id": uploads_bucket.id,
                    },
                }
            else:
                return {
                    "status": "warning",
                    "message": "Uploads bucket not found",
                    "available_buckets": [b.name for b in buckets],
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Storage access failed",
                "error": str(e),
            }

    def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration"""
        try:
            validation_result = validate_environment()

            if validation_result["valid"]:
                return {
                    "status": "healthy",
                    "message": "Environment configuration valid",
                    "warnings": validation_result.get("warnings", []),
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Environment configuration invalid",
                    "errors": validation_result.get("errors", []),
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Environment validation failed",
                "error": str(e),
            }

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        print("🔍 Running health checks...")
        print("=" * 50)

        checks = [
            ("Environment", self.check_environment),
            ("Database", self.check_database),
            ("Vector Database", self.check_vector_database),
            ("AI Services", self.check_ai_services),
            ("Storage", self.check_storage),
        ]

        for name, check_func in checks:
            print(f"Checking {name}...", end=" ")
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                self.results[name] = result

                if result["status"] == "healthy":
                    print("✅")
                elif result["status"] == "warning":
                    print("⚠️")
                else:
                    print("❌")

            except Exception as e:
                self.results[name] = {
                    "status": "error",
                    "message": f"Check failed: {str(e)}",
                }
                print("❌")

        return self.results

    def print_summary(self):
        """Print health check summary"""
        print("\n" + "=" * 50)
        print("🏥 HEALTH CHECK SUMMARY")
        print("=" * 50)

        total_checks = len(self.results)
        healthy_checks = sum(
            1 for r in self.results.values() if r["status"] == "healthy"
        )
        warning_checks = sum(
            1 for r in self.results.values() if r["status"] == "warning"
        )
        unhealthy_checks = sum(
            1 for r in self.results.values() if r["status"] in ["unhealthy", "error"]
        )

        print(f"Total checks: {total_checks}")
        print(f"✅ Healthy: {healthy_checks}")
        print(f"⚠️  Warnings: {warning_checks}")
        print(f"❌ Unhealthy: {unhealthy_checks}")

        if unhealthy_checks > 0:
            print("\n❌ Unhealthy components:")
            for name, result in self.results.items():
                if result["status"] in ["unhealthy", "error"]:
                    print(f"   - {name}: {result['message']}")

        if warning_checks > 0:
            print("\n⚠️  Components with warnings:")
            for name, result in self.results.items():
                if result["status"] == "warning":
                    print(f"   - {name}: {result['message']}")

        # Overall status
        if unhealthy_checks == 0:
            print("\n🎉 Overall Status: HEALTHY")
            return 0
        else:
            print("\n🚨 Overall Status: UNHEALTHY")
            return 1


async def main():
    """Main health check function"""
    print("🏥 Zentria Backend Health Check")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")

    checker = HealthChecker()
    await checker.run_all_checks()
    exit_code = checker.print_summary()

    print(f"\nCompleted at: {datetime.now(timezone.utc).isoformat()}")
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
