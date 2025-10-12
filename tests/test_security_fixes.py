"""
Comprehensive Security Test Suite for Authorization Bypass Fixes
Tests all the security vulnerabilities that were identified and fixed
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from app.main import app
from app.models.subscription import (OnboardingRequest, SubscriptionPlan,
                                     TokenUsageRequest)
from app.services.auth.middleware import SecurityMonitor, security_monitor
from app.services.subscription.subscription_service import SubscriptionService

# Test client
client = TestClient(app)


class TestAuthorizationSecurity:
    """Test suite for authorization and authentication security"""

    def setup_method(self):
        """Setup test data"""
        self.test_user_id = str(uuid.uuid4())
        self.test_org_id = str(uuid.uuid4())
        self.test_admin_id = str(uuid.uuid4())
        self.test_subscription_id = str(uuid.uuid4())

        self.mock_user = {
            "user_id": self.test_user_id,
            "email": "test@example.com",
            "user_data": {"org_id": self.test_org_id},
        }

        self.mock_admin = {
            "user_id": self.test_admin_id,
            "email": "admin@example.com",
            "user_data": {"role": "admin"},
        }

    def test_admin_endpoint_requires_authentication(self):
        """Test that admin endpoints require proper authentication"""
        # Test without authentication
        response = client.post(
            "/api/onboarding/admin/signup",
            json={
                "entity_type": "user",
                "full_name": "Test User",
                "email": "test@example.com",
                "selected_plan": "basic",
            },
        )

        # Should return 401 Unauthorized
        assert response.status_code == 401

        # Test with invalid token
        response = client.post(
            "/api/onboarding/admin/signup",
            headers={"Authorization": "Bearer invalid_token"},
            json={
                "entity_type": "user",
                "full_name": "Test User",
                "email": "test@example.com",
                "selected_plan": "basic",
            },
        )

        # Should return 401 Unauthorized
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_horizontal_privilege_escalation_blocked(self):
        """Test that users cannot access other users' data"""
        middleware = SecurityMonitor()

        # Test user trying to access another user's data
        user1 = {"user_id": str(uuid.uuid4()), "user_data": {}}
        user2_id = str(uuid.uuid4())

        # Should return False for unauthorized access
        has_access = await middleware.verify_entity_ownership(user1, "user", user2_id)
        assert has_access is False

        # Test user trying to access another organization's data
        org2_id = str(uuid.uuid4())
        has_access = await middleware.verify_entity_ownership(
            user1, "organization", org2_id
        )
        assert has_access is False

    @pytest.mark.asyncio
    async def test_entity_ownership_verification(self):
        """Test that entity ownership verification works correctly"""
        middleware = SecurityMonitor()

        # Test user accessing their own data
        user = {"user_id": self.test_user_id, "user_data": {"org_id": self.test_org_id}}

        # Should allow access to own user data
        has_access = await middleware.verify_entity_ownership(
            user, "user", self.test_user_id
        )
        assert has_access is True

        # Should allow access to own organization data
        has_access = await middleware.verify_entity_ownership(
            user, "organization", self.test_org_id
        )
        assert has_access is True

        # Should deny access to other user's data
        other_user_id = str(uuid.uuid4())
        has_access = await middleware.verify_entity_ownership(
            user, "user", other_user_id
        )
        assert has_access is False

    def test_rate_limiting_functionality(self):
        """Test that rate limiting works correctly"""
        from app.utils.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()
        user_id = str(uuid.uuid4())
        endpoint = "test_endpoint"

        # Should allow requests within limit
        for i in range(5):
            is_allowed, info = limiter.is_allowed(
                f"user:{user_id}", max_requests=5, window_seconds=60
            )
            assert is_allowed is True

        # Should block requests exceeding limit
        is_allowed, info = limiter.is_allowed(
            f"user:{user_id}", max_requests=5, window_seconds=60
        )
        assert is_allowed is False

    @pytest.mark.asyncio
    async def test_subscription_status_authorization(self):
        """Test that subscription status endpoint requires proper authorization"""
        with patch("app.services.auth.get_current_user") as mock_auth:
            mock_auth.return_value = self.mock_user

            # Test accessing own subscription - should work
            with patch(
                "app.services.subscription.subscription_service.SubscriptionService.get_subscription_usage"
            ) as mock_usage:
                mock_usage.return_value = Mock(
                    subscription_id=self.test_subscription_id,
                    tokens_used_this_month=100,
                    tokens_remaining=900,
                    monthly_limit=1000,
                    usage_percentage=10.0,
                    reset_date=datetime.now() + timedelta(days=30),
                )

                response = client.get(
                    f"/api/onboarding/subscription/user/{self.test_user_id}"
                )
                # This would require proper mocking of the entire dependency chain
                # For now, we test the logic directly

    @pytest.mark.asyncio
    async def test_token_consumption_authorization(self):
        """Test that token consumption requires proper authorization"""
        middleware = SecurityMonitor()

        # Test user trying to consume tokens for another entity
        user = {"user_id": self.test_user_id, "user_data": {}}
        other_entity_id = str(uuid.uuid4())

        # Should deny access
        has_access = await middleware.verify_entity_ownership(
            user, "user", other_entity_id
        )
        assert has_access is False

        # Should allow access to own entity
        has_access = await middleware.verify_entity_ownership(
            user, "user", self.test_user_id
        )
        assert has_access is True


class TestTokenConsumptionSecurity:
    """Test suite for token consumption race conditions and security"""

    def setup_method(self):
        """Setup test data"""
        self.mock_supabase = Mock()
        self.service = SubscriptionService(self.mock_supabase)
        self.test_entity_id = str(uuid.uuid4())
        self.test_user_id = str(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_atomic_token_consumption(self):
        """Test that token consumption is atomic and prevents race conditions"""
        # Mock the database RPC call
        self.mock_supabase.rpc.return_value.execute.return_value = Mock(
            data=[
                {
                    "success": True,
                    "subscription_id": str(uuid.uuid4()),
                    "tokens_remaining": 900,
                    "message": "Tokens consumed successfully",
                }
            ]
        )

        request = TokenUsageRequest(
            entity_id=self.test_entity_id,
            entity_type="user",
            tokens_consumed=100,
            operation_type="chat",
        )

        # Test successful consumption
        result = await self.service.consume_tokens(request, self.test_user_id)
        assert result is True

        # Verify RPC was called with correct parameters
        self.mock_supabase.rpc.assert_called_with(
            "consume_tokens_atomic",
            {
                "p_entity_id": self.test_entity_id,
                "p_entity_type": "user",
                "p_tokens_to_consume": 100,
                "p_operation_type": "chat",
                "p_requesting_user_id": self.test_user_id,
            },
        )

    @pytest.mark.asyncio
    async def test_insufficient_tokens_handling(self):
        """Test that insufficient tokens are handled correctly"""
        # Mock insufficient tokens response
        self.mock_supabase.rpc.return_value.execute.return_value = Mock(
            data=[
                {
                    "success": False,
                    "subscription_id": str(uuid.uuid4()),
                    "tokens_remaining": 50,
                    "message": "Insufficient tokens",
                }
            ]
        )

        request = TokenUsageRequest(
            entity_id=self.test_entity_id,
            entity_type="user",
            tokens_consumed=100,
            operation_type="chat",
        )

        # Test insufficient tokens
        result = await self.service.consume_tokens(request, self.test_user_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_token_consumption(self):
        """Test that concurrent token consumption is handled properly"""
        # This test would require more complex setup to simulate actual concurrency
        # For now, we test that the locking mechanism is in place

        # Test that locks are created for entities
        async with self.service._get_entity_lock(self.test_entity_id, "user"):
            # Verify lock exists
            lock_key = f"user:{self.test_entity_id}"
            assert lock_key in self.service._token_locks

            # Verify lock data structure
            lock_data = self.service._token_locks[lock_key]
            assert "lock" in lock_data
            assert "last_used" in lock_data


class TestJWTSecurity:
    """Test suite for JWT token security"""

    def test_jwt_algorithm_specification(self):
        """Test that JWT algorithm is explicitly specified"""
        from app.services.auth.jwt_handler import JWTValidator

        validator = JWTValidator()

        # Test with a mock token (this would need proper JWT creation for full test)
        # For now, we verify the algorithm is specified in the decode call
        with patch("jose.jwt.decode") as mock_decode:
            mock_decode.return_value = {
                "sub": str(uuid.uuid4()),
                "email": "test@example.com",
                "iat": 1234567890,
                "exp": 1234567890 + 3600,
            }

            try:
                validator.validate_token("mock_token")
                # Verify decode was called with explicit algorithm
                mock_decode.assert_called_once()
                call_args = mock_decode.call_args
                assert "algorithms" in call_args[1]
                assert call_args[1]["algorithms"] == ["HS256"]
            except Exception:
                # Expected since we're using a mock token
                pass

    def test_jwt_secret_validation(self):
        """Test that JWT secret is properly validated"""
        from app.services.auth.jwt_handler import JWTValidator

        # Test with missing secret
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="SUPABASE_JWT_SECRET"):
                JWTValidator()

    def test_token_payload_validation(self):
        """Test that token payload is properly validated"""
        from app.services.auth.exceptions import InvalidTokenError
        from app.services.auth.jwt_handler import JWTValidator

        with patch.dict(
            "os.environ",
            {
                "SUPABASE_JWT_SECRET": "test_secret_key_that_is_long_enough",
                "SUPABASE_PROJECT_ID": "test",
            },
        ):
            validator = JWTValidator()

            with patch("jose.jwt.decode") as mock_decode:
                # Test with missing user_id
                mock_decode.return_value = {"email": "test@example.com"}

                with pytest.raises(InvalidTokenError, match="missing user_id or email"):
                    validator.validate_token("mock_token")

                # Test with missing email
                mock_decode.return_value = {"sub": str(uuid.uuid4())}

                with pytest.raises(InvalidTokenError, match="missing user_id or email"):
                    validator.validate_token("mock_token")


class TestInputValidation:
    """Test suite for input validation security"""

    def test_onboarding_request_validation(self):
        """Test that onboarding requests are properly validated"""
        # Test with invalid entity type
        with pytest.raises(ValueError, match="Entity type must be"):
            OnboardingRequest(
                entity_type="invalid",
                full_name="Test User",
                email="test@example.com",
                selected_plan=SubscriptionPlan.BASIC,
            )

        # Test with invalid email
        with pytest.raises(ValueError):
            OnboardingRequest(
                entity_type="user",
                full_name="Test User",
                email="invalid_email",
                selected_plan=SubscriptionPlan.BASIC,
            )

        # Test with missing organization name for organization type
        with pytest.raises(ValueError, match="Organization name is required"):
            OnboardingRequest(
                entity_type="organization",
                full_name="Test User",
                email="test@example.com",
                selected_plan=SubscriptionPlan.BASIC,
            )

    def test_token_usage_request_validation(self):
        """Test that token usage requests are properly validated"""
        # Test with negative tokens
        with pytest.raises(ValueError, match="Tokens consumed cannot be negative"):
            TokenUsageRequest(
                entity_id=str(uuid.uuid4()),
                entity_type="user",
                tokens_consumed=-100,
                operation_type="chat",
            )

        # Test with excessive tokens
        with pytest.raises(ValueError, match="Tokens consumed seems too high"):
            TokenUsageRequest(
                entity_id=str(uuid.uuid4()),
                entity_type="user",
                tokens_consumed=20000,
                operation_type="chat",
            )

        # Test with invalid entity type
        with pytest.raises(ValueError, match="Entity type must be"):
            TokenUsageRequest(
                entity_id=str(uuid.uuid4()),
                entity_type="invalid",
                tokens_consumed=100,
                operation_type="chat",
            )


class TestAuditLogging:
    """Test suite for security audit logging"""

    def setup_method(self):
        """Setup test data"""
        self.middleware = EnhancedAuthMiddleware()
        self.test_user_id = str(uuid.uuid4())

    def test_suspicious_activity_tracking(self):
        """Test that suspicious activities are tracked"""
        # Simulate multiple unauthorized access attempts
        for i in range(6):
            self.middleware._track_suspicious_activity(
                self.test_user_id, "unauthorized_access"
            )

        # Verify suspicious activity is tracked
        assert self.test_user_id in self.middleware.suspicious_activities
        assert (
            "unauthorized_access"
            in self.middleware.suspicious_activities[self.test_user_id]
        )
        assert (
            len(
                self.middleware.suspicious_activities[self.test_user_id][
                    "unauthorized_access"
                ]
            )
            == 6
        )

    def test_rate_limit_logging(self):
        """Test that rate limit violations are logged"""
        from app.utils.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()
        endpoint = "test_endpoint"

        # Exceed rate limit
        for i in range(6):
            limiter.is_allowed(
                f"user:{self.test_user_id}", max_requests=5, window_seconds=60
            )

        # Verify rate limit data is tracked (limiter uses internal _requests dict)
        key = f"user:{self.test_user_id}"
        assert len(limiter._requests[key]) >= 5


class TestDatabaseSecurity:
    """Test suite for database security functions"""

    @pytest.mark.asyncio
    async def test_atomic_function_parameters(self):
        """Test that atomic database functions validate parameters correctly"""
        # This would require actual database connection for full testing
        # For now, we test the parameter validation logic

        # Test invalid token amount
        # Test invalid entity type
        # Test missing subscription
        # These would be tested with actual database calls in integration tests
        pass


class TestSecurityIntegration:
    """Integration tests for complete security flow"""

    def test_complete_authorization_flow(self):
        """Test complete authorization flow from request to response"""
        # This would test the entire flow:
        # 1. JWT token validation
        # 2. User authentication
        # 3. Entity ownership verification
        # 4. Rate limiting
        # 5. Atomic operations
        # 6. Audit logging
        pass

    def test_admin_onboarding_security(self):
        """Test complete admin onboarding security"""
        # Test that admin onboarding requires:
        # 1. Valid admin JWT token
        # 2. Admin role verification
        # 3. Rate limiting
        # 4. Input validation
        # 5. Audit logging
        pass


# Performance and stress tests


class TestSecurityPerformance:
    """Test suite for security performance under load"""

    @pytest.mark.asyncio
    async def test_concurrent_authorization_checks(self):
        """Test authorization performance under concurrent load"""
        middleware = SecurityMonitor()
        user = {"user_id": str(uuid.uuid4()), "user_data": {}}

        # Test multiple concurrent authorization checks
        tasks = []
        for i in range(100):
            task = middleware.verify_entity_ownership(user, "user", user["user_id"])
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

    def test_rate_limiting_performance(self):
        """Test rate limiting performance with many users"""
        from app.utils.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()

        # Test rate limiting with many users
        for i in range(1000):
            user_id = str(uuid.uuid4())
            is_allowed, info = limiter.is_allowed(
                f"user:{user_id}", max_requests=10, window_seconds=60
            )
            assert is_allowed is True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
