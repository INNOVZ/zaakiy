"""
Comprehensive test suite for the secure web scraper
Tests security features, error handling, and integration
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.scraping.web_scraper import (ScrapingConfig, SecureWebScraper,
                                               URLSecurityValidator, scrape_url_text)


class TestURLSecurityValidator:
    """Test URL security validation"""

    def test_valid_http_url(self):
        """Test that valid HTTP URLs pass validation"""
        valid, msg = URLSecurityValidator.validate_url("https://example.com")
        assert valid is True
        assert msg == ""

    def test_blocked_protocols(self):
        """Test that dangerous protocols are blocked"""
        dangerous_urls = [
            "file:///etc/passwd",
            "ftp://example.com/file",
            "gopher://example.com",
            "ldap://example.com",
            "ssh://example.com",
        ]

        for url in dangerous_urls:
            valid, msg = URLSecurityValidator.validate_url(url)
            assert valid is False
            assert "protocol" in msg.lower()

    def test_localhost_blocked(self):
        """Test that localhost and private IPs are blocked"""
        localhost_urls = [
            "http://localhost/admin",
            "http://127.0.0.1/",
            "http://0.0.0.0/",
            "http://127.1/",
            "https://127.0.1/",
        ]

        for url in localhost_urls:
            valid, msg = URLSecurityValidator.validate_url(url)
            assert valid is False
            assert any(
                term in msg.lower() for term in ["localhost", "private", "blocked"]
            )

    def test_private_ip_ranges(self):
        """Test that private IP ranges are blocked"""
        private_urls = [
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        ]

        for url in private_urls:
            valid, msg = URLSecurityValidator.validate_url(url)
            assert valid is False
            assert "private" in msg.lower()

    def test_dangerous_ports_blocked(self):
        """Test that dangerous ports are blocked"""
        dangerous_ports = [22, 23, 25, 1433, 3306, 5432, 6379]

        for port in dangerous_ports:
            url = f"http://example.com:{port}/"
            valid, msg = URLSecurityValidator.validate_url(url)
            assert valid is False
            assert "port" in msg.lower()

    def test_url_too_long(self):
        """Test that extremely long URLs are rejected"""
        long_url = "https://example.com/" + "a" * 3000
        valid, msg = URLSecurityValidator.validate_url(long_url)
        assert valid is False
        assert "too long" in msg.lower()

    def test_suspicious_patterns(self):
        """Test detection of suspicious URL patterns"""
        suspicious_urls = [
            "http://user@evil.com/redirect",  # Username in URL
            "http://example.com/../../../etc/passwd",  # Directory traversal
        ]

        for url in suspicious_urls:
            valid, msg = URLSecurityValidator.validate_url(url)
            assert valid is False
            assert "suspicious" in msg.lower()


class TestSecureWebScraper:
    """Test the secure web scraper functionality"""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance with test config"""
        config = ScrapingConfig(
            timeout=10,
            max_content_size=1024 * 1024,  # 1MB
            min_delay=0.1,
            max_delay=0.2,
            max_retries=1,
        )
        return SecureWebScraper(config)

    @pytest.mark.asyncio
    async def test_ssrf_protection(self, scraper):
        """Test that SSRF attacks are prevented"""
        malicious_urls = [
            "http://localhost:8080/admin",
            "http://127.0.0.1/secrets",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
        ]

        for url in malicious_urls:
            with pytest.raises(ValueError) as exc_info:
                await scraper.scrape_url_text(url)
            assert "security validation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_domain_blocking(self, scraper):
        """Test domain allow/block list functionality"""
        # Mock app config with blocked domains
        with patch.object(scraper, "app_config") as mock_config:
            mock_config.blocked_domains = ["blocked-site.com"]
            mock_config.allowed_domains = []
            mock_config.enable_ssrf_protection = True
            mock_config.respect_robots_txt = False

            with pytest.raises(ValueError) as exc_info:
                await scraper.scrape_url_text("https://blocked-site.com/page")
            assert "Domain is blocked" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_content_size_limit(self, scraper):
        """Test that oversized content is rejected"""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock a response with large content
            mock_response = MagicMock()
            mock_response.headers = {
                "content-type": "text/html",
                "content-length": str(100 * 1024 * 1024),  # 100MB
            }

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            with patch.object(
                scraper, "_fetch_with_retries", return_value=mock_response
            ):
                with pytest.raises(ValueError) as exc_info:
                    await scraper.scrape_url_text("https://example.com")
                assert "too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_content_type(self, scraper):
        """Test that non-HTML content types are rejected"""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "application/pdf"}
            mock_response.content = b"PDF content"

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            with patch.object(
                scraper, "_fetch_with_retries", return_value=mock_response
            ):
                with pytest.raises(ValueError) as exc_info:
                    await scraper.scrape_url_text("https://example.com")
                assert "Unsupported content type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, scraper):
        """Test that rate limiting is enforced"""
        domain = "example.com"

        # Make first request
        with patch("httpx.AsyncClient"), patch.object(
            scraper, "_fetch_with_retries"
        ) as mock_fetch, patch("asyncio.sleep") as mock_sleep:
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "text/html"}
            mock_response.text = "<html><body>Test content</body></html>"
            mock_fetch.return_value = mock_response

            # Mock app config
            with patch.object(scraper, "app_config") as mock_config:
                mock_config.enable_ssrf_protection = False
                mock_config.respect_robots_txt = False
                mock_config.blocked_domains = []
                mock_config.allowed_domains = []

            # First request
            await scraper.scrape_url_text(f"https://{domain}/page1")

            # Second request should trigger rate limiting
            await scraper.scrape_url_text(f"https://{domain}/page2")

            # Verify sleep was called for rate limiting
            assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_robots_txt_compliance(self, scraper):
        """Test robots.txt compliance"""
        with patch.object(scraper.robots_checker, "can_fetch", return_value=False):
            with patch.object(scraper, "app_config") as mock_config:
                mock_config.respect_robots_txt = True
                mock_config.enable_ssrf_protection = False

                with pytest.raises(ValueError) as exc_info:
                    await scraper.scrape_url_text("https://example.com/blocked")
                assert "blocked by robots.txt" in str(exc_info.value)


class TestBackwardsCompatibility:
    """Test that existing code still works with the new scraper"""

    @pytest.mark.asyncio
    async def test_scrape_url_text_function(self):
        """Test that the standalone function still works"""
        with patch("services.web_scraper.get_default_scraper") as mock_get_scraper:
            mock_scraper = MagicMock()
            mock_scraper.scrape_url_text.return_value = "Scraped content"
            mock_get_scraper.return_value = mock_scraper

            result = await scrape_url_text("https://example.com")
            assert result == "Scraped content"
            mock_scraper.scrape_url_text.assert_called_once_with("https://example.com")


class TestConfigurationIntegration:
    """Test integration with the configuration system"""

    def test_scraper_uses_app_config(self):
        """Test that scraper properly loads configuration"""
        with patch("services.web_scraper.get_web_scraping_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.timeout = 60
            mock_config.max_content_size = 10 * 1024 * 1024
            mock_config.min_delay_between_requests = 2.0
            mock_config.max_delay_between_requests = 5.0
            mock_config.max_retries = 5
            mock_get_config.return_value = mock_config

            scraper = SecureWebScraper()

            # Verify configuration was loaded
            assert scraper.config.timeout == 60
            assert scraper.config.max_content_size == 10 * 1024 * 1024
            assert scraper.config.min_delay == 2.0
            assert scraper.config.max_delay == 5.0
            assert scraper.config.max_retries == 5


if __name__ == "__main__":
    """Run security tests manually"""

    async def run_security_tests():
        """Run critical security tests"""
        print("🔒 Running Web Scraper Security Tests...")

        # Test SSRF protection
        print("\n1. Testing SSRF Protection...")
        validator = URLSecurityValidator()

        ssrf_urls = [
            "http://localhost/admin",
            "http://127.0.0.1/secrets",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/internal",
            "file:///etc/passwd",
            "ftp://evil.com/file",
        ]

        for url in ssrf_urls:
            valid, msg = validator.validate_url(url)
            status = "✅ BLOCKED" if not valid else "❌ ALLOWED"
            print(f"  {status}: {url} - {msg}")

        print("\n2. Testing Valid URLs...")
        valid_urls = [
            "https://example.com",
            "http://google.com/search",
            "https://github.com/user/repo",
        ]

        for url in valid_urls:
            valid, msg = validator.validate_url(url)
            status = "✅ ALLOWED" if valid else "❌ BLOCKED"
            print(f"  {status}: {url}")

        print("\n3. Testing Scraper with Mock...")
        scraper = SecureWebScraper()

        try:
            # This should fail due to SSRF protection
            await scraper.scrape_url_text("http://localhost:8080/admin")
            print("  ❌ SSRF test failed - localhost was allowed!")
        except ValueError as e:
            print(f"  ✅ SSRF protection working: {e}")

        print("\n🎉 Security tests completed!")

    # Run the tests
    asyncio.run(run_security_tests())
