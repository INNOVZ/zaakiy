"""
Enhanced secure web scraper ZaaKy's configuration and logging systems.
"""

import asyncio
import ipaddress
import random
import re
import socket
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from ...config.settings import get_performance_config, get_web_scraping_config
from ...utils.error_handlers import ErrorHandler, handle_errors
from ...utils.logging_config import LogContext, PerformanceLogger, get_logger
from .content_extractors import ContactExtractor
from .url_utils import (
    URLSanitizer,
    create_safe_error_message,
    create_safe_fetch_message,
    create_safe_success_message,
    log_domain_safely,
)

logger = get_logger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations"""

    timeout: int = 30
    max_content_size: int = 50 * 1024 * 1024  # 50MB
    min_delay: float = 1.0
    max_delay: float = 3.0
    max_retries: int = 3
    user_agents: List[str] = None
    allowed_content_types: Set[str] = None

    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ]

        if self.allowed_content_types is None:
            self.allowed_content_types = {
                "text/html",
                "text/plain",
                "application/xhtml+xml",
            }


class URLSecurityValidator:
    """Validates URLs for security vulnerabilities (SSRF protection)"""

    # Private IP ranges that should be blocked
    PRIVATE_IP_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),  # Localhost
        ipaddress.ip_network("169.254.0.0/16"),  # AWS metadata
        ipaddress.ip_network("::1/128"),  # IPv6 localhost
        ipaddress.ip_network("fc00::/7"),  # IPv6 private
        ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ]

    # Dangerous protocols
    BLOCKED_SCHEMES = {
        "file",
        "ftp",
        "gopher",
        "ldap",
        "dict",
        "sftp",
        "tftp",
        "telnet",
        "ssh",
        "jar",
        "netdoc",
    }

    # Dangerous ports
    BLOCKED_PORTS = {
        22,
        23,
        25,
        53,
        135,
        139,
        445,
        993,
        995,
        1433,
        1521,
        3306,
        5432,
        6379,
        27017,
        8080,
        8443,
        9200,
        11211,
    }

    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, str]:
        """
        Validate URL for security issues
        Returns: (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme.lower() in cls.BLOCKED_SCHEMES:
                return False, f"Blocked protocol: {parsed.scheme}"

            if parsed.scheme.lower() not in ["http", "https"]:
                return False, f"Only HTTP/HTTPS protocols allowed, got: {parsed.scheme}"

            # Check hostname
            hostname = parsed.hostname
            if not hostname:
                return False, "Invalid hostname"

            # Block localhost variations
            localhost_patterns = [
                "localhost",
                "0.0.0.0",
                "0x0.0x0.0x0.0x0",
                "0000",
                "0x0",
                "127.1",
                "127.0.1",
            ]
            if hostname.lower() in localhost_patterns:
                return False, f"Localhost access blocked: {hostname}"

            # Block IP addresses in private ranges
            try:
                ip = ipaddress.ip_address(hostname)
                for private_range in cls.PRIVATE_IP_RANGES:
                    if ip in private_range:
                        return False, f"Private IP access blocked: {ip}"
            except ValueError:
                # Not an IP address, check if it resolves to private IP
                try:
                    resolved_ips = socket.getaddrinfo(hostname, parsed.port)
                    for family, type, proto, canonname, sockaddr in resolved_ips:
                        ip_str = sockaddr[0]
                        try:
                            ip = ipaddress.ip_address(ip_str)
                            for private_range in cls.PRIVATE_IP_RANGES:
                                if ip in private_range:
                                    return (
                                        False,
                                        f"Hostname resolves to private IP: {hostname} -> {ip}",
                                    )
                        except ValueError:
                            continue
                except (socket.gaierror, socket.error):
                    return False, f"Cannot resolve hostname: {hostname}"

            # Check port
            port = parsed.port
            if port and port in cls.BLOCKED_PORTS:
                return False, f"Blocked port: {port}"

            # Additional security checks
            if len(url) > 2048:
                return False, "URL too long (>2048 characters)"

            # Check for URL redirection attempts
            suspicious_patterns = [
                r"@",  # Username in URL can be used for redirection
                r"%(?:2[fF]|5[cC])",  # Encoded slashes/backslashes
                r"(?:\/\.\.){2,}",  # Directory traversal
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, url):
                    return False, f"Suspicious URL pattern detected"

            return True, ""

        except Exception as e:
            return False, f"URL validation error: {str(e)}"


class RobotsTxtChecker:
    """Check robots.txt compliance"""

    def __init__(self):
        self.cache = {}  # Cache robots.txt content
        self.cache_ttl = 3600  # 1 hour cache

    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            # Check cache
            cache_key = robots_url
            now = time.time()
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if now - timestamp < self.cache_ttl:
                    if cached_data is None:
                        return True  # No robots.txt or error
                    return cached_data.can_fetch(user_agent, url)

            # Fetch robots.txt
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(robots_url)
                    if response.status_code == 200:
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.read()
                        # Parse the content
                        lines = response.text.splitlines()
                        for line in lines:
                            rp.read()

                        self.cache[cache_key] = (rp, now)
                        return rp.can_fetch(user_agent, url)
                    else:
                        # No robots.txt found, allow by default
                        self.cache[cache_key] = (None, now)
                        return True
            except Exception:
                # Error fetching robots.txt, allow by default but cache the result
                self.cache[cache_key] = (None, now)
                return True

        except Exception as e:
            logger.warning(
                f"Error checking robots.txt for {log_domain_safely(url)}: {type(e).__name__}"
            )
            return True  # Allow by default on error


class SecureWebScraper:
    """Secure web scraper with comprehensive protection and monitoring"""

    def __init__(self, config: Optional[ScrapingConfig] = None):
        # Use configuration from settings if not provided
        if config is None:
            app_config = get_web_scraping_config()
            config = ScrapingConfig(
                timeout=app_config.timeout,
                max_content_size=app_config.max_content_size,
                min_delay=app_config.min_delay_between_requests,
                max_delay=app_config.max_delay_between_requests,
                max_retries=app_config.max_retries,
            )

        self.config = config
        self.app_config = get_web_scraping_config()
        self.robots_checker = RobotsTxtChecker()
        self.request_times = {}  # Track request timing for rate limiting

    def _get_random_user_agent(self) -> str:
        """Get a random user agent to avoid detection"""
        # SECURITY NOTE: random.choice() is used here for non-cryptographic purposes
        # (selecting user agents for web scraping rotation, not security-critical)
        return random.choice(self.config.user_agents)

    async def _respect_rate_limit(self, domain: str):
        """Implement rate limiting per domain"""
        now = time.time()

        if domain in self.request_times:
            last_request = self.request_times[domain]
            time_since_last = now - last_request

            if time_since_last < self.config.min_delay:
                # SECURITY NOTE: random.uniform() for rate limiting delays (non-cryptographic)
                delay = random.uniform(self.config.min_delay, self.config.max_delay)
                logger.debug(f"Rate limiting: waiting {delay:.2f}s for {domain}")
                await asyncio.sleep(delay)

        self.request_times[domain] = time.time()

    async def _fetch_with_retries(
        self, client: httpx.AsyncClient, url: str
    ) -> httpx.Response:
        """Fetch URL with retry logic and exponential backoff"""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [404, 403, 401]:
                    # Don't retry on client errors
                    raise
                last_exception = e
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
                last_exception = e

            if attempt < self.config.max_retries - 1:
                # Exponential backoff
                # SECURITY NOTE: random.uniform() for exponential backoff jitter (non-cryptographic)
                delay = (2**attempt) + random.uniform(0, 1)
                logger.debug(
                    f"Retry {attempt + 1}/{self.config.max_retries} for {log_domain_safely(url)} after {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        raise last_exception

    def _extract_and_clean_text(self, html_content: str) -> str:
        """Extract and clean text from HTML, preserving contact information"""
        soup = BeautifulSoup(html_content, "html.parser")

        # FIRST: Extract contact information from headers/footers before removing them
        contact_info = self._extract_contact_information(soup)

        # Remove unwanted elements (but we already saved contact info)
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Extract text
        text = soup.get_text()

        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)

        # Prepend contact information to ensure it's included in the indexed content
        if contact_info:
            cleaned_text = contact_info + "\n\n" + cleaned_text

        return cleaned_text.strip()

    def _extract_contact_information(self, soup: BeautifulSoup) -> str:
        """
        Extract contact information (phone, email, address) from the entire page.

        Uses the shared ContactExtractor utility to avoid code duplication.
        """
        return ContactExtractor.extract_from_soup(soup, include_emoji=False)

    @handle_errors(context="secure_web_scraper.scrape_url")
    async def scrape_url_text(self, url: str, user_agent: Optional[str] = None) -> str:
        """
        Securely scrape text from a single URL with comprehensive protection
        """
        with LogContext(extra_context={"domain": log_domain_safely(url)}) as ctx:
            logger.info("Starting secure scrape of %s", create_safe_fetch_message(url))

            # Security validation
            if self.app_config.enable_ssrf_protection:
                is_valid, error_msg = URLSecurityValidator.validate_url(url)
                if not is_valid:
                    raise ValueError("URL security validation failed: %s", error_msg)

            # Domain allow/block list validation
            domain = urlparse(url).netloc.lower()
            if (
                self.app_config.blocked_domains
                and domain in self.app_config.blocked_domains
            ):
                raise ValueError("Domain is blocked: %s", domain)

            if (
                self.app_config.allowed_domains
                and domain not in self.app_config.allowed_domains
            ):
                raise ValueError("Domain not in allowed list: %s", domain)

            # Check robots.txt compliance
            user_agent_str = user_agent or self._get_random_user_agent()
            if (
                self.app_config.respect_robots_txt
                and not await self.robots_checker.can_fetch(url, user_agent_str)
            ):
                raise ValueError("URL blocked by robots.txt: %s", url)

            # Rate limiting
            domain = urlparse(url).netloc.lower()
            await self._respect_rate_limit(domain)

            with PerformanceLogger(f"scrape_url_{domain}"):
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(self.config.timeout),
                        headers={"User-Agent": user_agent_str},
                        limits=httpx.Limits(
                            max_connections=10, max_keepalive_connections=5
                        ),
                    ) as client:
                        response = await self._fetch_with_retries(client, url)

                        # Check content size
                        content_length = response.headers.get("content-length")
                        if (
                            content_length
                            and int(content_length) > self.config.max_content_size
                        ):
                            raise ValueError(
                                f"Content too large: {content_length} bytes"
                            )

                        # Check content type
                        content_type = response.headers.get("content-type", "").lower()
                        if not any(
                            ct in content_type
                            for ct in self.config.allowed_content_types
                        ):
                            raise ValueError(
                                f"Unsupported content type: {content_type}"
                            )

                        # Check actual response size
                        if len(response.content) > self.config.max_content_size:
                            raise ValueError(
                                f"Response too large: {len(response.content)} bytes"
                            )

                        # Extract text
                        text = self._extract_and_clean_text(response.text)

                        if len(text.strip()) < 10:
                            raise ValueError("No meaningful text content found")

                        logger.info(
                            "%s",
                            create_safe_success_message(
                                url, f"{len(text)} characters scraped"
                            ),
                        )
                        return text

                except httpx.HTTPStatusError as e:
                    logger.error(
                        "%s",
                        create_safe_error_message(
                            url, f"HTTP {e.response.status_code}"
                        ),
                    )
                    raise ValueError(
                        f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
                    )
                except httpx.TimeoutException:
                    logger.error("%s", create_safe_error_message(url, "timeout"))
                    raise ValueError(f"Request timeout after {self.config.timeout}s")
                except Exception as e:
                    logger.error(
                        "%s",
                        create_safe_error_message(
                            url, f"unexpected error: {type(e).__name__}"
                        ),
                    )
                    raise ValueError(f"Scraping failed: {str(e)}")

    @handle_errors(context="secure_web_scraper.scrape_website_recursive")
    async def scrape_website_recursive(
        self,
        start_url: str,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        same_domain_only: bool = True,
        concurrent_requests: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Recursively scrape a website with security protection and concurrency
        """
        # Use configuration defaults if not provided
        max_pages = max_pages or self.app_config.max_pages_per_site
        max_depth = max_depth or self.app_config.max_depth
        concurrent_requests = concurrent_requests or self.app_config.concurrent_requests

        with LogContext(
            extra_context={
                "domain": log_domain_safely(start_url),
                "max_pages": max_pages,
            }
        ):
            logger.info(
                "Starting recursive scrape: %s (max_pages=%s, max_depth=%s)",
                create_safe_fetch_message(start_url),
                max_pages,
                max_depth,
            )

            # Validate starting URL
            if self.app_config.enable_ssrf_protection:
                is_valid, error_msg = URLSecurityValidator.validate_url(start_url)
                if not is_valid:
                    raise ValueError(
                        f"Start URL security validation failed: {error_msg}"
                    )

            scraped_pages = {}
            visited_urls = set()
            url_queue = deque([(start_url, 0)])
            start_domain = urlparse(start_url).netloc.lower()

            # Semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_requests)

            async def scrape_single_url(
                url: str, depth: int
            ) -> Optional[Tuple[str, str]]:
                """Scrape a single URL with semaphore protection"""
                async with semaphore:
                    try:
                        text = await self.scrape_url_text(url)
                        return (url, text)
                    except Exception as e:
                        logger.warning(
                            "%s",
                            create_safe_error_message(
                                url, f"scraping failed: {type(e).__name__}"
                            ),
                        )
                        return None

        while url_queue and len(scraped_pages) < max_pages:
            current_url, depth = url_queue.popleft()

            # Skip if already visited or max depth reached
            if current_url in visited_urls or depth > max_depth:
                continue

            visited_urls.add(current_url)

            # Validate URL before scraping
            if self.app_config.enable_ssrf_protection:
                is_valid, error_msg = URLSecurityValidator.validate_url(current_url)
                if not is_valid:
                    logger.warning(
                        "Skipping invalid URL %s: %s",
                        log_domain_safely(current_url),
                        error_msg,
                    )
                    continue

            try:
                # Scrape current page
                result = await scrape_single_url(current_url, depth)
                if result:
                    url, text = result
                    scraped_pages[url] = text

                    # If not at max depth, find more links
                    if depth < max_depth:
                        # Parse the page again to find links (we already have the text)
                        try:
                            async with httpx.AsyncClient(
                                timeout=httpx.Timeout(10),
                                headers={"User-Agent": self._get_random_user_agent()},
                            ) as client:
                                response = await client.get(current_url)
                                soup = BeautifulSoup(response.text, "html.parser")

                                for link in soup.find_all("a", href=True):
                                    href = link["href"]
                                    absolute_url = urljoin(current_url, href)
                                    absolute_url, _ = urldefrag(
                                        absolute_url
                                    )  # Remove fragment

                                    # Security validation for new URL
                                    (
                                        is_link_valid,
                                        _,
                                    ) = URLSecurityValidator.validate_url(absolute_url)
                                    if not is_link_valid:
                                        continue

                                    # Domain restriction
                                    if (
                                        same_domain_only
                                        and urlparse(absolute_url).netloc.lower()
                                        != start_domain
                                    ):
                                        continue

                                    # Add to queue if not visited
                                    if (
                                        absolute_url not in visited_urls
                                        and absolute_url
                                        not in [u for u, d in url_queue]
                                    ):
                                        url_queue.append((absolute_url, depth + 1))

                        except Exception as e:
                            logger.warning(
                                "Failed to extract links from %s: %s",
                                log_domain_safely(current_url),
                                type(e).__name__,
                            )

            except Exception as e:
                logger.error(
                    "%s",
                    create_safe_error_message(
                        current_url, f"processing error: {type(e).__name__}"
                    ),
                )
                continue

        logger.info("Recursive scrape completed: %s pages scraped", len(scraped_pages))
        return scraped_pages


# Global scraper instance with default configuration
_default_scraper = None


def get_default_scraper() -> SecureWebScraper:
    """Get the default secure scraper instance"""
    global _default_scraper
    if _default_scraper is None:
        _default_scraper = SecureWebScraper()
    return _default_scraper


# Backwards compatibility functions
async def scrape_url_text(url: str) -> str:
    """
    Backwards compatible function for existing code
    Now uses secure scraper with all protections
    """
    scraper = get_default_scraper()
    return await scraper.scrape_url_text(url)


async def scrape_website_recursive(
    start_url: str,
    max_pages: int = 10,
    max_depth: int = 3,
    same_domain_only: bool = True,
) -> Dict[str, str]:
    """
    Backwards compatible function for existing code
    Now uses secure scraper with all protections
    """
    scraper = get_default_scraper()
    return await scraper.scrape_website_recursive(
        start_url, max_pages, max_depth, same_domain_only
    )


def _get_domain(url: str) -> str:
    """Extract domain from URL - kept for backwards compatibility"""
    return urlparse(url).netloc.lower()
