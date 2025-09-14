"services/web_scraper.py\n\n"

from urllib.parse import urlparse, urljoin, urldefrag
from urllib.robotparser import RobotFileParser
from typing import Dict, Set
import re
import time
from collections import deque
import requests
from bs4 import BeautifulSoup


def scrape_url_text(url: str) -> str:
    """
    Extract text content from a single URL.
    Used for single-page scraping.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator='\n')
        return text.strip()

    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to fetch URL {url}: {e}")
        return ""


def _get_domain(url: str) -> str:
    """Extract domain from URL"""
    return urlparse(url).netloc.lower()


def _normalize_link(base: str, link: str) -> str:
    """Convert relative links to absolute and remove fragments"""
    # Resolve relative links and remove fragment
    joined = urljoin(base, link)
    normalized, _ = urldefrag(joined)
    return normalized


def _is_valid_scheme(url: str) -> bool:
    """Check if URL has valid HTTP/HTTPS scheme"""
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


def _clean_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace"""
    # Collapse whitespace and remove multiple newlines
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()


def _can_fetch_robots(start_url: str, user_agent: str, target_url: str) -> bool:
    """Check if robots.txt allows fetching the target URL"""
    try:
        parsed = urlparse(start_url)
        root = f"{parsed.scheme}://{parsed.netloc}"

        rp = RobotFileParser()
        rp.set_url(urljoin(root, "/robots.txt"))
        rp.read()
        return rp.can_fetch(user_agent, target_url)
    except (requests.RequestException, OSError, ValueError):
        # If robots.txt can't be fetched or parsed, allow by default
        return True


def _extract_links(soup: BeautifulSoup, current_url: str, target_domain: str) -> Set[str]:
    """Extract valid internal links from a page"""
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # Skip non-web links
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        # Convert to absolute URL
        normalized = _normalize_link(current_url, href)

        # Validate scheme
        if not _is_valid_scheme(normalized):
            continue

        # Only internal links (same domain)
        if _get_domain(normalized) == target_domain:
            links.add(normalized)

    return links


def crawl_site(
    start_url: str,
    max_pages: int = 50,
    max_depth: int = 2,
    delay: float = 1.0,
    user_agent: str = "ZaaKyBot/1.0 (+https://zaaky.ai)"
) -> Dict[str, str]:
    """
    Crawl a website starting from start_url and return {url: text}.
    Only crawls internal links (same domain).

    Args:
        start_url: The homepage URL to start crawling from
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth to crawl (0 = start page only)
        delay: Delay between requests in seconds
        user_agent: User agent string to use

    Returns:
        Dict mapping URLs to their extracted text content
    """
    start_domain = _get_domain(start_url)
    visited: Set[str] = set()
    results: Dict[str, str] = {}

    # BFS queue: (url, depth)
    queue = deque([(start_url, 0)])

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    print(
        f"[Info] Starting crawl of {start_url} (max_pages={max_pages}, max_depth={max_depth})")

    while queue and len(results) < max_pages:
        url, depth = queue.popleft()

        # Skip if already visited
        if url in visited:
            continue

        # Skip if exceeded max depth
        if depth > max_depth:
            continue

        # Check robots.txt
        if not _can_fetch_robots(start_url, user_agent, url):
            print(f"[Info] Robots.txt disallows: {url}")
            visited.add(url)
            continue

        try:
            print(f"[Info] Crawling (depth {depth}): {url}")

            # Fetch the page
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract page title for better context
            title = soup.find("title")
            title_text = title.get_text().strip() if title else ""

            # Remove unwanted elements
            for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Extract text content
            text = soup.get_text(separator="\n")
            text = _clean_text(text)

            # Add title context if available
            if title_text and text:
                text = f"Page Title: {title_text}\n\n{text}"

            # Store result if we got meaningful content
            if text and len(text.strip()) > 50:  # Only store if substantial content
                results[url] = text
                print(f"[Success] Extracted {len(text)} characters from {url}")
            else:
                print(f"[Warning] No substantial content found at {url}")

            visited.add(url)

            # Find and queue new links if we haven't reached limits
            if len(results) < max_pages and depth < max_depth:
                new_links = _extract_links(soup, url, start_domain)

                for link in new_links:
                    if link not in visited:
                        queue.append((link, depth + 1))

                print(
                    f"[Info] Found {len(new_links)} internal links at depth {depth}")

        except requests.RequestException as e:
            print(f"[Error] Failed to fetch {url}: {e}")
            visited.add(url)  # Mark as visited to avoid retrying

        except (ValueError, AttributeError, TypeError) as e:
            print(f"[Error] Unexpected error processing {url}: {e}")
            visited.add(url)

        finally:
            # Polite crawling delay
            if delay > 0:
                time.sleep(delay)

    print(f"[Complete] Crawled {len(results)} pages from {start_url}")
    return results


def crawl_site_simple(start_url: str, max_pages: int = 10) -> Dict[str, str]:
    """
    Simplified website crawler with sensible defaults.
    Good for quick crawling of small to medium sites.
    """
    return crawl_site(
        start_url=start_url,
        max_pages=max_pages,
        max_depth=1,  # Only go 1 level deep
        delay=0.5,    # Faster crawling
        user_agent="ZaaKyBot/1.0 (Website Crawler)"
    )


def get_site_map(start_url: str, max_pages: int = 20) -> Dict[str, Dict]:
    """
    Get a site map with URLs and metadata (without full text extraction).
    Useful for understanding site structure before full crawling.
    """
    start_domain = _get_domain(start_url)
    visited: Set[str] = set()
    results: Dict[str, Dict] = {}

    queue = deque([(start_url, 0)])
    headers = {"User-Agent": "ZaaKyBot/1.0 (Site Mapper)"}

    while queue and len(results) < max_pages:
        url, depth = queue.popleft()

        if url in visited or depth > 2:
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract metadata
            title = soup.find("title")
            title_text = title.get_text().strip() if title else "No Title"

            description = soup.find("meta", attrs={"name": "description"})
            desc_text = description.get(
                "content", "").strip() if description else ""

            results[url] = {
                "title": title_text,
                "description": desc_text,
                "depth": depth,
                "status": response.status_code
            }

            visited.add(url)

            # Find links for next level
            if depth < 2:
                new_links = _extract_links(soup, url, start_domain)
                for link in new_links:
                    if link not in visited:
                        queue.append((link, depth + 1))

        except requests.RequestException as e:
            print(f"[Error] Failed to map {url}: {e}")
            visited.add(url)

        time.sleep(0.3)  # Light delay for mapping

    return results
