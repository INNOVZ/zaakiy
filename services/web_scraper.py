import httpx  # Replace requests with httpx
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.robotparser import RobotFileParser
from typing import Dict, Set
from collections import deque
from bs4 import BeautifulSoup


async def scrape_url_text(url: str) -> str:
    """Scrape text from a URL using async httpx"""
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                raise ValueError(f"Unsupported content type: {content_type}")

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            if len(text.strip()) < 10:
                raise ValueError("No meaningful text content found")

            return text.strip()

    except httpx.HTTPError as e:
        raise ValueError(f"Failed to fetch URL {url}: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Failed to process URL {url}: {str(e)}") from e


def _get_domain(url: str) -> str:
    """Extract domain from URL"""
    return urlparse(url).netloc.lower()


async def scrape_website_recursive(
    start_url: str,
    max_pages: int = 10,
    max_depth: int = 3,
    same_domain_only: bool = True
) -> Dict[str, str]:
    """Recursively scrape a website using async httpx"""
    scraped_pages = {}
    visited_urls = set()
    url_queue = deque([(start_url, 0)])  # (url, depth)
    start_domain = _get_domain(start_url)

    async with httpx.AsyncClient(
        timeout=30.0,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    ) as client:

        while url_queue and len(scraped_pages) < max_pages:
            current_url, depth = url_queue.popleft()

            # Skip if already visited or max depth reached
            if current_url in visited_urls or depth > max_depth:
                continue

            visited_urls.add(current_url)

            try:
                response = await client.get(current_url)
                response.raise_for_status()

                # Parse content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip()
                          for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)

                if len(clean_text.strip()) > 10:
                    scraped_pages[current_url] = clean_text.strip()

                # Find more links if not at max depth
                if depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(current_url, href)

                        # Remove fragment
                        absolute_url, _ = urldefrag(absolute_url)

                        # Skip if same domain only and different domain
                        if same_domain_only and _get_domain(absolute_url) != start_domain:
                            continue

                        # Skip if already visited or queued
                        if absolute_url not in visited_urls:
                            url_queue.append((absolute_url, depth + 1))

            except Exception as e:
                print(f"[Warning] Failed to scrape {current_url}: {e}")
                continue

    return scraped_pages
