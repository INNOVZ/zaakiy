import requests
from bs4 import BeautifulSoup


def scrape_url_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator='\n')
        return text.strip()

    except Exception as e:
        print(f"[Error] Failed to fetch URL {url}: {e}")
        return ""
