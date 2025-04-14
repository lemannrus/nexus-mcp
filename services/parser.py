import logging
from typing import Dict
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_webpage(url: str) -> Dict[str, str]:
    """
    Parse the given webpage and extract title, meta description, and main textual content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MCPBot/1.0; +https://example.com/bot)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch webpage: {e}")
        return {"error": f"Failed to load the page: {e}"}

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else "Title not found"

        description_tag = soup.find("meta", attrs={"name": "description"})
        description = (
            description_tag["content"].strip()
            if description_tag and "content" in description_tag.attrs
            else "Description not found"
        )

        paragraphs = soup.find_all("p")
        content = "\n\n".join(
            p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
        )

        return {"title": title, "description": description, "content": content}
    except Exception as e:
        logger.error(f"Failed to parse webpage: {e}")
        return {"error": f"Failed to parse content: {e}"}
