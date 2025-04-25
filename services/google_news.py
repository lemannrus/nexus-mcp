import requests
from datetime import datetime, timedelta
from typing import List, Tuple

from config import GOOGLE_API_KEY, GOOGLE_CX_ID
from services.logger import logger


def search_news_google(
    topic: str, days_back: int = 3, max_results: int = 10
) -> List[Tuple[str, str]]:
    """
    Searches for news articles using Google Programmable Search API.

    Args:
        topic: The news topic to search for.
        days_back: Number of days back to include in the search.
        max_results: Max number of news results to return.

    Returns:
        A list of tuples (title, url) for each article found.
    """
    try:
        date_from = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"{topic} news after:{date_from}"

        logger.debug(f"Querying Google API: '{query}'")

        results = []
        start = 1
        while len(results) < max_results:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "q": query,
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CX_ID,
                    "start": start,
                    "num": min(10, max_results - len(results)),
                    "safe": "active",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if "items" not in data:
                break

            for item in data["items"]:
                title = item.get("title")
                link = item.get("link")
                if title and link:
                    results.append((title, link))
            start += 10

        logger.info(f"Found {len(results)} news items for topic '{topic}'")
        return results

    except Exception as e:
        logger.error(f"News search failed: {e}")
        return []
