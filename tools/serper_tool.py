"""
tools/serper_tool.py
────────────────────
Serper API wrapper for Amazon product search and Google Trends.
Used by Amazon Research Agent, Trend Agent, and Review Agent.
"""

import os
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL = "https://google.serper.dev/search"
SERPER_SHOPPING_URL = "https://google.serper.dev/shopping"
MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def search_web(query: str, num_results: int = MAX_RESULTS) -> list[dict]:
    """General web search via Serper API."""
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set. Using mock data.")
        return _mock_web_results(query)

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(SERPER_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        organic = data.get("organic", [])
        results = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", idx + 1),
            }
            for idx, item in enumerate(organic[:num_results])
        ]
        logger.info(f"[Serper] Got {len(results)} results for: {query!r}")
        return results

    except Exception as e:
        logger.error(f"[Serper] Search failed: {e}")
        return _mock_web_results(query)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def search_shopping(query: str, num_results: int = MAX_RESULTS) -> list[dict]:
    """Shopping search via Serper API — returns product listings."""
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set. Using mock shopping data.")
        return _mock_shopping_results(query)

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(SERPER_SHOPPING_URL, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        shopping = data.get("shopping", [])
        results = [
            {
                "title": item.get("title", ""),
                "price": item.get("price", ""),
                "source": item.get("source", ""),
                "link": item.get("link", ""),
                "rating": item.get("rating", 0),
                "reviews": item.get("reviews", ""),
                "imageUrl": item.get("imageUrl", ""),
            }
            for item in shopping[:num_results]
        ]
        logger.info(f"[Serper Shopping] Got {len(results)} results for: {query!r}")
        return results

    except Exception as e:
        logger.error(f"[Serper Shopping] Failed: {e}")
        return _mock_shopping_results(query)


def search_amazon_products(category: str) -> list[dict]:
    """Search Amazon products for a given category."""
    query = f"site:amazon.com {category} best seller reviews price"
    return search_web(query)


def search_amazon_reviews(product: str) -> list[dict]:
    """Search Amazon reviews for a product."""
    query = f"amazon {product} customer reviews pros cons complaints 2024"
    return search_web(query)


def search_market_trends(category: str) -> list[dict]:
    """Search market trends for a category."""
    query = f"{category} market trends growth statistics 2024 2025"
    return search_web(query)


def search_competitor_pricing(category: str) -> list[dict]:
    """Search competitor pricing for a category."""
    return search_shopping(f"{category} best price compare")


# ─── Mock Data ────────────────────────────────────────────────────────────────

def _mock_web_results(query: str) -> list[dict]:
    return [
        {
            "title": f"[MOCK] Top {query} Products 2024 — Best Sellers",
            "link": "https://amazon.com/best-sellers",
            "snippet": (
                f"Top {query} products include premium options from leading brands. "
                "Average prices range from $20-$150. Customer ratings average 4.2/5. "
                "Market growing at 15% YoY with high demand in Q4."
            ),
            "position": 1,
        },
        {
            "title": f"[MOCK] {query} Market Analysis 2024",
            "link": "https://marketresearch.example.com",
            "snippet": (
                f"The {query} market is valued at $5.2B globally. "
                "Key players dominate 60% market share. "
                "Emerging trends include eco-friendly materials and smart features. "
                "Customer complaints focus on durability and battery life."
            ),
            "position": 2,
        },
        {
            "title": f"[MOCK] Best {query} Reviews & Buying Guide",
            "link": "https://reviews.example.com",
            "snippet": (
                "Customers love: Good value, fast shipping, quality build. "
                "Common complaints: Short battery life, poor customer service. "
                "Price sweet spot: $30-80 for best value. "
                "Top brands: Sony, Anker, JBL, Bose."
            ),
            "position": 3,
        },
    ]


def _mock_shopping_results(query: str) -> list[dict]:
    return [
        {
            "title": f"[MOCK] Premium {query} Pro",
            "price": "$49.99",
            "source": "Amazon",
            "link": "https://amazon.com",
            "rating": 4.5,
            "reviews": "2,847 reviews",
        },
        {
            "title": f"[MOCK] Budget {query} Basic",
            "price": "$19.99",
            "source": "Amazon",
            "link": "https://amazon.com",
            "rating": 4.1,
            "reviews": "5,231 reviews",
        },
        {
            "title": f"[MOCK] Professional {query} Elite",
            "price": "$89.99",
            "source": "Amazon",
            "link": "https://amazon.com",
            "rating": 4.7,
            "reviews": "1,204 reviews",
        },
    ]
