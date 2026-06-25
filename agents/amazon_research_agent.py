"""
agents/amazon_research_agent.py
────────────────────────────────
Amazon Research Agent — searches for top products in a category.

Workflow:
1. Search Amazon for top products using Serper
2. Search shopping results for product listings
3. Use Groq to analyze and structure findings
4. Return AmazonResearchData model
"""

import os
import json
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from models.state import ProductResearchState, AmazonResearchData
from tools.serper_tool import search_amazon_products, search_shopping

load_dotenv()


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.2,
    )


def amazon_research_node(state: dict) -> dict:
    """LangGraph node: Amazon Research Agent."""
    s = ProductResearchState(**state)
    logger.info(f"[Amazon] Researching: {s.product_category!r}")

    try:
        data = _run_amazon_research(s)
        logger.info(f"[Amazon] Found {len(data.top_products)} products.")
        return {"amazon_research": data.model_dump()}
    except Exception as e:
        logger.error(f"[Amazon] Error: {e}")
        return {
            "amazon_research": AmazonResearchData(error=str(e)).model_dump(),
            "errors": state.get("errors", []) + [f"Amazon research error: {e}"],
        }


def _run_amazon_research(s: ProductResearchState) -> AmazonResearchData:
    """Core Amazon research logic."""

    # Search for products
    web_results = search_amazon_products(s.product_category)
    shopping_results = search_shopping(f"{s.product_category} amazon best seller")

    # Format results for Groq
    def format_results(results: list[dict]) -> str:
        return "\n".join(
            f"[{i+1}] {r.get('title', '')} | "
            f"Price: {r.get('price', 'N/A')} | "
            f"Rating: {r.get('rating', 'N/A')} | "
            f"{r.get('snippet', r.get('reviews', ''))}"
            for i, r in enumerate(results[:8])
        )

    context = f"""
Product Category: {s.product_category}
Target Market: {s.target_market}
Budget Range: {s.budget_range}

=== WEB SEARCH RESULTS ===
{format_results(web_results)}

=== SHOPPING RESULTS ===
{format_results(shopping_results)}
"""

    llm = _get_llm()

    system_prompt = """You are an Amazon Product Research Specialist AI.
Analyze search results and extract structured product research data.

Return ONLY valid JSON:
{
  "category": "product category name",
  "top_products": [
    {
      "rank": 1,
      "title": "Product name",
      "brand": "Brand name",
      "price": "$XX.XX",
      "rating": 4.5,
      "reviews_count": "1,234 reviews",
      "key_features": ["Feature 1", "Feature 2", "Feature 3"],
      "why_popular": "Reason this product is popular"
    }
  ],
  "top_brands": ["Brand1", "Brand2", "Brand3", "Brand4", "Brand5"],
  "price_range": {
    "budget": "$10-30",
    "mid_range": "$30-70",
    "premium": "$70-200+"
  },
  "key_findings": [
    "Finding 1 with specific data",
    "Finding 2",
    "Finding 3",
    "Finding 4",
    "Finding 5"
  ],
  "total_found": 10
}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                raw = part
                break

    parsed = json.loads(raw)

    return AmazonResearchData(
        category=parsed.get("category", s.product_category),
        top_products=parsed.get("top_products", []),
        total_found=parsed.get("total_found", 0),
        top_brands=parsed.get("top_brands", []),
        price_range=parsed.get("price_range", {}),
        key_findings=parsed.get("key_findings", []),
    )
