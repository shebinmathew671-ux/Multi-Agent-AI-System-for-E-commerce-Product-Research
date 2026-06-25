"""
agents/price_analysis_agent.py
────────────────────────────────
Price Analysis Agent — analyzes pricing across the market.

Workflow:
1. Read top products from amazon_research state
2. Search for competitor pricing
3. Use Groq to analyze pricing strategy
4. Return PriceAnalysisData model
"""

import os
import json
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from models.state import ProductResearchState, PriceAnalysisData, AmazonResearchData
from tools.serper_tool import search_competitor_pricing

load_dotenv()


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.2,
    )


def price_analysis_node(state: dict) -> dict:
    """LangGraph node: Price Analysis Agent."""
    s = ProductResearchState(**state)
    logger.info(f"[Price] Analyzing pricing for: {s.product_category!r}")

    try:
        data = _run_price_analysis(s)
        logger.info("[Price] Analysis completed.")
        return {"price_analysis": data.model_dump()}
    except Exception as e:
        logger.error(f"[Price] Error: {e}")
        return {
            "price_analysis": PriceAnalysisData(error=str(e)).model_dump(),
            "errors": state.get("errors", []) + [f"Price analysis error: {e}"],
        }


def _run_price_analysis(s: ProductResearchState) -> PriceAnalysisData:
    """Core price analysis logic."""

    # Get products from previous agent
    amazon_data = None
    if s.amazon_research:
        ar = s.amazon_research
        if isinstance(ar, dict):
            ar = AmazonResearchData(**ar)
        amazon_data = ar

    # Search for more pricing data
    pricing_results = search_competitor_pricing(s.product_category)

    # Build context
    products_str = ""
    if amazon_data and amazon_data.top_products:
        products_str = json.dumps(amazon_data.top_products[:5], indent=2)

    pricing_str = "\n".join(
        f"- {r.get('title','')} | {r.get('price','')} | "
        f"Rating: {r.get('rating','')} | {r.get('reviews','')}"
        for r in pricing_results[:8]
    )

    # Currency mapping based on target market
    currency_map = {
        "India": "Indian Rupees (Rs). Convert all USD prices to INR (1 USD = Rs84). Example: $50 = Rs4,200",
        "UK": "British Pounds (GBP). Convert all USD prices to GBP (1 USD = 0.79 GBP). Example: $50 = 39 GBP",
        "Europe": "Euros (EUR). Convert all USD prices to EUR (1 USD = 0.92 EUR). Example: $50 = 46 EUR",
        "Australia": "Australian Dollars (AUD). Convert all USD prices to AUD (1 USD = 1.54 AUD). Example: $50 = 77 AUD",
        "United States": "US Dollars ($). Keep original USD prices.",
        "Global": "US Dollars ($). Keep original USD prices.",
    }

    currency_instruction = currency_map.get(s.target_market, "US Dollars ($)")

    context = f"""
Product Category: {s.product_category}
Target Market: {s.target_market}
Budget Range: {s.budget_range}
IMPORTANT: Show ALL prices in {currency_instruction}

=== TOP PRODUCTS WITH PRICES ===
{products_str}

=== COMPETITOR PRICING ===
{pricing_str}
"""

    llm = _get_llm()

    system_prompt = """You are a Pricing Strategy Expert AI for E-commerce.
Analyze product pricing data and provide strategic insights.

IMPORTANT CURRENCY RULE:
- If target market is India → use Rs (1 USD = Rs84). Example: $50 = Rs4,200
- If target market is UK → use GBP (1 USD = 0.79 GBP). Example: $50 = 39 GBP
- If target market is Europe → use EUR (1 USD = 0.92 EUR). Example: $50 = 46 EUR
- If target market is Australia → use AUD (1 USD = 1.54 AUD). Example: $50 = 77 AUD
- If target market is US/Global → use $ (keep original)

Always convert ALL prices to the correct currency for the target market!

Return ONLY valid JSON (no markdown, no extra text):
{
  "average_price": 45.99,
  "min_price": 9.99,
  "max_price": 199.99,
  "sweet_spot_price": "XX-XX in local currency (highest sales volume)",
  "price_segments": [
    {
      "segment": "Budget",
      "range": "XX-XX in local currency",
      "share": "35%",
      "characteristics": "Basic features, high volume"
    },
    {
      "segment": "Mid-Range",
      "range": "XX-XX in local currency",
      "share": "45%",
      "characteristics": "Best value, most popular"
    },
    {
      "segment": "Premium",
      "range": "XX-XX in local currency",
      "share": "20%",
      "characteristics": "Advanced features, loyal customers"
    }
  ],
  "pricing_strategy": [
    "Strategy recommendation 1 with specific price in local currency",
    "Strategy recommendation 2",
    "Strategy recommendation 3",
    "Strategy recommendation 4"
  ],
  "competitor_pricing": [
    {
      "brand": "Brand1",
      "price_range": "XX-XX in local currency",
      "positioning": "budget/mid/premium"
    },
    {
      "brand": "Brand2",
      "price_range": "XX-XX in local currency",
      "positioning": "budget/mid/premium"
    }
  ],
  "recommended_price_range": "XX-XX in local currency for maximum sales volume"
}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip code fences
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                raw = part
                break

    parsed = json.loads(raw)

    return PriceAnalysisData(
        average_price=float(parsed.get("average_price", 0)),
        min_price=float(parsed.get("min_price", 0)),
        max_price=float(parsed.get("max_price", 0)),
        sweet_spot_price=parsed.get("sweet_spot_price", ""),
        price_segments=parsed.get("price_segments", []),
        pricing_strategy=parsed.get("pricing_strategy", []),
        competitor_pricing=parsed.get("competitor_pricing", []),
        recommended_price_range=parsed.get("recommended_price_range", ""),
    )