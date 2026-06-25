"""
agents/trend_agent.py
──────────────────────
Trend Agent — analyzes market trends and growth opportunities.

Workflow:
1. Search for market trends using Serper
2. Find emerging patterns and seasonal data
3. Use Groq to synthesize trend insights
4. Return TrendData model
"""

import os
import json
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from models.state import ProductResearchState, TrendData
from tools.serper_tool import search_market_trends, search_web

load_dotenv()


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.3,
    )


def trend_agent_node(state: dict) -> dict:
    """LangGraph node: Trend Agent."""
    s = ProductResearchState(**state)
    logger.info(f"[Trend] Analyzing trends for: {s.product_category!r}")

    try:
        data = _run_trend_analysis(s)
        logger.info("[Trend] Analysis completed.")
        return {"trend_data": data.model_dump()}
    except Exception as e:
        logger.error(f"[Trend] Error: {e}")
        return {
            "trend_data": TrendData(error=str(e)).model_dump(),
            "errors": state.get("errors", []) + [f"Trend analysis error: {e}"],
        }


def _run_trend_analysis(s: ProductResearchState) -> TrendData:
    """Core trend analysis logic."""

    # Multiple searches for comprehensive trend data
    trend_results = search_market_trends(s.product_category)
    future_results = search_web(
        f"{s.product_category} future trends forecast 2025 2026 market size"
    )
    seasonal_results = search_web(
        f"{s.product_category} seasonal demand best time to sell amazon"
    )

    def format_results(results: list[dict]) -> str:
        return "\n".join(
            f"[{i+1}] {r.get('title','')} — {r.get('snippet','')}"
            for i, r in enumerate(results[:5])
        )

    context = f"""
Product Category: {s.product_category}
Target Market: {s.target_market}

=== MARKET TREND RESULTS ===
{format_results(trend_results)}

=== FUTURE FORECAST ===
{format_results(future_results)}

=== SEASONAL PATTERNS ===
{format_results(seasonal_results)}
"""

    llm = _get_llm()

    system_prompt = """You are a Market Trend Analysis Expert AI for E-commerce.
Analyze market trends and provide actionable insights for product sellers.

Return ONLY valid JSON:
{
  "market_size": "$X billion global market (2024)",
  "growth_rate": "X% CAGR expected through 2027",
  "trending_keywords": [
    "keyword1", "keyword2", "keyword3", "keyword4",
    "keyword5", "keyword6", "keyword7", "keyword8"
  ],
  "seasonal_patterns": [
    "Peak season: November-December (holiday shopping)",
    "Secondary peak: Back to school August-September",
    "Slow period: January-February post-holiday"
  ],
  "emerging_trends": [
    "Emerging trend 1 with specific detail",
    "Emerging trend 2",
    "Emerging trend 3",
    "Emerging trend 4"
  ],
  "market_opportunities": [
    "Opportunity 1 — specific and actionable",
    "Opportunity 2",
    "Opportunity 3",
    "Opportunity 4"
  ],
  "market_threats": [
    "Threat 1 — specific risk",
    "Threat 2",
    "Threat 3"
  ],
  "future_outlook": "2-3 sentence summary of where this market is heading"
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

    return TrendData(
        market_size=parsed.get("market_size", ""),
        growth_rate=parsed.get("growth_rate", ""),
        trending_keywords=parsed.get("trending_keywords", []),
        seasonal_patterns=parsed.get("seasonal_patterns", []),
        emerging_trends=parsed.get("emerging_trends", []),
        market_opportunities=parsed.get("market_opportunities", []),
        market_threats=parsed.get("market_threats", []),
        future_outlook=parsed.get("future_outlook", ""),
    )
