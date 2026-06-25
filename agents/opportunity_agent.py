"""
agents/opportunity_agent.py
────────────────────────────
Opportunity Agent — synthesizes all research into actionable opportunities.

Workflow:
1. Read all previous agent outputs
2. Identify best products to sell
3. Create go-to-market strategy
4. Return OpportunityData model
"""

import os
import json
from datetime import datetime
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from models.state import (
    ProductResearchState, OpportunityData,
    AmazonResearchData, PriceAnalysisData,
    ReviewAnalysisData, TrendData,
)

load_dotenv()


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.2,
    )


def opportunity_agent_node(state: dict) -> dict:
    """LangGraph node: Opportunity Agent (final synthesis)."""
    s = ProductResearchState(**state)
    logger.info(f"[Opportunity] Synthesizing research for: {s.product_category!r}")

    try:
        data = _build_opportunity_report(s)
        completed_at = datetime.utcnow().isoformat()
        logger.info("[Opportunity] Report finalized.")
        return {
            "opportunity_data": data.model_dump(),
            "completed_at": completed_at,
        }
    except Exception as e:
        logger.error(f"[Opportunity] Error: {e}")
        return {
            "opportunity_data": OpportunityData(error=str(e)).model_dump(),
            "errors": state.get("errors", []) + [f"Opportunity error: {e}"],
            "completed_at": datetime.utcnow().isoformat(),
        }


def _build_opportunity_report(s: ProductResearchState) -> OpportunityData:
    """Synthesize all agent outputs into opportunity report."""

    # Serialize all agent outputs
    def safe_dump(obj, model_class) -> str:
        if obj is None:
            return "Not available"
        if isinstance(obj, dict):
            obj = model_class(**obj)
        return json.dumps(obj.model_dump(), indent=2, default=str)

    amazon_str = safe_dump(s.amazon_research, AmazonResearchData)
    price_str = safe_dump(s.price_analysis, PriceAnalysisData)
    review_str = safe_dump(s.review_analysis, ReviewAnalysisData)
    trend_str = safe_dump(s.trend_data, TrendData)

    llm = _get_llm()

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

    system_prompt = """You are a Senior E-commerce Business Opportunity Analyst AI.
Synthesize all research data into a comprehensive opportunity report.

Return ONLY valid JSON (no markdown, no extra text):
{
  "executive_summary": "2-3 paragraph summary of the opportunity with specific numbers",
  "opportunity_score": 85,
  "recommended_products": [
    {
      "rank": 1,
      "product_idea": "Specific product to sell",
      "why": "Why this is a good opportunity",
      "estimated_price": "XX-XX in local currency",
      "profit_margin": "XX-XX%",
      "competition_level": "Low/Medium/High",
      "difficulty": "Easy/Medium/Hard"
    }
  ],
  "target_audience": "Specific description of ideal customer",
  "unique_selling_points": [
    "USP 1 — what makes your product different",
    "USP 2",
    "USP 3",
    "USP 4"
  ],
  "go_to_market_strategy": [
    "Step 1: Strategy action",
    "Step 2",
    "Step 3",
    "Step 4",
    "Step 5"
  ],
  "estimated_profit_margin": "XX-XX% based on market analysis",
  "investment_required": "XXX-XXXX in local currency to start",
  "time_to_market": "X-X weeks",
  "risk_assessment": [
    "Risk 1 with mitigation strategy",
    "Risk 2",
    "Risk 3"
  ],
  "action_plan": [
    {
      "phase": "Phase 1 (Week 1-2)",
      "actions": ["Action 1", "Action 2", "Action 3"],
      "goal": "What to achieve"
    },
    {
      "phase": "Phase 2 (Week 3-4)",
      "actions": ["Action 1", "Action 2"],
      "goal": "What to achieve"
    },
    {
      "phase": "Phase 3 (Month 2)",
      "actions": ["Action 1", "Action 2"],
      "goal": "What to achieve"
    }
  ],
  "final_verdict": "GO/NO-GO/PROCEED WITH CAUTION — specific reasoning with data points"
}

opportunity_score: 0-100 (0=terrible, 100=amazing opportunity)"""

    user_prompt = f"""
Product Category: {s.product_category}
Target Market: {s.target_market}
Budget Range: {s.budget_range}
IMPORTANT: Show ALL prices in {currency_instruction}

=== AMAZON RESEARCH ===
{amazon_str}

=== PRICE ANALYSIS ===
{price_str}

=== REVIEW ANALYSIS ===
{review_str}

=== TREND DATA ===
{trend_str}

Synthesize all data into a comprehensive opportunity report.
Be specific with numbers and actionable with recommendations.
Remember to convert ALL prices to the correct currency for {s.target_market}."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
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

    return OpportunityData(
        executive_summary=parsed.get("executive_summary", ""),
        opportunity_score=int(parsed.get("opportunity_score", 0)),
        recommended_products=parsed.get("recommended_products", []),
        target_audience=parsed.get("target_audience", ""),
        unique_selling_points=parsed.get("unique_selling_points", []),
        go_to_market_strategy=parsed.get("go_to_market_strategy", []),
        estimated_profit_margin=parsed.get("estimated_profit_margin", ""),
        investment_required=parsed.get("investment_required", ""),
        time_to_market=parsed.get("time_to_market", ""),
        risk_assessment=parsed.get("risk_assessment", []),
        action_plan=parsed.get("action_plan", []),
        final_verdict=parsed.get("final_verdict", ""),
    )