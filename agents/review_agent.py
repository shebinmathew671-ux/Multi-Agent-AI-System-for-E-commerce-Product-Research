"""
agents/review_agent.py
───────────────────────
Review Agent — analyzes customer reviews to find gaps and opportunities.

Workflow:
1. Search for customer reviews using Serper
2. Analyze common complaints and praises
3. Find quality gaps to exploit
4. Return ReviewAnalysisData model
"""

import os
import json
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from models.state import ProductResearchState, ReviewAnalysisData, AmazonResearchData
from tools.serper_tool import search_amazon_reviews, search_web

load_dotenv()


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.3,
    )


def review_agent_node(state: dict) -> dict:
    """LangGraph node: Review Agent."""
    s = ProductResearchState(**state)
    logger.info(f"[Review] Analyzing reviews for: {s.product_category!r}")

    try:
        data = _run_review_analysis(s)
        logger.info("[Review] Analysis completed.")
        return {"review_analysis": data.model_dump()}
    except Exception as e:
        logger.error(f"[Review] Error: {e}")
        return {
            "review_analysis": ReviewAnalysisData(error=str(e)).model_dump(),
            "errors": state.get("errors", []) + [f"Review analysis error: {e}"],
        }


def _run_review_analysis(s: ProductResearchState) -> ReviewAnalysisData:
    """Core review analysis logic."""

    # Search for reviews
    review_results = search_amazon_reviews(s.product_category)
    complaint_results = search_web(
        f"{s.product_category} amazon negative reviews common problems issues 2024"
    )

    def format_results(results: list[dict]) -> str:
        return "\n".join(
            f"[{i+1}] {r.get('title','')} — {r.get('snippet','')}"
            for i, r in enumerate(results[:6])
        )

    context = f"""
Product Category: {s.product_category}
Target Market: {s.target_market}

=== REVIEW SEARCH RESULTS ===
{format_results(review_results)}

=== COMPLAINT ANALYSIS ===
{format_results(complaint_results)}
"""

    llm = _get_llm()

    system_prompt = """You are a Customer Review Analysis Expert AI.
Analyze customer reviews to find gaps, opportunities and insights.

Return ONLY valid JSON:
{
  "overall_sentiment": "Mostly Positive/Mixed/Mostly Negative",
  "common_complaints": [
    "Complaint 1 with frequency indicator",
    "Complaint 2",
    "Complaint 3",
    "Complaint 4",
    "Complaint 5"
  ],
  "common_praises": [
    "Praise 1",
    "Praise 2",
    "Praise 3",
    "Praise 4",
    "Praise 5"
  ],
  "quality_gaps": [
    "Gap 1 — what customers wish existed",
    "Gap 2",
    "Gap 3",
    "Gap 4"
  ],
  "customer_needs": [
    "Unmet need 1",
    "Unmet need 2",
    "Unmet need 3",
    "Unmet need 4"
  ],
  "review_insights": [
    "Key insight 1 from review data",
    "Key insight 2",
    "Key insight 3"
  ],
  "improvement_opportunities": [
    "Opportunity 1 — specific product improvement",
    "Opportunity 2",
    "Opportunity 3",
    "Opportunity 4"
  ]
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

    return ReviewAnalysisData(
        overall_sentiment=parsed.get("overall_sentiment", ""),
        common_complaints=parsed.get("common_complaints", []),
        common_praises=parsed.get("common_praises", []),
        quality_gaps=parsed.get("quality_gaps", []),
        customer_needs=parsed.get("customer_needs", []),
        review_insights=parsed.get("review_insights", []),
        improvement_opportunities=parsed.get("improvement_opportunities", []),
    )
