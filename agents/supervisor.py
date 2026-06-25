"""
agents/supervisor.py
────────────────────
Supervisor Agent — orchestrates the product research workflow.
Receives user's product category and creates a routing plan.
"""

import os
import json
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from models.state import ProductResearchState

load_dotenv()


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.3,
    )


def supervisor_node(state: dict) -> dict:
    """LangGraph node: Supervisor Agent."""
    s = ProductResearchState(**state)
    logger.info(f"[Supervisor] Starting research for: {s.product_category!r}")

    try:
        plan, agents = _create_routing_plan(s)
        logger.info(f"[Supervisor] Plan created. Agents: {agents}")
        return {
            "supervisor_plan": plan,
            "active_agents": agents,
        }
    except Exception as e:
        logger.error(f"[Supervisor] Error: {e}")
        return {
            "supervisor_plan": f"Default plan — run all agents.",
            "active_agents": ["amazon_research", "price_analysis", "review_analysis", "trend", "opportunity"],
            "errors": state.get("errors", []) + [f"Supervisor error: {e}"],
        }


def _create_routing_plan(s: ProductResearchState) -> tuple[str, list[str]]:
    """Use Groq to create a research routing plan."""
    llm = _get_llm()

    system_prompt = """You are an E-commerce Product Research Supervisor AI.
Analyze the product category and create a research plan.

Return ONLY valid JSON:
{
  "routing_plan": "Brief explanation of research strategy",
  "active_agents": ["amazon_research", "price_analysis", "review_analysis", "trend", "opportunity"],
  "research_focus": {
    "amazon_research": "what to focus on",
    "price_analysis": "what to focus on",
    "review_analysis": "what to focus on",
    "trend": "what to focus on",
    "opportunity": "what to focus on"
  }
}"""

    user_prompt = f"""
Product Category: {s.product_category}
Target Market: {s.target_market}
Budget Range: {s.budget_range}

Create a comprehensive research plan for this product category.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
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
    plan = parsed.get("routing_plan", "Research all aspects of the product category.")
    agents = parsed.get("active_agents", [
        "amazon_research", "price_analysis",
        "review_analysis", "trend", "opportunity"
    ])

    return plan, agents
