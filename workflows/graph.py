"""
workflows/graph.py
───────────────────
LangGraph StateGraph for E-commerce Product Research Agent System.

Graph topology:
    supervisor
        ↓
    amazon_research
        ↓
    price_analysis
        ↓
    review_analysis
        ↓
    trend_analysis
        ↓
    opportunity
        ↓
       END
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional
from loguru import logger

from agents.supervisor import supervisor_node
from agents.amazon_research_agent import amazon_research_node
from agents.price_analysis_agent import price_analysis_node
from agents.review_agent import review_agent_node
from agents.trend_agent import trend_agent_node
from agents.opportunity_agent import opportunity_agent_node


class GraphState(TypedDict, total=False):
    # Input
    product_category: str
    target_market: str
    budget_range: str
    run_id: str
    started_at: str

    # Supervisor
    active_agents: list[str]
    supervisor_plan: str

    # Agent outputs
    amazon_research: Optional[dict]
    price_analysis: Optional[dict]
    review_analysis: Optional[dict]
    trend_data: Optional[dict]
    opportunity_data: Optional[dict]

    # Meta
    errors: list[str]
    completed_at: Optional[str]


def build_graph() -> StateGraph:
    """Build the LangGraph StateGraph."""
    graph = StateGraph(GraphState)

    # Register ALL nodes including supervisor
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("amazon_agent", amazon_research_node)
    graph.add_node("price_agent", price_analysis_node)
    graph.add_node("review_agent", review_agent_node)
    graph.add_node("trend_agent", trend_agent_node)
    graph.add_node("opportunity_agent", opportunity_agent_node)

    # Define edges
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "amazon_agent")
    graph.add_edge("amazon_agent", "price_agent")
    graph.add_edge("price_agent", "review_agent")
    graph.add_edge("review_agent", "trend_agent")
    graph.add_edge("trend_agent", "opportunity_agent")
    graph.add_edge("opportunity_agent", END)

    logger.info("[Graph] StateGraph compiled: 6 nodes, sequential execution.")
    return graph


_compiled_graph = None


def get_compiled_graph():
    """Return cached compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_graph()
        _compiled_graph = graph.compile()
        logger.info("[Graph] Graph compiled and cached.")
    return _compiled_graph


def run_product_research(
    product_category: str,
    target_market: str = "Global",
    budget_range: str = "Any",
    run_id: str = "",
) -> dict:
    """Execute the full product research workflow."""
    import uuid
    from datetime import datetime

    if not run_id:
        run_id = str(uuid.uuid4())

    initial_state: GraphState = {
        "product_category": product_category,
        "target_market": target_market,
        "budget_range": budget_range,
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "active_agents": [],
        "supervisor_plan": "",
        "amazon_research": None,
        "price_analysis": None,
        "review_analysis": None,
        "trend_data": None,
        "opportunity_data": None,
        "errors": [],
        "completed_at": None,
    }

    logger.info(f"[Graph] Starting run {run_id} | Category: {product_category!r}")

    app = get_compiled_graph()
    try:
        final_state = app.invoke(initial_state)
        logger.info(f"[Graph] Run {run_id} completed.")
        return dict(final_state)
    except Exception as e:
        logger.error(f"[Graph] Run {run_id} failed: {e}")
        raise
