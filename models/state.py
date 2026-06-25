"""
models/state.py
───────────────
Pydantic state models for the E-commerce Product Research Agent System.
Every agent reads from and writes to this shared state object.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class AmazonProduct(BaseModel):
    """Single product found by Amazon Research Agent."""
    title: str = ""
    price: str = ""
    rating: float = 0.0
    reviews_count: str = ""
    asin: str = ""
    brand: str = ""
    category: str = ""
    url: str = ""
    features: list[str] = Field(default_factory=list)


class AmazonResearchData(BaseModel):
    """Output from Amazon Research Agent."""
    category: str = ""
    top_products: list[dict[str, Any]] = Field(default_factory=list)
    total_found: int = 0
    top_brands: list[str] = Field(default_factory=list)
    price_range: dict[str, Any] = Field(default_factory=dict)
    key_findings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class PriceAnalysisData(BaseModel):
    """Output from Price Analysis Agent."""
    average_price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    sweet_spot_price: str = ""
    price_segments: list[dict[str, Any]] = Field(default_factory=list)
    pricing_strategy: list[str] = Field(default_factory=list)
    competitor_pricing: list[dict[str, Any]] = Field(default_factory=list)
    recommended_price_range: str = ""
    error: Optional[str] = None


class ReviewAnalysisData(BaseModel):
    """Output from Review Agent."""
    overall_sentiment: str = ""
    common_complaints: list[str] = Field(default_factory=list)
    common_praises: list[str] = Field(default_factory=list)
    quality_gaps: list[str] = Field(default_factory=list)
    customer_needs: list[str] = Field(default_factory=list)
    review_insights: list[str] = Field(default_factory=list)
    improvement_opportunities: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class TrendData(BaseModel):
    """Output from Trend Agent."""
    market_size: str = ""
    growth_rate: str = ""
    trending_keywords: list[str] = Field(default_factory=list)
    seasonal_patterns: list[str] = Field(default_factory=list)
    emerging_trends: list[str] = Field(default_factory=list)
    market_opportunities: list[str] = Field(default_factory=list)
    market_threats: list[str] = Field(default_factory=list)
    future_outlook: str = ""
    error: Optional[str] = None


class OpportunityData(BaseModel):
    """Final output from Opportunity Agent."""
    executive_summary: str = ""
    opportunity_score: int = 0
    recommended_products: list[dict[str, Any]] = Field(default_factory=list)
    target_audience: str = ""
    unique_selling_points: list[str] = Field(default_factory=list)
    go_to_market_strategy: list[str] = Field(default_factory=list)
    estimated_profit_margin: str = ""
    investment_required: str = ""
    time_to_market: str = ""
    risk_assessment: list[str] = Field(default_factory=list)
    action_plan: list[dict[str, Any]] = Field(default_factory=list)
    final_verdict: str = ""
    error: Optional[str] = None


class ProductResearchState(BaseModel):
    """Master LangGraph state for product research workflow."""

    # Input
    product_category: str = Field(
        description="Product category to research e.g. 'Wireless Earbuds'"
    )
    target_market: str = Field(
        default="Global",
        description="Target market e.g. 'US', 'India', 'Global'"
    )
    budget_range: str = Field(
        default="Any",
        description="Budget range for product e.g. '$10-50'"
    )
    run_id: str = Field(default="")
    started_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Supervisor
    supervisor_plan: str = Field(default="")
    active_agents: list[str] = Field(default_factory=list)

    # Agent outputs
    amazon_research: Optional[AmazonResearchData] = None
    price_analysis: Optional[PriceAnalysisData] = None
    review_analysis: Optional[ReviewAnalysisData] = None
    trend_data: Optional[TrendData] = None
    opportunity_data: Optional[OpportunityData] = None

    # Meta
    errors: list[str] = Field(default_factory=list)
    completed_at: Optional[str] = None

    class Config:
        extra = "allow"
