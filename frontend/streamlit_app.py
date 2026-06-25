"""
frontend/streamlit_app.py
──────────────────────────
Streamlit UI for E-commerce Product Research Agent System.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Product Research AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
}
.main-header h1 { margin: 0; font-size: 2.2rem; }
.main-header p  { margin: 0.5rem 0 0 0; opacity: 0.9; }

.agent-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.4rem 0;
    border-left: 4px solid #f5576c;
}
.agent-card h4 { margin: 0 0 0.3rem 0; color: #2d3748; font-size: 0.95rem; }
.agent-card p  { margin: 0; color: #718096; font-size: 0.85rem; }

.kpi-card {
    background: linear-gradient(135deg, #fff5f7, #ffeef2);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #fed7e2;
}
.kpi-label { font-size: 0.85rem; color: #718096; margin-bottom: 0.3rem; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #e53e3e; }

.opportunity-score {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin: 1rem 0;
}
.score-number { font-size: 4rem; font-weight: 900; }
.score-label  { font-size: 1rem; opacity: 0.9; }

.verdict-go {
    background: linear-gradient(135deg, #48bb78, #38a169);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.verdict-caution {
    background: linear-gradient(135deg, #ed8936, #dd6b20);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.verdict-nogo {
    background: linear-gradient(135deg, #fc8181, #e53e3e);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.product-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem;
    margin: 0.7rem 0;
    border-top: 4px solid #f5576c;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Agent Pipeline")
        agents = [
            ("🎯", "Supervisor", "Routes & coordinates"),
            ("🔍", "Amazon Research", "Finds top products"),
            ("💰", "Price Analysis", "Compares pricing"),
            ("⭐", "Review Agent", "Analyzes reviews"),
            ("📊", "Trend Agent", "Market trends"),
            ("🧠", "Opportunity Agent", "Best products to sell"),
        ]
        for icon, name, desc in agents:
            st.markdown(f"""
            <div class="agent-card">
                <h4>{icon} {name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📋 Research History")

        if st.button("🔄 Refresh"):
            st.rerun()

        history = fetch_history()
        if history:
            for run in history[:8]:
                status_emoji = {"completed": "✅", "failed": "❌", "running": "⏳"}.get(run.get("status", ""), "❓")
                with st.expander(f"{status_emoji} {run.get('product_category', '')[:25]}"):
                    st.caption(f"Market: {run.get('target_market', '')}")
                    st.caption(f"Status: {run.get('status', '')}")
                    st.caption(f"Date: {run.get('created_at', '')[:10]}")
        else:
            st.info("No research yet!")

        st.markdown("---")
        st.markdown("### 🔧 System Status")
        check_health()


def fetch_history():
    try:
        r = requests.get(f"{API_BASE}/api/runs", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def check_health():
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success("✅ API Online")
            st.caption(f"Groq API: {'✅' if data.get('groq_api_configured') else '❌'}")
            st.caption(f"Serper API: {'✅' if data.get('serper_api_configured') else '⚠️ Using mock'}")
    except Exception:
        st.error("❌ API Offline")
        st.caption("Start FastAPI server first!")


# ─── Main Page ────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🛒 E-commerce Product Research AI</h1>
        <p>6 AI agents research any product category and find your best selling opportunity</p>
    </div>
    """, unsafe_allow_html=True)


def render_input_form():
    st.markdown("### 🔍 What Product Do You Want to Research?")

    if "category_text" not in st.session_state:
        st.session_state["category_text"] = ""

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        category = st.text_input(
            "Product Category",
            value=st.session_state["category_text"],
            placeholder="e.g. Wireless Earbuds, Yoga Mats, Phone Cases...",
        )
        st.session_state["category_text"] = category

    with col2:
        target_market = st.selectbox(
            "Target Market",
            ["Global", "United States", "India", "UK", "Europe", "Australia"],
        )

    with col3:
        if target_market == "India":
            price_options = [
                "Any",
                "₹100-300",
                "₹300-500",
                "₹500-1,000",
                "₹1,000-2,000",
                "₹2,000-5,000",
                "₹5,000-10,000",
                "₹10,000-20,000",
                "₹20,000+",
            ]
        elif target_market == "UK":
            price_options = [
                "Any",
                "£4-16",
                "£16-40",
                "£40-79",
                "£79-158",
                "£158+",
            ]
        elif target_market == "Europe":
            price_options = [
                "Any",
                "€5-18",
                "€18-46",
                "€46-92",
                "€92-184",
                "€184+",
            ]
        elif target_market == "Australia":
            price_options = [
                "Any",
                "A$8-31",
                "A$31-77",
                "A$77-154",
                "A$154-308",
                "A$308+",
            ]
        elif target_market == "United States":
            price_options = [
                "Any",
                "$5-20",
                "$20-50",
                "$50-100",
                "$100-200",
                "$200+",
            ]
        else:
            price_options = [
                "Any",
                "$5-20",
                "$20-50",
                "$50-100",
                "$100-200",
                "$200+",
            ]

        budget_range = st.selectbox(
            "Price Range",
            price_options,
        )

    # Quick presets
    st.markdown("**💡 Popular Categories:**")
    cols = st.columns(6)
    presets = [
        "Wireless Earbuds",
        "Yoga Mats",
        "Phone Cases",
        "LED Lights",
        "Water Bottles",
        "Laptop Stands",
    ]
    for i, preset in enumerate(presets):
        with cols[i]:
            if st.button(f"🔍 {preset}", use_container_width=True, key=f"cat_{i}"):
                st.session_state["category_text"] = preset
                st.rerun()

    return st.session_state["category_text"], target_market, budget_range


def render_progress():
    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        (15, "🎯 Supervisor creating research plan..."),
        (30, "🔍 Amazon Research Agent finding products..."),
        (45, "💰 Price Analysis Agent comparing pricing..."),
        (60, "⭐ Review Agent analyzing customer feedback..."),
        (75, "📊 Trend Agent researching market trends..."),
        (90, "🧠 Opportunity Agent finding best products..."),
        (100, "✅ Research complete!"),
    ]

    for pct, msg in steps:
        progress_bar.progress(pct)
        status_text.markdown(f"**{msg}**")
        time.sleep(0.5)

    return progress_bar, status_text


# ─── Results ──────────────────────────────────────────────────────────────────

def render_results(result: dict):
    """Render all research results."""

    agents_used = result.get("active_agents", [])
    st.markdown("### ✅ Research Complete!")

    cols = st.columns(5)
    agent_list = [
        ("amazon_research", "🔍 Amazon"),
        ("price_analysis", "💰 Pricing"),
        ("review_analysis", "⭐ Reviews"),
        ("trend", "📊 Trends"),
        ("opportunity", "🧠 Opportunity"),
    ]
    for i, (key, label) in enumerate(agent_list):
        with cols[i]:
            st.metric(label, "✅ Done")

    st.markdown("---")

    tabs = st.tabs([
        "🧠 Opportunity", "🔍 Amazon", "💰 Pricing",
        "⭐ Reviews", "📊 Trends", "📋 Raw JSON"
    ])

    with tabs[0]:
        render_opportunity_tab(result.get("opportunity_data"))

    with tabs[1]:
        render_amazon_tab(result.get("amazon_research"))

    with tabs[2]:
        render_pricing_tab(result.get("price_analysis"))

    with tabs[3]:
        render_reviews_tab(result.get("review_analysis"))

    with tabs[4]:
        render_trends_tab(result.get("trend_data"))

    with tabs[5]:
        render_json_tab(result)


def render_opportunity_tab(data: dict | None):
    if not data:
        st.warning("Opportunity data not available.")
        return

    st.markdown("## 🧠 Product Opportunity Report")

    score = data.get("opportunity_score", 0)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="opportunity-score">
            <div class="score-number">{score}</div>
            <div class="score-label">Opportunity Score / 100</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="margin-top:0.5rem;">
            <div class="kpi-label">Investment Required</div>
            <div class="kpi-value" style="font-size:1.2rem;">
                {data.get('investment_required', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="margin-top:0.5rem;">
            <div class="kpi-label">Profit Margin</div>
            <div class="kpi-value" style="font-size:1.2rem;">
                {data.get('estimated_profit_margin', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    verdict = data.get("final_verdict", "")
    if verdict:
        if "GO" in verdict.upper() and "NO" not in verdict.upper():
            css_class = "verdict-go"
        elif "CAUTION" in verdict.upper():
            css_class = "verdict-caution"
        else:
            css_class = "verdict-nogo"
        st.markdown(f'<div class="{css_class}">🎯 {verdict}</div>',
                   unsafe_allow_html=True)

    if data.get("executive_summary"):
        st.markdown("### 📝 Executive Summary")
        st.markdown(data["executive_summary"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Recommended Products to Sell")
        for prod in data.get("recommended_products", []):
            st.markdown(f"""
            <div class="product-card">
                <div style="color:#e53e3e;font-weight:700;">
                    #{prod.get('rank','')} {prod.get('product_idea','')}
                </div>
                <div style="color:#000000;margin:0.5rem 0;">
                    {prod.get('why','')}
                </div>
                <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                    <span style="background:#fff5f7;color:#e53e3e;
                    padding:2px 8px;border-radius:4px;font-size:0.85rem;">
                        💰 {prod.get('estimated_price','')}
                    </span>
                    <span style="background:#f0fff4;color:#276749;
                    padding:2px 8px;border-radius:4px;font-size:0.85rem;">
                        📈 {prod.get('profit_margin','')}
                    </span>
                    <span style="background:#ebf8ff;color:#2b6cb0;
                    padding:2px 8px;border-radius:4px;font-size:0.85rem;">
                        🏁 {prod.get('competition_level','')} Competition
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### ✨ Unique Selling Points")
        for usp in data.get("unique_selling_points", []):
            st.markdown(f"⭐ {usp}")

        st.markdown("### 🎯 Target Audience")
        st.info(data.get("target_audience", ""))

        st.markdown("### ⏱️ Time to Market")
        st.success(f"**{data.get('time_to_market', 'N/A')}**")

    st.markdown("---")

    st.markdown("### 🚀 Go-to-Market Strategy")
    for step in data.get("go_to_market_strategy", []):
        st.markdown(f"→ {step}")

    st.markdown("---")

    st.markdown("### 📅 Action Plan")
    action_plan = data.get("action_plan", [])
    if action_plan:
        cols = st.columns(len(action_plan))
        for i, phase in enumerate(action_plan):
            with cols[i]:
                st.markdown(f"**{phase.get('phase','')}**")
                st.caption(f"Goal: {phase.get('goal','')}")
                for action in phase.get("actions", []):
                    st.markdown(f"• {action}")

    st.markdown("---")
    st.markdown("### ⚠️ Risk Assessment")
    for risk in data.get("risk_assessment", []):
        st.markdown(f"🔴 {risk}")


def render_amazon_tab(data: dict | None):
    if not data:
        st.warning("Amazon research data not available.")
        return

    st.markdown("## 🔍 Amazon Product Research")

    st.markdown("### 💡 Key Findings")
    for finding in data.get("key_findings", []):
        st.markdown(f"🔎 {finding}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        brands = data.get("top_brands", [])
        if brands:
            st.markdown("### 🏢 Top Brands")
            for i, brand in enumerate(brands, 1):
                st.markdown(f"{i}. **{brand}**")

    with col2:
        price_range = data.get("price_range", {})
        if price_range:
            st.markdown("### 💰 Price Segments")
            for segment, price in price_range.items():
                st.markdown(f"**{segment.title()}:** {price}")

    st.markdown("---")
    st.markdown("### 🛒 Top Products Found")
    for prod in data.get("top_products", []):
        with st.expander(
            f"#{prod.get('rank','')} {prod.get('title','')} — "
            f"{prod.get('price','')} ⭐{prod.get('rating','')}"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Brand:** {prod.get('brand','')}")
                st.markdown(f"**Price:** {prod.get('price','')}")
                st.markdown(f"**Rating:** ⭐ {prod.get('rating','')}")
                st.markdown(f"**Reviews:** {prod.get('reviews_count','')}")
            with col2:
                st.markdown("**Key Features:**")
                for feat in prod.get("key_features", []):
                    st.markdown(f"• {feat}")
            st.markdown(f"**Why Popular:** {prod.get('why_popular','')}")


def render_pricing_tab(data: dict | None):
    if not data:
        st.warning("Pricing data not available.")
        return

    st.markdown("## 💰 Price Analysis")

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Price</div>
            <div class="kpi-value">{data.get('average_price', 0):.0f}</div>
        </div>""", unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Min Price</div>
            <div class="kpi-value">{data.get('min_price', 0):.0f}</div>
        </div>""", unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Max Price</div>
            <div class="kpi-value">{data.get('max_price', 0):.0f}</div>
        </div>""", unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Sweet Spot</div>
            <div class="kpi-value" style="font-size:1rem;">
                {data.get('sweet_spot_price', 'N/A')}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    segments = data.get("price_segments", [])
    if segments:
        st.markdown("### 📊 Price Segment Distribution")
        df_seg = pd.DataFrame(segments)
        if "segment" in df_seg.columns and "share" in df_seg.columns:
            try:
                df_seg["share_num"] = df_seg["share"].str.replace("%", "").astype(float)
                fig = px.pie(
                    df_seg, names="segment", values="share_num",
                    title="Market Share by Price Segment",
                    color_discrete_sequence=["#f093fb", "#f5576c", "#4facfe"],
                )
                fig.update_layout(paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        for seg in segments:
            with st.expander(f"💲 {seg.get('segment','')} — {seg.get('range','')}"):
                st.markdown(f"**Market Share:** {seg.get('share','')}")
                st.markdown(f"**Characteristics:** {seg.get('characteristics','')}")

    st.markdown("### 💡 Pricing Strategy Recommendations")
    for strategy in data.get("pricing_strategy", []):
        st.markdown(f"→ {strategy}")

    st.markdown("---")
    st.success(f"**Recommended Price Range:** {data.get('recommended_price_range', 'N/A')}")


def render_reviews_tab(data: dict | None):
    if not data:
        st.warning("Review data not available.")
        return

    st.markdown("## ⭐ Review Analysis")

    sentiment = data.get("overall_sentiment", "Mixed")
    sentiment_color = (
        "#48bb78" if "Positive" in sentiment
        else "#e53e3e" if "Negative" in sentiment
        else "#ed8936"
    )
    st.markdown(
        f'<div style="background:{sentiment_color};color:white;padding:0.8rem;'
        f'border-radius:8px;text-align:center;font-weight:700;margin-bottom:1rem;">'
        f'Overall Customer Sentiment: {sentiment}</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👍 Common Praises")
        for praise in data.get("common_praises", []):
            st.markdown(f"✅ {praise}")

        st.markdown("### 💡 Customer Needs")
        for need in data.get("customer_needs", []):
            st.markdown(f"🎯 {need}")

    with col2:
        st.markdown("### 👎 Common Complaints")
        for complaint in data.get("common_complaints", []):
            st.markdown(f"❌ {complaint}")

        st.markdown("### 🔧 Quality Gaps")
        for gap in data.get("quality_gaps", []):
            st.markdown(f"⚠️ {gap}")

    st.markdown("---")
    st.markdown("### 🚀 Improvement Opportunities")
    for opp in data.get("improvement_opportunities", []):
        st.markdown(f"💡 {opp}")

    st.markdown("### 🔎 Key Insights")
    for insight in data.get("review_insights", []):
        st.markdown(f"🔍 {insight}")


def render_trends_tab(data: dict | None):
    if not data:
        st.warning("Trend data not available.")
        return

    st.markdown("## 📊 Market Trends")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Market Size</div>
            <div class="kpi-value" style="font-size:1.2rem;">
                {data.get('market_size', 'N/A')}
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Growth Rate</div>
            <div class="kpi-value" style="font-size:1.2rem;">
                {data.get('growth_rate', 'N/A')}
            </div>
        </div>""", unsafe_allow_html=True)

    keywords = data.get("trending_keywords", [])
    if keywords:
        st.markdown("### 🔑 Trending Keywords")
        kw_html = " ".join(
            f'<span style="background:#fff5f7;color:#e53e3e;padding:4px 10px;'
            f'border-radius:20px;margin:3px;display:inline-block;">{k}</span>'
            for k in keywords
        )
        st.markdown(kw_html, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📅 Seasonal Patterns")
        for pattern in data.get("seasonal_patterns", []):
            st.markdown(f"🗓️ {pattern}")

        st.markdown("### ⚠️ Market Threats")
        for threat in data.get("market_threats", []):
            st.markdown(f"🔴 {threat}")

    with col2:
        st.markdown("### 🚀 Emerging Trends")
        for trend in data.get("emerging_trends", []):
            st.markdown(f"📈 {trend}")

        st.markdown("### 🟢 Market Opportunities")
        for opp in data.get("market_opportunities", []):
            st.markdown(f"✅ {opp}")

    st.markdown("---")
    st.markdown("### 🔭 Future Outlook")
    st.info(data.get("future_outlook", ""))


def render_json_tab(result: dict):
    st.markdown("## 📋 Raw JSON Output")
    json_str = json.dumps(result, indent=2, default=str)
    st.download_button(
        label="⬇️ Download JSON",
        data=json_str,
        file_name=f"product_research_{result.get('run_id','')[:8]}.json",
        mime="application/json",
    )
    st.code(json_str, language="json")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_header()

    category, target_market, budget_range = render_input_form()

    st.markdown("---")

    if "research_result" not in st.session_state:
        st.session_state["research_result"] = None
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button(
            "🔍 Start Product Research",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.get("is_running", False) or not category.strip(),
        )

    if run_clicked and category.strip():
        st.session_state["is_running"] = True
        st.session_state["research_result"] = None

        with st.spinner(""):
            progress_bar, status_text = render_progress()

            try:
                response = requests.post(
                    f"{API_BASE}/api/research",
                    json={
                        "product_category": category,
                        "target_market": target_market,
                        "budget_range": budget_range,
                    },
                    timeout=300,
                )

                progress_bar.empty()
                status_text.empty()

                if response.status_code == 200:
                    data = response.json()
                    st.session_state["research_result"] = data.get("result", {})
                    st.session_state["is_running"] = False
                    st.rerun()
                else:
                    st.error(f"API Error {response.status_code}: {response.text[:300]}")
                    st.session_state["is_running"] = False

            except requests.exceptions.Timeout:
                st.error("⏰ Timed out. Research takes 90-120 seconds. Try again.")
                st.session_state["is_running"] = False
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {API_BASE}. Start FastAPI first!")
                st.session_state["is_running"] = False
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state["is_running"] = False

    if st.session_state.get("research_result"):
        render_results(st.session_state["research_result"])
    elif not run_clicked:
        st.markdown("---")
        st.markdown("### 🤖 How It Works")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
            **1️⃣ Enter a product category**
            Type any product you're interested in selling.
            Be specific for better results!
            """)
        with cols[1]:
            st.markdown("""
            **2️⃣ 5 agents research automatically**
            Amazon research, pricing, reviews,
            market trends — all automated!
            """)
        with cols[2]:
            st.markdown("""
            **3️⃣ Get your opportunity report**
            Opportunity score, recommended products,
            go-to-market strategy, action plan!
            """)


if __name__ == "__main__":
    main()