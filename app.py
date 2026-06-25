"""
app.py
───────
FastAPI backend for E-commerce Product Research Agent System.
"""

import os
import uuid
import traceback
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from loguru import logger
from dotenv import load_dotenv

from database.db import init_db, get_db, create_run, update_run_completed, update_run_failed, list_runs, get_run
from workflows.graph import run_product_research

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting E-commerce Product Research AI System...")
    init_db()
    logger.info("Database initialized. Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="E-commerce Product Research AI Agent",
    description="Multi-Agent AI system for e-commerce product research using LangGraph + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    product_category: str = Field(..., min_length=3, max_length=200)
    target_market: str = Field(default="Global")
    budget_range: str = Field(default="Any")

    model_config = {"json_schema_extra": {
        "example": {
            "product_category": "Wireless Earbuds",
            "target_market": "US",
            "budget_range": "$20-100",
        }
    }}


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    message: str
    result: dict | None = None
    error: str | None = None


class RunSummary(BaseModel):
    run_id: str
    product_category: str
    target_market: str
    status: str
    created_at: str
    completed_at: str | None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "E-commerce Product Research AI Agent",
        "version": "1.0.0",
        "agents": ["supervisor", "amazon_research", "price_analysis", "review_analysis", "trend", "opportunity"],
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY")),
        "serper_api_configured": bool(os.getenv("SERPER_API_KEY")),
    }


@app.post("/api/research", response_model=ResearchResponse)
def research(request: ResearchRequest, db: Session = Depends(get_db)):
    """Trigger a full product research analysis."""
    run_id = str(uuid.uuid4())
    logger.info(f"[API] New research | category={request.product_category!r}")

    create_run(
        db,
        product_category=request.product_category,
        target_market=request.target_market,
        budget_range=request.budget_range,
    )

    try:
        final_state = run_product_research(
            product_category=request.product_category,
            target_market=request.target_market,
            budget_range=request.budget_range,
            run_id=run_id,
        )

        # Update DB
        runs = list_runs(db, limit=5)
        for r in runs:
            if r.status == "running" and r.product_category == request.product_category:
                update_run_completed(
                    db,
                    run_id=r.run_id,
                    result_json=final_state,
                    supervisor_plan=final_state.get("supervisor_plan", ""),
                )
                run_id = r.run_id
                break

        return ResearchResponse(
            run_id=run_id,
            status="completed",
            message="Research completed successfully.",
            result=final_state,
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"[API] Research failed: {error_msg}")

        runs = list_runs(db, limit=5)
        for r in runs:
            if r.status == "running":
                update_run_failed(db, run_id=r.run_id, error=error_msg)
                break

        raise HTTPException(status_code=500, detail=f"Research failed: {error_msg}")


@app.get("/api/runs", response_model=list[RunSummary])
def list_research_runs(limit: int = 20, db: Session = Depends(get_db)):
    runs = list_runs(db, limit=limit)
    return [
        RunSummary(
            run_id=r.run_id,
            product_category=r.product_category,
            target_market=r.target_market or "Global",
            status=r.status,
            created_at=r.created_at,
            completed_at=r.completed_at,
        )
        for r in runs
    ]


@app.get("/api/runs/{run_id}", response_model=ResearchResponse)
def get_research_run(run_id: str, db: Session = Depends(get_db)):
    import json
    run = get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    result = None
    if run.result_json:
        try:
            result = json.loads(run.result_json)
        except Exception:
            result = None

    return ResearchResponse(
        run_id=run.run_id,
        status=run.status,
        message=f"Run status: {run.status}",
        result=result,
        error=run.error_log,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
        log_level="info",
    )
