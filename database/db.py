"""
database/db.py
──────────────
SQLAlchemy SQLite persistence for product research runs.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./product_research.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False)
    product_category = Column(String(200), nullable=False)
    target_market = Column(String(100), nullable=True)
    budget_range = Column(String(100), nullable=True)
    status = Column(String(20), default="running")
    result_json = Column(Text, nullable=True)
    supervisor_plan = Column(Text, nullable=True)
    error_log = Column(Text, nullable=True)
    created_at = Column(String(32), default=lambda: datetime.utcnow().isoformat())
    completed_at = Column(String(32), nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Database initialized.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_run(db: Session, *, product_category: str, target_market: str, budget_range: str) -> str:
    run_id = str(uuid.uuid4())
    run = ResearchRun(
        run_id=run_id,
        product_category=product_category,
        target_market=target_market,
        budget_range=budget_range,
        status="running",
    )
    db.add(run)
    db.commit()
    return run_id


def update_run_completed(db: Session, run_id: str, *, result_json: dict, supervisor_plan: str = "") -> None:
    run = db.query(ResearchRun).filter(ResearchRun.run_id == run_id).first()
    if not run:
        return
    run.status = "completed"
    run.result_json = json.dumps(result_json, ensure_ascii=False)
    run.supervisor_plan = supervisor_plan
    run.completed_at = datetime.utcnow().isoformat()
    db.commit()


def update_run_failed(db: Session, run_id: str, error: str) -> None:
    run = db.query(ResearchRun).filter(ResearchRun.run_id == run_id).first()
    if not run:
        return
    run.status = "failed"
    run.error_log = error
    run.completed_at = datetime.utcnow().isoformat()
    db.commit()


def get_run(db: Session, run_id: str) -> Optional[ResearchRun]:
    return db.query(ResearchRun).filter(ResearchRun.run_id == run_id).first()


def list_runs(db: Session, limit: int = 20) -> list[ResearchRun]:
    return db.query(ResearchRun).order_by(ResearchRun.id.desc()).limit(limit).all()
