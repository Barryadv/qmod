from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from qmod.dto import RunOutputDTO

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "company"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    yfticker: Mapped[str] = mapped_column(String(32), unique=True, index=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[Optional[str]] = mapped_column(DateTime, index=True)  # naive UTC
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    signal: Mapped[str] = mapped_column(String(8))
    score: Mapped[float] = mapped_column(Float)
    chart_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    company_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("date", "ticker", "model_version", name="uq_signal_key"),
    )


def init_engine_and_session(db_url: str) -> Session:
    """
    Initialize engine, create tables if missing, and return a Session.
    """
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return Session(engine, future=True)


def get_or_create_company(session: Session, yfticker: str) -> int:
    """
    Return company.id for yfticker, creating if necessary.
    """
    yfticker = yfticker.upper().strip()
    stmt = select(Company).where(Company.yfticker == yfticker)
    obj = session.scalar(stmt)
    if obj:
        return obj.id
    obj = Company(yfticker=yfticker)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj.id


def upsert_signal(session: Session, run_output: RunOutputDTO, company_id: int | None) -> int:
    """
    Upsert signals row on (date, ticker, model_version). Returns row id.
    """
    key_stmt = select(Signal).where(
        Signal.date == run_output.date,
        Signal.ticker == run_output.ticker,
        Signal.model_version == run_output.model_version,
    )
    row = session.scalar(key_stmt)

    if row is None:
        row = Signal(
            date=run_output.date,
            ticker=run_output.ticker,
            signal=run_output.signal,
            score=float(run_output.score),
            chart_base64=run_output.chart_base64,
            explanation=run_output.explanation,
            model_version=run_output.model_version,
            company_id=company_id,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        logger.info("Inserted signal id=%s %s %s", row.id, row.ticker, row.date)
        return row.id

    # Update existing
    row.signal = run_output.signal
    row.score = float(run_output.score)
    row.chart_base64 = run_output.chart_base64
    row.explanation = run_output.explanation
    row.company_id = company_id
    session.commit()
    logger.info("Updated signal id=%s %s %s", row.id, row.ticker, row.date)
    return row.id
