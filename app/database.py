from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class TransactionPrediction(Base):
    __tablename__ = "transaction_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    amount: Mapped[float] = mapped_column(Float, nullable=False)
    transaction_hour: Mapped[int] = mapped_column(Integer, nullable=False)
    merchant_category: Mapped[str] = mapped_column(String(100), nullable=False)
    foreign_transaction: Mapped[bool] = mapped_column(Boolean, nullable=False)
    location_mismatch: Mapped[bool] = mapped_column(Boolean, nullable=False)
    device_trust_score: Mapped[float] = mapped_column(Float, nullable=False)
    velocity_last_24h: Mapped[int] = mapped_column(Integer, nullable=False)
    cardholder_age: Mapped[int] = mapped_column(Integer, nullable=False)

    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
