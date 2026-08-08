from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from .config import API_TITLE, API_VERSION, METADATA_PATH
from .database import TransactionPrediction, get_db, init_db
from .model import fraud_model
from .schemas import (
    MetricsResponse,
    PredictionResponse,
    TransactionRecord,
    TransactionRequest,
)
import joblib


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Real-time AI fraud prediction API using the trained XGBoost model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def root():
    return {
        "message": "AI Fraud Detection API is running",
        "docs": "/docs",
        "model": "XGBoost",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "XGBoost",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: TransactionRequest,
    db: Session = Depends(get_db),
):
    start = time.perf_counter()

    try:
        result = fraud_model.predict(request.model_dump())
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        record = TransactionPrediction(
            amount=request.amount,
            transaction_hour=request.transaction_hour,
            merchant_category=request.merchant_category,
            foreign_transaction=bool(request.foreign_transaction),
            location_mismatch=bool(request.location_mismatch),
            device_trust_score=request.device_trust_score,
            velocity_last_24h=request.velocity_last_24h,
            cardholder_age=request.cardholder_age,
            prediction=result["prediction"],
            fraud_probability=result["fraud_probability"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "transaction_id": record.id,
            **result,
            "latency_ms": latency_ms,
        }

    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc


@app.get("/transactions", response_model=list[TransactionRecord])
def transactions(
    limit: int = Query(100, ge=1, le=5000),
    fraud_only: bool = False,
    db: Session = Depends(get_db),
):
    query = db.query(TransactionPrediction)

    if fraud_only:
        query = query.filter(TransactionPrediction.prediction == 1)

    rows = query.order_by(desc(TransactionPrediction.created_at)).limit(limit).all()

    return [
        {
            "transaction_id": row.id,
            "amount": row.amount,
            "transaction_hour": row.transaction_hour,
            "merchant_category": row.merchant_category,
            "foreign_transaction": int(row.foreign_transaction),
            "location_mismatch": int(row.location_mismatch),
            "device_trust_score": row.device_trust_score,
            "velocity_last_24h": row.velocity_last_24h,
            "cardholder_age": row.cardholder_age,
            "prediction": row.prediction,
            "fraud_probability": row.fraud_probability,
            "risk_score": row.risk_score,
            "risk_level": row.risk_level,
            "latency_ms": row.latency_ms,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@app.get("/metrics", response_model=MetricsResponse)
def metrics(db: Session = Depends(get_db)):
    total = db.query(func.count(TransactionPrediction.id)).scalar() or 0
    fraud = (
        db.query(func.count(TransactionPrediction.id))
        .filter(TransactionPrediction.prediction == 1)
        .scalar()
        or 0
    )

    avg_risk = db.query(func.avg(TransactionPrediction.risk_score)).scalar()
    avg_latency = db.query(func.avg(TransactionPrediction.latency_ms)).scalar()

    metadata = joblib.load(METADATA_PATH)
    model_metrics = metadata.get("metrics", {})

    return {
        "total_transactions": int(total),
        "fraud_transactions": int(fraud),
        "fraud_rate": round((fraud / total * 100) if total else 0, 2),
        "average_risk_score": round(float(avg_risk or 0), 2),
        "average_latency_ms": round(float(avg_latency or 0), 3),
        "model_metrics": model_metrics,
    }
