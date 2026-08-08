from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Human-readable transaction fields accepted by the API."""

    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., ge=0, le=10_000_000, description="Transaction amount")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction, 0-23")
    merchant_category: str = Field(
        ..., min_length=1, max_length=100, description="Merchant category"
    )
    foreign_transaction: int = Field(
        ..., ge=0, le=1, description="1 if foreign transaction, otherwise 0"
    )
    location_mismatch: int = Field(
        ..., ge=0, le=1, description="1 if transaction location mismatches normal location"
    )
    device_trust_score: float = Field(
        ..., ge=0, le=100, description="Device trust score from 0 to 100"
    )
    velocity_last_24h: int = Field(
        ..., ge=0, le=10000, description="Number of recent transactions"
    )
    cardholder_age: int = Field(..., ge=18, le=120, description="Cardholder age")


class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: Literal[0, 1]
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    latency_ms: float = Field(..., ge=0)


class TransactionRecord(PredictionResponse):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int
    created_at: str


class MetricsResponse(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_rate: float
    average_risk_score: float
    average_latency_ms: float
    model_metrics: dict
