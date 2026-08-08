from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, FEATURES_PATH, METADATA_PATH


class FraudModel:
    """Loads the trained XGBoost artifact and performs production inference."""

    REQUIRED_USER_FEATURES = [
        "amount",
        "transaction_hour",
        "merchant_category",
        "foreign_transaction",
        "location_mismatch",
        "device_trust_score",
        "velocity_last_24h",
        "cardholder_age",
    ]

    def __init__(self) -> None:
        for path in (MODEL_PATH, FEATURES_PATH, METADATA_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(f"Required model artifact not found: {path}")

        artifact: Dict[str, Any] = joblib.load(MODEL_PATH)
        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["model"]
        self.feature_names = list(
            joblib.load(FEATURES_PATH)
        )
        self.metadata = joblib.load(METADATA_PATH)

        # These thresholds were learned from the training dataset.
        self.amount_threshold = self._extract_threshold(
            self.metadata.get("feature_engineering", {}),
            "high_value_transaction",
            default=530.2085,
        )
        self.velocity_threshold = self._extract_threshold(
            self.metadata.get("feature_engineering", {}),
            "high_velocity",
            default=4.0,
        )

        self.prediction_threshold = float(
            artifact.get("threshold", self.metadata.get("threshold", 0.50))
        )

        risk_thresholds = artifact.get(
            "risk_thresholds",
            {"LOW_MAX": 0.30, "MEDIUM_MAX": 0.70, "HIGH_MIN": 0.70},
        )
        self.low_max = float(risk_thresholds.get("LOW_MAX", 0.30))
        self.medium_max = float(risk_thresholds.get("MEDIUM_MAX", 0.70))

    @staticmethod
    def _extract_threshold(
        feature_engineering: Dict[str, Any],
        feature_name: str,
        default: float,
    ) -> float:
        value = feature_engineering.get(feature_name)
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Handles strings such as "amount >= 530.2085".
            try:
                return float(value.split(">=")[-1].strip())
            except (ValueError, IndexError):
                pass

        return default

    def _engineer_features(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        row = {
            key: transaction[key]
            for key in self.REQUIRED_USER_FEATURES
        }
        df = pd.DataFrame([row])

        df["log_amount"] = np.log1p(df["amount"])
        df["high_value_transaction"] = (
            df["amount"] >= self.amount_threshold
        ).astype(int)
        df["low_device_trust"] = (
            df["device_trust_score"] < 40
        ).astype(int)
        df["high_velocity"] = (
            df["velocity_last_24h"] >= self.velocity_threshold
        ).astype(int)

        # Exact feature order used during model training.
        return df[self.feature_names]

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        df = self._engineer_features(transaction)
        encoded = self.preprocessor.transform(df)

        probability = float(self.model.predict_proba(encoded)[0, 1])
        prediction = int(probability >= self.prediction_threshold)
        risk_score = round(probability * 100.0, 2)

        if probability < self.low_max:
            risk_level = "LOW"
        elif probability < self.medium_max:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "prediction": prediction,
            "fraud_probability": round(probability, 6),
            "risk_score": risk_score,
            "risk_level": risk_level,
        }


fraud_model = FraudModel()
