from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "fraud_detection_xgboost.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "fraud_model_features.pkl"
METADATA_PATH = ARTIFACTS_DIR / "fraud_model_metadata.pkl"

# Default is SQLite so the project runs immediately.
# For PostgreSQL, set:
# DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/fraud_db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'fraud_predictions.db'}"
)

API_TITLE = "AI Fraud Detection API"
API_VERSION = "1.0.0"
PREDICTION_THRESHOLD = 0.50
