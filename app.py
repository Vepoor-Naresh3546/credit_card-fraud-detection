"""FastAPI inference service for the first local fraud-detection milestone."""
from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone
import json
import os
import sqlite3
import time

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DB_PATH = ROOT / "data" / "prediction_audit.sqlite3"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError("Model files not found. Run `python generate_data.py` and `python train.py` first.")

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
threshold = float(metadata["threshold"])
features = metadata["feature_columns"]

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Educational capstone API. Demo data and local credentials only; not production-ready.",
    version=metadata.get("model_version", "0.1.0"),
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
FRAUD_API_KEY = os.getenv("FRAUD_API_KEY", "dev-key-change-me")
PREDICTIONS = Counter("fraud_prediction_total", "Number of fraud decisions", ["decision"])
LATENCY = Histogram("fraud_inference_latency_seconds", "Time spent handling prediction requests")
app.mount("/metrics", make_asgi_app())


class Transaction(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    transactions_last_1h: int = Field(ge=0, le=1000)
    avg_amount_24h: float = Field(gt=0, le=1_000_000)
    amount_to_avg_ratio: float = Field(ge=0, le=1_000_000)
    is_international: int = Field(ge=0, le=1)
    new_device: int = Field(ge=0, le=1)
    merchant_risk_score: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    threshold: float
    decision: str
    model_version: str


def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    if not api_key or api_key != FRAUD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send X-API-Key header.",
        )
    return api_key


def initialize_audit_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                decision TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
        """)
        connection.commit()


@app.on_event("startup")
def startup_event():
    initialize_audit_db()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata.get("model_version", "unknown")}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, _: str = Depends(verify_api_key)):
    started = time.perf_counter()
    decision = "error"
    try:
        row = transaction.model_dump()
        model_input = pd.DataFrame([{name: row[name] for name in features}], columns=features)
        probability = float(model.predict_proba(model_input)[0, 1])
        decision = "fraud" if probability >= threshold else "legitimate"
        PREDICTIONS.labels(decision=decision).inc()

        # Keep a minimal prediction audit trail; do not store the entire input payload.
        with closing(sqlite3.connect(DB_PATH)) as connection:
            connection.execute(
                "INSERT INTO predictions (transaction_id, created_at, fraud_probability, decision, model_version) VALUES (?, ?, ?, ?, ?)",
                (
                    transaction.transaction_id,
                    datetime.now(timezone.utc).isoformat(),
                    probability,
                    decision,
                    metadata.get("model_version", "unknown"),
                ),
            )
            connection.commit()

        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(probability, 6),
            threshold=round(threshold, 4),
            decision=decision,
            model_version=metadata.get("model_version", "unknown"),
        )
    finally:
        LATENCY.observe(time.perf_counter() - started)
