from __future__ import annotations

import argparse
from collections import Counter
import os
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_platform.config import ROOT
from fraud_platform.quality import validate_dataframe

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
REQUIRED_COLUMNS = [*FEATURES, "Class"]
DATASET = ROOT / "data" / "creditcard.csv"
REPORT_DIR = ROOT / "reports" / "pipeline"
GATE_LIMITS = {
    "f1": float(os.getenv("MIN_MODEL_F1", "0.65")),
    "recall": float(os.getenv("MIN_MODEL_RECALL", "0.70")),
    "precision": float(os.getenv("MIN_MODEL_PRECISION", "0.60")),
}


def validate_dataset(path: Path = DATASET) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    counts: Counter[int] = Counter()
    row_count = 0
    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=50_000)):
        missing = sorted(set(REQUIRED_COLUMNS) - set(chunk.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if list(chunk.columns) != REQUIRED_COLUMNS:
            raise ValueError("Dataset columns or order differ from the real-model contract")
        validate_dataframe(chunk, str(chunk_number))
        if chunk[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError("Dataset contains null model inputs or labels")
        labels = set(chunk["Class"].unique())
        if not labels.issubset({0, 1}):
            raise ValueError(f"Unexpected Class labels: {sorted(labels)}")
        counts.update(chunk["Class"].astype(int).value_counts().to_dict())
        row_count += len(chunk)
    if not row_count or set(counts) != {0, 1}:
        raise ValueError("Dataset must contain both classes and at least one row")
    return {
        "rows": row_count,
        "fraud_count": counts[1],
        "legitimate_count": counts[0],
        "feature_count": len(FEATURES),
    }


def prepare_manifest() -> dict[str, Any]:
    report = validate_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "data_manifest.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_features() -> dict[str, Any]:
    metadata_path = ROOT / "models" / "creditcard_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != FEATURES:
        raise ValueError("Model metadata feature order differs from the training contract")
    return {"feature_count": len(FEATURES), "model_version": metadata["model_version"]}


def evaluate_model_metrics(metrics: dict[str, Any], limits: dict[str, float] = GATE_LIMITS) -> bool:
    return all(float(metrics.get(metric, 0.0)) >= limit for metric, limit in limits.items())


def evaluation_gate() -> dict[str, Any]:
    pointer_path = ROOT / "reports" / "candidates" / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    candidate_dir = Path(pointer["candidate_dir"])
    if not candidate_dir.is_absolute():
        candidate_dir = ROOT / candidate_dir
    metrics_path = candidate_dir / "creditcard_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = {
        "model_version": pointer["model_version"],
        "passed": evaluate_model_metrics(metrics),
        "required_metrics": GATE_LIMITS,
        "observed_metrics": {
            name: metrics.get(name) for name in GATE_LIMITS
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "evaluation_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Candidate failed the evaluation gate: {result['observed_metrics']}")
    return result


def record_candidate_in_mlflow() -> dict[str, str]:
    pointer_path = ROOT / "reports" / "candidates" / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    candidate_dir = Path(pointer["candidate_dir"])
    if not candidate_dir.is_absolute():
        candidate_dir = ROOT / candidate_dir
    metadata_path = candidate_dir / "creditcard_model_metadata.json"
    metrics_path = candidate_dir / "creditcard_metrics.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    from train_real import log_candidate_to_mlflow

    log_candidate_to_mlflow(
        joblib.load(candidate_dir / "creditcard_model.joblib"),
        metadata,
        metrics,
        pointer["model_version"],
        metadata_path,
        metrics_path,
        run_name="airflow-candidate-artifact-recording",
    )
    return {"model_version": pointer["model_version"], "tracking": "recorded"}


def aggregate_analytics(path: Path = DATASET) -> dict[str, Any]:
    transaction_count = 0
    fraud_count = 0
    amount_sum = 0.0
    for chunk in pd.read_csv(path, usecols=["Amount", "Class"], chunksize=50_000):
        transaction_count += len(chunk)
        fraud_count += int(chunk["Class"].sum())
        amount_sum += float(chunk["Amount"].sum())
    result = {
        "transactions": transaction_count,
        "fraud_transactions": fraud_count,
        "fraud_rate": fraud_count / transaction_count if transaction_count else 0,
        "mean_amount": amount_sum / transaction_count if transaction_count else 0,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "dataset_analytics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable fraud training pipeline tasks")
    parser.add_argument(
        "task",
        choices=["validate", "prepare", "features", "gate", "track", "analytics"],
    )
    args = parser.parse_args()
    functions = {
        "validate": validate_dataset,
        "prepare": prepare_manifest,
        "features": validate_features,
        "gate": evaluation_gate,
        "track": record_candidate_in_mlflow,
        "analytics": aggregate_analytics,
    }
    print(json.dumps(functions[args.task](), indent=2))


if __name__ == "__main__":
    main()
