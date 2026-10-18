"""Train a baseline fraud classifier on Kaggle's creditcard.csv dataset.

Separate artifact names are used so the existing synthetic-data API remains intact.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import os
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "creditcard_model.joblib"
METADATA_PATH = MODEL_DIR / "creditcard_model_metadata.json"
METRICS_PATH = REPORTS_DIR / "creditcard_metrics.json"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"
MINIMUM_METRICS = {
    "f1": float(os.getenv("MIN_MODEL_F1", "0.65")),
    "recall": float(os.getenv("MIN_MODEL_RECALL", "0.70")),
    "precision": float(os.getenv("MIN_MODEL_PRECISION", "0.60")),
}


def passes_evaluation_gate(metrics: dict) -> bool:
    return all(float(metrics.get(name, 0.0)) >= minimum for name, minimum in MINIMUM_METRICS.items())


def _backup_and_promote(
    candidate_model: Path,
    candidate_metadata: Path,
    candidate_metrics: Path,
    model_version: str,
) -> None:
    backup_dir = ROOT / "models" / "backups" / model_version
    backup_dir.mkdir(parents=True, exist_ok=False)
    for current in (MODEL_PATH, METADATA_PATH, METRICS_PATH):
        if current.is_file():
            shutil.copy2(current, backup_dir / current.name)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (candidate_model, MODEL_PATH),
        (candidate_metadata, METADATA_PATH),
        (candidate_metrics, METRICS_PATH),
    ):
        temporary = destination.with_name(f".{destination.name}.{model_version}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a classification threshold using validation predictions only."""
    candidates = np.linspace(0.01, 0.99, 99)
    scores = [
        f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 6),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "test_rows": int(len(y_true)),
        "test_fraud_count": int(y_true.sum()),
        "test_fraud_rate_percent": round(float(y_true.mean() * 100), 6),
    }


def log_candidate_to_mlflow(
    pipeline: Pipeline,
    metadata: dict,
    metrics: dict,
    model_version: str,
    metadata_path: Path,
    metrics_path: Path,
    run_name: str = "logistic-regression-real-data",
) -> None:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    )
    mlflow.set_experiment("creditcard-fraud-baseline")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": metadata.get("model_type", "LogisticRegression"),
            "imbalance_strategy": metadata.get(
                "imbalance_strategy",
                "LogisticRegression class_weight=balanced",
            ),
            "preprocessing": metadata.get("preprocessing", "StandardScaler"),
            "threshold_selection": "maximum validation F1",
            "split": "70/15/15 stratified train/validation/test",
            "feature_count": len(FEATURES),
            "dataset": metadata["dataset"],
            "candidate_model_version": model_version,
        })
        for name, value in metadata.get("hyperparameters", {}).items():
            mlflow.log_param(name, value)
        tracked_metrics = {
            key: float(value) for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        tracked_metrics.update(
            {
                name: float(value)
                for name, value in zip(
                    ("true_negatives", "false_positives", "false_negatives", "true_positives"),
                    metrics["confusion_matrix_tn_fp_fn_tp"],
                    strict=True,
                )
            }
        )
        mlflow.log_metrics(tracked_metrics)
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(metrics_path))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="creditcard_model",
            registered_model_name=(
                os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "CreditCardFraud")
                if passes_evaluation_gate(metrics)
                else None
            ),
        )
        print(f"MLflow model artifact logged: {model_info.model_uri}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put creditcard.csv in the project's data folder."
        )

    print("Loading Kaggle credit card dataset...")
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    if df.isnull().any().any():
        raise ValueError("Missing values found. Inspect and clean the dataset before training.")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected Class values 0 (legitimate) and 1 (fraud).")

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    print(f"Rows: {len(df):,} | Features: {len(FEATURES)}")
    print(f"Fraud rows: {int(y.sum()):,} ({y.mean() * 100:.4f}%)")
    print("Splitting data into stratified train/validation/test partitions (70/15/15)...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=500, random_state=42
        )),
    ])

    print("Training Logistic Regression baseline (this may take a little while)...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
        for warning in caught:
            print(f"Training warning: {warning.message}")

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    threshold = choose_f1_threshold(y_val, val_probabilities)
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_probabilities, threshold)
    metrics.update({
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "features_count": int(len(FEATURES)),
        "dataset": "Kaggle Credit Card Fraud Detection (creditcard.csv)",
        "model": "StandardScaler + LogisticRegression(class_weight=balanced)",
    })

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    model_version = f"creditcard-{run_timestamp}"
    candidate_dir = MODEL_DIR / "candidates" / model_version
    candidate_dir.mkdir(parents=True, exist_ok=False)
    candidate_model_path = candidate_dir / MODEL_PATH.name
    candidate_metadata_path = candidate_dir / METADATA_PATH.name
    candidate_metrics_path = candidate_dir / METRICS_PATH.name
    joblib.dump(pipeline, candidate_model_path)
    metadata = {
        "model_version": model_version,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "threshold": threshold,
        "dataset": "Kaggle Credit Card Fraud Detection (creditcard.csv)",
        "model_type": "StandardScaler + LogisticRegression(class_weight=balanced)",
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "evaluation_gate": MINIMUM_METRICS,
        "warning": "Evaluate on operational data before making production claims.",
    }
    candidate_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    candidate_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    latest_pointer = REPORTS_DIR / "candidates" / "latest.json"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(
        json.dumps({"model_version": model_version, "candidate_dir": str(candidate_dir)}, indent=2),
        encoding="utf-8",
    )

    if os.getenv("MLFLOW_LOG_DURING_TRAINING", "true").lower() == "true":
        try:
            log_candidate_to_mlflow(
                pipeline,
                metadata,
                metrics,
                model_version,
                candidate_metadata_path,
                candidate_metrics_path,
            )
            print("MLflow run logged in experiment: creditcard-fraud-baseline")
        except Exception as exc:
            warnings.warn(f"MLflow tracking did not complete ({exc}); local candidate was saved.")
    else:
        print("MLflow tracking deferred to the orchestration task.")

    if not passes_evaluation_gate(metrics):
        print(f"Evaluation gate failed: required metrics are {MINIMUM_METRICS}")
    elif os.getenv("PROMOTE_MODEL", "false").lower() == "true":
        _backup_and_promote(
            candidate_model_path,
            candidate_metadata_path,
            candidate_metrics_path,
            model_version,
        )
        print(f"Promoted {model_version}; previous artifacts backed up under models/backups/")
    else:
        print("Evaluation gate passed; candidate remains unpromoted. Set PROMOTE_MODEL=true to promote.")

    print("\nTraining complete.")
    print(f"Candidate model:    {candidate_model_path}")
    print(f"Candidate metadata: {candidate_metadata_path}")
    print(f"Candidate metrics:  {candidate_metrics_path}")
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nThe existing app.py still serves the separate synthetic-data demo.")


if __name__ == "__main__":
    main()
