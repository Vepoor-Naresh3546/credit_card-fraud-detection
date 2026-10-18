"""Train and evaluate the first fraud classifier; optionally log to MLflow."""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
FEATURES = [
    "amount",
    "hour",
    "transactions_last_1h",
    "avg_amount_24h",
    "amount_to_avg_ratio",
    "is_international",
    "new_device",
    "merchant_risk_score",
]


def best_f1_threshold(y_true, probabilities):
    # Tune on validation data only; do not tune on the final test set.
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric_report(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, predictions, labels=[0, 1]).ravel().tolist(),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python generate_data.py` before training.")

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["is_fraud"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Missing values found in model features or target.")

    X = df[FEATURES]
    y = df["is_fraud"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Target must contain both classes: 0=legitimate, 1=fraud.")

    # 70% train, 15% validation (threshold choice), 15% untouched test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_probabilities)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    metrics = metric_report(y_test, test_probabilities, threshold)
    metrics["train_rows"] = int(len(X_train))
    metrics["validation_rows"] = int(len(X_val))
    metrics["test_rows"] = int(len(X_test))
    metrics["test_fraud_rate_percent"] = round(float(y_test.mean() * 100), 3)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_version": "0.1.0",
        "feature_columns": FEATURES,
        "threshold": threshold,
        "training_data": "synthetic_rule_generated_demo",
        "warning": "Synthetic evaluation is not evidence of real-world fraud performance.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # MLflow is optional for the first local run. It is listed in requirements.txt.
    try:
        import mlflow
        mlflow.set_experiment("fraud-detection-capstone")
        with mlflow.start_run():
            mlflow.log_params({
                "model": "RandomForestClassifier",
                "n_estimators": 180,
                "max_depth": 14,
                "class_weight": "balanced_subsample",
                "threshold_selection": "maximum validation F1",
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(
    sk_model=model,
    name="fraud_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
            mlflow.log_artifact(str(METADATA_PATH))
            mlflow.log_artifact(str(METRICS_PATH))
        print("MLflow run logged. Start UI with: mlflow ui --port 5000")
    except ImportError:
        warnings.warn("MLflow is not installed; model and metrics were still saved.")
    except Exception as exc:
        warnings.warn(f"MLflow tracking did not complete ({exc}); local artifacts were saved.")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Test metrics saved to: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
