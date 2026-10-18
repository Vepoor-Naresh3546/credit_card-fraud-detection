from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fraud_platform.config import ROOT
from train_real import (
    FEATURES,
    MINIMUM_METRICS,
    _backup_and_promote,
    choose_f1_threshold,
    evaluate,
    log_candidate_to_mlflow,
    passes_evaluation_gate,
)


def _random_forest() -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        ]
    )


def _xgboost(params: dict[str, Any], scale_pos_weight: float) -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline(
        [
            (
                "model",
                XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        ]
    )


def _save_candidate(
    model: Pipeline,
    model_name: str,
    threshold: float,
    metrics: dict[str, Any],
    training_rows: int,
    validation_rows: int,
    baseline: list[float],
    hyperparameters: dict[str, Any],
) -> tuple[str, Path]:
    version = (
        f"{model_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    candidate_dir = ROOT / "models" / "candidates" / version
    candidate_dir.mkdir(parents=True, exist_ok=False)
    model_path = candidate_dir / "creditcard_model.joblib"
    metadata_path = candidate_dir / "creditcard_model_metadata.json"
    metrics_path = candidate_dir / "creditcard_metrics.json"
    joblib.dump(model, model_path)
    metadata = {
        "model_version": version,
        "feature_columns": FEATURES,
        "target_column": "Class",
        "threshold": threshold,
        "dataset": "Kaggle Credit Card Fraud Detection (creditcard.csv)",
        "model_type": model_name,
        "preprocessing": "raw numeric model features",
        "hyperparameters": hyperparameters,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "test_rows": metrics["test_rows"],
        "random_state": 42,
        "imbalance_strategy": (
            "RandomForest class_weight=balanced_subsample"
            if model_name == "RandomForest"
            else "XGBoost scale_pos_weight derived from training labels"
        ),
        "explanation_baseline": baseline,
        "evaluation_gate": MINIMUM_METRICS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_candidate_to_mlflow(
        model,
        metadata,
        metrics,
        version,
        metadata_path,
        metrics_path,
        run_name=f"{model_name}-candidate",
    )
    return version, candidate_dir


def run_benchmarks(
    dataset_path: Path,
    *,
    optuna_trials: int = 10,
    include_xgboost: bool = True,
    promote: bool = False,
) -> list[dict[str, Any]]:
    if optuna_trials < 1:
        raise ValueError("optuna_trials must be at least 1")
    frame = pd.read_csv(dataset_path, usecols=[*FEATURES, "Class"])
    if frame.isna().any().any() or set(frame["Class"].unique()) != {0, 1}:
        raise ValueError("Training data must have complete features and binary labels")

    X = frame[FEATURES]
    y = frame["Class"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    baseline = X_train.mean().astype(float).tolist()
    candidates: list[tuple[Pipeline, str, float, dict[str, Any]]] = []

    forest = _random_forest()
    forest.fit(X_train, y_train)
    forest_threshold = choose_f1_threshold(
        y_val, forest.predict_proba(X_val)[:, 1]
    )
    candidates.append((forest, "RandomForest", forest_threshold, {}))

    if include_xgboost:
        import optuna

        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.3, log=True
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 12.0
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            estimator = _xgboost(params, scale_pos_weight)
            estimator.fit(X_train, y_train)
            probabilities = estimator.predict_proba(X_val)[:, 1]
            threshold = choose_f1_threshold(y_val, probabilities)
            return float(
                evaluate(y_val, probabilities, threshold)["f1"]
            )

        study.optimize(objective, n_trials=optuna_trials)
        best_model = _xgboost(study.best_params, scale_pos_weight)
        best_model.fit(X_train, y_train)
        probabilities = best_model.predict_proba(X_val)[:, 1]
        best_threshold = choose_f1_threshold(y_val, probabilities)
        candidates.append(
            (best_model, "XGBoost-Optuna", best_threshold, study.best_params)
        )

    results: list[dict[str, Any]] = []
    for model, name, threshold, params in candidates:
        test_probabilities = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, test_probabilities, threshold)
        metrics.update(
            {
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_val)),
                "features_count": len(FEATURES),
                "dataset": "Kaggle Credit Card Fraud Detection (creditcard.csv)",
                "model": name,
            }
        )
        version, candidate_dir = _save_candidate(
            model,
            name,
            threshold,
            metrics,
            len(X_train),
            len(X_val),
            baseline,
            params,
        )
        results.append(
            {
                "model_version": version,
                "candidate_dir": str(candidate_dir),
                "model": name,
                "hyperparameters": params,
                "metrics": metrics,
                "passed_evaluation_gate": passes_evaluation_gate(metrics),
            }
        )

    selected = max(
        results,
        key=lambda result: (
            result["passed_evaluation_gate"],
            result["metrics"]["f1"],
        ),
    )
    latest_path = ROOT / "reports" / "candidates" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "model_version": selected["model_version"],
                "candidate_dir": selected["candidate_dir"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if promote:
        if not selected["passed_evaluation_gate"]:
            raise RuntimeError("No advanced candidate passed all promotion gates.")
        selected_dir = Path(selected["candidate_dir"])
        _backup_and_promote(
            selected_dir / "creditcard_model.joblib",
            selected_dir / "creditcard_model_metadata.json",
            selected_dir / "creditcard_metrics.json",
            selected["model_version"],
        )
        selected["promoted"] = True
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and Optuna-tuned XGBoost candidates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "creditcard.csv",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the best candidate only if all configured evaluation gates pass.",
    )
    args = parser.parse_args()
    results = run_benchmarks(
        args.input,
        optuna_trials=args.trials,
        include_xgboost=not args.skip_xgboost,
        promote=args.promote,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
