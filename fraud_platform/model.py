from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class FraudModel:
    estimator: Any
    metadata: dict[str, Any]
    feature_columns: tuple[str, ...]
    threshold: float
    model_version: str

    @classmethod
    def load(
        cls,
        model_path: Path,
        metadata_path: Path,
        threshold_override: float | None = None,
    ) -> "FraudModel":
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = tuple(metadata.get("feature_columns", ()))
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("Model metadata must declare unique feature_columns")

        estimator = joblib.load(model_path)
        if not callable(getattr(estimator, "predict_proba", None)):
            raise TypeError("The configured model must implement predict_proba")
        expected_count = getattr(estimator, "n_features_in_", len(feature_columns))
        if expected_count != len(feature_columns):
            raise ValueError(
                f"Model expects {expected_count} features; metadata lists {len(feature_columns)}"
            )
        fitted_features = getattr(estimator, "feature_names_in_", None)
        if fitted_features is not None and tuple(fitted_features) != feature_columns:
            raise ValueError("Model feature order does not match model metadata")

        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(metadata["threshold"])
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Classification threshold must be between 0 and 1")
        return cls(
            estimator=estimator,
            metadata=metadata,
            feature_columns=feature_columns,
            threshold=threshold,
            model_version=str(metadata.get("model_version", "unknown")),
        )

    def predict(self, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(rows, columns=self.feature_columns)
        probabilities = self.estimator.predict_proba(frame)[:, 1]
        return [
            {
                "fraud_probability": float(probability),
                "decision": "fraud" if probability >= self.threshold else "legitimate",
            }
            for probability in probabilities
        ]

    def explain(self, row: dict[str, float]) -> dict[str, Any]:
        try:
            import numpy as np
            import shap
        except ImportError as exc:
            raise RuntimeError(
                "SHAP explanations require the optional requirements-advanced-ml.txt "
                "dependencies."
            ) from exc

        steps = getattr(self.estimator, "named_steps", {})
        classifier = self.estimator[-1] if steps else self.estimator
        scaler = steps.get("scaler")
        baseline_values = self.metadata.get("explanation_baseline")
        if baseline_values is None:
            baseline_values = getattr(scaler, "mean_", None)
        if baseline_values is None or len(baseline_values) != len(self.feature_columns):
            raise RuntimeError("This model has no verified feature baseline for SHAP.")

        instance = pd.DataFrame([row], columns=self.feature_columns)
        model_input = scaler.transform(instance) if scaler is not None else instance.to_numpy()
        probabilities = self.estimator.predict_proba(instance)[:, 1]

        if hasattr(classifier, "coef_"):
            background = np.asarray(baseline_values, dtype=float).reshape(1, -1)
            if scaler is not None:
                background = scaler.transform(
                    pd.DataFrame(background, columns=self.feature_columns)
                )
            explainer = shap.LinearExplainer(
                classifier,
                shap.maskers.Independent(background, max_samples=1),
            )
            explanation = explainer.shap_values(model_input)
            contributions = np.asarray(explanation).reshape(-1)
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
            output = float(classifier.decision_function(model_input)[0])
            output_scale = "log_odds"
            method = "SHAP LinearExplainer; training-feature mean baseline"
        elif hasattr(classifier, "feature_importances_"):
            background = pd.DataFrame(
                [baseline_values], columns=self.feature_columns
            ).to_numpy()
            if scaler is not None:
                background = scaler.transform(
                    pd.DataFrame(background, columns=self.feature_columns)
                )
            explainer = shap.TreeExplainer(
                classifier,
                data=background,
                model_output="probability",
            )
            explanation = explainer.shap_values(model_input)
            if isinstance(explanation, list):
                contributions = np.asarray(explanation[1]).reshape(-1)
            else:
                values = np.asarray(explanation)
                contributions = values.reshape(-1, len(self.feature_columns))[-1]
            expected = np.asarray(explainer.expected_value).reshape(-1)
            base_value = float(expected[-1])
            output = float(probabilities[0])
            output_scale = "fraud_probability"
            method = "SHAP TreeExplainer; training-feature mean baseline"
        else:
            raise RuntimeError("SHAP is only enabled for supported linear/tree models.")

        return {
            "output": output,
            "fraud_probability": float(probabilities[0]),
            "base_value": base_value,
            "feature_values": {
                feature: float(value)
                for feature, value in zip(
                    self.feature_columns, contributions, strict=True
                )
            },
            "output_scale": output_scale,
            "method": method,
        }
