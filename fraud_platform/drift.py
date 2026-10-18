from __future__ import annotations

from collections import deque
from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from prometheus_client import Gauge

from fraud_platform.model import FraudModel


FEATURE_PSI = Gauge(
    "creditcard_feature_drift_psi",
    "Population Stability Index by model feature over the recent prediction window",
    ["feature"],
)
FEATURE_KL = Gauge(
    "creditcard_feature_drift_kl_divergence",
    "KL divergence by model feature over the recent prediction window",
    ["feature"],
)


@dataclass(frozen=True)
class FeatureBins:
    edges: tuple[float, ...]
    expected: tuple[float, ...]


class DriftMonitor:
    def __init__(
        self,
        feature_bins: dict[str, FeatureBins],
        *,
        window_size: int = 1000,
        minimum_samples: int = 100,
    ) -> None:
        if window_size < 1 or not 1 <= minimum_samples <= window_size:
            raise ValueError("Invalid drift sample window settings.")
        if not feature_bins:
            raise ValueError("At least one feature baseline is required.")
        self._bins = feature_bins
        self._window_size = window_size
        self._minimum_samples = minimum_samples
        self._history: deque[tuple[int, ...]] = deque(maxlen=window_size)
        self._counts = {
            name: [0] * len(bins.expected)
            for name, bins in feature_bins.items()
        }
        self._lock = Lock()

    @classmethod
    def from_file(
        cls,
        path: Path,
        expected_features: tuple[str, ...],
        **kwargs: Any,
    ) -> "DriftMonitor":
        content = json.loads(path.read_text(encoding="utf-8"))
        raw_bins = content.get("feature_bins", {})
        if set(raw_bins) != set(expected_features):
            raise ValueError("Drift baseline feature names do not match the model contract.")
        feature_bins: dict[str, FeatureBins] = {}
        for feature in expected_features:
            entry = raw_bins[feature]
            edges = tuple(float(value) for value in entry["edges"])
            expected = tuple(float(value) for value in entry["expected"])
            if len(edges) + 1 != len(expected) or len(expected) < 2:
                raise ValueError(f"Invalid drift bins for feature {feature}.")
            if any(not math.isfinite(edge) for edge in edges):
                raise ValueError(f"Non-finite drift boundary for feature {feature}.")
            if any(left >= right for left, right in zip(edges, edges[1:])):
                raise ValueError(f"Drift boundaries must increase for feature {feature}.")
            if any(value < 0 for value in expected) or not math.isclose(
                sum(expected), 1.0, abs_tol=1e-6
            ):
                raise ValueError(f"Invalid expected proportions for feature {feature}.")
            feature_bins[feature] = FeatureBins(edges=edges, expected=expected)
        return cls(feature_bins, **kwargs)

    def observe(self, row: dict[str, float]) -> dict[str, tuple[float, float]] | None:
        if set(row) != set(self._bins):
            raise ValueError("Observed features do not match the configured drift baseline.")
        indexes = tuple(
            self._bin_index(self._bins[feature].edges, float(row[feature]))
            for feature in self._bins
        )
        with self._lock:
            if len(self._history) == self._window_size:
                previous = self._history[0]
                for (feature, counts), index in zip(
                    self._counts.items(), previous, strict=True
                ):
                    counts[index] -= 1
            self._history.append(indexes)
            for (feature, counts), index in zip(
                self._counts.items(), indexes, strict=True
            ):
                counts[index] += 1
            if len(self._history) < self._minimum_samples:
                return None

            scores: dict[str, tuple[float, float]] = {}
            for feature, counts in self._counts.items():
                expected = self._bins[feature].expected
                actual_total = sum(counts)
                actual = [
                    (count + 1e-6) / (actual_total + len(counts) * 1e-6)
                    for count in counts
                ]
                reference = [
                    (value + 1e-6) / (sum(expected) + len(expected) * 1e-6)
                    for value in expected
                ]
                kl = sum(
                    observed * math.log(observed / baseline)
                    for observed, baseline in zip(actual, reference, strict=True)
                )
                psi = sum(
                    (observed - baseline) * math.log(observed / baseline)
                    for observed, baseline in zip(actual, reference, strict=True)
                )
                FEATURE_PSI.labels(feature=feature).set(psi)
                FEATURE_KL.labels(feature=feature).set(kl)
                scores[feature] = (psi, kl)
            return scores

    @staticmethod
    def _bin_index(edges: tuple[float, ...], value: float) -> int:
        if not math.isfinite(value):
            raise ValueError("Drift observations must be finite.")
        return bisect_right(edges, value)


def build_drift_baseline(
    dataset_path: Path,
    model: FraudModel,
    destination: Path,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    import numpy as np
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(
        dataset_path,
        usecols=[*model.feature_columns, "Class"],
    )
    if frame.isna().any().any():
        raise ValueError("Baseline training features and labels must not contain nulls.")
    if not np.isfinite(frame[list(model.feature_columns)].to_numpy(dtype=float)).all():
        raise ValueError("Baseline training features must contain only finite values.")
    X_train, _, _, _ = train_test_split(
        frame[list(model.feature_columns)],
        frame["Class"],
        test_size=0.30,
        random_state=42,
        stratify=frame["Class"],
    )
    bins: dict[str, dict[str, list[float]]] = {}
    for feature in model.feature_columns:
        values = X_train[feature].to_numpy(dtype=float)
        inner_edges = np.unique(
            np.quantile(values, np.linspace(0, 1, bin_count + 1)[1:-1])
        )
        counts, _ = np.histogram(
            values,
            bins=np.concatenate(([-np.inf], inner_edges, [np.inf])),
        )
        bins[feature] = {
            "edges": [float(value) for value in inner_edges],
            "expected": (counts / counts.sum()).astype(float).tolist(),
        }
    result = {
        "model_version": model.model_version,
        "training_rows": int(len(X_train)),
        "random_state": 42,
        "binning": "unique decile boundaries fitted only on stratified training split",
        "feature_bins": bins,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
