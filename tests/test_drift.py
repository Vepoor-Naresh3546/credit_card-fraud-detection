from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_platform.config import ROOT
from fraud_platform.drift import DriftMonitor, FeatureBins, build_drift_baseline
from fraud_platform.model import FraudModel


def test_drift_monitor_detects_distribution_shift() -> None:
    monitor = DriftMonitor(
        {"V1": FeatureBins(edges=(0.0,), expected=(0.5, 0.5))},
        window_size=4,
        minimum_samples=2,
    )
    assert monitor.observe({"V1": 1.0}) is None
    scores = monitor.observe({"V1": 1.0})
    assert scores is not None
    assert scores["V1"][0] > 0.2


def test_training_baseline_is_reusable_and_matches_model_features(tmp_path: Path) -> None:
    model = FraudModel.load(
        ROOT / "models" / "creditcard_model.joblib",
        ROOT / "models" / "creditcard_model_metadata.json",
    )
    rng = np.random.default_rng(9)
    labels = [0] * 80 + [1] * 20
    frame = pd.DataFrame(
        rng.normal(size=(len(labels), len(model.feature_columns))),
        columns=model.feature_columns,
    )
    frame["Class"] = labels
    dataset = tmp_path / "train.csv"
    frame.to_csv(dataset, index=False)
    output = tmp_path / "baseline.json"

    baseline = build_drift_baseline(dataset, model, output)
    monitor = DriftMonitor.from_file(output, model.feature_columns, window_size=5, minimum_samples=1)

    assert baseline["training_rows"] == 70
    assert set(baseline["feature_bins"]) == set(model.feature_columns)
    assert monitor.observe({feature: 0.0 for feature in model.feature_columns}) is not None
    assert json.loads(output.read_text(encoding="utf-8"))["model_version"] == model.model_version


def test_drift_monitor_rejects_nonfinite_values() -> None:
    monitor = DriftMonitor(
        {"V1": FeatureBins(edges=(0.0,), expected=(0.5, 0.5))},
        window_size=2,
        minimum_samples=1,
    )
    with pytest.raises(ValueError, match="finite"):
        monitor.observe({"V1": float("nan")})
