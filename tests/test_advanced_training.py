from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fraud_platform.advanced_training as advanced
from train_real import FEATURES


def test_random_forest_and_optuna_candidates_use_separate_splits(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("xgboost")
    pytest.importorskip("optuna")

    rng = np.random.default_rng(42)
    labels = np.array([0] * 120 + [1] * 30)
    data = pd.DataFrame(
        rng.normal(size=(len(labels), len(FEATURES))),
        columns=FEATURES,
    )
    data["Class"] = labels
    dataset = tmp_path / "small.csv"
    data.to_csv(dataset, index=False)
    monkeypatch.setattr(advanced, "ROOT", tmp_path)
    monkeypatch.setattr(
        advanced,
        "log_candidate_to_mlflow",
        lambda *_args, **_kwargs: None,
    )

    results = advanced.run_benchmarks(
        dataset,
        optuna_trials=1,
        include_xgboost=True,
        promote=False,
    )

    assert {result["model"] for result in results} == {
        "RandomForest",
        "XGBoost-Optuna",
    }
    for result in results:
        metrics = result["metrics"]
        assert metrics["train_rows"] == 105
        assert metrics["validation_rows"] == 22
        assert metrics["test_rows"] == 23
        assert np.isclose(metrics["threshold"], np.linspace(0.01, 0.99, 99)).any()
        assert (tmp_path / "models" / "candidates" / result["model_version"]).is_dir()
    metadata_path = tmp_path / "models" / "candidates" / results[1]["model_version"]
    assert (metadata_path / "creditcard_model_metadata.json").is_file()
