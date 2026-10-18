from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_platform.spark_jobs import DATASET_COLUMNS, FEATURE_COLUMNS, validate_dataset_columns

ROOT = Path(__file__).resolve().parents[1]


def test_spark_dataset_contract_matches_model_metadata() -> None:
    metadata = json.loads(
        (ROOT / "models" / "creditcard_model_metadata.json").read_text(encoding="utf-8")
    )
    assert list(FEATURE_COLUMNS) == metadata["feature_columns"]
    assert set(DATASET_COLUMNS) == set(metadata["feature_columns"] + ["Class"])


def test_spark_schema_validation_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_columns(list(DATASET_COLUMNS[1:]))


def test_feast_source_uses_only_existing_dataset_fields() -> None:
    source_builder = (ROOT / "scripts" / "build_feast_source.py").read_text(encoding="utf-8")
    assert "FEATURE_COLUMNS" in source_builder
    assert "customer_id" not in source_builder
    assert "card_id" not in source_builder
