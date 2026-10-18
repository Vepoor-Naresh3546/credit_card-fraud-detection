from __future__ import annotations

import pandas as pd
import pytest


gx = pytest.importorskip("great_expectations")

from fraud_platform.quality import validate_dataframe


def make_frame() -> pd.DataFrame:
    columns = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"]
    return pd.DataFrame(
        [[0.0] * 30 + [0], [1.0] * 30 + [1]],
        columns=columns,
    )


def test_quality_gate_accepts_expected_transaction_schema() -> None:
    validate_dataframe(make_frame(), "valid")


def test_quality_gate_rejects_unexpected_labels() -> None:
    frame = make_frame()
    frame.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="checks failed"):
        validate_dataframe(frame, "invalid-label")
