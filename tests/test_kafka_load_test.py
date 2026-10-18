from __future__ import annotations

import pytest

from scripts.kafka_load_test import percentile, summarize_load


def test_percentiles_interpolate_and_empty_is_explicit() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25
    assert percentile([], 95) is None


def test_load_summary_reports_backlog_and_target_failure() -> None:
    result = summarize_load(
        sent=10,
        completed_ids={"one", "two", "three"},
        dead_letter_ids=set(),
        latencies_ms=[10.0, 20.0, 30.0],
        elapsed_seconds=10,
        target_per_minute=100,
    )
    assert result["backlog"] == 7
    assert result["errors"] == 7
    assert result["latency_ms"]["p95"] == pytest.approx(29)
    assert result["target_met"] is False
