from __future__ import annotations

from pathlib import Path

from fraud_platform.training import evaluate_model_metrics


def test_evaluation_gate_requires_all_configured_metrics() -> None:
    assert evaluate_model_metrics({"f1": 0.7, "recall": 0.8, "precision": 0.65})
    assert not evaluate_model_metrics({"f1": 0.7, "recall": 0.69, "precision": 0.65})


def test_airflow_dag_source_is_syntax_valid_without_airflow_install() -> None:
    path = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "creditcard_ml_pipeline.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
