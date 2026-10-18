from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from fraud_platform.config import settings
from fraud_platform.config import normalize_database_url
from fraud_platform.db import database
from fraud_platform.db.models import Base, PredictionAudit


def configure_database(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    test_settings = replace(settings, database_url=f"sqlite:///{path.as_posix()}")
    monkeypatch.setattr(database, "settings", test_settings)
    database.get_engine.cache_clear()
    database.get_session_factory.cache_clear()
    Base.metadata.create_all(database.get_engine())


def clear_database() -> None:
    database.get_engine.cache_clear()
    database.get_session_factory.cache_clear()


@pytest.mark.parametrize(
    ("database_url", "expected_url"),
    [
        ("postgres://user:password@host/db", "postgresql+psycopg://user:password@host/db"),
        ("postgresql://user:password@host/db", "postgresql+psycopg://user:password@host/db"),
        ("postgresql+psycopg://user:password@host/db", "postgresql+psycopg://user:password@host/db"),
        ("sqlite:///local.sqlite3", "sqlite:///local.sqlite3"),
    ],
)
def test_database_url_normalizes_supported_postgresql_schemes(
    database_url: str, expected_url: str
) -> None:
    assert normalize_database_url(database_url) == expected_url


def test_prediction_audit_is_idempotent_by_event_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_database(monkeypatch, tmp_path / "idempotency.sqlite3")
    record = {
        "event_id": "event-idempotency-test",
        "transaction_id": "transaction-idempotency-test",
        "request_id": "request-idempotency-test",
        "created_at": datetime.now(timezone.utc),
        "fraud_probability": 0.1,
        "threshold": 0.99,
        "decision": "legitimate",
        "model_version": "test-model",
    }
    try:
        database.record_predictions([record, record])
        database.record_predictions([record])
        with database.get_session_factory()() as session:
            count = session.scalar(select(func.count()).select_from(PredictionAudit))
        assert count == 1
    finally:
        clear_database()


def test_prediction_audit_retention_deletes_only_expired_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_database(monkeypatch, tmp_path / "retention.sqlite3")
    now = datetime.now(timezone.utc)
    records = [
        {
            "event_id": None,
            "transaction_id": "expired-audit",
            "request_id": "request-expired",
            "created_at": now.replace(year=now.year - 1),
            "fraud_probability": 0.1,
            "threshold": 0.99,
            "decision": "legitimate",
            "model_version": "test-model",
        },
        {
            "event_id": None,
            "transaction_id": "recent-audit",
            "request_id": "request-recent",
            "created_at": now,
            "fraud_probability": 0.2,
            "threshold": 0.99,
            "decision": "legitimate",
            "model_version": "test-model",
        },
    ]
    try:
        database.record_predictions(records)
        assert database.purge_expired_prediction_audits(90) == 1
        with database.get_session_factory()() as session:
            remaining = list(session.scalars(select(PredictionAudit.transaction_id)))
        assert remaining == ["recent-audit"]
    finally:
        clear_database()
