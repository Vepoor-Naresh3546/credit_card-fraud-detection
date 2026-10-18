from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from fraud_platform.config import normalize_database_url, settings
from fraud_platform.db.models import ModelDeployment, PredictionAudit


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = normalize_database_url(settings.database_url)
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite:///"):
        db_path = url.removeprefix("sqlite:///")
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT_SECONDS", "10")),
        )
    return create_engine(url, **kwargs)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def check_database() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(select(1))
            table_names = set(inspect(connection).get_table_names())
        return {"prediction_audit", "model_deployments"}.issubset(table_names)
    except SQLAlchemyError:
        return False


def record_prediction(
    *,
    event_id: str | None,
    transaction_id: str,
    request_id: str,
    created_at: Any,
    fraud_probability: float,
    threshold: float,
    decision: str,
    model_version: str,
) -> None:
    record_predictions(
        [
            {
                "event_id": event_id,
                "transaction_id": transaction_id,
                "request_id": request_id,
                "created_at": created_at,
                "fraud_probability": fraud_probability,
                "threshold": threshold,
                "decision": decision,
                "model_version": model_version,
            }
        ]
    )


def record_predictions(records: list[dict[str, Any]]) -> None:
    factory = get_session_factory()
    with factory() as session:
        with session.begin():
            event_ids = [record["event_id"] for record in records if record["event_id"]]
            existing_ids = set()
            if event_ids:
                existing_ids = set(
                    session.scalars(
                        select(PredictionAudit.event_id).where(
                            PredictionAudit.event_id.in_(event_ids)
                        )
                    )
                )
            seen_ids = set(existing_ids)
            new_records = []
            for record in records:
                event_id = record["event_id"]
                if event_id and event_id in seen_ids:
                    continue
                new_records.append(PredictionAudit(**record))
                if event_id:
                    seen_ids.add(event_id)
            session.add_all(
                new_records
            )


def record_deployment(
    *, model_version: str, deployed_at: Any, artifact_uri: str, threshold: float, metadata: dict
) -> None:
    factory = get_session_factory()
    with factory() as session:
        with session.begin():
            session.merge(
                ModelDeployment(
                    model_version=model_version,
                    deployed_at=deployed_at,
                    artifact_uri=artifact_uri,
                    threshold=threshold,
                    metadata_json=metadata,
                )
            )


def purge_expired_prediction_audits(retention_days: int = 90) -> int:
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    with get_session_factory()() as session:
        with session.begin():
            result = session.execute(
                delete(PredictionAudit).where(PredictionAudit.created_at < cutoff)
            )
            return int(result.rowcount or 0)
