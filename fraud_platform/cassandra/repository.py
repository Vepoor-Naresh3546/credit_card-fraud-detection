from __future__ import annotations

from datetime import date, timezone
import re
from typing import Any

from fraud_platform.config import settings


class CassandraEventRepository:
    def __init__(self, session: Any, cluster: Any, keyspace: str) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", keyspace):
            raise ValueError("Cassandra keyspace must be a valid identifier")
        self._session = session
        self._cluster = cluster
        self._keyspace = keyspace
        self._insert = session.prepare(
            f"INSERT INTO {keyspace}.scored_events_by_day "
            "(event_date, event_timestamp, event_id, transaction_id, decision, "
            "fraud_probability, threshold, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )

    @classmethod
    def connect(cls) -> "CassandraEventRepository":
        from cassandra.auth import PlainTextAuthProvider
        from cassandra.cluster import Cluster

        kwargs: dict[str, Any] = {
            "contact_points": list(settings.cassandra_hosts),
            "port": settings.cassandra_port,
            "connect_timeout": settings.request_timeout_seconds,
        }
        if settings.cassandra_username and settings.cassandra_password:
            kwargs["auth_provider"] = PlainTextAuthProvider(
                username=settings.cassandra_username,
                password=settings.cassandra_password,
            )
        cluster = Cluster(**kwargs)
        session = cluster.connect()
        return cls(session=session, cluster=cluster, keyspace=settings.cassandra_keyspace)

    def check_health(self) -> bool:
        self._session.execute("SELECT release_version FROM system.local").one()
        return True

    def save_scored_event(self, result: dict[str, Any]) -> None:
        timestamp = result["event_timestamp"].astimezone(timezone.utc)
        self._session.execute(
            self._insert,
            (
                date(timestamp.year, timestamp.month, timestamp.day),
                timestamp,
                result["event_id"],
                result["transaction_id"],
                result["decision"],
                result["fraud_probability"],
                result["threshold"],
                result["model_version"],
            ),
            timeout=settings.request_timeout_seconds,
        )

    def close(self) -> None:
        self._cluster.shutdown()
