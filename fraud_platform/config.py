from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


@dataclass(frozen=True)
class Settings:
    model_path: Path
    metadata_path: Path
    drift_baseline_path: Path
    database_url: str
    api_key: str
    threshold: float | None
    max_batch_size: int
    max_request_bytes: int
    max_requests_per_minute: int
    request_timeout_seconds: float
    kafka_bootstrap_servers: str
    kafka_raw_topic: str
    kafka_scored_topic: str
    kafka_alerts_topic: str
    kafka_dlq_topic: str
    kafka_consumer_group: str
    cassandra_hosts: tuple[str, ...]
    cassandra_port: int
    cassandra_keyspace: str
    cassandra_username: str
    cassandra_password: str

    @classmethod
    def from_env(cls) -> "Settings":
        threshold_value = os.getenv("FRAUD_THRESHOLD")
        threshold = float(threshold_value) if threshold_value else None
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("FRAUD_THRESHOLD must be between 0 and 1")
        batch_size = int(os.getenv("MAX_BATCH_SIZE", "100"))
        if not 1 <= batch_size <= 1000:
            raise ValueError("MAX_BATCH_SIZE must be between 1 and 1000")
        max_request_bytes = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
        if not 1024 <= max_request_bytes <= 10_485_760:
            raise ValueError("MAX_REQUEST_BYTES must be between 1024 and 10485760")
        requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "120"))
        if not 1 <= requests_per_minute <= 100_000:
            raise ValueError("MAX_REQUESTS_PER_MINUTE must be between 1 and 100000")

        database_url = os.getenv(
            "DATABASE_URL",
            f"sqlite:///{(ROOT / 'data' / 'fraud_platform.sqlite3').as_posix()}",
        )
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "creditcard_model.joblib"))),
            metadata_path=Path(
                os.getenv(
                    "MODEL_METADATA_PATH",
                    str(ROOT / "models" / "creditcard_model_metadata.json"),
                )
            ),
            drift_baseline_path=Path(
                os.getenv(
                    "DRIFT_BASELINE_PATH",
                    str(ROOT / "models" / "creditcard_feature_baseline.json"),
                )
            ),
            database_url=database_url,
            api_key=os.getenv("FRAUD_API_KEY", ""),
            threshold=threshold,
            max_batch_size=batch_size,
            max_request_bytes=max_request_bytes,
            max_requests_per_minute=requests_per_minute,
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            kafka_raw_topic=os.getenv("KAFKA_RAW_TOPIC", "transactions.raw"),
            kafka_scored_topic=os.getenv("KAFKA_SCORED_TOPIC", "transactions.scored"),
            kafka_alerts_topic=os.getenv("KAFKA_ALERTS_TOPIC", "fraud.alerts"),
            kafka_dlq_topic=os.getenv("KAFKA_DLQ_TOPIC", "transactions.dlq"),
            kafka_consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", "fraud-scoring-v1"),
            cassandra_hosts=tuple(
                host.strip()
                for host in os.getenv("CASSANDRA_HOSTS", "localhost").split(",")
                if host.strip()
            ),
            cassandra_port=int(os.getenv("CASSANDRA_PORT", "9042")),
            cassandra_keyspace=os.getenv("CASSANDRA_KEYSPACE", "fraud_platform"),
            cassandra_username=os.getenv("CASSANDRA_USERNAME", ""),
            cassandra_password=os.getenv("CASSANDRA_PASSWORD", ""),
        )


settings = Settings.from_env()
