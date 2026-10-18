from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Callable

from prometheus_client import Counter, Histogram, start_http_server
from pydantic import ValidationError

from fraud_platform.config import settings
from fraud_platform.db.database import record_prediction
from fraud_platform.model import FraudModel
from fraud_platform.streaming.events import TransactionEvent
from fraud_platform.streaming.kafka_io import KafkaEventProducer

logger = logging.getLogger("fraud_platform.worker")
WORKER_EVENTS = Counter(
    "creditcard_worker_events_total",
    "Kafka events processed by outcome",
    ["outcome"],
)
WORKER_LATENCY = Histogram(
    "creditcard_worker_processing_duration_seconds",
    "Kafka event processing duration",
)
MAX_PROCESS_ATTEMPTS = int(os.getenv("KAFKA_MAX_PROCESS_ATTEMPTS", "3"))
if not 1 <= MAX_PROCESS_ATTEMPTS <= 10:
    raise ValueError("KAFKA_MAX_PROCESS_ATTEMPTS must be between 1 and 10")


def process_event(
    event: TransactionEvent,
    *,
    model: FraudModel,
    publisher: Any,
    cassandra_repository: Any | None = None,
    audit_writer: Callable[..., None] = record_prediction,
) -> dict[str, Any]:
    feature_row = {
        feature: getattr(event.transaction, feature)
        for feature in model.feature_columns
    }
    result = model.predict([feature_row])[0]
    scored_event = {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "event_timestamp": event.event_timestamp.isoformat(),
        "transaction_id": event.transaction.transaction_id,
        **result,
        "threshold": model.threshold,
        "model_version": model.model_version,
    }
    audit_writer(
        event_id=event.event_id,
        transaction_id=event.transaction.transaction_id,
        request_id=f"kafka:{event.event_id}",
        created_at=event.event_timestamp,
        fraud_probability=result["fraud_probability"],
        threshold=model.threshold,
        decision=result["decision"],
        model_version=model.model_version,
    )
    if cassandra_repository is not None:
        cassandra_repository.save_scored_event(
            {**scored_event, "event_timestamp": event.event_timestamp}
        )
    publisher.publish(settings.kafka_scored_topic, event.event_id, scored_event)
    if result["decision"] == "fraud":
        publisher.publish(settings.kafka_alerts_topic, event.event_id, scored_event)
    return scored_event


def _dead_letter(producer: KafkaEventProducer, raw_value: bytes, error_code: str) -> None:
    envelope = {
        "error_code": error_code,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "original_event": raw_value.decode("utf-8", errors="replace"),
    }
    producer.publish(settings.kafka_dlq_topic, f"dlq-{time.time_ns()}", envelope)


def process_with_retries(
    event: TransactionEvent,
    raw_value: bytes,
    *,
    model: FraudModel,
    publisher: Any,
    cassandra_repository: Any | None = None,
    audit_writer: Callable[..., None] = record_prediction,
    max_attempts: int = MAX_PROCESS_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    dead_letter_writer: Callable[[KafkaEventProducer, bytes, str], None] = _dead_letter,
) -> str:
    if not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be between 1 and 10")
    for attempt in range(1, max_attempts + 1):
        try:
            process_event(
                event,
                model=model,
                publisher=publisher,
                cassandra_repository=cassandra_repository,
                audit_writer=audit_writer,
            )
            return "processed"
        except Exception as exc:
            logger.warning(
                "event_processing_attempt_failed",
                extra={
                    "event_id": event.event_id,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                },
            )
            if attempt == max_attempts:
                dead_letter_writer(publisher, raw_value, type(exc).__name__)
                return "dead_lettered"
            sleep(min(2 ** (attempt - 1), 10))
    raise RuntimeError("Event processing exited without a terminal outcome.")


def run_worker() -> None:
    from kafka import KafkaConsumer

    start_http_server(int(os.getenv("WORKER_METRICS_PORT", "9101")))
    producer = KafkaEventProducer()
    model = FraudModel.load(settings.model_path, settings.metadata_path, settings.threshold)
    cassandra_repository = None
    if os.getenv("CASSANDRA_ENABLED", "false").lower() == "true":
        from fraud_platform.cassandra.repository import CassandraEventRepository

        cassandra_repository = CassandraEventRepository.connect()
        cassandra_repository.check_health()
    consumer = KafkaConsumer(
        settings.kafka_raw_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        group_id=settings.kafka_consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=1,
        consumer_timeout_ms=0,
    )
    try:
        for message in consumer:
            started = time.perf_counter()
            raw_value = message.value
            try:
                event = TransactionEvent.model_validate_json(raw_value)
                outcome = process_with_retries(
                    event,
                    raw_value,
                    model=model,
                    publisher=producer,
                    cassandra_repository=cassandra_repository,
                )
                consumer.commit()
                WORKER_EVENTS.labels(outcome=outcome).inc()
            except ValidationError:
                _dead_letter(producer, raw_value, "event_validation_error")
                consumer.commit()
                WORKER_EVENTS.labels(outcome="invalid").inc()
                logger.warning("invalid_event_dead_lettered", extra={"error_type": "ValidationError"})
            except Exception:
                logger.exception("event_was_not_acknowledged")
                WORKER_EVENTS.labels(outcome="failed").inc()
                raise
            finally:
                WORKER_LATENCY.observe(time.perf_counter() - started)
    finally:
        consumer.close()
        producer.close()
        if cassandra_repository is not None:
            cassandra_repository.close()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_worker()
