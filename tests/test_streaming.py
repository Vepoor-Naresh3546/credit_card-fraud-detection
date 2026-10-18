from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from fraud_platform.api import MODEL
from fraud_platform.streaming.events import TransactionEvent
from fraud_platform.streaming.worker import process_event, process_with_retries


def valid_event() -> TransactionEvent:
    from fraud_platform.api import CreditCardTransaction

    transaction = CreditCardTransaction.model_validate(
        {
            "transaction_id": "stream-test-001",
            **{name: 0.5 for name in MODEL.feature_columns},
        }
    )
    return TransactionEvent(
        schema_version="1.0",
        event_id="3ed35a4f-d6e4-4a67-9d98-4ff39b3f20c7",
        event_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        transaction=transaction,
    )


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, dict[str, Any]]] = []

    def publish(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        self.messages.append((topic, key, payload))


class FakeCassandra:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def save_scored_event(self, result: dict[str, Any]) -> None:
        self.events.append(result)


def test_event_requires_timezone_and_known_schema_version() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            schema_version="1.0",
            event_id="invalid",
            event_timestamp=datetime(2026, 1, 1),
            transaction=valid_event().transaction,
        )


def test_worker_scores_persists_and_publishes_with_idempotency_key() -> None:
    event = valid_event()
    publisher = FakePublisher()
    cassandra = FakeCassandra()
    audits: list[dict[str, Any]] = []
    result = process_event(
        event,
        model=MODEL,
        publisher=publisher,
        cassandra_repository=cassandra,
        audit_writer=lambda **kwargs: audits.append(kwargs),
    )
    assert result["event_id"] == event.event_id
    assert result["transaction_id"] == event.transaction.transaction_id
    assert 0 <= result["fraud_probability"] <= 1
    assert audits[0]["event_id"] == event.event_id
    assert cassandra.events[0]["event_id"] == event.event_id
    assert publisher.messages[0][1] == event.event_id
    assert publisher.messages[0][0] == "transactions.scored"
    if result["decision"] == "fraud":
        assert publisher.messages[1][0] == "fraud.alerts"


def test_worker_retries_failed_publish_before_processing() -> None:
    event = valid_event()

    class FlakyPublisher(FakePublisher):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True

        def publish(self, topic: str, key: str, payload: dict[str, Any]) -> None:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary broker failure")
            super().publish(topic, key, payload)

    publisher = FlakyPublisher()
    outcome = process_with_retries(
        event,
        event.model_dump_json().encode(),
        model=MODEL,
        publisher=publisher,
        audit_writer=lambda **_kwargs: None,
        max_attempts=2,
        sleep=lambda _seconds: None,
    )
    assert outcome == "processed"
    assert publisher.messages[0][0] == "transactions.scored"


def test_worker_dead_letters_after_retry_exhaustion() -> None:
    event = valid_event()
    failed_messages: list[tuple[bytes, str]] = []

    class FailedPublisher:
        def publish(self, _topic: str, _key: str, _payload: dict[str, Any]) -> None:
            raise RuntimeError("broker unavailable")

    raw_event = event.model_dump_json().encode()
    outcome = process_with_retries(
        event,
        raw_event,
        model=MODEL,
        publisher=FailedPublisher(),
        audit_writer=lambda **_kwargs: None,
        max_attempts=2,
        sleep=lambda _seconds: None,
        dead_letter_writer=lambda _producer, value, code: failed_messages.append((value, code)),
    )
    assert outcome == "dead_lettered"
    assert failed_messages == [(raw_event, "RuntimeError")]


def test_model_loader_confirms_exact_feature_contract() -> None:
    assert MODEL.feature_columns == tuple(
        ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    )
    assert 0 <= MODEL.threshold <= 1
