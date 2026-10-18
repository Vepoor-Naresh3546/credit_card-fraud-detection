from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize_load(
    *,
    sent: int,
    completed_ids: set[str],
    dead_letter_ids: set[str],
    latencies_ms: list[float],
    elapsed_seconds: float,
    target_per_minute: int,
) -> dict[str, Any]:
    completed = len(completed_ids)
    dead_lettered = len(dead_letter_ids)
    backlog = max(sent - completed, 0)
    return {
        "target_events_per_minute": target_per_minute,
        "sent_events": sent,
        "scored_events": completed - dead_lettered,
        "dead_letter_events": dead_lettered,
        "errors": dead_lettered + backlog,
        "backlog": backlog,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "observed_events_per_minute": round(
            completed / elapsed_seconds * 60 if elapsed_seconds else 0, 2
        ),
        "latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "target_met": (
            completed == sent
            and dead_lettered == 0
            and completed / elapsed_seconds * 60 >= target_per_minute
            if elapsed_seconds
            else False
        ),
        "measurement_note": (
            "End-to-end event age is measured using producer event_timestamp and "
            "consumer receive time on this host; clocks must be synchronized."
        ),
    }


def run_load_test(
    *,
    events: int,
    target_per_minute: int,
    timeout_seconds: int,
    bootstrap_servers: str,
    raw_topic: str,
    scored_topic: str,
    dlq_topic: str,
    transaction_template: dict[str, Any],
) -> dict[str, Any]:
    from kafka import KafkaConsumer, KafkaProducer

    if events < 1 or target_per_minute < 1 or timeout_seconds < 1:
        raise ValueError("events, target_per_minute, and timeout_seconds must be positive")

    group_id = f"fraud-load-test-{uuid4()}"
    consumer = KafkaConsumer(
        scored_topic,
        dlq_topic,
        bootstrap_servers=bootstrap_servers.split(","),
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        consumer_timeout_ms=0,
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        acks="all",
        retries=5,
        value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode(),
        key_serializer=lambda value: value.encode(),
    )
    sent_ids: set[str] = set()
    completed_ids: set[str] = set()
    dead_letter_ids: set[str] = set()
    latencies_ms: list[float] = []
    sent = 0
    try:
        subscription_deadline = time.monotonic() + 30
        while not consumer.assignment() and time.monotonic() < subscription_deadline:
            consumer.poll(timeout_ms=500)
        if not consumer.assignment():
            raise TimeoutError("Kafka load-test consumer did not receive a topic assignment.")
        started = time.monotonic()
        start_wall_time = time.time()
        interval = 60.0 / target_per_minute
        next_send = started
        while sent < events:
            now = time.monotonic()
            if now < next_send:
                time.sleep(next_send - now)
            event_id = str(uuid4())
            event_timestamp = datetime.now(timezone.utc).isoformat()
            transaction = {
                **transaction_template,
                "transaction_id": f"load-{event_id}",
            }
            payload = {
                "schema_version": "1.0",
                "event_id": event_id,
                "event_timestamp": event_timestamp,
                "transaction": transaction,
            }
            producer.send(raw_topic, key=event_id, value=payload).get(timeout=10)
            sent_ids.add(event_id)
            sent += 1
            next_send += interval
        producer.flush(timeout=30)

        deadline = time.monotonic() + timeout_seconds
        while completed_ids != sent_ids and time.monotonic() < deadline:
            for message in consumer.poll(timeout_ms=500, max_records=1000).values():
                for record in message:
                    payload = json.loads(record.value)
                    if record.topic == dlq_topic:
                        original = payload.get("original_event")
                        if isinstance(original, str):
                            try:
                                event_id = json.loads(original).get("event_id")
                            except json.JSONDecodeError:
                                event_id = None
                        else:
                            event_id = None
                        if event_id in sent_ids:
                            dead_letter_ids.add(event_id)
                            completed_ids.add(event_id)
                        continue
                    event_id = payload.get("event_id")
                    if event_id not in sent_ids or event_id in completed_ids:
                        continue
                    completed_ids.add(event_id)
                    event_timestamp = datetime.fromisoformat(
                        payload["event_timestamp"].replace("Z", "+00:00")
                    )
                    age_ms = (
                        datetime.now(timezone.utc) - event_timestamp
                    ).total_seconds() * 1000
                    latencies_ms.append(max(age_ms, 0.0))
    finally:
        producer.close(timeout=10)
        consumer.close()

    elapsed = max(time.monotonic() - started, 1e-9)
    result = summarize_load(
        sent=sent,
        completed_ids=completed_ids,
        dead_letter_ids=dead_letter_ids,
        latencies_ms=latencies_ms,
        elapsed_seconds=elapsed,
        target_per_minute=target_per_minute,
    )
    result["bootstrap_servers"] = bootstrap_servers
    result["raw_topic"] = raw_topic
    result["scored_topic"] = scored_topic
    result["dlq_topic"] = dlq_topic
    result["start_time_epoch_seconds"] = start_wall_time
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Kafka fraud scoring throughput, latency, errors, and backlog."
    )
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--target-per-minute", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--template", type=Path, default=Path("examples/transaction.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/kafka_load_test.json"))
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
    )
    parser.add_argument("--raw-topic", default=os.getenv("KAFKA_RAW_TOPIC", "transactions.raw"))
    parser.add_argument(
        "--scored-topic",
        default=os.getenv("KAFKA_SCORED_TOPIC", "transactions.scored"),
    )
    parser.add_argument("--dlq-topic", default=os.getenv("KAFKA_DLQ_TOPIC", "transactions.dlq"))
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    result = run_load_test(
        events=args.events,
        target_per_minute=args.target_per_minute,
        timeout_seconds=args.timeout_seconds,
        bootstrap_servers=args.bootstrap_servers,
        raw_topic=args.raw_topic,
        scored_topic=args.scored_topic,
        dlq_topic=args.dlq_topic,
        transaction_template=template,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["target_met"]:
        raise SystemExit("Load-test target was not met; inspect errors, latency, and backlog.")


if __name__ == "__main__":
    main()
