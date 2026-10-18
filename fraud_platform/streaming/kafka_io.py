from __future__ import annotations

import json
from typing import Any

from fraud_platform.config import settings


class KafkaEventProducer:
    def __init__(self) -> None:
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
            acks="all",
            retries=5,
            request_timeout_ms=int(settings.request_timeout_seconds * 1000),
            max_block_ms=int(settings.request_timeout_seconds * 1000),
        )

    def publish(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        future = self._producer.send(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"),
        )
        future.get(timeout=settings.request_timeout_seconds)

    def publish_raw(self, topic: str, key: str, payload: bytes) -> None:
        future = self._producer.send(topic, key=key.encode("utf-8"), value=payload)
        future.get(timeout=settings.request_timeout_seconds)

    def close(self) -> None:
        self._producer.flush(timeout=settings.request_timeout_seconds)
        self._producer.close(timeout=settings.request_timeout_seconds)
