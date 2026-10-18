from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from fraud_platform.api import CreditCardTransaction, MODEL
from fraud_platform.config import settings
from fraud_platform.streaming.events import TransactionEvent
from fraud_platform.streaming.kafka_io import KafkaEventProducer


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish one fraud transaction event")
    parser.add_argument("--transaction-json", type=Path, required=True)
    parser.add_argument("--event-id", default=None, help="Reuse this ID when retrying an event")
    args = parser.parse_args()
    transaction = CreditCardTransaction.model_validate_json(
        args.transaction_json.read_text(encoding="utf-8")
    )
    event = TransactionEvent(
        schema_version="1.0",
        event_id=args.event_id or str(uuid4()),
        event_timestamp=datetime.now(timezone.utc),
        transaction=transaction,
    )
    producer = KafkaEventProducer()
    try:
        producer.publish(
            settings.kafka_raw_topic,
            event.event_id,
            event.model_dump(mode="json"),
        )
    finally:
        producer.close()
    print(json.dumps({"published": True, "event_id": event.event_id, "model_version": MODEL.model_version}))


if __name__ == "__main__":
    main()
