from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_platform.api import CreditCardTransaction


class TransactionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=36, max_length=36)
    event_timestamp: datetime
    transaction: CreditCardTransaction

    @field_validator("event_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("Unsupported event schema version")
        return value

    @field_validator("event_id")
    @classmethod
    def require_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except ValueError as exc:
            raise ValueError("event_id must be a UUID") from exc
