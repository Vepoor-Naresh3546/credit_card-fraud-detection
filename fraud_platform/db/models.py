from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import DateTime, Float, Index, JSON, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_prediction_audit_event_id"),
        Index("ix_prediction_audit_created_at", "created_at"),
        Index("ix_prediction_audit_decision_created_at", "decision", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    transaction_id: Mapped[str] = mapped_column(String(100), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)


class ModelDeployment(Base):
    __tablename__ = "model_deployments"
    __table_args__ = (Index("ix_model_deployments_deployed_at", "deployed_at"),)

    model_version: Mapped[str] = mapped_column(String(100), primary_key=True)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False)
