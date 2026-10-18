"""Create prediction audit and model deployment tables.

Revision ID: 0001_initial_schema
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_audit",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=100), nullable=True),
        sa.Column("transaction_id", sa.String(length=100), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", name="uq_prediction_audit_event_id"),
    )
    op.create_index("ix_prediction_audit_created_at", "prediction_audit", ["created_at"])
    op.create_index(
        "ix_prediction_audit_decision_created_at",
        "prediction_audit",
        ["decision", "created_at"],
    )
    op.create_table(
        "model_deployments",
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact_uri", sa.String(length=2048), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("model_version"),
    )
    op.create_index(
        "ix_model_deployments_deployed_at",
        "model_deployments",
        ["deployed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_deployments_deployed_at", table_name="model_deployments")
    op.drop_table("model_deployments")
    op.drop_index("ix_prediction_audit_decision_created_at", table_name="prediction_audit")
    op.drop_index("ix_prediction_audit_created_at", table_name="prediction_audit")
    op.drop_table("prediction_audit")
