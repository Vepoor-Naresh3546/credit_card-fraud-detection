from __future__ import annotations

from datetime import timedelta
import logging
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/fraud-platform"
logger = logging.getLogger("airflow.task")


def report_failure(context: dict) -> None:
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    logger.error(
        "Fraud ML pipeline task failed: dag=%s task=%s error_type=%s",
        getattr(task_instance, "dag_id", "unknown"),
        getattr(task_instance, "task_id", "unknown"),
        type(exception).__name__ if exception else "unknown",
    )
    webhook = os.getenv("AIRFLOW_ALERT_WEBHOOK")
    if webhook:
        payload = json.dumps(
            {
                "dag_id": getattr(task_instance, "dag_id", "unknown"),
                "task_id": getattr(task_instance, "task_id", "unknown"),
                "error_type": type(exception).__name__ if exception else "unknown",
                "execution_date": str(context.get("logical_date", "unknown")),
            }
        ).encode("utf-8")
        try:
            request = Request(
                webhook,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                if not 200 <= response.status < 300:
                    logger.error("Failure alert webhook returned HTTP %s", response.status)
        except (OSError, URLError):
            logger.exception("Failure alert webhook delivery failed")


DEFAULT_ARGS = {
    "owner": "fraud-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=90),
    "on_failure_callback": report_failure,
}

with DAG(
    dag_id="creditcard_fraud_training",
    description="Validate data, train a candidate, gate evaluation, and record analytics",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@weekly",
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["fraud", "mlops"],
) as dag:
    dataset_validation = BashOperator(
        task_id="dataset_validation",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.training validate",
    )
    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.training prepare",
    )
    feature_validation = BashOperator(
        task_id="feature_validation",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.training features",
    )
    model_training = BashOperator(
        task_id="model_training_candidate",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "PROMOTE_MODEL=false MLFLOW_LOG_DURING_TRAINING=false python train_real.py"
        ),
    )
    mlflow_recording = BashOperator(
        task_id="mlflow_metric_and_artifact_recording",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.training track",
    )
    evaluation_gate = BashOperator(
        task_id="evaluation_gate",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.training gate",
    )
    batch_analytics = BashOperator(
        task_id="batch_analytics",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.training analytics",
    )
    completion = BashOperator(
        task_id="pipeline_completion",
        bash_command="echo 'Fraud dataset validation, model evaluation gate, and analytics completed.'",
    )

    dataset_validation >> data_preparation >> feature_validation
    feature_validation >> model_training >> mlflow_recording
    mlflow_recording >> evaluation_gate >> batch_analytics >> completion

with DAG(
    dag_id="creditcard_audit_retention",
    description="Delete prediction audit metadata beyond the configured retention period",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["fraud", "retention"],
) as retention_dag:
    purge_prediction_audits = BashOperator(
        task_id="purge_expired_prediction_audits",
        bash_command=f"cd {PROJECT_ROOT} && python -m fraud_platform.retention",
    )
