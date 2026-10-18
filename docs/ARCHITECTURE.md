# Fraud detection platform architecture

## Data flow

```text
                         +--------------------------+
                         | Airflow ML pipeline      |
                         | validate -> prepare ->   |
                         | train -> eval gate       |
                         +------------+-------------+
                                      | model + metrics
                                      v
                                 MLflow tracking

HTTP transaction --> FastAPI --> existing sklearn pipeline
                     |  |             |
                     |  |             +--> score + configured threshold decision
                     |  +--> PostgreSQL prediction/deployment metadata
                     +--> Prometheus /metrics --> Grafana
                     +--> rolling in-memory feature PSI/KL --> Prometheus alert
                     |
                     +--> Kafka transactions.raw (optional asynchronous path)
                              |
                         scoring worker
                              +--> transactions.scored
                              +--> fraud.alerts (above threshold)
                              +--> PostgreSQL minimal relational audit
                              +--> Cassandra event history
                              +--> transactions.dlq after exhausted retries

creditcard.csv --> Spark validation, analytics and batch scoring
       +--------> Feast offline Parquet source using exact model feature columns
```

## Storage responsibilities

- **PostgreSQL** is the relational source of truth for prediction audit metadata and
  model deployment records. Raw model inputs are not stored.
- **Cassandra** stores append-oriented scored transaction events for event/time-window
  access patterns. The schema is partitioned by UTC day and event ID; records are not
  blindly duplicated from synchronous HTTP predictions.
- **MLflow** stores training runs, metrics, model artifacts and model version metadata.
- **Feast** uses the real dataset columns as offline features. Since the dataset has no
  stable card/customer key, the offline entity is a dataset row key and online lookup
  is not inserted into live scoring.
- **Drift monitoring** compares finite live model inputs against decile boundaries
  fitted on the model's training partition. It keeps a bounded in-memory window and
  exports PSI/KL gauges; no transaction feature values are retained or labeled in
  metrics.
- **SQLite** is supported only for local API development/tests. Compose configures
  PostgreSQL.

## Deployments

- Docker Compose keeps the API/PostgreSQL/Prometheus/Grafana core small and places Kafka,
  Cassandra, Spark, Airflow and BentoML behind opt-in profiles.
- Kubernetes manifests deploy stateless API and scoring worker workloads. PostgreSQL,
  Cassandra, Kafka, MLflow and monitoring should use managed services or separately
  operated stateful deployments in production. No public database service is created.
- Prometheus scrapes low-cardinality API, prediction, worker and dependency metrics.
  Grafana dashboards query only metrics defined by this code.
- BentoML wraps the existing model as an optional service; Seldon Core 2 uses a
  separately packaged MLServer sklearn artifact. Neither is a verified deployment,
  and FastAPI remains the primary serving contract.

## Trust and operational boundaries

- `FRAUD_API_KEY` protects prediction routes; readiness/health and metrics contain no
  transaction values or credentials.
- Predictions return an uncalibrated model score and a threshold decision. Thresholds
  are selected on validation data, not the held-out test set.
- API quotas are process-local and keyed by client IP; a production multi-replica
  deployment needs a shared gateway or distributed limiter.
- PSI drift alerts indicate feature-distribution changes only. They do not prove
  degraded predictive performance or trigger model retraining.
- Local `creditcard.csv` includes labels and should not be placed in container images.
- Docker Compose and Kubernetes files are configuration deliverables; actual service
  readiness must be verified on a machine with Docker/WSL2 or a Kubernetes cluster.
