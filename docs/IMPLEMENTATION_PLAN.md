# Platform inventory and implementation plan

## Verified starting point

| Area | Current state | Evidence |
|---|---|---|
| Real fraud dataset | Present; 284,807 rows, 31 columns, zero nulls, 492 fraud labels | `data/creditcard.csv` inspected with pandas |
| Real model | Present; Logistic Regression pipeline, 30 ordered features, threshold 0.99 | `models/creditcard_model.joblib`, `models/creditcard_model_metadata.json` |
| Evaluation | Present; validation-selected threshold and held-out metrics | `reports/creditcard_metrics.json` |
| Real API | Partially implemented; single prediction, API key default, SQLite audit, Prometheus | `app_real.py` |
| Synthetic demo API/training | Present and separate; preserve unchanged | `app.py`, `train.py`, `generate_data.py` |
| MLflow | Partially implemented; local tracking artifacts/database exist | `train_real.py`, `mlflow.db`, `mlruns/` |
| PostgreSQL/Alembic | Implemented; migration locally applied against separate SQLite DB | `fraud_platform/db/`, `alembic/` |
| Cassandra/Kafka | Implemented; integration tests use fakes | `fraud_platform/cassandra/`, `fraud_platform/streaming/` |
| Spark/Feast/Airflow/Great Expectations | Spark/Feast/Airflow code/config added but runtimes unavailable; Great Expectations schema runtime-checked | `fraud_platform/spark_jobs.py`, `feature_repo/`, `airflow/dags/`, `fraud_platform/quality.py` |
| Prometheus/Grafana config | Config/dashboard added; scrape and rendering not integration-tested | `monitoring/` |
| Docker/Kubernetes/CI | Manifests/workflow added; Compose and cluster deployment not validated locally | `Dockerfile*`, `docker-compose.yml`, `k8s/`, `.github/workflows/ci.yml` |
| Tests/Git | Tests added; no Git repository is initialized | `tests/`; project root |
| Runtime constraints | Python 3.12 core `.venv`; Feast 0.48.0 installed and `feast apply` works in isolated `.venv-feast`; Docker/Java/kubectl unavailable | Selected interpreter, Feast CLI/materialization run |

The PDF brief is available as `03_Machine_Learning_AI.pdf`. It requires the broader stack listed in the user-provided project brief. The existing project is a starter, not a deployment of that full stack.

## Target architecture

```text
HTTP clients
    |
    v
FastAPI (API key, validation, request IDs, metrics)
    | synchronous scoring using unchanged sklearn pipeline
    +--> PostgreSQL (prediction metadata, deployment metadata)
    +--> optional Kafka producer --> transactions.raw
                                  --> scoring worker --> transactions.scored
                                                       +--> fraud.alerts
                                                       +--> Cassandra event history
    |
    +--> /metrics --> Prometheus --> provisioned Grafana dashboard

creditcard.csv --> Spark validation/analytics/batch score outputs
       |                  |
       +--> Feast offline Parquet source (same 30 model features; no fabricated IDs)

Airflow DAG: validate --> prepare/features --> train/evaluate gate --> MLflow
                                                      |
                                                      +--> optional batch scoring

Docker Compose: core services plus opt-in `streaming`, `analytics`, and `airflow`
profiles. Kubernetes manifests deploy the stateless API/worker and reference external
stateful services by default; stateful products are not falsely claimed deployed here.
```

## Technology-to-file map

| Requirement | Concrete implementation |
|---|---|
| FastAPI/model serving | `app_real.py`, `fraud_platform/model.py`, `fraud_platform/api.py` |
| PostgreSQL/SQLAlchemy/Alembic | `fraud_platform/db/`, `alembic.ini`, `alembic/` |
| Cassandra | `fraud_platform/cassandra/`, `cassandra/schema.cql` |
| Kafka | `fraud_platform/streaming/`, `schemas/transaction-event.schema.json` |
| Spark | `fraud_platform/spark_jobs.py` |
| Feast | `feature_repo/` |
| Airflow | `airflow/dags/` |
| MLflow/evaluation gate | `train_real.py`, `fraud_platform/training.py` |
| Advanced candidate training | `fraud_platform/advanced_training.py` |
| SHAP prediction explanations | `fraud_platform/model.py`, `fraud_platform/api.py` |
| Prometheus/Grafana | `monitoring/prometheus/`, `monitoring/grafana/` |
| Feature drift observations | `fraud_platform/drift.py`, `models/creditcard_feature_baseline.json` |
| BentoML/Seldon | `fraud_platform/bento_service.py`, `Dockerfile.bentoml`, `scripts/package_seldon_model.py`, `k8s/seldon-core2-model.yaml` |
| Docker | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.env.example` |
| Kubernetes | `k8s/` |
| Tests/CI | `tests/`, `.github/workflows/ci.yml` |
| Architecture/runbook | `docs/ARCHITECTURE.md`, `README.md` |

## Ordered implementation checklist

- [x] Inspect workspace, PDF, real dataset/model, existing API/training, dependencies and local tooling.
- [x] Establish shared configuration and model contract; preserve existing model artifacts and synthetic API.
- [x] Harden API, add readiness/batch prediction, structured request logging and measurable metrics.
- [x] Add SQLAlchemy audit/deployment records, Alembic migration, local SQLite mode and PostgreSQL health.
- [x] Add Cassandra event repository and query-oriented schema.
- [x] Add versioned event schema, Kafka producer/scoring worker, retries/DLQ and idempotent persistence.
- [x] Add Spark validation, analytics, feature preparation and local batch scoring jobs.
- [x] Apply Great Expectations data-quality suites on Airflow dataset-validation runs.
- [x] Add Feast entity/view/offline source and local feature repository commands.
- [x] Add modular Airflow DAG and evaluation gate.
- [x] Expand MLflow logging and explicit promotion/rollback gates without test leakage.
- [x] Add Random Forest and Optuna/XGBoost training candidates; require evaluation gates and explicit promotion.
- [x] Add bounded SHAP explanations and a training-only feature-drift baseline with Prometheus gauges/alert.
- [x] Add a repeatable Kafka load test for the 10,000-events/minute target.
- [x] Validate a dedicated Feast 0.48 environment and apply/materialize the sample local feature repository.
- [x] Enforce API request-size limits on streamed bodies without `Content-Length`.
- [x] Provision Prometheus scrape/alerts and Grafana dashboard using real metrics.
- [x] Add pinned service configuration and profile-based Docker Compose.
- [x] Add stateless Kubernetes manifests and external stateful-service references.
- [x] Add tests, lightweight CI, security defaults and full runbook.
- [x] Run focused tests and syntax/config validation; identify services not runnable on this host.

## PDF requirements traceability

| PDF requirement | Implementation and evidence | Status / remaining work |
|---|---|---|
| FastAPI model serving, validation, score and threshold | `fraud_platform/api.py`, `fraud_platform/model.py`; API tests include auth, invalid inputs, size limits, batches and scoring | Implemented and locally tested; database integration tested with SQLite, not PostgreSQL |
| API inference latency target (p99 below 100 ms) | `fraud_platform/api.py` exports request/inference latency histograms | Metric is implemented; no representative load test or p99 result is available |
| Logistic Regression and Random Forest baselines | Existing Logistic Regression artifact/report; `fraud_platform/advanced_training.py`; small-fixture candidate test | Implemented; no new full-dataset candidate metrics were produced in this audit |
| XGBoost tuning with Optuna | `fraud_platform/advanced_training.py`; test runs one Optuna trial against a small fixture | Implemented and fixture-tested; no full benchmark or promotion run |
| Class imbalance handling; SMOTE or cost-sensitive training | Logistic Regression `class_weight`; RF `balanced_subsample`; XGBoost `scale_pos_weight` computed from training labels | Implemented; no resampling occurs before split or on validation/test data |
| PyTorch model implementation | No PyTorch model or training path exists | Missing; it is listed in the PDF technology table but was not added |
| Validation threshold optimization and held-out evaluation | `train_real.py`, `fraud_platform/advanced_training.py`; tests validate split sizes/gates | Implemented; existing held-out metrics are below; advanced candidates unmeasured on full data |
| SHAP explanations from prediction API | `FraudModel.explain`, `POST /predict?explain=true`; tests check output contract and additive log-odds | Implemented and tested for active Logistic Regression; other model types and production latency not integration-tested |
| Kafka transaction processing and 10,000 events/minute | `fraud_platform/streaming/`, event schema, `scripts/kafka_load_test.py`; worker tests use fakes | Processing logic tested with fakes; no broker/load test run, so target achievement is unverified |
| PySpark preparation, aggregation and analytics | `fraud_platform/spark_jobs.py`; sample-contract tests | Code and contract tests exist; Spark runtime not run (Java unavailable). Customer velocity/graph features are blocked by absent IDs/history |
| Feast offline/online features and point-in-time retrieval | `feature_repo/`, builder script; isolated Feast 0.48 environment; `feast apply` succeeds on sample | Offline definition/apply verified; `online=False` by design, local materialization has zero online views. No meaningful live entity key or verified online serving |
| Great Expectations quality checks/failure handling | `fraud_platform/quality.py`; quality tests; Airflow task failure callback | Validation and surfaced failures tested locally; notification webhook not exercised |
| PostgreSQL relational records/migrations | SQLAlchemy models/repositories and Alembic; disposable SQLite migration and DB tests | Schema/migrations and SQLite behavior tested; PostgreSQL server unavailable for integration check |
| Cassandra event history | `fraud_platform/cassandra/`, CQL schema, repository tests with fakes | Code/schema and fake-backed behavior tested; no Cassandra service integration |
| BentoML and Seldon Core model deployment | `fraud_platform/bento_service.py`, `Dockerfile.bentoml`; packaging script and Seldon Core 2 manifest | Packaging/configuration implemented but not run; Bento dependency/container and Seldon cluster/object-store integration unverified |
| MLflow tracking/model registry | `train_real.py`, `fraud_platform/training.py`, local tracking configuration | Logging code exists; no new end-to-end tracking run or remote registry integration during this audit |
| Prometheus/Grafana monitoring | `monitoring/`; API/worker counters, request latency, PSI/KL gauges, dashboard and alert rules | Config files parse and metrics are exposed locally; live scrape/dashboard not run without Docker |
| Airflow orchestration | `airflow/dags/creditcard_ml_pipeline.py`, isolated image | DAG source exists; import/task execution unverified because Airflow and Docker are unavailable |
| Docker/Compose and Kubernetes deployment | Dockerfiles, `docker-compose.yml`, `k8s/` | Static manifests can be parsed; no Docker/Compose/kubectl/cluster installed, so image builds and readiness are unverified |
| Public demo hosting | `render.yaml`, Render instructions in `README.md` | Blueprint YAML parses and provisions a Docker web service plus managed PostgreSQL; no GitHub/Render account was connected, so public deployment is not yet created |
| A/B, shadow/champion comparison, canary rollback | No implementation found in this project | Missing; no traffic split, confidence comparison, or automatic canary rollback |
| Drift monitoring, performance alerts and retraining triggers | `fraud_platform/drift.py`, baseline artifact, Prometheus PSI alert | Feature-distribution drift implemented; labeled performance/drift-triggered retraining is missing |
| API quotas | Process-local IP-based sliding-window limit in `fraud_platform/api.py` | Basic quota tested; not distributed or per API key, so unsuitable as a multi-replica production quota |
| 90-day audit retention | `fraud_platform/retention.py`, database purge function and daily Airflow DAG | Purge logic tested; scheduler/service execution not verified |
| PII encryption and secure transport | API/database deployment configuration and environment-based secrets | No application-level PII persistence is needed for predictions, but encryption-at-rest and end-to-end TLS are deployment responsibilities and were not verified |
| Demographic fairness audits | No demographic fields exist in the benchmark | Blocked until approved demographic attributes and governance requirements are supplied |

These files/configurations provide a runnable development implementation, not evidence
that a production deployment or external-service integration is operational. The
original real-data model and dataset remain preserved.

### Verified existing model metrics

The existing held-out report (`reports/creditcard_metrics.json`) records threshold
`0.99`, precision `0.674157`, recall `0.810811`, F1 `0.736196`, ROC-AUC `0.967972`,
average precision `0.792873`, and confusion matrix TN/FP/FN/TP =
`42619/29/14/60`. These belong to the existing model/report; no competing model was
trained on the full dataset as part of this audit.

## Final verification status

| Status | Scope |
|---|---|
| Passed locally | 34 tests; core `pip check`; Python compilation; disposable SQLite Alembic upgrade; Feast 0.48 import/apply and local incremental-materialization command; 24 YAML and 14 JSON files parsed; Render Blueprint YAML parsed |
| Not run against services | PostgreSQL, Kafka end-to-end/load, Cassandra, Spark/Java, Airflow, MLflow server, Prometheus/Grafana containers, BentoML, Seldon, Docker Compose and Kubernetes |
| Partial / blocked | Feast online serving, distributed Spark inference, Kafka 10k/minute observed performance, A/B/canary, performance-based retraining, customer velocity/graph features, and demographic fairness |

## Known risks and boundaries

- Kafka, Cassandra, PostgreSQL, Airflow, Spark, and Prometheus/Grafana container checks need Docker/WSL2; Docker is not installed on the inspected host.
- Spark requires a compatible Java runtime; Java is not installed on the inspected host.
- Kubernetes deployment needs `kubectl` and a cluster; neither is installed.
- Airflow has strict dependency constraints and is deliberately isolated from the API's Python dependencies in Compose.
- Feast uses a separate ignored `.venv-feast`; its local SQLite/Parquet provider is for development, not a distributed production online store.
- Core `pip check` passes, but pip still warns about leftover invalid `~vicorn` backup metadata in `.venv\Lib\site-packages`; it was not removed because the environment may be shared by running processes.
- The Kaggle dataset has no card/customer identifier or trustworthy event-time history. No customer-level velocity or graph features will be invented.
- Model probabilities have not been calibrated. API results are scores, not calibrated risk guarantees.
- The current model artifact is small and remains unchanged. The 150 MB source dataset will not be copied into the API image.
- Without Git initialized, this workspace has no clean baseline or rollback mechanism; changes are additive wherever possible.

## Exact PowerShell validation commands

From the project root, using the selected core interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q app_real.py train_real.py fraud_platform scripts airflow\dags feature_repo\creditcard_fraud
.\.venv\Scripts\python.exe -m pytest -q tests
$env:DATABASE_URL = "sqlite:///./data/audit_migration_tmp.sqlite3"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
```

For a local integration run, install Docker Desktop with WSL2, Java 17 for Spark, and
`kubectl` plus a cluster for Kubernetes; then use the Compose, Kafka load-test, Spark,
Airflow and cluster commands in `README.md` and `k8s/README.md`. Never point audit
migration commands at a production database.
