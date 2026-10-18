# Credit Card Fraud Detection Platform

This repository extends the original real-data Logistic Regression demo into a
modular ML platform. The real Kaggle model stays the serving model unless a
separately trained candidate passes evaluation and an operator explicitly promotes
it. The independent synthetic-data demo (`app.py`, `train.py`) is retained.

## Verified data and model

- `data/creditcard.csv` is the Kaggle benchmark: **284,807 rows**, 30 numeric model
  inputs (`Time`, `V1`–`V28`, `Amount`) plus `Class`, zero missing values, and 492
  fraud rows. This was verified from the file, not inferred from its name.
- `models/creditcard_model.joblib` is a `StandardScaler` + class-balanced Logistic
  Regression pipeline. Its metadata declares the same 30 features, in order, and an
  F1-selected validation threshold of `0.99`.
- Existing held-out report: precision 0.674, recall 0.811, F1 0.736, ROC-AUC 0.968,
  average precision 0.793. These are benchmark results, not a production guarantee.
  Probability calibration has not been established.
- `data/sample_creditcard.csv` contains 50 real benchmark rows (40 legitimate and
  10 fraud) for quick Spark/data-contract checks. It is not a training split.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for data flow, service boundaries,
and storage responsibilities. The implementation inventory and remaining verification
checklist are in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

```text
HTTP --> FastAPI --> preserved sklearn pipeline --> PostgreSQL audit metadata
           |                                             ^
           +--> /metrics --> Prometheus --> Grafana      |
Kafka transactions.raw --> scoring worker ---------------+
                               +--> transactions.scored / fraud.alerts
                               +--> Cassandra scored-event history
                               +--> transactions.dlq after bounded retries

Kaggle CSV --> Spark validation/analytics/driver-batch scoring
           --> Feast offline feature source (same model columns; no card IDs)

Airflow validates/trains/evaluates candidates --> MLflow tracking
Docker Compose runs local services; Kubernetes deploys stateless API/worker
```

## Technology and implementation map

| Technology | Implemented in |
|---|---|
| FastAPI, Pydantic, Uvicorn | `fraud_platform/api.py`, `app_real.py` |
| Existing preprocessing/model pipeline | `fraud_platform/model.py`, `models/` |
| SQLAlchemy, PostgreSQL, Alembic | `fraud_platform/db/`, `alembic/` |
| Cassandra | `fraud_platform/cassandra/`, `cassandra/schema.cql` |
| Kafka | `fraud_platform/streaming/`, `schemas/transaction-event.schema.json` |
| Apache Spark | `fraud_platform/spark_jobs.py`, `requirements-spark.txt` |
| Great Expectations quality gate | `fraud_platform/quality.py`, `requirements-quality.txt` |
| Feast | `feature_repo/`, `scripts/build_feast_source.py` |
| Airflow | `airflow/dags/creditcard_ml_pipeline.py` |
| MLflow and evaluation gate | `train_real.py`, `fraud_platform/training.py` |
| Random Forest, XGBoost and Optuna candidates | `fraud_platform/advanced_training.py`, `requirements-advanced-ml.txt` |
| SHAP API explanations | `fraud_platform/model.py`, `fraud_platform/api.py` |
| Prometheus / Grafana | `monitoring/` |
| Feature drift metrics and alert | `fraud_platform/drift.py`, `monitoring/prometheus/alerts.yml` |
| BentoML service | `fraud_platform/bento_service.py`, `Dockerfile.bentoml` |
| Seldon Core 2 packaging/manifest | `scripts/package_seldon_model.py`, `k8s/seldon-core2-model.yaml` |
| Docker / Compose | `Dockerfile*`, `docker-compose.yml` |
| Kubernetes | `k8s/` |
| Tests / CI | `tests/`, `.github/workflows/ci.yml` |

XGBoost/Optuna candidate training, SHAP explanations, BentoML packaging, and Seldon
Core 2 packaging are implemented, but their optional serving integrations have not
been deployed here. PyTorch, Spark Structured Streaming, distributed Spark inference,
and a production online Feast lookup are not implemented. See the PDF traceability
table in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for tested,
partially implemented, and blocked requirements.

## Local Python setup

Requires Python 3.12. From PowerShell in this project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:FRAUD_API_KEY = "replace-with-a-long-random-local-key"
# Optional override; otherwise the model metadata threshold (0.99) is used.
# $env:FRAUD_THRESHOLD = "0.99"
python -m alembic upgrade head
python -m uvicorn app_real:app --reload --port 8001
```

### Native Windows desktop application

The Tkinter desktop app uses the existing `app_real.py` FastAPI service and the
active Kaggle-model artifacts. From Windows CMD, run these commands from the project
root:

```cmd
cd /d "C:\Users\nares\Downloads\ML Project"
.venv\Scripts\activate.bat
python run_app.py
```

The launcher starts the API on `127.0.0.1:8001`, waits for `/ready`, and opens the
desktop window without opening a browser. It reuses a matching ready API if one is
already running and stops only a backend process it started when the window closes.
Set `FRAUD_API_KEY` in the environment to use a chosen local key; if it is unset, the
launcher generates a temporary random key for a new local backend. When connecting
to an already-running API without an environment key, the app asks for it in a
masked prompt. Optional launcher settings are `FRAUD_DESKTOP_PORT` and
`FRAUD_DESKTOP_STARTUP_TIMEOUT` (seconds). The active model metadata and model file
must be present, and the API's configured audit database must be available for
readiness and predictions. If that database has not been initialized yet, run
`python -m alembic upgrade head` once with the same database environment before
launching the desktop app.

Open:

- Swagger UI: `http://127.0.0.1:8001/docs`
- Liveness: `http://127.0.0.1:8001/health`
- Readiness (requires a migrated, reachable database): `http://127.0.0.1:8001/ready`
- Relational DB health: `http://127.0.0.1:8001/health/database`
- Prometheus metrics: `http://127.0.0.1:8001/metrics`

Without `DATABASE_URL`, local mode uses a separate SQLAlchemy SQLite database at
`data/fraud_platform.sqlite3`; it does not reuse or replace the original prediction
audit database. For PostgreSQL, set `DATABASE_URL` to a SQLAlchemy URL such as
`postgresql+psycopg://user:password@localhost:5432/fraud_platform` before running
Alembic and the API. URL-encode special characters in credentials.

Authorize with `X-API-Key` in Swagger. The key must be configured using
`FRAUD_API_KEY`; prediction requests return 503 if server authentication has not been
configured. The API accepts exactly the model's declared numeric features and
`transaction_id`. Batches contain 1–100 transactions by default (`MAX_BATCH_SIZE`).
Requests larger than 1 MiB are rejected by default (`MAX_REQUEST_BYTES`).
The prediction score is not asserted to be calibrated.
The default process-local client quota is `MAX_REQUESTS_PER_MINUTE=120`; it is not a
shared quota across API replicas. The request-size middleware also checks streamed
payloads without a `Content-Length` header.

### Example request

```json
{
  "transaction_id": "TX-DEMO-001",
  "Time": 406.0,
  "V1": 0.1, "V2": 0.1, "V3": 0.1, "V4": 0.1, "V5": 0.1, "V6": 0.1,
  "V7": 0.1, "V8": 0.1, "V9": 0.1, "V10": 0.1, "V11": 0.1, "V12": 0.1,
  "V13": 0.1, "V14": 0.1, "V15": 0.1, "V16": 0.1, "V17": 0.1, "V18": 0.1,
  "V19": 0.1, "V20": 0.1, "V21": 0.1, "V22": 0.1, "V23": 0.1, "V24": 0.1,
  "V25": 0.1, "V26": 0.1, "V27": 0.1, "V28": 0.1,
  "Amount": 12.5
}
```

```powershell
$headers = @{ "X-API-Key" = $env:FRAUD_API_KEY }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/predict `
  -Headers $headers -ContentType "application/json" `
  -InFile .\examples\transaction.json
```

## Train, evaluate, and promote safely

```powershell
python train_real.py
```

Training splits data into stratified train/validation/test sets. It selects the
classification threshold on validation predictions only and evaluates on the held-out
test set. Each run saves versioned candidate model/metadata/metrics below
`models/candidates/` and `reports/candidates/`; the active serving artifacts are not
overwritten by default. MLflow stores parameters, metrics, confusion-matrix/report
artifacts, feature metadata, and a registered candidate when the evaluation gate passes.

Promotion requires both passing configured minimum metrics and explicit consent:

```powershell
$env:PROMOTE_MODEL = "true"
python train_real.py
```

The defaults require F1 >= 0.65, recall >= 0.70, and precision >= 0.60. Configure with
`MIN_MODEL_F1`, `MIN_MODEL_RECALL`, and `MIN_MODEL_PRECISION`. A promotion first writes
the previous model, metadata, and report under `models/backups/<candidate-version>/`.
To roll back manually, copy the intended backup model and metadata back into `models/`
while the API is stopped, then restart it. No model is promoted automatically by Airflow.

The default persistent MLflow tracking URI points to `mlflow.db` in the project root;
set `MLFLOW_TRACKING_URI` to use a remote tracking server and suitable artifact store.
Treat Joblib artifacts as trusted executable pickle data: load only artifacts from a
trusted source and keep the training/serving scikit-learn versions compatible.

Candidate comparison with the Random Forest baseline and Optuna-tuned XGBoost is
available separately; it writes candidate artifacts and never replaces the active
model unless explicitly requested and every promotion gate passes:

```powershell
python -m pip install -r requirements-advanced-ml.txt
python -m fraud_platform.advanced_training --trials 10
```

This full-data benchmark can be resource intensive. The automated test uses only a
small synthetic fixture; no advanced candidate was trained on the full dataset during
this audit, so no comparative performance claim is made. The existing Logistic
Regression benchmark remains the active model.

SHAP is optional and can be requested for a single transaction with
`POST /predict?explain=true`; install `requirements-advanced-ml.txt` first. The API
returns per-feature contributions with an explicit output scale (log-odds for the
current Logistic Regression model). SHAP values explain the model score, not causality.

### Feature drift

The checked-in baseline is computed from only the same stratified 70% training
partition used by the real-data training contract; it includes model inputs, not raw
transactions or labels. To regenerate it after changing the dataset or feature
contract, run:

```powershell
python -m scripts.build_drift_baseline
```

Successful predictions update rolling-window PSI/KL gauges at `/metrics`. Prometheus
alerts when a feature's PSI remains above 0.2 for 15 minutes. This is an input
distribution warning, not a performance/degradation diagnosis: the dataset has no
online labels, and alerts do not automatically retrain or roll back a model.

## PostgreSQL and migrations

The API writes minimal audit metadata (IDs, UTC time, score, threshold, decision, model
version); it does not persist the 30 input features. `model_deployments` records model
version metadata. PostgreSQL is used in Compose; SQLite is only for local
development/tests.

```powershell
$env:DATABASE_URL = "postgresql+psycopg://fraud:password@localhost:5432/fraud_platform"
python -m alembic upgrade head
python -m alembic current
```

Alembic owns schema changes. The application reports readiness false if the database or
required tables are unavailable. Prediction writes are transactional; if the audit
database fails, the API returns 503 rather than silently returning an unrecorded result.

## Kafka and Cassandra (optional local profile)

Install the Python clients when running the worker outside Docker:

```powershell
pip install -r requirements-streaming.txt
```

The documented topics are `transactions.raw`, `transactions.scored`, `fraud.alerts`,
and `transactions.dlq`. Events have a UUID `event_id`, timezone-aware
`event_timestamp`, schema version `1.0`, and validated model transaction. See
`schemas/transaction-event.schema.json`.

Start the broker, Cassandra and worker with Compose (full command below), then publish
an event using:

```powershell
python -m fraud_platform.streaming.publish_event --transaction-json .\examples\transaction.json
```

Use `--event-id <uuid>` to retain the same id when retrying a source event. The worker
commits the Kafka offset only after processing or successful DLQ publication. Retries
are bounded; PostgreSQL uses a unique event ID and Cassandra uses an idempotent primary
key. Delivery is at-least-once: downstream event consumers should deduplicate by
`event_id`. Cassandra stores scored events partitioned by UTC event day and clustered
by event timestamp; PostgreSQL is for relational audit/deployment queries. The
benchmark has no customer/card IDs, so no customer-history partitions or velocity
features are invented. Protect the DLQ topic because it retains the original event for
replay.

## Spark jobs

Spark uses lazy imports so the API does not require Java/Spark. Install
`requirements-spark.txt` in a compatible Python environment and install Java 17:

```powershell
pip install -r requirements-spark.txt
python -m fraud_platform.spark_jobs validate --input data\sample_creditcard.csv `
  --output data\processed\sample_creditcard
python -m fraud_platform.spark_jobs aggregate --input data\processed\sample_creditcard `
  --output reports\sample_spark_summary.json
python -m fraud_platform.spark_jobs score --input data\processed\sample_creditcard `
  --output reports\sample_batch_scores.csv
```

`validate` writes accepted Parquet rows and a separate rejected-record path.
`aggregate` summarizes transaction and fraud counts. `score` reuses the persisted
scikit-learn preprocessing pipeline, but iterates Spark rows on the driver in bounded
chunks; it is **not distributed inference**. Dataset labels are preserved; input
features are never fitted or normalized outside the saved pipeline.

## Feast feature store

Feast is isolated from the API so the live prediction path never performs a lookup
without a genuine entity key. The public data has no card/customer key. `dataset_row_id`
is only a surrogate row key for this offline benchmark. Feast requires timestamps; the
builder converts the real relative `Time` seconds into a UTC Unix-epoch-relative
ordering key. That is explicitly not a real-world calendar timestamp or historical
customer activity.

```powershell
py -3.12 -m venv .venv-feast
.\.venv-feast\Scripts\Activate.ps1
python -m pip install -r requirements-feast.txt
python -c "import feast, pandas, pyarrow; print(feast.__version__, pandas.__version__, pyarrow.__version__)"
python -m scripts.build_feast_source --input data\sample_creditcard.csv `
  --output feature_repo\data\creditcard_features.parquet
Set-Location feature_repo
feast apply
feast materialize-incremental 2026-01-01T00:00:00
Set-Location ..
```

Use the full `data/creditcard.csv` instead of the sample for complete offline
materialization. Local SQLite online storage is for development only.

## Airflow

The Compose Airflow profile uses a separate PostgreSQL metadata database and the
`LocalExecutor`. It runs validation, manifest preparation, model-feature validation,
candidate training, evaluation gate, chunked batch analytics, and completion/failure
reporting. Tasks pass paths/manifests rather than data frames through XCom. The gate
blocks the DAG before analytics/completion if the candidate misses the configured
minimums. Training records MLflow runs but leaves the active model untouched.

Run it with the `airflow` Compose profile below; the web UI is
`http://127.0.0.1:8081`. Configure the Airflow UI, metadata DB, and webserver secret
in `.env`; use a secrets backend for production. Airflow's package constraints are
isolated in `Dockerfile.airflow` because they conflict with a general-purpose API
environment. That image also installs the pinned Great Expectations dependency for
dataset validation. To run the data-validation pipeline directly in a local Python
environment, install `requirements-quality.txt` before running
`python -m fraud_platform.training validate`.

## Docker Compose

Docker Desktop with WSL2 is required on Windows. Copy `.env.example` to `.env` and
replace all placeholders (never commit `.env`). Local endpoints bind to loopback.
Approximate memory guidance: 4 GB for core, 8 GB with streaming/Cassandra, 10+ GB with
Airflow or Spark; actual resource needs depend on data and host. Use URL-safe
alphanumeric PostgreSQL passwords in this Compose example because the connection URL
is assembled from environment values; other deployments should URL-encode credentials.

```powershell
Copy-Item .env.example .env
# Edit .env and set distinct local passwords and a long random FRAUD_API_KEY.
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

## Share a public demo with Render

The API can be deployed with the root `render.yaml` Blueprint. The Blueprint creates
a public Docker web service and a private managed PostgreSQL database, sets a generated
API key as a Render secret, and uses `/ready` as the health check. The free PostgreSQL
plan is temporary and may expire; check Render's current plan terms before creating
resources. Keep the generated `FRAUD_API_KEY` private and share it separately from the
public app URL. Anyone with that key can submit prediction requests.

1. Create a private GitHub repository and upload the project. Do not upload `.env`,
   credentials, local databases, or private keys; `.gitignore` excludes these.
2. In Render, choose **New > Blueprint**, connect the repository, and deploy
   `render.yaml`. Review the plan/pricing and database lifetime before confirming.
3. Wait until the web service is Live and `/ready` succeeds. The shareable Swagger URL
   is `https://<your-render-service>.onrender.com/docs`.
4. In Render's service environment settings, reveal/copy `FRAUD_API_KEY` and send it
   to the recipient through a private channel. In Swagger, click **Authorize** and
   enter that key before trying `POST /predict`.

Render deployments, pricing, and availability are controlled by the account owner;
this project workspace cannot publish or administer the service on your behalf. The
local API URL `http://127.0.0.1:8001` only works on this computer and is not shareable.

Core URLs: API `http://127.0.0.1:8001/docs`, Prometheus
`http://127.0.0.1:9090`, Grafana `http://127.0.0.1:3000`. Grafana user is `admin`;
password comes from `GRAFANA_ADMIN_PASSWORD`.

Opt-in services:

```powershell
docker compose --profile mlflow up --build -d
docker compose --profile streaming up --build -d
docker compose --profile spark run --rm spark
docker compose --profile airflow up --build -d
```

Kafka UI is `http://127.0.0.1:8080`; MLflow is `http://127.0.0.1:5000`; Airflow is
`http://127.0.0.1:8081`. Stop containers without removing persisted volumes:

```powershell
docker compose --profile streaming --profile mlflow --profile spark --profile airflow down
```

Persistent volumes are intentionally retained. Removing volumes would delete local
database/tracking data; inspect and back up before any `down -v`.

## Kubernetes

`k8s/` deploys the stateless API and Kafka scoring worker with probes, resource bounds,
least-privilege service account settings, ConfigMap, Secret references and CPU HPA.
PostgreSQL, Kafka, Cassandra, MLflow, Prometheus and Grafana are external/independently
operated services; databases are not exposed by these manifests. The local kind guide,
secret creation, validation, ingress opt-in, and rollback commands are in
[k8s/README.md](k8s/README.md). No cluster was available to verify deployment.

## Tests, monitoring and security

```powershell
python -m pytest -q tests
python -m compileall -q app_real.py train_real.py fraud_platform scripts airflow\dags feature_repo\creditcard_fraud
```

Prometheus scrapes measured API request/latency/error/prediction metrics and the
Kafka worker success/failure/dead-letter counters plus rolling feature PSI/KL. It
does not claim consumer lag.
Grafana provisions a dashboard and Prometheus data source automatically. Alerts cover
API down, elevated API 5xx rate, DLQ events, and sustained feature PSI drift.

The reproducible Kafka load test targets the brief's 10,000 events/minute and reports
observed throughput, end-to-end event-age p50/p95/p99, errors, dead letters, and
backlog. It requires a running Kafka worker stack and has not been run in this
environment:

```powershell
python -m scripts.kafka_load_test --events 10000 --target-per-minute 10000 `
  --timeout-seconds 180 --output reports\kafka_load_test.json
```

BentoML is an optional local serving alternative (port 3001); FastAPI remains the
verified primary API:

```powershell
docker compose --profile bentoml up --build -d bentoml
```

The Seldon Core 2 manifest is intentionally excluded from default Kustomize resources.
Package the active model with `python -m scripts.package_seldon_model`, upload that
directory to private object storage, replace the manifest placeholder URI, and deploy
only into a cluster with the matching Seldon Core 2 CRDs/runtime.

The Airflow daily retention DAG deletes audit metadata older than 90 days by default
(`PREDICTION_AUDIT_RETENTION_DAYS`). Customer velocity/graph features and demographic
fairness audits remain blocked because the benchmark has no customer/card IDs or
demographic attributes; do not treat dataset row IDs as people or cards.

API logs contain request ID, route template, status, and duration only; request bodies,
features, credentials, and identifiers are not used as metric labels. Keep credentials
in environment variables/secrets, keep `.env` and local database files untracked, and
do not expose API, database, broker, or dashboards publicly without authentication and
TLS. The sample Compose/Kafka/Cassandra configuration is for local development, not a
production security baseline.

## Status and limitations

The API/SQLite test path, dataset contracts, model scoring, evaluation gate, event
processing logic with fakes, database migration, and Great Expectations schema
checks are locally testable. Docker, Java, and `kubectl` were absent during
implementation, and Kafka/Cassandra/Spark/Feast/Airflow services have not been
launched or integration-tested here. Consult the checklist in
`docs/IMPLEMENTATION_PLAN.md` for evidence and remaining prerequisites.
