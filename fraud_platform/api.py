from __future__ import annotations

from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timezone
import hmac
import json
import logging
import os
import re
from threading import Lock
import time
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy.exc import SQLAlchemyError

from fraud_platform.config import settings
from fraud_platform.db.database import (
    check_database,
    record_deployment,
    record_predictions,
)
from fraud_platform.drift import DriftMonitor
from fraud_platform.model import FraudModel

MODEL = FraudModel.load(settings.model_path, settings.metadata_path, settings.threshold)
REQUESTS = Counter(
    "creditcard_api_requests_total",
    "HTTP requests handled by the fraud API",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "creditcard_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "method"],
)
PREDICTIONS = Counter(
    "creditcard_prediction_total",
    "Number of fraud model decisions",
    ["decision"],
)
INFERENCE_LATENCY = Histogram(
    "creditcard_inference_latency_seconds",
    "Time spent in model inference and audit persistence",
)
RATE_LIMITED = Counter(
    "creditcard_api_rate_limited_total",
    "Requests rejected by the process-local client rate limit",
)
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_MAX_RATE_LIMIT_CLIENTS = 10_000
_rate_limit_lock = Lock()
_client_request_times: dict[str, deque[float]] = {}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
request_id_pattern = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "endpoint"):
            payload["endpoint"] = record.endpoint
        if hasattr(record, "status_code"):
            payload["status_code"] = record.status_code
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = record.duration_ms
        return json.dumps(payload, separators=(",", ":"))


logger = logging.getLogger("fraud_platform.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
logger.propagate = False

try:
    DRIFT_MONITOR: DriftMonitor | None = DriftMonitor.from_file(
        settings.drift_baseline_path,
        MODEL.feature_columns,
        window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "1000")),
        minimum_samples=int(os.getenv("DRIFT_MINIMUM_SAMPLES", "100")),
    )
except FileNotFoundError:
    DRIFT_MONITOR = None
    logger.warning("feature_drift_baseline_unavailable")


def _is_rate_limited(client_id: str, now: float) -> bool:
    with _rate_limit_lock:
        if client_id not in _client_request_times:
            if len(_client_request_times) >= _MAX_RATE_LIMIT_CLIENTS:
                oldest_client = min(
                    _client_request_times,
                    key=lambda key: _client_request_times[key][-1],
                )
                del _client_request_times[oldest_client]
            _client_request_times[client_id] = deque()
        request_times = _client_request_times[client_id]
        cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
        while request_times and request_times[0] <= cutoff:
            request_times.popleft()
        if len(request_times) >= settings.max_requests_per_minute:
            return True
        request_times.append(now)
        return False


class CreditCardTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str = Field(..., min_length=1, max_length=100)
    Time: float = Field(..., strict=True, ge=0, allow_inf_nan=False)
    V1: float = Field(..., strict=True, allow_inf_nan=False)
    V2: float = Field(..., strict=True, allow_inf_nan=False)
    V3: float = Field(..., strict=True, allow_inf_nan=False)
    V4: float = Field(..., strict=True, allow_inf_nan=False)
    V5: float = Field(..., strict=True, allow_inf_nan=False)
    V6: float = Field(..., strict=True, allow_inf_nan=False)
    V7: float = Field(..., strict=True, allow_inf_nan=False)
    V8: float = Field(..., strict=True, allow_inf_nan=False)
    V9: float = Field(..., strict=True, allow_inf_nan=False)
    V10: float = Field(..., strict=True, allow_inf_nan=False)
    V11: float = Field(..., strict=True, allow_inf_nan=False)
    V12: float = Field(..., strict=True, allow_inf_nan=False)
    V13: float = Field(..., strict=True, allow_inf_nan=False)
    V14: float = Field(..., strict=True, allow_inf_nan=False)
    V15: float = Field(..., strict=True, allow_inf_nan=False)
    V16: float = Field(..., strict=True, allow_inf_nan=False)
    V17: float = Field(..., strict=True, allow_inf_nan=False)
    V18: float = Field(..., strict=True, allow_inf_nan=False)
    V19: float = Field(..., strict=True, allow_inf_nan=False)
    V20: float = Field(..., strict=True, allow_inf_nan=False)
    V21: float = Field(..., strict=True, allow_inf_nan=False)
    V22: float = Field(..., strict=True, allow_inf_nan=False)
    V23: float = Field(..., strict=True, allow_inf_nan=False)
    V24: float = Field(..., strict=True, allow_inf_nan=False)
    V25: float = Field(..., strict=True, allow_inf_nan=False)
    V26: float = Field(..., strict=True, allow_inf_nan=False)
    V27: float = Field(..., strict=True, allow_inf_nan=False)
    V28: float = Field(..., strict=True, allow_inf_nan=False)
    Amount: float = Field(..., strict=True, ge=0, le=1_000_000, allow_inf_nan=False)


if tuple(CreditCardTransaction.model_fields) != ("transaction_id", *MODEL.feature_columns):
    raise ValueError("API request feature order does not match the loaded model metadata")


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud: bool
    fraud_probability: float
    threshold: float
    decision: str
    model_version: str
    request_id: str
    explanation: dict[str, Any] | None = None


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[CreditCardTransaction] = Field(
        min_length=1, max_length=settings.max_batch_size
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str


class RequestBodyTooLarge(Exception):
    pass


class RequestBodySizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(send, 400, "Invalid Content-Length")
                return
            if declared_length < 0:
                await self._reject(send, 400, "Invalid Content-Length")
                return
            if declared_length > self.max_bytes:
                await self._reject(send, 413, "Request body exceeds the configured size limit.")
                return

        received_bytes = 0

        async def receive_limited() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, receive_limited, send)
        except RequestBodyTooLarge:
            await self._reject(send, 413, "Request body exceeds the configured size limit.")

    @staticmethod
    async def _reject(send: Any, status_code: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    try:
        record_deployment(
            model_version=MODEL.model_version,
            deployed_at=datetime.now(timezone.utc),
            artifact_uri=str(settings.model_path),
            threshold=MODEL.threshold,
            metadata={
                "dataset": MODEL.metadata.get("dataset", "unknown"),
                "feature_count": len(MODEL.feature_columns),
            },
        )
    except SQLAlchemyError:
        logger.warning("model_deployment_record_unavailable")
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Fraud inference using the preserved Kaggle Logistic Regression pipeline. "
        "Scores are uncalibrated unless separately evaluated; they are not guarantees."
    ),
    version=MODEL.model_version,
    lifespan=lifespan,
)
app.add_middleware(
    RequestBodySizeLimitMiddleware,
    max_bytes=settings.max_request_bytes,
)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def request_observability(request: Request, call_next: Any):
    request_id = request.headers.get("X-Request-ID", "")
    if not request_id_pattern.fullmatch(request_id):
        request_id = str(uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    try:
        client_id = request.client.host if request.client is not None else "unknown"
        limited = (
            request.url.path in {"/predict", "/predict/batch"}
            and _is_rate_limited(client_id, time.monotonic())
        )
        if limited:
            RATE_LIMITED.inc()
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Request quota exceeded; retry after the rate-limit window.",
                    "request_id": request_id,
                },
                headers={"Retry-After": str(int(_RATE_LIMIT_WINDOW_SECONDS))},
            )
        else:
            response = await call_next(request)
        status_code = response.status_code
    except Exception:
        logger.exception(
            "unhandled_request_error",
            extra={"request_id": request_id, "endpoint": "unmatched"},
        )
        raise
    finally:
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        REQUESTS.labels(endpoint=endpoint, method=request.method, status=str(status_code)).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method=request.method).observe(elapsed)
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": round(elapsed * 1000, 3),
            },
        )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "loc": error.get("loc", ()),
                    "msg": error.get("msg", "Invalid value"),
                    "type": error.get("type", "value_error"),
                }
                for error in exc.errors()
            ],
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction authentication is not configured on this server.",
        )
    if not api_key or not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


def prediction_results(
    transactions: list[CreditCardTransaction],
    request_id: str,
    include_explanation: bool = False,
) -> list[PredictionResponse]:
    started = time.perf_counter()
    try:
        scored = MODEL.predict(
            [
                {feature: getattr(transaction, feature) for feature in MODEL.feature_columns}
                for transaction in transactions
            ]
        )
        explanations = (
            [
                MODEL.explain(
                    {
                        feature: getattr(transaction, feature)
                        for feature in MODEL.feature_columns
                    }
                )
                for transaction in transactions
            ]
            if include_explanation
            else [None] * len(transactions)
        )
        now = datetime.now(timezone.utc)
        try:
            record_predictions(
                [
                    {
                        "event_id": None,
                        "transaction_id": transaction.transaction_id,
                        "request_id": request_id,
                        "created_at": now,
                        "fraud_probability": result["fraud_probability"],
                        "threshold": MODEL.threshold,
                        "decision": result["decision"],
                        "model_version": MODEL.model_version,
                    }
                    for transaction, result in zip(transactions, scored, strict=True)
                ]
            )
        except SQLAlchemyError as exc:
            logger.error("prediction_audit_write_failed", extra={"request_id": request_id})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction audit storage is unavailable; the result was not returned.",
            ) from exc
        if DRIFT_MONITOR is not None:
            for transaction in transactions:
                DRIFT_MONITOR.observe(
                    {
                        feature: getattr(transaction, feature)
                        for feature in MODEL.feature_columns
                    }
                )
        responses: list[PredictionResponse] = []
        for transaction, result, explanation in zip(
            transactions, scored, explanations, strict=True
        ):
            decision = result["decision"]
            PREDICTIONS.labels(decision=decision).inc()
            responses.append(
                PredictionResponse(
                    transaction_id=transaction.transaction_id,
                    fraud=decision == "fraud",
                    fraud_probability=round(result["fraud_probability"], 6),
                    threshold=round(MODEL.threshold, 6),
                    decision=decision,
                    model_version=MODEL.model_version,
                    request_id=request_id,
                    explanation=explanation,
                )
            )
        return responses
    finally:
        INFERENCE_LATENCY.observe(time.perf_counter() - started)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    if not check_database():
        raise HTTPException(status_code=503, detail="Database is not ready")
    return {
        "status": "ready",
        "model_version": MODEL.model_version,
        "feature_count": len(MODEL.feature_columns),
        "database": "available",
    }


@app.get("/health/database")
def database_health() -> dict[str, str]:
    if not check_database():
        raise HTTPException(status_code=503, detail="Relational database is unavailable")
    return {"status": "ok", "database": settings.database_url.split(":", 1)[0]}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    transaction: CreditCardTransaction,
    request: Request,
    explain: bool = False,
    _: str = Depends(verify_api_key),
) -> PredictionResponse:
    try:
        return prediction_results(
            [transaction],
            request.state.request_id,
            include_explanation=explain,
        )[0]
    except RuntimeError as exc:
        if explain:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        raise


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    batch: BatchPredictionRequest,
    request: Request,
    _: str = Depends(verify_api_key),
) -> BatchPredictionResponse:
    predictions = prediction_results(batch.transactions, request.state.request_id)
    return BatchPredictionResponse(predictions=predictions, model_version=MODEL.model_version)
