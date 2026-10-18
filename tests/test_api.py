from __future__ import annotations

import json
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import fraud_platform.api as api
from fraud_platform.api import MODEL, app
from fraud_platform.config import settings


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    test_settings = replace(
        settings,
        api_key="test-secret",
        database_url=f"sqlite:///{(tmp_path / 'api-test.sqlite3').as_posix()}",
    )
    from fraud_platform.db import database

    monkeypatch.setattr(api, "settings", test_settings)
    monkeypatch.setattr(database, "settings", test_settings)
    database.get_engine.cache_clear()
    database.get_session_factory.cache_clear()
    from fraud_platform.db.models import Base

    Base.metadata.create_all(database.get_engine())
    with TestClient(app) as test_client:
        yield test_client
    database.get_engine.cache_clear()
    database.get_session_factory.cache_clear()


def transaction_payload() -> dict[str, object]:
    return {
        "transaction_id": "test-transaction-001",
        **{feature: 0.25 for feature in MODEL.feature_columns},
    }


def test_health_ready_and_metrics(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    assert client.get("/health/database").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_predict_requires_api_key(client: TestClient) -> None:
    response = client.post("/predict", json=transaction_payload())
    assert response.status_code == 401


def test_predict_returns_valid_score_and_request_id(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json=transaction_payload(),
        headers={"X-API-Key": "test-secret", "X-Request-ID": "request-test-1"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["transaction_id"] == "test-transaction-001"
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"fraud", "legitimate"}
    assert result["fraud"] is (result["decision"] == "fraud")
    assert result["request_id"] == "request-test-1"
    assert response.headers["X-Request-ID"] == "request-test-1"


def test_predict_rejects_missing_extra_and_nonfinite_features(client: TestClient) -> None:
    headers = {"X-API-Key": "test-secret"}
    incomplete = transaction_payload()
    incomplete.pop(MODEL.feature_columns[0])
    assert client.post("/predict", json=incomplete, headers=headers).status_code == 422

    extra = transaction_payload()
    extra["not_a_model_feature"] = 2
    assert client.post("/predict", json=extra, headers=headers).status_code == 422

    nonfinite = transaction_payload()
    nonfinite[MODEL.feature_columns[0]] = float("nan")
    response = client.post(
        "/predict",
        content=json.dumps(nonfinite, allow_nan=True),
        headers={**headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_batch_prediction(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json={"transactions": [transaction_payload(), transaction_payload()]},
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 2
    assert all(item["model_version"] == MODEL.model_version for item in response.json()["predictions"])


def test_batch_prediction_enforces_configured_limit(client: TestClient) -> None:
    payload = transaction_payload()
    response = client.post(
        "/predict/batch",
        json={"transactions": [payload] * (settings.max_batch_size + 1)},
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 422


def test_rejects_request_body_over_configured_size(client: TestClient) -> None:
    response = client.post(
        "/predict",
        content=b"{}",
        headers={
            "X-API-Key": "test-secret",
            "Content-Type": "application/json",
            "Content-Length": str(settings.max_request_bytes + 1),
        },
    )
    assert response.status_code == 413


def test_streamed_request_body_limit_without_content_length() -> None:
    async def run_middleware() -> list[dict[str, object]]:
        messages = iter(
            [
                {"type": "http.request", "body": b"12345", "more_body": True},
                {"type": "http.request", "body": b"67890", "more_body": False},
            ]
        )
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return next(messages)

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def downstream(_scope, limited_receive, _downstream_send) -> None:
            assert _scope["type"] == "http"
            assert callable(_downstream_send)
            await limited_receive()
            await limited_receive()

        middleware = api.RequestBodySizeLimitMiddleware(downstream, max_bytes=8)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"transfer-encoding", b"chunked")],
            },
            receive,
            send,
        )
        return sent

    response_messages = asyncio.run(run_middleware())
    assert response_messages[0]["status"] == 413


def test_prediction_can_include_bounded_shap_explanation(client: TestClient) -> None:
    pytest.importorskip("shap")
    response = client.post(
        "/predict?explain=true",
        json=transaction_payload(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    explanation = result["explanation"]
    assert len(explanation["feature_values"]) == len(MODEL.feature_columns)
    assert explanation["output_scale"] == "log_odds"
    assert explanation["fraud_probability"] == pytest.approx(
        result["fraud_probability"], abs=1e-5
    )
    assert sum(explanation["feature_values"].values()) + explanation["base_value"] == pytest.approx(
        explanation["output"], abs=1e-4
    )


def test_rate_limit_returns_429(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    limited_settings = replace(api.settings, max_requests_per_minute=1)
    monkeypatch.setattr(api, "settings", limited_settings)
    api._client_request_times.pop("testclient", None)
    assert client.get("/health").status_code == 200
    first = client.post(
        "/predict",
        json=transaction_payload(),
        headers={"X-API-Key": "test-secret"},
    )
    second = client.post(
        "/predict",
        json=transaction_payload(),
        headers={"X-API-Key": "test-secret"},
    )
    assert first.status_code == 200
    assert second.status_code == 429
    api._client_request_times.pop("testclient", None)
