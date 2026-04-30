
"""Integration tests for automated ML assistant FastAPI endpoints."""

from dataclasses import replace
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from app import main as main_module
from app.main import app, storage

client = TestClient(app)


def _assert_error_contract(
    response,
    expected_status: int,
    expected_code: str,
    expected_message: str | None = None,
    expect_details: bool = False,
) -> None:
    """Validate normalized API error contract fields."""

    assert response.status_code == expected_status
    payload = response.json()
    assert isinstance(payload.get("error"), dict)
    assert payload["error"]["code"] == expected_code
    assert isinstance(payload["error"]["message"], str)
    if expected_message is not None:
        assert payload["error"]["message"] == expected_message
    assert isinstance(payload["error"]["request_id"], str)
    assert payload["error"]["request_id"]
    if expect_details:
        assert "details" in payload["error"]


def _override_settings(monkeypatch, **changes):
    """Apply temporary runtime setting overrides for endpoint security tests."""

    monkeypatch.setattr(main_module, "settings", replace(main_module.settings, **changes))


@pytest.fixture(autouse=True)
def reset_storage():
    """Clear persisted logs between tests for deterministic assertions."""

    with storage._connect() as connection:  # pylint: disable=protected-access
        connection.execute("DELETE FROM inference_logs")
        connection.execute("DELETE FROM annotation_tasks")
        connection.commit()

    yield

    with storage._connect() as connection:  # pylint: disable=protected-access
        connection.execute("DELETE FROM inference_logs")
        connection.execute("DELETE FROM annotation_tasks")
        connection.commit()


def test_health_endpoint_reports_backend_status():
    """Health endpoint should return uptime-safe diagnostics."""

    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["storage_backend"] in {"sqlite", "postgresql"}
    assert payload["rate_limiter_backend"] in {"in_memory", "redis"}
    assert isinstance(payload["rate_limiter_available"], bool)
    assert "db_available" in payload
    assert "total_logs" in payload


def test_probe_alias_endpoints_return_runtime_state():
    """Readiness and probe aliases should be available for deployment checks."""

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    health_alias = client.get("/healthz")
    assert health_alias.status_code == 200
    assert health_alias.json()["status"] in {"ok", "degraded"}

    ready_alias = client.get("/readyz")
    assert ready_alias.status_code == 200
    assert ready_alias.json()["status"] == "ready"


def test_log_ingest_and_summary_flow():
    """Posting a log should reflect in summary output and aggregates."""

    timestamp = datetime.now(timezone.utc).isoformat()
    response = client.post(
        "/log",
        json={
            "model_name": "risk-model-v1",
            "latency_ms": 123.4,
            "prediction": "approved",
            "confidence": 0.82,
            "timestamp": timestamp,
            "metadata": {"tenant": "demo"},
        },
    )

    assert response.status_code == 200
    ingest_payload = response.json()
    assert ingest_payload["status"] == "accepted"
    assert ingest_payload["log_id"] > 0

    summary_response = client.get("/summary?limit=10")
    assert summary_response.status_code == 200

    summary_payload = summary_response.json()
    assert summary_payload["total_items"] == 1
    assert len(summary_payload["items"]) == 1
    assert summary_payload["prediction_distribution"]["approved"] == 1


def test_export_csv_returns_expected_content_type():
    """CSV export endpoint should return attachment metadata and CSV body."""

    timestamp = datetime.now(timezone.utc).isoformat()
    client.post(
        "/log",
        json={
            "model_name": "risk-model-v1",
            "latency_ms": 44.2,
            "prediction": "review",
            "confidence": 0.45,
            "timestamp": timestamp,
        },
    )

    response = client.get("/export?format=csv&limit=10")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "model_name,latency_ms,prediction,confidence,timestamp" in response.text


def test_dataset_diagnostics_returns_profiles_and_hints():
    """Diagnostics endpoint should report column quality and numeric summaries."""

    response = client.post(
        "/diagnostics",
        json={
            "target_column": "target",
            "rows": [
                {"feature_a": 10, "feature_b": 5, "target": 1, "city": "chennai"},
                {"feature_a": 12, "feature_b": 7, "target": 1, "city": "madurai"},
                {"feature_a": 8, "feature_b": 2, "target": 0, "city": "chennai"},
                {"feature_a": None, "feature_b": 6, "target": 1, "city": ""},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 4
    assert payload["column_count"] == 4
    assert "feature_a" in payload["numeric_summary"]
    assert "task_hints" in payload
    assert isinstance(payload["task_hints"], list)


def test_recommendations_ranks_models_for_task_type():
    """Recommendations endpoint should return ranked families for supported task type."""

    diagnostics_response = client.post(
        "/diagnostics",
        json={
            "target_column": "label",
            "rows": [
                {"f1": 1.2, "f2": 0.3, "label": 1},
                {"f1": 0.4, "f2": 0.8, "label": 0},
                {"f1": 1.8, "f2": 0.5, "label": 1},
            ],
        },
    )
    diagnostics_payload = diagnostics_response.json()

    response = client.post(
        "/recommendations",
        json={
            "task_type": "classification",
            "diagnostics": diagnostics_payload,
            "objective": "maximize precision at low latency",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "classification"
    assert payload["signal_source"] == "rule-based+historical-telemetry"
    assert len(payload["recommendations"]) >= 3
    assert payload["recommendations"][0]["score"] >= payload["recommendations"][1]["score"]


def test_annotation_task_lifecycle_supports_corrections():
    """Annotation endpoints should create tasks and apply corrections deterministically."""

    create_response = client.post(
        "/annotations/tasks",
        json={
            "dataset_name": "surveillance-sample",
            "model_name": "yolov5s",
            "candidates": [
                {
                    "candidate_id": "frame-1",
                    "image_uri": "s3://bucket/frame-1.jpg",
                    "predicted_label": "person",
                    "confidence": 0.88,
                    "bbox": [11.0, 22.0, 120.0, 190.0],
                },
                {
                    "candidate_id": "frame-2",
                    "image_uri": "s3://bucket/frame-2.jpg",
                    "predicted_label": "bicycle",
                    "confidence": 0.72,
                    "bbox": [31.0, 44.0, 210.0, 260.0],
                },
            ],
        },
    )

    assert create_response.status_code == 200
    task_payload = create_response.json()
    assert task_payload["status"] == "pending"
    task_id = task_payload["task_id"]

    update_response = client.post(
        f"/annotations/tasks/{task_id}/corrections",
        json={
            "corrections": [
                {
                    "candidate_id": "frame-1",
                    "final_label": "pedestrian",
                    "approved": False,
                    "notes": "refined for taxonomy",
                }
            ]
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["status"] == "in_review"
    assert updated["corrections_count"] == 1

    list_response = client.get("/annotations/tasks?status=in_review")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["total"] == 1
    assert list_payload["tasks"][0]["task_id"] == task_id


def test_annotation_task_list_supports_offset_pagination():
    """Annotation listing should support offset pagination for larger queues."""

    for index in range(3):
        create_response = client.post(
            "/annotations/tasks",
            json={
                "dataset_name": f"dataset-{index}",
                "model_name": "yolov5s",
                "candidates": [
                    {
                        "candidate_id": f"frame-{index}",
                        "image_uri": f"s3://bucket/frame-{index}.jpg",
                        "predicted_label": "person",
                        "confidence": 0.8,
                        "bbox": [10.0, 20.0, 100.0, 200.0],
                    }
                ],
            },
        )
        assert create_response.status_code == 200

    first_page = client.get("/annotations/tasks?limit=1&offset=0")
    second_page = client.get("/annotations/tasks?limit=1&offset=1")

    assert first_page.status_code == 200
    assert second_page.status_code == 200
    first_payload = first_page.json()
    second_payload = second_page.json()
    assert first_payload["total"] == 3
    assert second_payload["total"] == 3
    assert len(first_payload["tasks"]) == 1
    assert len(second_payload["tasks"]) == 1
    assert first_payload["tasks"][0]["task_id"] != second_payload["tasks"][0]["task_id"]


def test_diagnostics_requires_api_key_when_configured(monkeypatch):
    """Diagnostics endpoint should enforce API key when runtime key is configured."""

    _override_settings(monkeypatch, api_key="secret-key", enforce_api_key=True)

    unauthorized = client.post(
        "/diagnostics",
        json={
            "target_column": "target",
            "rows": [{"feature_a": 10, "target": 1}, {"feature_a": 8, "target": 0}],
        },
    )
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    authorized = client.post(
        "/diagnostics",
        headers={"X-API-Key": "secret-key"},
        json={
            "target_column": "target",
            "rows": [{"feature_a": 10, "target": 1}, {"feature_a": 8, "target": 0}],
        },
    )
    assert authorized.status_code == 200


def test_health_is_public_even_when_api_key_enabled(monkeypatch):
    """Health endpoint should stay unauthenticated for uptime probes."""

    _override_settings(monkeypatch, api_key="secret-key", enforce_api_key=True)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}

    ready_response = client.get("/ready")
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"


def test_phase1_auth_required_contract(monkeypatch):
    """Protected endpoints should require API key with normalized unauthorized errors."""

    _override_settings(monkeypatch, api_key="phase1-secret", enforce_api_key=True)

    unauthorized = client.get("/summary")
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    invalid_key = client.get("/summary", headers={"X-API-Key": "wrong"})
    _assert_error_contract(
        invalid_key,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    authorized = client.get("/summary", headers={"X-API-Key": "phase1-secret"})
    assert authorized.status_code == 200


def test_phase1_error_contract_response():
    """Request validation should return normalized validation error payload."""

    response = client.post("/diagnostics", json={"target_column": "target"})
    _assert_error_contract(
        response,
        expected_status=422,
        expected_code="validation_error",
        expected_message="request_validation_failed",
        expect_details=True,
    )


def test_error_responses_include_request_and_security_headers(monkeypatch):
    """Error responses should carry request tracing and baseline security headers."""

    _override_settings(monkeypatch, api_key="header-secret", enforce_api_key=True)
    response = client.post(
        "/diagnostics",
        json={
            "target_column": "target",
            "rows": [{"feature_a": 10, "target": 1}, {"feature_a": 8, "target": 0}],
        },
    )

    assert response.status_code == 401
    assert response.headers.get("X-Request-ID")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"


def test_diagnostics_rejects_oversized_request_body(monkeypatch):
    """Request middleware should reject payloads larger than configured limit."""

    _override_settings(monkeypatch, max_request_body_bytes=80)

    response = client.post(
        "/diagnostics",
        json={
            "target_column": "target",
            "rows": [
                {"feature_a": 10, "target": 1, "note": "payload"},
                {"feature_a": 8, "target": 0, "note": "payload"},
            ],
        },
    )
    _assert_error_contract(
        response,
        expected_status=413,
        expected_code="payload_too_large",
        expected_message="request_too_large",
    )
    assert response.headers.get("X-Request-ID")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"


def test_log_rate_limit_returns_429_with_retry_after(monkeypatch):
    """Log ingest should return normalized 429 responses with Retry-After guidance."""

    _override_settings(monkeypatch, rate_limit_per_minute=1)
    timestamp = datetime.now(timezone.utc).isoformat()

    first = client.post(
        "/log",
        headers={"X-Forwarded-For": "198.51.100.44"},
        json={
            "model_name": "risk-model-v1",
            "latency_ms": 22.5,
            "prediction": "approved",
            "confidence": 0.91,
            "timestamp": timestamp,
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/log",
        headers={"X-Forwarded-For": "198.51.100.44"},
        json={
            "model_name": "risk-model-v1",
            "latency_ms": 24.1,
            "prediction": "review",
            "confidence": 0.74,
            "timestamp": timestamp,
        },
    )
    _assert_error_contract(
        second,
        expected_status=429,
        expected_code="rate_limited",
        expected_message="rate_limited",
    )
    assert second.headers.get("Retry-After") == "60"
