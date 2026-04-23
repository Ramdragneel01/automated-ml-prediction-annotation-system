
"""Integration tests for automated ML assistant FastAPI endpoints."""

from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from app.main import app, storage

client = TestClient(app)


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
    assert "db_available" in payload
    assert "total_logs" in payload


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
