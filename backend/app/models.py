
"""Pydantic models used by the automated ML and annotation API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class InferenceLog(BaseModel):
    """Represents one model inference event entering observability pipeline."""

    model_name: str = Field(min_length=1, max_length=120)
    latency_ms: float = Field(ge=0)
    prediction: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone(cls, value: datetime) -> datetime:
        """Ensure timestamp is timezone aware for deterministic time operations."""

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class SummaryResponse(BaseModel):
    """Response model for returning sampled recent logs and aggregate context."""

    items: list[InferenceLog]
    drift_flag: bool
    avg_confidence: float | None
    total_items: int
    prediction_distribution: dict[str, int]
    drift_threshold: float


class NumericSummary(BaseModel):
    """Describes a numeric feature with compact descriptive statistics."""

    mean: float
    min: float
    max: float
    stddev: float


class ColumnDiagnostics(BaseModel):
    """Summarizes basic quality and type information for one dataset column."""

    name: str = Field(min_length=1, max_length=120)
    detected_type: Literal["numeric", "categorical", "boolean", "text", "mixed", "empty"]
    missing_count: int = Field(ge=0)
    missing_ratio: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)


class DatasetDiagnosticsRequest(BaseModel):
    """Input payload used to compute dataset diagnostics and feature-level hints."""

    rows: list[dict[str, Any]] = Field(min_length=1, max_length=5000)
    target_column: str | None = Field(default=None, min_length=1, max_length=120)


class DatasetDiagnosticsResponse(BaseModel):
    """Computed diagnostics used by model recommendation and UI previews."""

    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    column_profiles: list[ColumnDiagnostics]
    missing_by_column: dict[str, int]
    numeric_summary: dict[str, NumericSummary]
    correlation_to_target: dict[str, float]
    task_hints: list[str]


class ModelRecommendation(BaseModel):
    """One ranked model-family recommendation with score and rationale."""

    model_family: str = Field(min_length=1, max_length=120)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)


class ModelRecommendationRequest(BaseModel):
    """Input payload for ranking model families from diagnostics and task context."""

    task_type: Literal[
        "classification",
        "regression",
        "clustering",
        "time_series",
        "matrix_factorization",
        "anomaly",
    ]
    diagnostics: DatasetDiagnosticsResponse
    objective: str | None = Field(default=None, max_length=300)


class ModelRecommendationResponse(BaseModel):
    """Ranked recommendation response for model family selection."""

    task_type: str
    recommendations: list[ModelRecommendation]
    signal_source: str


class AnnotationCandidate(BaseModel):
    """Represents one pre-labeled image candidate from CV model inference."""

    candidate_id: str = Field(min_length=1, max_length=120)
    image_uri: str = Field(min_length=1, max_length=500)
    predicted_label: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)


class AnnotationCorrection(BaseModel):
    """Captures one reviewer correction on a pre-labeled candidate."""

    candidate_id: str = Field(min_length=1, max_length=120)
    final_label: str = Field(min_length=1, max_length=120)
    approved: bool
    notes: str | None = Field(default=None, max_length=500)


class AnnotationTaskCreateRequest(BaseModel):
    """Input payload for creating a new annotation review task."""

    dataset_name: str = Field(min_length=1, max_length=120)
    model_name: str = Field(min_length=1, max_length=120)
    candidates: list[AnnotationCandidate] = Field(min_length=1, max_length=500)


class AnnotationTaskUpdateRequest(BaseModel):
    """Input payload for submitting corrections and optional status updates."""

    corrections: list[AnnotationCorrection] = Field(min_length=1, max_length=500)
    status: Literal["pending", "in_review", "completed"] | None = None


class AnnotationTask(BaseModel):
    """Represents one annotation task with candidates and accumulated corrections."""

    task_id: int = Field(ge=1)
    dataset_name: str
    model_name: str
    status: Literal["pending", "in_review", "completed"]
    created_at: datetime
    candidates: list[AnnotationCandidate]
    corrections: list[AnnotationCorrection]
    corrections_count: int = Field(ge=0)


class AnnotationTaskListResponse(BaseModel):
    """List response wrapper for annotation task dashboards."""

    tasks: list[AnnotationTask]
    total: int = Field(ge=0)
