
"""SQLite-backed persistence layer for telemetry and annotation workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import sqlite3
from statistics import mean
import threading
from typing import Any

from .models import (
    AnnotationCorrection,
    AnnotationTask,
    AnnotationTaskCreateRequest,
    InferenceLog,
)


class LogStorage:
    """Stores telemetry and annotation tasks in SQLite for API workflows."""

    def __init__(self, db_path: str) -> None:
        """Initialize storage and ensure schema exists before serving requests."""

        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @property
    def db_path(self) -> str:
        """Expose configured database path for diagnostics."""

        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection configured for row-based access."""

        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        """Create tables and indexes required by ingestion and annotation queries."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_inference_logs_model_ts
                ON inference_logs (model_name, timestamp DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS annotation_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    corrections_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_annotation_tasks_status_created
                ON annotation_tasks (status, created_at DESC)
                """
            )
            connection.commit()

    def insert_log(self, payload: InferenceLog) -> int:
        """Persist one inference record and return generated primary key."""

        metadata_json = json.dumps(payload.metadata or {}, ensure_ascii=True)
        timestamp = payload.timestamp.astimezone(timezone.utc).isoformat()

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO inference_logs (
                    model_name,
                    latency_ms,
                    prediction,
                    confidence,
                    timestamp,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.model_name,
                    payload.latency_ms,
                    payload.prediction,
                    payload.confidence,
                    timestamp,
                    metadata_json,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_logs(self, limit: int, model_name: str | None = None) -> list[InferenceLog]:
        """Fetch most recent logs optionally filtered by model name."""

        if model_name:
            query = (
                "SELECT model_name, latency_ms, prediction, confidence, timestamp, metadata_json "
                "FROM inference_logs WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (model_name, limit)
        else:
            query = (
                "SELECT model_name, latency_ms, prediction, confidence, timestamp, metadata_json "
                "FROM inference_logs ORDER BY timestamp DESC LIMIT ?"
            )
            params = (limit,)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        items = [self._row_to_inference_log(row) for row in rows]
        items.reverse()
        return items

    def count_logs(self, model_name: str | None = None) -> int:
        """Return total count of logs optionally filtered by model name."""

        with self._connect() as connection:
            if model_name:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM inference_logs WHERE model_name = ?",
                    (model_name,),
                ).fetchone()
            else:
                row = connection.execute("SELECT COUNT(*) AS count FROM inference_logs").fetchone()

        return int(row["count"] if row else 0)

    def is_available(self) -> bool:
        """Return True when SQLite backend can be queried successfully."""

        try:
            with self._connect() as connection:
                connection.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def insert_annotation_task(self, payload: AnnotationTaskCreateRequest) -> int:
        """Create a new annotation task with model pre-label candidates."""

        created_at = datetime.now(timezone.utc).isoformat()
        candidates_json = json.dumps(
            [item.model_dump(mode="json") for item in payload.candidates],
            ensure_ascii=True,
        )

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO annotation_tasks (
                    dataset_name,
                    model_name,
                    status,
                    created_at,
                    candidates_json,
                    corrections_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.dataset_name,
                    payload.model_name,
                    "pending",
                    created_at,
                    candidates_json,
                    "[]",
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_annotation_tasks(
        self,
        limit: int = 100,
        status: str | None = None,
        offset: int = 0,
    ) -> list[AnnotationTask]:
        """Return recent annotation tasks with optional status filtering and offset pagination."""

        if status:
            query = (
                "SELECT id, dataset_name, model_name, status, created_at, candidates_json, corrections_json "
                "FROM annotation_tasks WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            params: tuple[Any, ...] = (status, limit, offset)
        else:
            query = (
                "SELECT id, dataset_name, model_name, status, created_at, candidates_json, corrections_json "
                "FROM annotation_tasks ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            params = (limit, offset)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return [self._row_to_annotation_task(row) for row in rows]

    def count_annotation_tasks(self, status: str | None = None) -> int:
        """Return total annotation task count with optional status filtering."""

        with self._connect() as connection:
            if status:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM annotation_tasks WHERE status = ?",
                    (status,),
                ).fetchone()
            else:
                row = connection.execute("SELECT COUNT(*) AS count FROM annotation_tasks").fetchone()
        return int(row["count"] if row else 0)

    def get_annotation_task(self, task_id: int) -> AnnotationTask | None:
        """Fetch a single annotation task by primary key."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, dataset_name, model_name, status, created_at, candidates_json, corrections_json
                FROM annotation_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

        if not row:
            return None
        return self._row_to_annotation_task(row)

    def append_annotation_corrections(
        self,
        task_id: int,
        corrections: list[AnnotationCorrection],
        status: str | None = None,
    ) -> AnnotationTask | None:
        """Append human corrections to a task and transition status safely."""

        correction_payload = [item.model_dump(mode="json") for item in corrections]

        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, dataset_name, model_name, status, created_at, candidates_json, corrections_json
                FROM annotation_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
            if not row:
                return None

            existing_corrections = json.loads(row["corrections_json"] or "[]")
            updated_corrections = existing_corrections + correction_payload

            candidate_count = len(json.loads(row["candidates_json"] or "[]"))
            next_status = status
            if next_status is None:
                next_status = "completed" if len(updated_corrections) >= candidate_count else "in_review"

            connection.execute(
                """
                UPDATE annotation_tasks
                SET corrections_json = ?, status = ?
                WHERE id = ?
                """,
                (
                    json.dumps(updated_corrections, ensure_ascii=True),
                    next_status,
                    task_id,
                ),
            )
            connection.commit()

            updated = connection.execute(
                """
                SELECT id, dataset_name, model_name, status, created_at, candidates_json, corrections_json
                FROM annotation_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

        return self._row_to_annotation_task(updated) if updated else None

    def get_model_family_outcomes(self, limit: int = 5000) -> dict[str, float]:
        """Estimate historical model-family outcome score from telemetry metadata."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT metadata_json
                FROM inference_logs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        family_scores: dict[str, list[float]] = {}
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            family = metadata.get("model_family")
            score = metadata.get("outcome_score")
            if not isinstance(family, str):
                continue
            if not isinstance(score, (int, float)):
                continue

            normalized_score = float(score)
            if normalized_score < 0.0 or normalized_score > 1.0:
                continue

            family_scores.setdefault(family, []).append(normalized_score)

        return {family: mean(scores) for family, scores in family_scores.items() if scores}

    @staticmethod
    def _row_to_inference_log(row: sqlite3.Row) -> InferenceLog:
        """Convert one SQLite row into InferenceLog domain model."""

        metadata = json.loads(row["metadata_json"] or "{}")
        timestamp = datetime.fromisoformat(row["timestamp"])
        return InferenceLog(
            model_name=row["model_name"],
            latency_ms=row["latency_ms"],
            prediction=row["prediction"],
            confidence=row["confidence"],
            timestamp=timestamp,
            metadata=metadata,
        )

    @staticmethod
    def _row_to_annotation_task(row: sqlite3.Row) -> AnnotationTask:
        """Convert one SQLite row into AnnotationTask domain model."""

        candidates = json.loads(row["candidates_json"] or "[]")
        corrections = json.loads(row["corrections_json"] or "[]")
        created_at = datetime.fromisoformat(row["created_at"])

        return AnnotationTask(
            task_id=int(row["id"]),
            dataset_name=row["dataset_name"],
            model_name=row["model_name"],
            status=row["status"],
            created_at=created_at,
            candidates=candidates,
            corrections=corrections,
            corrections_count=len(corrections),
        )
