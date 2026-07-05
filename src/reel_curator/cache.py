from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from reel_curator.models import (
    ANALYSIS_CACHE_VERSION,
    ScoreBreakdown,
    VideoAnalysis,
    VideoMetadata,
    VisionResult,
)


class AnalysisCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir = self.cache_dir / "thumbnails"
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "analysis.sqlite3"
        self._init_db()

    def get(self, cache_key: str) -> VideoAnalysis | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM analyses WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if not row:
            return None
        return analysis_from_dict(json.loads(row[0]))

    def set(self, cache_key: str, analysis: VideoAnalysis) -> None:
        payload = json.dumps(analysis.as_dict(), sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses(cache_key, path, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    path = excluded.path,
                    payload = excluded.payload
                """,
                (cache_key, analysis.metadata.path, payload),
            )

    def update_selection(self, path: str, favorite: bool, rejected: bool) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT cache_key, payload FROM analyses WHERE path = ?", (path,)
            ).fetchall()
            for cache_key, payload in rows:
                data = json.loads(payload)
                data["favorite"] = favorite
                data["rejected"] = rejected
                conn.execute(
                    "UPDATE analyses SET payload = ? WHERE cache_key = ?",
                    (json.dumps(data, sort_keys=True), cache_key),
                )

    def all(self) -> list[VideoAnalysis]:
        with self._connect() as conn:
            rows = conn.execute("SELECT cache_key, payload FROM analyses ORDER BY path").fetchall()

        by_path: dict[str, tuple[int, VideoAnalysis]] = {}
        for cache_key, payload in rows:
            analysis = analysis_from_dict(json.loads(payload))
            priority = 1 if str(cache_key).startswith(ANALYSIS_CACHE_VERSION) else 0
            existing = by_path.get(analysis.metadata.path)
            if existing is None or priority >= existing[0]:
                by_path[analysis.metadata.path] = (priority, analysis)
        return [analysis for _, analysis in by_path.values()]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    cache_key TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_path ON analyses(path)")


def analysis_from_dict(data: dict[str, Any]) -> VideoAnalysis:
    metadata = VideoMetadata(
        path=data["path"],
        filename=data["filename"],
        size_bytes=int(data["size_bytes"]),
        modified_time=float(data["modified_time"]),
        duration_seconds=float(data["duration_seconds"]),
        width=int(data["width"]),
        height=int(data["height"]),
        fps=float(data["fps"]),
        codec=data.get("codec"),
        created_at=data.get("created_at"),
        rotation_degrees=int(data.get("rotation_degrees", 0)),
    )
    score_data = data.get("score", {})
    score = ScoreBreakdown(
        image_quality=float(score_data.get("image_quality", 0.0)),
        stability=float(score_data.get("stability", 0.0)),
        lighting=float(score_data.get("lighting", 0.0)),
        scenic_content=float(score_data.get("scenic_content", 0.0)),
        people_priority=float(score_data.get("people_priority", 0.0)),
        uniqueness=float(score_data.get("uniqueness", 0.0)),
        composition=float(score_data.get("composition", 0.0)),
        reel_usefulness=float(score_data.get("reel_usefulness", 0.0)),
        overall=float(score_data.get("overall", data.get("quality_score", 0.0))),
        explanation=str(score_data.get("explanation", "")),
    )
    vision = VisionResult(
        labels=list(data.get("vision_labels", [])),
        people_count=int(data.get("people_count", 0)),
        animal_count=int(data.get("animal_count", 0)),
        text_detected=bool(data.get("text_detected", False)),
        face_count=int(data.get("face_count", 0)),
    )
    return VideoAnalysis(
        metadata=metadata,
        brightness=float(data.get("brightness", 0.0)),
        sharpness=float(data.get("sharpness", 0.0)),
        stability=float(data.get("stability", 0.0)),
        motion=float(data.get("motion", 0.0)),
        scene_changes=int(data.get("scene_changes", 0)),
        perceptual_hash=str(data.get("perceptual_hash", "")),
        thumbnail_path=str(data.get("thumbnail_path", "")),
        best_frame_second=float(data.get("best_frame_second", 0.0)),
        tags=list(data.get("tags", [])),
        vision=vision,
        score=score,
        priority_people_matches=list(data.get("priority_people_matches", [])),
        priority_people_score=float(data.get("priority_people_score", 0.0)),
        duplicate_group=data.get("duplicate_group"),
        duplicate_representative=bool(data.get("duplicate_representative", False)),
        favorite=bool(data.get("favorite", False)),
        rejected=bool(data.get("rejected", False)),
    )
