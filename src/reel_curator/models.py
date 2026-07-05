from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANALYSIS_CACHE_VERSION = "orientation-v3"


@dataclass(slots=True)
class VideoIdentity:
    path: Path
    size_bytes: int
    mtime_ns: int

    @property
    def cache_key(self) -> str:
        return (
            f"{ANALYSIS_CACHE_VERSION}::{self.path.resolve()}::"
            f"{self.size_bytes}::{self.mtime_ns}"
        )


@dataclass(slots=True)
class VideoMetadata:
    path: str
    filename: str
    size_bytes: int
    modified_time: float
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str | None = None
    created_at: str | None = None
    rotation_degrees: int = 0

    @property
    def orientation(self) -> str:
        if self.height > self.width:
            return "portrait"
        if self.width > self.height:
            return "landscape"
        return "square"

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(slots=True)
class VisionResult:
    labels: list[str] = field(default_factory=list)
    confidence_by_label: dict[str, float] = field(default_factory=dict)
    people_count: int = 0
    animal_count: int = 0
    text_detected: bool = False
    face_count: int = 0


@dataclass(slots=True)
class ScoreBreakdown:
    image_quality: float
    stability: float
    lighting: float
    scenic_content: float
    people_priority: float
    uniqueness: float
    composition: float
    reel_usefulness: float
    overall: float
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_quality": self.image_quality,
            "stability": self.stability,
            "lighting": self.lighting,
            "scenic_content": self.scenic_content,
            "people_priority": self.people_priority,
            "uniqueness": self.uniqueness,
            "composition": self.composition,
            "reel_usefulness": self.reel_usefulness,
            "overall": self.overall,
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class VideoAnalysis:
    metadata: VideoMetadata
    brightness: float
    sharpness: float
    stability: float
    motion: float
    scene_changes: int
    perceptual_hash: str
    thumbnail_path: str
    best_frame_second: float
    tags: list[str]
    vision: VisionResult
    score: ScoreBreakdown
    priority_people_matches: list[str] = field(default_factory=list)
    priority_people_score: float = 0.0
    duplicate_group: str | None = None
    duplicate_representative: bool = False
    favorite: bool = False
    rejected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.metadata.path,
            "filename": self.metadata.filename,
            "size_bytes": self.metadata.size_bytes,
            "modified_time": self.metadata.modified_time,
            "created_at": self.metadata.created_at,
            "duration_seconds": self.metadata.duration_seconds,
            "width": self.metadata.width,
            "height": self.metadata.height,
            "resolution": self.metadata.resolution_label,
            "fps": self.metadata.fps,
            "codec": self.metadata.codec,
            "rotation_degrees": self.metadata.rotation_degrees,
            "orientation": self.metadata.orientation,
            "brightness": self.brightness,
            "sharpness": self.sharpness,
            "stability": self.stability,
            "motion": self.motion,
            "scene_changes": self.scene_changes,
            "perceptual_hash": self.perceptual_hash,
            "thumbnail_path": self.thumbnail_path,
            "best_frame_second": self.best_frame_second,
            "tags": self.tags,
            "vision_labels": self.vision.labels,
            "people_count": self.vision.people_count,
            "animal_count": self.vision.animal_count,
            "text_detected": self.vision.text_detected,
            "face_count": self.vision.face_count,
            "quality_score": self.score.overall,
            "score": self.score.as_dict(),
            "priority_people_matches": self.priority_people_matches,
            "priority_people_score": self.priority_people_score,
            "duplicate_group": self.duplicate_group,
            "duplicate_representative": self.duplicate_representative,
            "favorite": self.favorite,
            "rejected": self.rejected,
        }
