from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

from reel_curator.models import VideoMetadata

LOGGER = logging.getLogger(__name__)


def probe_video(path: Path) -> VideoMetadata:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            return _probe_with_ffprobe(path, ffprobe)
        except Exception as exc:  # pragma: no cover - fallback path depends on local ffprobe
            LOGGER.warning("ffprobe failed for %s: %s", path, exc)
    return _probe_with_opencv(path)


def _probe_with_ffprobe(path: Path, ffprobe: str) -> VideoMetadata:
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload: dict[str, Any] = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
    fmt = payload.get("format", {})
    fps = _parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    tags = fmt.get("tags", {}) | video_stream.get("tags", {})
    raw_width = int(video_stream.get("width") or 0)
    raw_height = int(video_stream.get("height") or 0)
    rotation = _rotation_degrees(video_stream)
    width, height = _display_dimensions(raw_width, raw_height, rotation)
    stat = path.stat()
    return VideoMetadata(
        path=str(path),
        filename=path.name,
        size_bytes=stat.st_size,
        modified_time=stat.st_mtime,
        duration_seconds=float(video_stream.get("duration") or fmt.get("duration") or 0.0),
        width=width,
        height=height,
        fps=fps,
        codec=video_stream.get("codec_name"),
        created_at=tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate"),
        rotation_degrees=rotation,
    )


def _probe_with_opencv(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    stat = path.stat()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=str(path),
        filename=path.name,
        size_bytes=stat.st_size,
        modified_time=stat.st_mtime,
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
    )


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denom = float(denominator)
    return float(numerator) / denom if denom else 0.0


def _rotation_degrees(video_stream: dict[str, Any]) -> int:
    tags = video_stream.get("tags", {}) or {}
    rotate = tags.get("rotate")
    if rotate is not None:
        return _normalize_rotation(float(rotate))

    for item in video_stream.get("side_data_list", []) or []:
        if "rotation" in item:
            return _normalize_rotation(float(item["rotation"]))
        displaymatrix = item.get("displaymatrix")
        if isinstance(displaymatrix, str) and "rotation" in displaymatrix.lower():
            tail = displaymatrix.lower().split("rotation", 1)[-1]
            digits = "".join(ch for ch in tail if ch.isdigit() or ch in ".-")
            if digits:
                return _normalize_rotation(float(digits))
    return 0


def _normalize_rotation(value: float) -> int:
    return int(round(value)) % 360


def _display_dimensions(width: int, height: int, rotation: int) -> tuple[int, int]:
    if rotation in {90, 270}:
        return height, width
    return width, height
