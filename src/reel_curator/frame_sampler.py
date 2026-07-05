from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class SampledFrame:
    second: float
    frame_bgr: np.ndarray


def sample_frames(
    path: Path,
    sample_count: int,
    rotation_degrees: int = 0,
    display_size: tuple[int, int] | None = None,
) -> list[SampledFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return []

    start = int(frame_count * 0.05)
    end = max(start + 1, int(frame_count * 0.95))
    positions = np.linspace(start, end, num=min(sample_count, max(1, end - start)), dtype=int)
    frames: list[SampledFrame] = []
    for frame_index in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok and frame is not None:
            frame = _rotate_frame(frame, rotation_degrees, display_size)
            frames.append(SampledFrame(second=float(frame_index) / fps, frame_bgr=frame))
    cap.release()
    return frames


def save_thumbnail(
    frame: np.ndarray, output_path: Path, max_size: tuple[int, int] = (640, 640)
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    image.save(output_path, format="JPEG", quality=88, optimize=True)


def _rotate_frame(
    frame: np.ndarray,
    rotation_degrees: int,
    display_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if display_size and _matches_display_size(frame, display_size):
        return frame

    rotation = rotation_degrees % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _matches_display_size(frame: np.ndarray, display_size: tuple[int, int]) -> bool:
    display_width, display_height = display_size
    frame_height, frame_width = frame.shape[:2]
    return frame_width == display_width and frame_height == display_height
