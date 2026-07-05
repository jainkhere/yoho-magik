from __future__ import annotations

import cv2
import numpy as np

from reel_curator.frame_sampler import SampledFrame
from reel_curator.models import VideoMetadata, VisionResult

TAG_ALIASES = {
    "mountain": "mountains",
    "mountains": "mountains",
    "water": "water",
    "ocean": "water",
    "lake": "water",
    "river": "water",
    "beach": "beach",
    "sunset": "sunset/sunrise",
    "sunrise": "sunset/sunrise",
    "food": "food",
    "building": "buildings",
    "architecture": "buildings",
    "animal": "animals",
    "person": "people",
    "people": "people",
    "face": "people",
    "indoor": "indoor",
    "outdoor": "outdoor",
    "text": "text",
}


def generate_tags(
    metadata: VideoMetadata, frames: list[SampledFrame], vision: VisionResult
) -> list[str]:
    tags: set[str] = {metadata.orientation}

    for label in vision.labels:
        normalized = label.lower()
        for needle, tag in TAG_ALIASES.items():
            if needle in normalized:
                tags.add(tag)

    if vision.people_count or vision.face_count:
        tags.add("people")
        if vision.face_count == 1:
            tags.add("selfie-candidate")
        elif vision.face_count >= 3:
            tags.add("group-shot")
    if vision.animal_count:
        tags.add("animals")
    if vision.text_detected:
        tags.add("text")

    heuristic = _visual_heuristics(frames)
    tags.update(heuristic)
    return sorted(tags)


def scenic_score_from_tags(tags: list[str]) -> float:
    scenic_tags = {"mountains", "beach", "water", "sunset/sunrise", "outdoor"}
    content_tags = {"food", "buildings", "animals", "people"}
    scenic = len(scenic_tags.intersection(tags)) / 3.0
    content = len(content_tags.intersection(tags)) / 5.0
    return min(1.0, 0.75 * scenic + 0.25 * content)


def _visual_heuristics(frames: list[SampledFrame]) -> set[str]:
    tags: set[str] = set()
    if not frames:
        return tags
    blue_ratios = []
    green_ratios = []
    warm_ratios = []
    edge_densities = []
    for item in frames:
        hsv = cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        vivid = (sat > 45) & (val > 50)
        blue_ratios.append(float(np.mean((hue > 85) & (hue < 130) & vivid)))
        green_ratios.append(float(np.mean((hue > 35) & (hue < 85) & vivid)))
        warm_ratios.append(float(np.mean(((hue < 18) | (hue > 165)) & vivid)))
        edges = cv2.Canny(cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2GRAY), 80, 160)
        edge_densities.append(float(np.mean(edges > 0)))

    if float(np.mean(blue_ratios)) > 0.18:
        tags.add("water")
        tags.add("outdoor")
    if float(np.mean(green_ratios)) > 0.16:
        tags.add("outdoor")
    if float(np.mean(warm_ratios)) > 0.13:
        tags.add("sunset/sunrise")
    if float(np.mean(edge_densities)) > 0.12:
        tags.add("buildings")
    if not tags:
        tags.add("uncategorized")
    return tags
