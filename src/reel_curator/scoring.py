from __future__ import annotations

from reel_curator.models import ScoreBreakdown, VideoMetadata
from reel_curator.tagging import scenic_score_from_tags

DEFAULT_SCORE_WEIGHTS = {
    "people": 30.0,
    "image_quality": 20.0,
    "stability": 14.0,
    "lighting": 10.0,
    "scenic_content": 10.0,
    "uniqueness": 7.0,
    "composition": 4.0,
    "reel_usefulness": 5.0,
}

SCORE_PRESETS = {
    "People first": DEFAULT_SCORE_WEIGHTS,
    "Balanced": {
        "people": 20.0,
        "image_quality": 20.0,
        "stability": 15.0,
        "lighting": 12.0,
        "scenic_content": 15.0,
        "uniqueness": 8.0,
        "composition": 5.0,
        "reel_usefulness": 5.0,
    },
    "Scenery first": {
        "people": 12.0,
        "image_quality": 18.0,
        "stability": 12.0,
        "lighting": 12.0,
        "scenic_content": 28.0,
        "uniqueness": 8.0,
        "composition": 5.0,
        "reel_usefulness": 5.0,
    },
    "Cinematic/stable": {
        "people": 14.0,
        "image_quality": 24.0,
        "stability": 24.0,
        "lighting": 12.0,
        "scenic_content": 10.0,
        "uniqueness": 6.0,
        "composition": 6.0,
        "reel_usefulness": 4.0,
    },
    "Reel energy": {
        "people": 22.0,
        "image_quality": 16.0,
        "stability": 10.0,
        "lighting": 8.0,
        "scenic_content": 12.0,
        "uniqueness": 8.0,
        "composition": 4.0,
        "reel_usefulness": 20.0,
    },
}


def calculate_score(
    metadata: VideoMetadata,
    brightness: float,
    sharpness: float,
    stability: float,
    motion: float,
    tags: list[str],
    uniqueness: float,
    people_priority: float = 0.0,
    weights: dict[str, float] | None = None,
) -> ScoreBreakdown:
    normalized_weights = normalize_score_weights(weights or DEFAULT_SCORE_WEIGHTS)
    image_quality = _pct(0.78 * sharpness + 0.22 * _brightness_to_quality(brightness))
    stability_score = _pct(stability)
    lighting = _pct(_brightness_to_quality(brightness))
    scenic = _pct(scenic_score_from_tags(tags))
    people = _pct(_people_priority_score(tags, people_priority))
    unique = _pct(uniqueness)
    composition = _pct(_composition_score(metadata, tags))
    reel = _pct(_reel_usefulness(metadata, motion, tags))

    overall = (
        people * normalized_weights["people"]
        + image_quality * normalized_weights["image_quality"]
        + stability_score * normalized_weights["stability"]
        + lighting * normalized_weights["lighting"]
        + scenic * normalized_weights["scenic_content"]
        + unique * normalized_weights["uniqueness"]
        + composition * normalized_weights["composition"]
        + reel * normalized_weights["reel_usefulness"]
    )
    explanation = (
        f"Overall = {_format_weight(normalized_weights['people'])}% people priority, "
        f"{_format_weight(normalized_weights['image_quality'])}% image quality, "
        f"{_format_weight(normalized_weights['stability'])}% stability, "
        f"{_format_weight(normalized_weights['lighting'])}% lighting, "
        f"{_format_weight(normalized_weights['scenic_content'])}% scenic/content signals, "
        f"{_format_weight(normalized_weights['uniqueness'])}% uniqueness, "
        f"{_format_weight(normalized_weights['composition'])}% composition, and "
        f"{_format_weight(normalized_weights['reel_usefulness'])}% travel-reel usefulness. "
        "Reference face matches receive the "
        "strongest people-priority boost."
    )
    return ScoreBreakdown(
        image_quality=round(image_quality, 1),
        stability=round(stability_score, 1),
        lighting=round(lighting, 1),
        scenic_content=round(scenic, 1),
        people_priority=round(people, 1),
        uniqueness=round(unique, 1),
        composition=round(composition, 1),
        reel_usefulness=round(reel, 1),
        overall=round(overall, 1),
        explanation=explanation,
    )


def _brightness_to_quality(raw_brightness: float) -> float:
    normalized = raw_brightness / 255.0
    return max(0.0, min(1.0, 1.0 - abs(normalized - 0.58) / 0.58))


def _composition_score(metadata: VideoMetadata, tags: list[str]) -> float:
    score = 0.6
    if metadata.orientation == "portrait":
        score += 0.25
    if metadata.width >= 1080 or metadata.height >= 1080:
        score += 0.1
    if "text" in tags:
        score -= 0.1
    return max(0.0, min(1.0, score))


def _people_priority_score(tags: list[str], people_priority: float) -> float:
    score = people_priority
    if "people" in tags:
        score = max(score, 0.68)
    if "selfie-candidate" in tags:
        score = max(score, 0.74)
    if "group-shot" in tags:
        score = max(score, 0.82)
    if any(tag.startswith("priority-person:") for tag in tags):
        score = max(score, 0.96)
    return max(0.0, min(1.0, score))


def _reel_usefulness(metadata: VideoMetadata, motion: float, tags: list[str]) -> float:
    duration = metadata.duration_seconds
    duration_score = (
        1.0
        if 2.0 <= duration <= 18.0
        else max(0.25, 1.0 - abs(duration - 10.0) / 45.0)
    )
    motion_score = max(0.25, min(1.0, motion * 1.35))
    travel_tags = {"mountains", "water", "beach", "sunset/sunrise", "food"}
    content_bonus = 0.2 if travel_tags.intersection(tags) else 0
    return max(0.0, min(1.0, duration_score * 0.55 + motion_score * 0.25 + content_bonus))


def _pct(value: float) -> float:
    return max(0.0, min(100.0, value * 100.0))


def normalize_score_weights(weights: dict[str, float]) -> dict[str, float]:
    merged = DEFAULT_SCORE_WEIGHTS | weights
    total = sum(max(0.0, value) for value in merged.values())
    if total <= 0:
        return {key: value / 100.0 for key, value in DEFAULT_SCORE_WEIGHTS.items()}
    return {key: max(0.0, value) / total for key, value in merged.items()}


def _format_weight(weight: float) -> str:
    return f"{weight * 100:.0f}"
