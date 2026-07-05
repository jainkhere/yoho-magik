from __future__ import annotations

from collections import defaultdict

from reel_curator.metrics import hamming_distance_hex
from reel_curator.models import VideoAnalysis
from reel_curator.scoring import calculate_score


def assign_duplicate_groups(
    analyses: list[VideoAnalysis], hamming_threshold: int
) -> list[VideoAnalysis]:
    groups: list[list[VideoAnalysis]] = []
    for analysis in sorted(analyses, key=lambda item: item.metadata.path):
        matched_group: list[VideoAnalysis] | None = None
        for group in groups:
            if any(
                hamming_distance_hex(analysis.perceptual_hash, other.perceptual_hash)
                <= hamming_threshold
                for other in group
            ):
                matched_group = group
                break
        if matched_group is None:
            groups.append([analysis])
        else:
            matched_group.append(analysis)

    for index, group in enumerate(groups, start=1):
        group_id = f"D{index:04d}" if len(group) > 1 else None
        representative = max(group, key=lambda item: item.score.overall)
        uniqueness = 1.0 if len(group) == 1 else max(0.35, 1.0 / len(group))
        for item in group:
            item.duplicate_group = group_id
            item.duplicate_representative = item is representative
            item.score = calculate_score(
                metadata=item.metadata,
                brightness=item.brightness,
                sharpness=item.sharpness,
                stability=item.stability,
                motion=item.motion,
                tags=item.tags,
                uniqueness=uniqueness,
                people_priority=item.priority_people_score,
            )
    return analyses


def duplicate_summary(analyses: list[VideoAnalysis]) -> dict[str, list[VideoAnalysis]]:
    groups: dict[str, list[VideoAnalysis]] = defaultdict(list)
    for item in analyses:
        if item.duplicate_group:
            groups[item.duplicate_group].append(item)
    return dict(groups)
