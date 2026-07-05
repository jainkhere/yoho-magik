from reel_curator.duplicates import assign_duplicate_groups
from reel_curator.models import ScoreBreakdown, VideoAnalysis, VideoMetadata, VisionResult


def _analysis(name: str, phash: str, score: float) -> VideoAnalysis:
    metadata = VideoMetadata(
        path=f"/tmp/{name}",
        filename=name,
        size_bytes=100,
        modified_time=1,
        duration_seconds=8,
        width=1080,
        height=1920,
        fps=30,
    )
    return VideoAnalysis(
        metadata=metadata,
        brightness=140,
        sharpness=0.8,
        stability=0.8,
        motion=0.5,
        scene_changes=1,
        perceptual_hash=phash,
        thumbnail_path="/tmp/thumb.jpg",
        best_frame_second=1,
        tags=["portrait", "mountains"],
        vision=VisionResult(),
        score=ScoreBreakdown(80, 80, 80, 80, 0, 100, 80, 80, score, ""),
    )


def test_assign_duplicate_groups_marks_representative() -> None:
    analyses = [
        _analysis("a.mov", "0000000000000000", 70),
        _analysis("b.mov", "0000000000000001", 90),
        _analysis("c.mov", "ffffffffffffffff", 50),
    ]

    grouped = assign_duplicate_groups(analyses, hamming_threshold=2)

    duplicate_items = [item for item in grouped if item.duplicate_group]
    assert len(duplicate_items) == 2
    assert sum(item.duplicate_representative for item in duplicate_items) == 1
    assert grouped[2].duplicate_group is None
