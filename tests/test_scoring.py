from reel_curator.models import VideoMetadata
from reel_curator.scoring import calculate_score


def test_portrait_scenic_clip_scores_higher_than_poor_clip() -> None:
    metadata = VideoMetadata(
        path="/tmp/a.mov",
        filename="a.mov",
        size_bytes=100,
        modified_time=1,
        duration_seconds=8,
        width=1080,
        height=1920,
        fps=30,
    )

    good = calculate_score(
        metadata=metadata,
        brightness=145,
        sharpness=0.9,
        stability=0.9,
        motion=0.6,
        tags=["portrait", "people", "mountains", "water", "outdoor"],
        uniqueness=1.0,
        people_priority=0.8,
    )
    poor = calculate_score(
        metadata=metadata,
        brightness=20,
        sharpness=0.1,
        stability=0.1,
        motion=0.05,
        tags=["portrait"],
        uniqueness=0.3,
        people_priority=0.0,
    )

    assert good.overall > poor.overall
    assert good.overall <= 100


def test_people_priority_is_largest_score_signal() -> None:
    metadata = VideoMetadata(
        path="/tmp/a.mov",
        filename="a.mov",
        size_bytes=100,
        modified_time=1,
        duration_seconds=8,
        width=1080,
        height=1920,
        fps=30,
    )

    people = calculate_score(
        metadata=metadata,
        brightness=130,
        sharpness=0.5,
        stability=0.5,
        motion=0.4,
        tags=["portrait", "people"],
        uniqueness=0.7,
        people_priority=0.9,
    )
    scenic_no_people = calculate_score(
        metadata=metadata,
        brightness=130,
        sharpness=0.5,
        stability=0.5,
        motion=0.4,
        tags=["portrait", "mountains", "water", "outdoor"],
        uniqueness=0.7,
        people_priority=0.0,
    )

    assert people.overall > scenic_no_people.overall
