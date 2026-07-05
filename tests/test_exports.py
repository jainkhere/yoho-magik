from reel_curator.exports import select_for_export
from reel_curator.models import ScoreBreakdown, VideoAnalysis, VideoMetadata, VisionResult


def _analysis(name: str, score: float, favorite: bool = False) -> VideoAnalysis:
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
        perceptual_hash="0",
        thumbnail_path="/tmp/thumb.jpg",
        best_frame_second=1,
        tags=["portrait"],
        vision=VisionResult(),
        score=ScoreBreakdown(80, 80, 80, 80, 0, 100, 80, 80, score, ""),
        favorite=favorite,
    )


def test_select_for_export_top_n_and_favorites() -> None:
    analyses = [_analysis("a.mov", 50), _analysis("b.mov", 90, favorite=True)]

    assert [item.metadata.filename for item in select_for_export(analyses, "top_n", 1)] == ["b.mov"]
    assert [item.metadata.filename for item in select_for_export(analyses, "favorites")] == [
        "b.mov"
    ]
