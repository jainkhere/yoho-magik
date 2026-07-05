from pathlib import Path

from reel_curator.scanner import scan_videos


def test_scan_videos_finds_common_video_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.MOV").write_bytes(b"x")
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("nope")

    videos = scan_videos(tmp_path)

    assert [item.path.name for item in videos] == ["a.MOV", "b.mp4"]
