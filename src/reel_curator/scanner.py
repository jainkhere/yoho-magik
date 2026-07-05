from __future__ import annotations

from pathlib import Path

from reel_curator.models import VideoIdentity

VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def scan_videos(folder: Path) -> list[VideoIdentity]:
    folder = folder.expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    videos: list[VideoIdentity] = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        stat = path.stat()
        videos.append(VideoIdentity(path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return sorted(videos, key=lambda item: str(item.path).lower())
