from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from reel_curator.scoring import DEFAULT_SCORE_WEIGHTS


class AppConfig(BaseModel):
    input_folder: Path = Path("/Users/kunjain/Movies/Alberta videos")
    cache_dir: Path = Path(".reel_curator_cache")
    export_dir: Path = Path("exports")
    workers: int = Field(default=4, ge=1)
    sample_frames: int = Field(default=12, ge=3, le=60)
    duplicate_hamming_threshold: int = Field(default=10, ge=0, le=64)
    min_quality: float = Field(default=0, ge=0, le=100)
    enable_apple_vision: bool = True
    scan_landscape_videos: bool = False
    prioritize_people: bool = True
    face_match_threshold: float = Field(default=0.72, ge=0.1, le=0.99)
    scoring_preset: str = "People first"
    scoring_weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_SCORE_WEIGHTS.copy())

    def resolved(self, base_dir: Path) -> AppConfig:
        data = self.model_dump()
        for key in ("cache_dir", "export_dir"):
            path = Path(data[key])
            data[key] = path if path.is_absolute() else base_dir / path
        data["input_folder"] = Path(data["input_folder"]).expanduser()
        return AppConfig(**data)


def load_config(path: Path = Path("config.yaml")) -> AppConfig:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return AppConfig(**raw).resolved(path.parent.resolve())
    return AppConfig().resolved(Path.cwd())


def save_config(config: AppConfig, path: Path = Path("config.yaml")) -> None:
    data = config.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)
