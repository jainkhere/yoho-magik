from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from reel_curator.analyzers.vision import AppleVisionAnalyzer
from reel_curator.cache import AnalysisCache
from reel_curator.config import AppConfig
from reel_curator.duplicates import assign_duplicate_groups
from reel_curator.face_priority import FacePriorityMatcher
from reel_curator.frame_sampler import sample_frames, save_thumbnail
from reel_curator.metrics import (
    motion_and_stability,
    perceptual_hash,
    raw_brightness,
    scene_change_count,
    select_best_frame,
    sharpness_score,
)
from reel_curator.models import VideoAnalysis, VideoIdentity, VisionResult
from reel_curator.scanner import scan_videos
from reel_curator.scoring import calculate_score
from reel_curator.tagging import generate_tags
from reel_curator.video_probe import probe_video

LOGGER = logging.getLogger(__name__)


def analyze_library(config: AppConfig, force: bool = False) -> list[VideoAnalysis]:
    cache = AnalysisCache(config.cache_dir)
    identities = scan_videos(config.input_folder)
    vision = AppleVisionAnalyzer(enabled=config.enable_apple_vision)
    analyses: list[VideoAnalysis] = []
    pending: list[VideoIdentity] = []
    for identity in identities:
        cached = None if force else cache.get(identity.cache_key)
        if cached:
            if config.scan_landscape_videos or cached.metadata.orientation != "landscape":
                analyses.append(cached)
        else:
            if config.scan_landscape_videos:
                pending.append(identity)
            else:
                metadata = probe_video(identity.path)
                if metadata.orientation != "landscape":
                    pending.append(identity)

    if pending:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(
                    _analyze_one, identity, config, cache.thumbnail_dir, vision
                ): identity
                for identity in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing videos"):
                identity = futures[future]
                try:
                    analysis = future.result()
                except Exception:
                    LOGGER.exception("Failed to analyze %s", identity.path)
                    continue
                cache.set(identity.cache_key, analysis)
                analyses.append(analysis)

    analyses = assign_duplicate_groups(analyses, config.duplicate_hamming_threshold)
    for identity in identities:
        match = next((item for item in analyses if item.metadata.path == str(identity.path)), None)
        if match:
            cache.set(identity.cache_key, match)
    return sorted(analyses, key=lambda item: item.score.overall, reverse=True)


def _analyze_one(
    identity: VideoIdentity,
    config: AppConfig,
    thumbnail_dir: Path,
    vision: AppleVisionAnalyzer,
) -> VideoAnalysis:
    metadata = probe_video(identity.path)
    frames = sample_frames(
        identity.path,
        config.sample_frames,
        metadata.rotation_degrees,
        display_size=(metadata.width, metadata.height),
    )
    best = select_best_frame(frames)
    if best is None:
        raise ValueError(f"No readable frames: {identity.path}")

    thumb_name = hashlib.sha1(identity.cache_key.encode("utf-8")).hexdigest() + ".jpg"
    thumbnail_path = thumbnail_dir / thumb_name
    save_thumbnail(best.frame_bgr, thumbnail_path)

    vision_result = vision.analyze_image(thumbnail_path) if vision.available else VisionResult()
    face_matcher = FacePriorityMatcher(
        config.cache_dir / "priority_people", threshold=config.face_match_threshold
    )
    face_match = (
        face_matcher.match_frame(best.frame_bgr) if config.prioritize_people else None
    )
    if face_match and face_match.face_count > vision_result.face_count:
        vision_result.face_count = face_match.face_count
        vision_result.people_count = max(vision_result.people_count, face_match.face_count)
    brightness = raw_brightness(frames)
    sharpness = sharpness_score(frames)
    motion, stability = motion_and_stability(frames)
    tags = generate_tags(metadata, frames, vision_result)
    priority_matches = face_match.matched_names if face_match else []
    priority_score = face_match.score if face_match else 0.0
    for name in priority_matches:
        tags.append(f"priority-person:{name}")
    tags = sorted(set(tags))
    score = calculate_score(
        metadata=metadata,
        brightness=brightness,
        sharpness=sharpness,
        stability=stability,
        motion=motion,
        tags=tags,
        uniqueness=1.0,
        people_priority=priority_score,
    )

    return VideoAnalysis(
        metadata=metadata,
        brightness=round(brightness, 2),
        sharpness=round(sharpness, 3),
        stability=round(stability, 3),
        motion=round(motion, 3),
        scene_changes=scene_change_count(frames),
        perceptual_hash=perceptual_hash(best.frame_bgr),
        thumbnail_path=str(thumbnail_path),
        best_frame_second=round(best.second, 2),
        tags=tags,
        vision=vision_result,
        score=score,
        priority_people_matches=priority_matches,
        priority_people_score=round(priority_score, 3),
    )
