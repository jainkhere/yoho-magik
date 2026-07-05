from __future__ import annotations

import cv2
import numpy as np

from reel_curator.frame_sampler import SampledFrame


def brightness_score(frames: list[SampledFrame]) -> float:
    if not frames:
        return 0.0
    values = []
    for item in frames:
        hsv = cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2HSV)
        values.append(float(np.mean(hsv[:, :, 2])) / 255.0)
    avg = float(np.mean(values))
    return _clamp01(1.0 - abs(avg - 0.58) / 0.58)


def raw_brightness(frames: list[SampledFrame]) -> float:
    if not frames:
        return 0.0
    return float(np.mean([np.mean(cv2.cvtColor(f.frame_bgr, cv2.COLOR_BGR2GRAY)) for f in frames]))


def sharpness_score(frames: list[SampledFrame]) -> float:
    if not frames:
        return 0.0
    variances = []
    for item in frames:
        gray = cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2GRAY)
        variances.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    median_variance = float(np.median(variances))
    return _clamp01(median_variance / 650.0)


def motion_and_stability(frames: list[SampledFrame]) -> tuple[float, float]:
    if len(frames) < 2:
        return 0.0, 0.5
    motion_values: list[float] = []
    shake_values: list[float] = []
    previous = _small_gray(frames[0].frame_bgr)
    for item in frames[1:]:
        current = _small_gray(item.frame_bgr)
        diff = cv2.absdiff(previous, current)
        motion_values.append(float(np.mean(diff)) / 255.0)

        points = cv2.goodFeaturesToTrack(previous, maxCorners=120, qualityLevel=0.01, minDistance=8)
        if points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
            if next_points is not None and status is not None:
                valid = status.reshape(-1) == 1
                if np.any(valid):
                    shifts = next_points[valid] - points[valid]
                    shake_values.append(float(np.std(shifts)))
        previous = current

    motion = _clamp01(float(np.mean(motion_values)) * 8.0) if motion_values else 0.0
    shake = float(np.median(shake_values)) if shake_values else 8.0
    stability = _clamp01(1.0 - shake / 18.0)
    return motion, stability


def scene_change_count(frames: list[SampledFrame]) -> int:
    if len(frames) < 2:
        return 0
    count = 0
    previous_hist = _histogram(frames[0].frame_bgr)
    for item in frames[1:]:
        hist = _histogram(item.frame_bgr)
        distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        if distance > 0.45:
            count += 1
        previous_hist = hist
    return count


def perceptual_hash(frame_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if value else "0" for value in diff.flatten())
    return f"{int(bits, 2):016x}"


def hamming_distance_hex(left: str, right: str) -> int:
    if not left or not right:
        return 64
    return (int(left, 16) ^ int(right, 16)).bit_count()


def select_best_frame(frames: list[SampledFrame]) -> SampledFrame | None:
    if not frames:
        return None
    best: tuple[float, SampledFrame] | None = None
    for item in frames:
        single = [item]
        gray = cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2GRAY)
        sharp = _clamp01(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 650.0)
        bright = brightness_score(single)
        hsv = cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1])) / 255.0
        score = 0.55 * sharp + 0.30 * bright + 0.15 * _clamp01(saturation * 1.8)
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best else frames[0]


def _small_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)


def _histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
