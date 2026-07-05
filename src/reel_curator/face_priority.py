from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FaceReference:
    name: str
    path: Path
    vector: np.ndarray


@dataclass(slots=True)
class FaceReferenceAddResult:
    accepted: bool
    name: str
    path: Path | None
    face_count: int
    message: str


@dataclass(slots=True)
class FaceMatchResult:
    matched_names: list[str]
    score: float
    face_count: int


class FacePriorityMatcher:
    def __init__(self, reference_dir: Path, threshold: float = 0.72) -> None:
        self.reference_dir = reference_dir
        self.threshold = threshold
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.detector = _face_detector()
        self.references = self._load_references()

    def match_frame(self, frame_bgr: np.ndarray) -> FaceMatchResult:
        vectors = self._face_vectors(frame_bgr)
        if not vectors:
            return FaceMatchResult(matched_names=[], score=0.0, face_count=0)

        matched: dict[str, float] = {}
        for vector in vectors:
            for reference in self.references:
                similarity = _cosine_similarity(vector, reference.vector)
                if similarity >= self.threshold:
                    matched[reference.name] = max(matched.get(reference.name, 0.0), similarity)

        score = min(1.0, 0.45 + 0.18 * len(vectors) + 0.22 * len(matched)) if vectors else 0.0
        if matched:
            score = min(1.0, score + 0.2)
        return FaceMatchResult(
            matched_names=sorted(matched, key=matched.get, reverse=True),
            score=score,
            face_count=len(vectors),
        )

    def add_reference_image(
        self, name: str, image_bytes: bytes, suffix: str
    ) -> FaceReferenceAddResult:
        clean_name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -_").strip()
        clean_name = clean_name or "person"
        normalized_bytes = _normalize_image_bytes(image_bytes)
        if normalized_bytes is None:
            return FaceReferenceAddResult(
                accepted=False,
                name=clean_name,
                path=None,
                face_count=0,
                message="Could not read image.",
            )
        image = _decode_image(normalized_bytes)
        if image is None:
            return FaceReferenceAddResult(
                accepted=False,
                name=clean_name,
                path=None,
                face_count=0,
                message="Could not read image.",
            )
        vectors = self._face_vectors(image)
        if not vectors:
            return FaceReferenceAddResult(
                accepted=False,
                name=clean_name,
                path=None,
                face_count=0,
                message="No clear face found.",
            )

        digest = hashlib.sha1(normalized_bytes).hexdigest()[:10]
        path = self.reference_dir / f"{clean_name}__{digest}.jpg"
        path.write_bytes(normalized_bytes)
        self.references = self._load_references()
        return FaceReferenceAddResult(
            accepted=True,
            name=clean_name,
            path=path,
            face_count=len(vectors),
            message=f"Accepted {len(vectors)} face(s).",
        )

    def roster(self) -> dict[str, list[Path]]:
        people: dict[str, list[Path]] = {}
        for reference in self.references:
            people.setdefault(reference.name, []).append(reference.path)
        return people

    def _load_references(self) -> list[FaceReference]:
        references: list[FaceReference] = []
        for path in sorted(self.reference_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic", ".webp"}:
                continue
            image = _decode_image(path.read_bytes())
            if image is None:
                continue
            vectors = self._face_vectors(image)
            if not vectors:
                LOGGER.info("No face found in reference image %s", path)
                continue
            name = path.stem.split("__", 1)[0]
            references.append(FaceReference(name=name, path=path, vector=vectors[0]))
        return references

    def _face_vectors(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(40, 40)
        )
        vectors: list[np.ndarray] = []
        for x, y, w, h in faces:
            crop = gray[y : y + h, x : x + w]
            resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
            equalized = cv2.equalizeHist(resized)
            vector = equalized.astype(np.float32).reshape(-1)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vectors.append(vector / norm)
        return vectors


def _face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")
    return detector


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    normalized = _normalize_image_bytes(image_bytes)
    if normalized is None:
        return None
    data = np.frombuffer(normalized, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalized_reference_image(path: Path) -> bytes | None:
    return _normalize_image_bytes(path.read_bytes())


def _normalize_image_bytes(image_bytes: bytes) -> bytes | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            output = io.BytesIO()
            oriented.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue()
    except Exception:
        LOGGER.exception("Could not normalize reference image")
        return None
