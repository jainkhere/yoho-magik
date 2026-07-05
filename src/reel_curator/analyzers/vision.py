from __future__ import annotations

import logging
from pathlib import Path

from reel_curator.models import VisionResult

LOGGER = logging.getLogger(__name__)


class AppleVisionAnalyzer:
    """Thin optional adapter around Apple's local Vision framework.

    PyObjC does not expose every Vision API with identical signatures across macOS releases, so this
    adapter is intentionally conservative. It detects faces, animals, text rectangles, and common
    scene classifications when the local framework supports them. Failures return an empty result
    rather than interrupting the scan.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.available = False
        if not enabled:
            return
        try:
            import Quartz  # type: ignore  # noqa: F401
            import Vision  # type: ignore  # noqa: F401

            self.available = True
        except Exception as exc:  # pragma: no cover - depends on macOS/PyObjC
            LOGGER.info("Apple Vision unavailable: %s", exc)

    def analyze_image(self, image_path: Path) -> VisionResult:
        if not self.enabled or not self.available:
            return VisionResult()
        try:
            return self._analyze_image(image_path)
        except Exception as exc:  # pragma: no cover - depends on macOS/PyObjC
            LOGGER.warning("Apple Vision failed for %s: %s", image_path, exc)
            return VisionResult()

    def _analyze_image(self, image_path: Path) -> VisionResult:  # pragma: no cover - macOS specific
        import Quartz  # type: ignore
        import Vision  # type: ignore

        url = Quartz.NSURL.fileURLWithPath_(str(image_path))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        result = VisionResult()

        requests = []
        face_request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        text_request = Vision.VNRecognizeTextRequest.alloc().init()
        requests.extend([face_request, text_request])

        if hasattr(Vision, "VNRecognizeAnimalsRequest"):
            animal_request = Vision.VNRecognizeAnimalsRequest.alloc().init()
            requests.append(animal_request)
        else:
            animal_request = None

        if hasattr(Vision, "VNClassifyImageRequest"):
            classify_request = Vision.VNClassifyImageRequest.alloc().init()
            requests.append(classify_request)
        else:
            classify_request = None

        ok, error = handler.performRequests_error_(requests, None)
        if not ok:
            LOGGER.debug("Vision request failed for %s: %s", image_path, error)
            return result

        result.face_count = len(face_request.results() or [])
        result.people_count = result.face_count
        result.text_detected = bool(text_request.results())

        if animal_request is not None:
            result.animal_count = len(animal_request.results() or [])

        labels: list[str] = []
        confidence: dict[str, float] = {}
        if classify_request is not None:
            for observation in classify_request.results() or []:
                label = str(observation.identifier())
                conf = float(observation.confidence())
                if conf >= 0.25:
                    labels.append(label)
                    confidence[label] = conf
        result.labels = labels
        result.confidence_by_label = confidence
        return result
