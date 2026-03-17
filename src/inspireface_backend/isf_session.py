"""
InspireFace backend wrapper.

Provides a unified interface to InspireFace SDK that matches the pipeline's
expected API, replacing separate ONNX-based detector, aligner, anti-spoof,
recognizer, and tracker modules.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import inspireface as isf

logger = logging.getLogger(__name__)

# Mapping detect mode strings to InspireFace constants
DETECT_MODES = {
    "always_detect": isf.HF_DETECT_MODE_ALWAYS_DETECT,
    "light_track": isf.HF_DETECT_MODE_LIGHT_TRACK,
}


@dataclass
class ISFFaceDetection:
    """Face detection result from InspireFace, compatible with pipeline."""
    bbox: np.ndarray           # [x1, y1, x2, y2]
    score: float               # detection confidence
    landmarks: np.ndarray      # (5, 2) five key points
    track_id: int = -1         # tracking ID (-1 if no tracking)
    track_count: int = 0       # number of frames this track has been active
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    # Internal: keep original InspireFace face object for feature extraction
    _isf_face: object = field(default=None, repr=False)


@dataclass
class ISFLivenessResult:
    """Liveness check result from InspireFace."""
    is_real: bool
    confidence: float
    label: str  # "real" or "spoof"


@dataclass
class ISFQualityResult:
    """Quality check result from InspireFace."""
    passed: bool
    quality_confidence: float
    reason: str


class InspireFaceBackend:
    """
    Unified InspireFace backend replacing multiple ONNX modules.

    Provides:
    - Face detection (with optional built-in tracking)
    - Face quality assessment
    - RGB liveness detection (anti-spoofing)
    - Face feature extraction (recognition embeddings)
    """

    def __init__(
        self,
        model_pack: str = "Pikachu",
        detect_mode: str = "light_track",
        detect_pixel_level: int = 320,
        max_detect_num: int = 20,
        enable_recognition: bool = True,
        enable_liveness: bool = True,
        enable_quality: bool = True,
        enable_mask_detect: bool = False,
        enable_interaction: bool = False,
        confidence_threshold: float = 0.5,
        liveness_threshold: float = 0.5,
        quality_threshold: float = 0.3,
    ):
        self.model_pack = model_pack
        self.liveness_threshold = liveness_threshold
        self.quality_threshold = quality_threshold
        self.enable_liveness = enable_liveness
        self.enable_quality = enable_quality
        self.enable_recognition = enable_recognition

        # Initialize InspireFace SDK
        logger.info(f"Loading InspireFace model pack: {model_pack}")
        ret = isf.reload(model_pack)
        if not ret:
            raise RuntimeError(
                f"Failed to load InspireFace model pack '{model_pack}'. "
                "Ensure the model is downloaded."
            )

        # Build feature flags
        opt = isf.HF_ENABLE_NONE
        if enable_recognition:
            opt |= isf.HF_ENABLE_FACE_RECOGNITION
        if enable_liveness:
            opt |= isf.HF_ENABLE_LIVENESS
        if enable_quality:
            opt |= isf.HF_ENABLE_QUALITY
        if enable_mask_detect:
            opt |= isf.HF_ENABLE_MASK_DETECT
        if enable_interaction:
            opt |= isf.HF_ENABLE_INTERACTION

        # Build pipeline feature flags for face_pipeline calls
        self._pipeline_features = isf.HF_ENABLE_NONE
        if enable_liveness:
            self._pipeline_features |= isf.HF_ENABLE_LIVENESS
        if enable_quality:
            self._pipeline_features |= isf.HF_ENABLE_QUALITY

        # Resolve detect mode
        mode = DETECT_MODES.get(detect_mode, isf.HF_DETECT_MODE_LIGHT_TRACK)

        # Create session
        self.session = isf.InspireFaceSession(
            opt, mode,
            max_detect_num=max_detect_num,
            detect_pixel_level=detect_pixel_level,
        )
        self.session.set_detection_confidence_threshold(confidence_threshold)

        # Configure tracking settings if using light_track mode
        if detect_mode == "light_track":
            self.session.set_track_mode_smooth_ratio(0.06)
            self.session.set_track_mode_num_smooth_cache_frame(15)
            self.session.set_filter_minimum_face_pixel_size(0)
            self.session.set_track_model_detect_interval(0)
            self.session.set_track_lost_recovery_mode(True)

        logger.info(
            f"InspireFace initialized: model={model_pack}, mode={detect_mode}, "
            f"recognition={enable_recognition}, liveness={enable_liveness}, "
            f"quality={enable_quality}"
        )

    def detect(self, image: np.ndarray) -> list[ISFFaceDetection]:
        """
        Detect faces in an image (with tracking if enabled).

        Args:
            image: BGR image (H, W, 3)

        Returns:
            List of ISFFaceDetection with bbox, landmarks, track_id, pose
        """
        faces = self.session.face_detection(image)

        detections = []
        for face in faces:
            x1, y1, x2, y2 = face.location

            # Get five key points (standard face landmarks)
            five_pts = self.session.get_face_five_key_points(face)

            det = ISFFaceDetection(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                score=face.detection_confidence,
                landmarks=five_pts.astype(np.float32) if five_pts is not None else np.zeros((5, 2), dtype=np.float32),
                track_id=getattr(face, 'track_id', -1),
                track_count=getattr(face, 'track_count', 0),
                roll=face.roll,
                yaw=face.yaw,
                pitch=face.pitch,
                _isf_face=face,
            )
            detections.append(det)

        logger.debug(f"Detected {len(detections)} faces")
        return detections

    def run_pipeline(
        self, image: np.ndarray, detections: list[ISFFaceDetection]
    ) -> list[dict]:
        """
        Run face pipeline (liveness + quality) on detected faces.

        Args:
            image: BGR image
            detections: list of ISFFaceDetection from detect()

        Returns:
            List of dicts with 'liveness_confidence', 'quality_confidence'
            per face, in the same order as detections.
        """
        if not self._pipeline_features or not detections:
            # Return defaults if no pipeline features enabled
            return [
                {"liveness_confidence": 1.0, "quality_confidence": 1.0}
                for _ in detections
            ]

        # Extract original InspireFace face objects
        isf_faces = [d._isf_face for d in detections]

        try:
            extends = self.session.face_pipeline(
                image, isf_faces, self._pipeline_features
            )

            results = []
            for ext in extends:
                result = {
                    "liveness_confidence": getattr(ext, 'rgb_liveness_confidence', 1.0),
                    "quality_confidence": getattr(ext, 'quality_confidence', 1.0),
                }
                results.append(result)

            return results
        except Exception as e:
            logger.debug(f"face_pipeline batch call failed: {e}, using defaults")
            return [
                {"liveness_confidence": 1.0, "quality_confidence": 1.0}
                for _ in detections
            ]

    def check_liveness(
        self, image: np.ndarray, detection: ISFFaceDetection
    ) -> ISFLivenessResult:
        """
        Check if a face is real (not a spoof) using InspireFace liveness.

        Args:
            image: BGR image
            detection: single ISFFaceDetection

        Returns:
            ISFLivenessResult
        """
        if not self.enable_liveness:
            return ISFLivenessResult(is_real=True, confidence=1.0, label="real")

        try:
            extends = self.session.face_pipeline(
                image, [detection._isf_face], isf.HF_ENABLE_LIVENESS
            )

            if extends:
                confidence = getattr(extends[0], 'rgb_liveness_confidence', 1.0)
                is_real = confidence >= self.liveness_threshold
                return ISFLivenessResult(
                    is_real=is_real,
                    confidence=confidence,
                    label="real" if is_real else "spoof",
                )
        except Exception as e:
            logger.debug(f"Liveness check failed: {e}")

        return ISFLivenessResult(is_real=True, confidence=0.5, label="unknown")

    def check_quality(
        self, image: np.ndarray, detection: ISFFaceDetection
    ) -> ISFQualityResult:
        """
        Check face quality using InspireFace quality model.

        Args:
            image: BGR image
            detection: single ISFFaceDetection

        Returns:
            ISFQualityResult
        """
        if not self.enable_quality:
            return ISFQualityResult(
                passed=True, quality_confidence=1.0, reason="disabled"
            )

        try:
            extends = self.session.face_pipeline(
                image, [detection._isf_face], isf.HF_ENABLE_QUALITY
            )

            if extends:
                confidence = getattr(extends[0], 'quality_confidence', 1.0)
                passed = confidence >= self.quality_threshold
                return ISFQualityResult(
                    passed=passed,
                    quality_confidence=confidence,
                    reason="ok" if passed else "low_quality",
                )
        except Exception as e:
            logger.debug(f"Quality check failed: {e}")

        return ISFQualityResult(
            passed=True, quality_confidence=0.5, reason="unknown"
        )

    def extract_feature(
        self, image: np.ndarray, detection: ISFFaceDetection
    ) -> np.ndarray:
        """
        Extract face recognition feature vector.

        InspireFace handles alignment internally.

        Args:
            image: BGR image
            detection: single ISFFaceDetection

        Returns:
            Feature vector as numpy array (L2-normalized)
        """
        feature = self.session.face_feature_extract(
            image, detection._isf_face
        )

        # Ensure L2 normalization
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        return feature.flatten()

    def compare_features(
        self, feature1: np.ndarray, feature2: np.ndarray
    ) -> float:
        """
        Compare two feature vectors and return similarity score.

        Args:
            feature1, feature2: feature vectors from extract_feature()

        Returns:
            Cosine similarity score
        """
        return float(np.dot(feature1.flatten(), feature2.flatten()))
