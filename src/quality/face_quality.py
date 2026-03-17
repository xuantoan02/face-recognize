"""
Face quality filter.

Filters out low-quality face detections before running expensive
recognition and anti-spoof inference. Checks face size, blur level,
and face angle (yaw/pitch estimation from landmarks).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceQualityFilter:
    """
    Filter face detections by quality criteria.

    Rejects faces that are:
    - Too small (below min_face_size)
    - Too blurry (Laplacian variance below threshold)
    - Too angled (estimated yaw/pitch exceeds threshold)
    """

    def __init__(
        self,
        min_face_size: int = 80,
        blur_threshold: float = 50.0,
        max_yaw: float = 45.0,
        max_pitch: float = 35.0,
        enabled: bool = True,
    ):
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.enabled = enabled

        logger.info(
            f"FaceQualityFilter: min_size={min_face_size}, "
            f"blur_threshold={blur_threshold}, "
            f"max_yaw={max_yaw}, max_pitch={max_pitch}, "
            f"enabled={enabled}"
        )

    def check_size(self, bbox: np.ndarray) -> bool:
        """Check if face bounding box meets minimum size."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return min(w, h) >= self.min_face_size

    def check_blur(self, face_crop: np.ndarray) -> bool:
        """
        Check if face is sharp enough using Laplacian variance.
        Higher variance = sharper image.
        """
        if face_crop.size == 0:
            return False
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= self.blur_threshold

    def check_angle(self, landmarks: np.ndarray) -> bool:
        """
        Estimate face angle from 5-point landmarks.
        Uses eye and nose positions to estimate yaw and pitch.

        Args:
            landmarks: (5, 2) - left_eye, right_eye, nose, mouth_left, mouth_right
        """
        if landmarks.shape[0] < 5:
            return True  # Cannot check, pass by default

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        # Estimate yaw from eye-nose triangle
        eye_center = (left_eye + right_eye) / 2.0
        eye_width = np.linalg.norm(right_eye - left_eye)

        if eye_width < 1.0:
            return False

        # Horizontal offset of nose from eye center
        nose_offset_x = (nose[0] - eye_center[0]) / eye_width
        estimated_yaw = abs(nose_offset_x) * 90.0  # rough approximation

        # Vertical ratio for pitch estimation
        eye_nose_dist = nose[1] - eye_center[1]
        nose_ratio = eye_nose_dist / eye_width
        # Normal ratio is ~0.5-0.7, extremes suggest pitch
        if nose_ratio < 0.2:
            estimated_pitch = 30.0  # looking up
        elif nose_ratio > 1.0:
            estimated_pitch = 30.0  # looking down
        else:
            estimated_pitch = 0.0

        return estimated_yaw <= self.max_yaw and estimated_pitch <= self.max_pitch

    def check(
        self,
        bbox: np.ndarray,
        face_crop: np.ndarray | None = None,
        landmarks: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """
        Run all quality checks on a face detection.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            face_crop: optional BGR face crop for blur check
            landmarks: optional (5, 2) landmarks for angle check

        Returns:
            (passed, reason): True if passed, reason string if failed
        """
        if not self.enabled:
            return True, "disabled"

        if not self.check_size(bbox):
            return False, "too_small"

        if face_crop is not None and not self.check_blur(face_crop):
            return False, "too_blurry"

        if landmarks is not None and not self.check_angle(landmarks):
            return False, "bad_angle"

        return True, "ok"
