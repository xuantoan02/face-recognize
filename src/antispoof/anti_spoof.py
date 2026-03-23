"""
Anti-spoofing module using MiniFASNetV2 ONNX model.

Detects presentation attacks (printed photos, phone screens, 3D masks)
using the Silent-Face-Anti-Spoofing approach with multi-scale analysis.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.image_utils import crop_face_with_margin

logger = logging.getLogger(__name__)


@dataclass
class SpoofResult:
    """Anti-spoofing result for a single face."""
    is_real: bool
    confidence: float  # confidence that face is real
    label: str         # "real", "fake_2d", "fake_3d"


class AntiSpoof:
    """
    MiniFASNetV2-based face anti-spoofing.

    Uses multi-scale face crops following the Silent-Face approach.
    Each scale provides a different context for liveness detection.
    """

    def __init__(
        self,
        model_paths: list[str],
        input_size: int = 80,
        threshold: float = 0.8,
        scales: list[float] | None = None,
        num_threads: int = 2,
        providers: Optional[list[str]] = None,
    ):
        self.input_size = input_size
        self.threshold = threshold
        self.scales = scales or [2.7, 4.0]

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Load ONNX sessions
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sessions = []
        for path in model_paths:
            try:
                session = ort.InferenceSession(
                    path, sess_options,
                    providers=providers
                )
                self.sessions.append(session)
                logger.info(f"AntiSpoof model loaded: {path}")
            except Exception as e:
                logger.warning(f"Failed to load anti-spoof model {path}: {e}")

        if not self.sessions:
            logger.warning("No anti-spoof models loaded. Anti-spoofing will be disabled.")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess cropped face for MiniFASNetV2.

        Args:
            crop: BGR face crop

        Returns:
            input_tensor: (1, 3, input_size, input_size) float32
        """
        # Resize to model input size
        resized = cv2.resize(crop, (self.input_size, self.input_size))

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))  # (3, H, W)
        tensor = np.expand_dims(tensor, axis=0)         # (1, 3, H, W)

        return tensor

    def check(self, image: np.ndarray, bbox: np.ndarray) -> SpoofResult:
        """
        Check if a detected face is real or spoofed.

        Args:
            image: original BGR image
            bbox: [x1, y1, x2, y2] face bounding box

        Returns:
            SpoofResult with is_real, confidence, and label
        """
        if not self.sessions:
            # No models loaded → assume real
            return SpoofResult(is_real=True, confidence=1.0, label="real")

        predictions = []

        for i, session in enumerate(self.sessions):
            # Use corresponding scale, or default
            scale = self.scales[i] if i < len(self.scales) else self.scales[-1]

            # Crop face with margin
            crop = crop_face_with_margin(image, bbox, scale=scale)
            if crop.size == 0:
                continue

            # Preprocess
            input_tensor = self._preprocess(crop)

            # Run inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_tensor})

            # Output is typically (1, 3) for [fake_2d, fake_3d, real] logits
            logits = outputs[0][0]

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            predictions.append(probs)

        if not predictions:
            return SpoofResult(is_real=True, confidence=0.5, label="unknown")

        # Average predictions across scales/models
        avg_probs = np.mean(predictions, axis=0)

        # Interpret: last class is typically "real"
        if len(avg_probs) == 3:
            # [fake_2d, fake_3d, real]
            real_prob = float(avg_probs[2])
            fake_2d_prob = float(avg_probs[0])
            fake_3d_prob = float(avg_probs[1])
        elif len(avg_probs) == 2:
            # [fake, real]
            real_prob = float(avg_probs[1])
            fake_2d_prob = float(avg_probs[0])
            fake_3d_prob = 0.0
        else:
            real_prob = float(avg_probs[-1])
            fake_2d_prob = 1.0 - real_prob
            fake_3d_prob = 0.0

        is_real = real_prob >= self.threshold

        if is_real:
            label = "real"
        elif fake_3d_prob > fake_2d_prob:
            label = "fake_3d"
        else:
            label = "fake_2d"

        return SpoofResult(
            is_real=is_real,
            confidence=real_prob,
            label=label,
        )
