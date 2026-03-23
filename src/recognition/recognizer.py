"""
Face recognition using MobileFaceNet ONNX model.

MobileFaceNet is a lightweight face recognition model that generates
compact face embeddings for identity verification.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.recognition.base_recognizer import BaseRecognizer

logger = logging.getLogger(__name__)


class FaceRecognizer(BaseRecognizer):
    """
    MobileFaceNet-based face recognizer.

    Takes aligned 112x112 face crops and produces L2-normalized embeddings.

    Preprocessing:
        1. Resize to input_size
        2. BGR → RGB
        3. Normalize: /255 then (x - 0.5) / 0.5
        4. HWC → CHW (or keep NHWC based on model input)
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        num_threads: int = 2,
        providers: Optional[list[str]] = None,
    ):
        super().__init__(
            model_path=model_path,
            input_size=input_size,
            num_threads=num_threads,
            providers=providers,
        )

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for MobileFaceNet.

        Args:
            aligned_face: BGR aligned face (112, 112, 3)

        Returns:
            input_tensor: (1, 3, 112, 112) or (1, 112, 112, 3) float32
        """
        # Ensure correct size
        if aligned_face.shape[:2] != (self.input_size, self.input_size):
            aligned_face = cv2.resize(
                aligned_face,
                (self.input_size, self.input_size)
            )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Normalize: scale to [0, 1] then standardize
        normalized = rgb.astype(np.float32) / 255.0
        # Standard normalization for face recognition
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Check expected input format
        if self.input_shape and len(self.input_shape) == 4:
            if self.input_shape[1] == 3:
                # NCHW format
                tensor = np.transpose(normalized, (2, 0, 1))
            else:
                # NHWC format
                tensor = normalized
        else:
            # Default to NCHW
            tensor = np.transpose(normalized, (2, 0, 1))

        tensor = np.expand_dims(tensor, axis=0)  # (1, 3, H, W)
        return tensor
