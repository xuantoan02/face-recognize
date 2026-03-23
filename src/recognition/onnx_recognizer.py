"""
Face recognition using generic ONNX model.

Based on testmodel.py inference logic. Supports any ONNX face recognition
model that takes NCHW input and produces embedding vectors.

Preprocessing: (pixel - 127.5) / 128.0
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.recognition.base_recognizer import BaseRecognizer

logger = logging.getLogger(__name__)


class ONNXRecognizer(BaseRecognizer):
    """
    Generic ONNX-based face recognizer.

    Takes aligned face crops and produces L2-normalized embeddings.

    Preprocessing (from testmodel.py):
        1. BGR → RGB
        2. Resize to model input size
        3. Normalize: (pixel - 127.5) / 128.0
        4. HWC → CHW, add batch dim → NCHW
    """

    def __init__(
        self,
        model_path: str="models/w600k_mbf.onnx",
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
        Preprocess face image following testmodel.py convention.

        Args:
            aligned_face: BGR face image (H, W, 3).

        Returns:
            NCHW float32 tensor (1, 3, H, W).
        """
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))

        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # -> NCHW
        return img
