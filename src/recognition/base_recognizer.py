"""
Abstract base class for face recognizers.

All face recognizer implementations should inherit from BaseRecognizer
and implement the _preprocess method with model-specific preprocessing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.image_utils import l2_normalize

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Face recognition result."""
    person_id: str
    name: str
    similarity: float
    embedding: np.ndarray


class BaseRecognizer(ABC):
    """
    Abstract base recognizer using ONNX Runtime.

    Subclasses must implement ``_preprocess`` to define model-specific
    image normalization and tensor layout.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        num_threads: int = 2,
        providers: Optional[list[str]] = None,
    ):
        """
        Initialize the ONNX session and cache I/O metadata.

        Args:
            model_path: Path to the .onnx model file.
            input_size: Spatial size (height & width) expected by the model.
            num_threads: Number of intra-op threads for ONNX Runtime.
            providers: Execution providers. Defaults to ``["CPUExecutionProvider"]``.
        """
        self.input_size = input_size
        self.model_path = model_path

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Build session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Create inference session
        self.session = ort.InferenceSession(
            model_path, sess_options, providers=providers
        )

        # Cache I/O metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Detect embedding dimension from output shape
        output_shape = self.session.get_outputs()[0].shape
        self.embedding_dim = (
            output_shape[-1] if isinstance(output_shape[-1], int) else 512
        )

        logger.info(
            "%s loaded: input=%dx%d, embedding_dim=%d, model=%s",
            self.__class__.__name__,
            input_size,
            input_size,
            self.embedding_dim,
            model_path,
        )

    # ------------------------------------------------------------------
    # Abstract – subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Convert an aligned BGR face image to an input tensor.

        Args:
            aligned_face: BGR face image (H, W, 3).

        Returns:
            NCHW (or NHWC) float32 tensor with batch dim, e.g. (1, 3, H, W).
        """
        ...

    # ------------------------------------------------------------------
    # Shared inference logic
    # ------------------------------------------------------------------

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalized embedding from an aligned face image.

        Args:
            aligned_face: BGR aligned face (H, W, 3).

        Returns:
            1-D L2-normalized embedding vector.
        """
        input_tensor = self._preprocess(aligned_face)

        # Run inference
        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor}
        )

        # Get embedding and L2 normalize
        embedding = outputs[0].flatten()
        embedding = l2_normalize(embedding.reshape(1, -1)).flatten()
        return embedding

    def get_embeddings_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of aligned faces.

        Args:
            aligned_faces: List of BGR aligned face images.

        Returns:
            (N, embedding_dim) L2-normalized embeddings.
        """
        if not aligned_faces:
            return np.array([])

        embeddings = []
        for face in aligned_faces:
            emb = self.get_embedding(face)
            embeddings.append(emb)

        return np.array(embeddings)
