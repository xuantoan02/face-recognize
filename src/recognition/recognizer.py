"""
Face recognition using MobileFaceNet ONNX model.

MobileFaceNet is a lightweight face recognition model that generates
compact face embeddings for identity verification.
"""

import logging
from dataclasses import dataclass

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


class FaceRecognizer:
    """
    MobileFaceNet-based face recognizer.

    Takes aligned 112x112 face crops and produces L2-normalized embeddings.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        num_threads: int = 2,
    ):
        self.input_size = input_size

        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Detect embedding dimension from output shape
        output_shape = self.session.get_outputs()[0].shape
        self.embedding_dim = output_shape[-1] if output_shape[-1] else 512

        logger.info(
            f"FaceRecognizer loaded: input={input_size}x{input_size}, "
            f"embedding_dim={self.embedding_dim}, model={model_path}"
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

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face.

        Args:
            aligned_face: BGR aligned face (112, 112, 3)

        Returns:
            embedding: L2-normalized embedding vector
        """
        input_tensor = self._preprocess(aligned_face)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Get embedding and L2 normalize
        embedding = outputs[0].flatten()
        embedding = l2_normalize(embedding.reshape(1, -1)).flatten()

        return embedding

    def get_embeddings_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple aligned faces.

        Args:
            aligned_faces: list of BGR aligned faces (112, 112, 3)

        Returns:
            embeddings: (N, embedding_dim) L2-normalized
        """
        if not aligned_faces:
            return np.array([])

        embeddings = []
        for face in aligned_faces:
            emb = self.get_embedding(face)
            embeddings.append(emb)

        return np.array(embeddings)
