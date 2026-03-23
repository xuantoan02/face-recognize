"""
BlazeFace face detector using ONNX Runtime.

BlazeFace is a lightweight face detection model from Google MediaPipe,
optimized for real-time performance on CPU.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.image_utils import nms

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox: np.ndarray       # [x1, y1, x2, y2] in original image coords
    score: float           # detection confidence
    landmarks: np.ndarray  # (5, 2) - eyes, nose, mouth corners


# BlazeFace back camera anchors generation
def _generate_anchors(input_size: int = 256) -> np.ndarray:
    """
    Generate SSD anchors for BlazeFace back camera model.
    Following MediaPipe anchor generation spec.
    """
    strides = [16, 32]  # back camera uses 2 strides
    anchors_per_stride = [1, 1]  # back camera: 1 anchor per grid cell (after merge)

    anchors = []
    for stride, num_anchors in zip(strides, anchors_per_stride):
        grid_h = input_size // stride
        grid_w = input_size // stride

        for y in range(grid_h):
            for x in range(grid_w):
                for _ in range(num_anchors):
                    cx = (x + 0.5) / grid_w
                    cy = (y + 0.5) / grid_h
                    anchors.append([cx, cy])

    return np.array(anchors, dtype=np.float32)


def _generate_blazeface_anchors(input_size: int = 128) -> np.ndarray:
    """
    Generate SSD anchors for BlazeFace front camera model (128x128).
    """
    strides = [8, 16]
    num_anchors_per_stride = [2, 6]

    anchors = []
    for stride, num_anchors in zip(strides, num_anchors_per_stride):
        grid_h = input_size // stride
        grid_w = input_size // stride
        for y in range(grid_h):
            for x in range(grid_w):
                for _ in range(num_anchors):
                    cx = (x + 0.5) * stride / input_size
                    cy = (y + 0.5) * stride / input_size
                    anchors.append([cx, cy])

    return np.array(anchors, dtype=np.float32)


class FaceDetector:
    """
    BlazeFace face detector.

    Supports both front (128x128) and back (256x256) camera models.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 128,
        confidence_threshold: float = 0.7,
        nms_iou_threshold: float = 0.3,
        num_threads: int = 2,
        providers: Optional[list[str]] = None,
    ):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Detect model variant from input size
        if input_size == 256:
            self.anchors = _generate_anchors(256)
            self.num_keypoints = 6
        else:
            self.anchors = _generate_blazeface_anchors(128)
            self.num_keypoints = 6

        logger.info(
            f"FaceDetector loaded: input={input_size}x{input_size}, "
            f"anchors={len(self.anchors)}, model={model_path}"
        )

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Preprocess image for BlazeFace.

        Returns:
            input_tensor: (1, input_size, input_size, 3) float32 [-1, 1]
            meta: dict with 'scale', 'pad_x', 'pad_y', 'orig_h', 'orig_w'
        """
        orig_h, orig_w = image.shape[:2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize with aspect ratio preservation + padding
        scale = self.input_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize to [-1, 1]
        input_tensor = (padded.astype(np.float32) - 127.5) / 127.5

        # Add batch dimension: (1, H, W, 3)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        meta = {
            "scale": scale,
            "pad_x": 0,
            "pad_y": 0,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "new_h": new_h,
            "new_w": new_w,
        }
        return input_tensor, meta

    def _decode_predictions(
        self,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
        meta: dict,
    ) -> list[FaceDetection]:
        """
        Decode raw model outputs to FaceDetection objects.

        Args:
            raw_boxes: (1, num_anchors, 16/18) - center offsets + size + keypoints
            raw_scores: (1, num_anchors, 1) - raw logits
        """
        # Squeeze batch dim
        boxes = raw_boxes[0]    # (num_anchors, 16+)
        scores = raw_scores[0]  # (num_anchors, 1)

        # Apply sigmoid to scores
        scores = 1.0 / (1.0 + np.exp(-scores))
        scores = scores.flatten()

        # Filter by confidence
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return []

        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_anchors = self.anchors[mask]

        # Decode boxes: center format relative to anchors
        cx = filtered_boxes[:, 0] / self.input_size + filtered_anchors[:, 0]
        cy = filtered_boxes[:, 1] / self.input_size + filtered_anchors[:, 1]
        w = filtered_boxes[:, 2] / self.input_size
        h = filtered_boxes[:, 3] / self.input_size

        # Convert to [x1, y1, x2, y2] in normalized coords
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Decode keypoints
        all_keypoints = []
        for i in range(min(self.num_keypoints, (filtered_boxes.shape[1] - 4) // 2)):
            kp_x = filtered_boxes[:, 4 + i * 2] / self.input_size + filtered_anchors[:, 0]
            kp_y = filtered_boxes[:, 4 + i * 2 + 1] / self.input_size + filtered_anchors[:, 1]
            all_keypoints.append(np.stack([kp_x, kp_y], axis=-1))

        # Stack to (N, num_kp, 2)
        if all_keypoints:
            keypoints = np.stack(all_keypoints, axis=1)
        else:
            keypoints = np.zeros((len(filtered_scores), 6, 2), dtype=np.float32)

        # Scale back to original image coords
        scale = meta["scale"]
        orig_h = meta["orig_h"]
        orig_w = meta["orig_w"]

        det_boxes = np.stack([x1, y1, x2, y2], axis=-1) * self.input_size / scale
        det_boxes[:, [0, 2]] = np.clip(det_boxes[:, [0, 2]], 0, orig_w)
        det_boxes[:, [1, 3]] = np.clip(det_boxes[:, [1, 3]], 0, orig_h)

        det_keypoints = keypoints * self.input_size / scale

        # NMS
        keep = nms(det_boxes, filtered_scores, self.nms_iou_threshold)

        detections = []
        for idx in keep:
            # Extract 5 key landmarks: right_eye, left_eye, nose, mouth_right, mouth_left
            kps = det_keypoints[idx]
            # BlazeFace gives 6 keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear
            # We use first 4 + remap to 5-point format
            landmarks_5 = np.zeros((5, 2), dtype=np.float32)
            if kps.shape[0] >= 4:
                landmarks_5[0] = kps[0]  # right eye
                landmarks_5[1] = kps[1]  # left eye
                landmarks_5[2] = kps[2]  # nose tip
                landmarks_5[3] = kps[3]  # mouth center → approximate mouth right
                landmarks_5[4] = kps[3]  # mouth center → approximate mouth left
                # Offset mouth corners slightly from center
                mouth_width = np.abs(kps[1][0] - kps[0][0]) * 0.4
                landmarks_5[3][0] -= mouth_width  # mouth right
                landmarks_5[4][0] += mouth_width  # mouth left

            det = FaceDetection(
                bbox=det_boxes[idx],
                score=float(filtered_scores[idx]),
                landmarks=landmarks_5,
            )
            detections.append(det)

        return detections

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            list of FaceDetection
        """
        input_tensor, meta = self._preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # BlazeFace outputs: regressors (boxes) and classificators (scores)
        if len(outputs) == 2:
            raw_boxes = outputs[0]
            raw_scores = outputs[1]
        else:
            # Some ONNX exports may have different output ordering
            raw_boxes = outputs[0]
            raw_scores = outputs[1] if len(outputs) > 1 else outputs[0]

        detections = self._decode_predictions(raw_boxes, raw_scores, meta)

        logger.debug(f"Detected {len(detections)} faces")
        return detections
