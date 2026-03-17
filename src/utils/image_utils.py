"""
Utility functions for image preprocessing and post-processing.
"""

import numpy as np
import cv2


def resize_image(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float]:
    """
    Resize image keeping aspect ratio, padding with zeros.

    Args:
        image: BGR image (H, W, 3)
        target_size: target square size

    Returns:
        resized_image: (target_size, target_size, 3)
        scale: scale factor used
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale


def normalize_image(image: np.ndarray, mean: float = 127.5, std: float = 127.5) -> np.ndarray:
    """
    Normalize image to [-1, 1] range.
    """
    return (image.astype(np.float32) - mean) / std


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Non-Maximum Suppression.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def crop_face_with_margin(
    image: np.ndarray,
    bbox: np.ndarray,
    scale: float = 2.7
) -> np.ndarray:
    """
    Crop face region with margin for anti-spoofing.

    Args:
        image: original BGR image
        bbox: [x1, y1, x2, y2]
        scale: scale factor for crop region

    Returns:
        cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1
    face_size = max(bw, bh)

    new_size = face_size * scale
    half = new_size / 2

    nx1 = int(max(0, cx - half))
    ny1 = int(max(0, cy - half))
    nx2 = int(min(w, cx + half))
    ny2 = int(min(h, cy + half))

    crop = image[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        crop = image[int(y1):int(y2), int(x1):int(x2)]
    return crop


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings along last axis."""
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norm + 1e-10)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(emb1.flatten(), emb2.flatten()))
