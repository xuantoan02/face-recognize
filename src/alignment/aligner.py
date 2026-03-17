"""
Face alignment using 5-point landmarks.

Performs affine transformation to align detected faces to a standard
template (ArcFace 112x112) for consistent recognition.
"""

import numpy as np
import cv2


# Standard 5-point landmarks for ArcFace 112x112 alignment
# Order: left_eye, right_eye, nose, mouth_left, mouth_right
ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # mouth left
    [70.7299, 92.2041],   # mouth right
], dtype=np.float32)


class FaceAligner:
    """
    Align face crops using 5-point landmarks to a standard template.
    Uses similarity transform (rotation, scale, translation).
    """

    def __init__(self, output_size: int = 112):
        self.output_size = output_size

        # Scale template to output size
        if output_size == 112:
            self.template = ARCFACE_TEMPLATE_112.copy()
        else:
            scale = output_size / 112.0
            self.template = ARCFACE_TEMPLATE_112.copy() * scale

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        Align face using 5-point landmarks.

        Args:
            image: original BGR image (H, W, 3)
            landmarks: (5, 2) array of [x, y] for each landmark
                order: left_eye, right_eye, nose, mouth_left, mouth_right

        Returns:
            aligned: (output_size, output_size, 3) BGR aligned face
        """
        # Note: BlazeFace gives landmarks in order:
        # right_eye, left_eye, nose, mouth_right, mouth_left
        # We need to remap to ArcFace order:
        # left_eye, right_eye, nose, mouth_left, mouth_right
        src_pts = np.array([
            landmarks[1],  # left eye (BlazeFace idx 1)
            landmarks[0],  # right eye (BlazeFace idx 0)
            landmarks[2],  # nose
            landmarks[4],  # mouth left (BlazeFace idx 4)
            landmarks[3],  # mouth right (BlazeFace idx 3)
        ], dtype=np.float32)

        dst_pts = self.template.copy()

        # Estimate similarity transform using partial affine (4 DOF)
        tform, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.LMEDS
        )

        if tform is None:
            # Fallback: use full affine
            tform, _ = cv2.estimateAffine2D(src_pts, dst_pts)

        if tform is None:
            # Last resort: simple crop and resize from bbox
            return self._fallback_crop(image, landmarks)

        # Apply affine warp
        aligned = cv2.warpAffine(
            image,
            tform,
            (self.output_size, self.output_size),
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned

    def _fallback_crop(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """Fallback: crop face region from landmarks and resize."""
        h, w = image.shape[:2]

        x_min = max(0, int(landmarks[:, 0].min() - 20))
        y_min = max(0, int(landmarks[:, 1].min() - 30))
        x_max = min(w, int(landmarks[:, 0].max() + 20))
        y_max = min(h, int(landmarks[:, 1].max() + 30))

        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            crop = image

        aligned = cv2.resize(crop, (self.output_size, self.output_size))
        return aligned
