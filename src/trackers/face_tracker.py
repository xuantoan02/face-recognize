"""
SORT (Simple Online and Realtime Tracking) face tracker.

Lightweight tracker using IoU-based association and Kalman filter
for predicting bounding box motion. Pure NumPy/SciPy implementation.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single tracked face with state."""
    track_id: int
    bbox: np.ndarray           # [x1, y1, x2, y2]
    age: int = 0               # frames since creation
    hits: int = 1              # total detections matched
    time_since_update: int = 0 # frames since last match

    # Recognition state (set by pipeline)
    spoof_checked: bool = False
    is_real: bool = False
    spoof_confidence: float = 0.0
    spoof_label: str = ""
    recognized: bool = False
    person_id: str = ""
    person_name: str = ""
    similarity: float = 0.0
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))

    # Kalman filter state
    _kf_state: np.ndarray = field(default_factory=lambda: np.array([]))
    _kf_P: np.ndarray = field(default_factory=lambda: np.array([]))


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    ar = w / (h + 1e-6)
    return np.array([cx, cy, area, ar])


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, area, aspect_ratio] back to [x1, y1, x2, y2]."""
    w = np.sqrt(max(z[2] * z[3], 0))
    h = z[2] / (w + 1e-6)
    return np.array([
        z[0] - w / 2.0,
        z[1] - h / 2.0,
        z[0] + w / 2.0,
        z[1] + h / 2.0,
    ])


def _iou_batch(bb_a: np.ndarray, bb_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bb_a: (N, 4) boxes
        bb_b: (M, 4) boxes

    Returns:
        iou_matrix: (N, M)
    """
    if len(bb_a) == 0 or len(bb_b) == 0:
        return np.zeros((len(bb_a), len(bb_b)))

    xx1 = np.maximum(bb_a[:, 0:1], bb_b[:, 0:1].T)
    yy1 = np.maximum(bb_a[:, 1:2], bb_b[:, 1:2].T)
    xx2 = np.minimum(bb_a[:, 2:3], bb_b[:, 2:3].T)
    yy2 = np.minimum(bb_a[:, 3:4], bb_b[:, 3:4].T)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = (bb_a[:, 2] - bb_a[:, 0]) * (bb_a[:, 3] - bb_a[:, 1])
    area_b = (bb_b[:, 2] - bb_b[:, 0]) * (bb_b[:, 3] - bb_b[:, 1])

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    return inter / (union + 1e-6)


class KalmanBoxTracker:
    """
    Simple Kalman filter for tracking a bounding box.

    State: [cx, cy, area, aspect_ratio, vx, vy, va]
    Measurement: [cx, cy, area, aspect_ratio]
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        # State: [cx, cy, s, r, vx, vy, vs] (7D)
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # cx += vx
        self.F[1, 5] = 1  # cy += vy
        self.F[2, 6] = 1  # s += vs

        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Measurement noise
        self.R = np.eye(self.dim_z) * 10.0
        self.R[2, 2] = 100.0  # area measurement noise higher

        # Process noise
        self.Q = np.eye(self.dim_x) * 1.0
        self.Q[4, 4] = 0.01
        self.Q[5, 5] = 0.01
        self.Q[6, 6] = 0.0001

        # Initial state
        z = _bbox_to_z(bbox)
        self.x = np.zeros(self.dim_x)
        self.x[:4] = z

        # Initial covariance
        self.P = np.eye(self.dim_x) * 100.0
        self.P[4, 4] = 1000.0
        self.P[5, 5] = 1000.0
        self.P[6, 6] = 1000.0

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        """Predict next state and return bbox."""
        # Area must stay positive
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return _z_to_bbox(self.x[:4])

    def update(self, bbox: np.ndarray):
        """Update state with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = _bbox_to_z(bbox)

        # Kalman update
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_bbox(self) -> np.ndarray:
        """Get current bounding box estimate."""
        return _z_to_bbox(self.x[:4])


class SORTTracker:
    """
    SORT multi-object tracker for faces.

    Associates detections to existing tracks using IoU + Hungarian algorithm.
    Uses Kalman filter for motion prediction.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._trackers: list[KalmanBoxTracker] = []
        self._tracks: dict[int, Track] = {}
        self.frame_count = 0

        logger.info(
            f"SORTTracker initialized: max_age={max_age}, "
            f"min_hits={min_hits}, iou_threshold={iou_threshold}"
        )

    def update(self, detections: list[np.ndarray]) -> list[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: list of [x1, y1, x2, y2] bounding boxes

        Returns:
            list of active Track objects
        """
        self.frame_count += 1

        # Get predicted bboxes from existing trackers
        predicted_bboxes = []
        to_remove = []
        for i, trk in enumerate(self._trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_remove.append(i)
            else:
                predicted_bboxes.append(pred)

        # Remove invalid trackers
        for i in reversed(to_remove):
            del self._trackers[i]

        if not detections:
            det_array = np.empty((0, 4))
        else:
            det_array = np.array(detections)

        pred_array = np.array(predicted_bboxes) if predicted_bboxes else np.empty((0, 4))

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate(
            det_array, pred_array
        )

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self._trackers[trk_idx].update(det_array[det_idx])

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(det_array[det_idx])
            self._trackers.append(trk)

        # Build output tracks
        active_tracks = []
        trackers_to_remove = []

        for i, trk in enumerate(self._trackers):
            bbox = trk.get_bbox()

            if trk.time_since_update > self.max_age:
                trackers_to_remove.append(i)
                # Clean up track state
                if trk.id in self._tracks:
                    del self._tracks[trk.id]
                continue

            # Only return tracks with enough hits
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                if trk.id not in self._tracks:
                    self._tracks[trk.id] = Track(
                        track_id=trk.id,
                        bbox=bbox,
                    )

                track = self._tracks[trk.id]
                track.bbox = bbox
                track.age = trk.age
                track.hits = trk.hits
                track.time_since_update = trk.time_since_update
                active_tracks.append(track)

        # Remove dead trackers
        for i in reversed(trackers_to_remove):
            del self._trackers[i]

        return active_tracks

    def _associate(
        self,
        detections: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associate detections to tracked objects using IoU + Hungarian.

        Returns:
            matched: list of (detection_idx, tracker_idx) pairs
            unmatched_detections: list of detection indices
            unmatched_trackers: list of tracker indices
        """
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))

        iou_matrix = _iou_batch(detections, predictions)

        # Hungarian algorithm (minimize cost = maximize IoU)
        if min(iou_matrix.shape) > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        else:
            row_indices = np.array([])
            col_indices = np.array([])

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(predictions)))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((int(r), int(c)))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    def reset(self):
        """Reset all tracks."""
        self._trackers.clear()
        self._tracks.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
