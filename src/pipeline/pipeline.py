"""
Pipeline orchestrator that chains all face recognition stages via InspireFace.

Optimized flow with InspireFace built-in tracking:
    detect+track → quality filter → liveness (once per track) →
    recognize (once per track) → match DB

InspireFace handles detection, tracking, alignment, and feature extraction
internally, greatly simplifying the pipeline while improving performance.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.inspireface_backend.isf_session import InspireFaceBackend, ISFFaceDetection
from src.recognition.embedding_cache import EmbeddingCache
from src.database.vector_db import FaceDatabase, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class AttendanceRecord:
    """Single attendance recognition result."""
    person_id: str
    name: str
    similarity: float
    is_real: bool
    spoof_confidence: float
    spoof_label: str
    timestamp: float
    frame_idx: int
    track_id: int = -1
    bbox: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class PipelineMetrics:
    """Performance metrics for the pipeline."""
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    quality_ms: float = 0.0
    antispoof_ms: float = 0.0
    recognition_ms: float = 0.0
    matching_ms: float = 0.0
    total_ms: float = 0.0
    faces_detected: int = 0
    tracks_active: int = 0
    tracks_recognized: int = 0
    tracks_cached: int = 0
    faces_filtered: int = 0


@dataclass
class FrameResult:
    """All recognition results for a single frame."""
    frame_idx: int
    detections: list[ISFFaceDetection]
    records: list[AttendanceRecord]
    processing_time_ms: float
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline powered by InspireFace.

    Key advantages over the previous ONNX-based approach:
    1. InspireFace handles detection + tracking in one call
    2. Built-in face alignment (no separate aligner needed)
    3. Optimized C/C++ backend with MNN inference
    4. Liveness and quality via face_pipeline()
    5. Track-based caching for recognition results
    """

    def __init__(
        self,
        backend: InspireFaceBackend,
        database: FaceDatabase,
        recognition_threshold: float = 0.4,
        anti_spoof_enabled: bool = True,
        quality_filter_enabled: bool = True,
        max_yaw: float = 45.0,
        max_pitch: float = 35.0,
        min_face_size: int = 80,
    ):
        self.backend = backend
        self.database = database
        self.recognition_threshold = recognition_threshold
        self.anti_spoof_enabled = anti_spoof_enabled
        self.quality_filter_enabled = quality_filter_enabled
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_face_size = min_face_size

        # Embedding cache for track-based recognition
        self._embed_cache = EmbeddingCache()

        # Track spoofing state: {track_id: {"checked": bool, "is_real": bool, ...}}
        self._track_spoof_state: dict[int, dict] = {}

        # Cumulative metrics
        self._total_metrics = PipelineMetrics()

        logger.info(
            f"Pipeline initialized (InspireFace): "
            f"anti_spoof={'ON' if anti_spoof_enabled else 'OFF'}, "
            f"quality={'ON' if quality_filter_enabled else 'OFF'}, "
            f"recognition_threshold={recognition_threshold}, "
            f"db_size={database.size()}"
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> FrameResult:
        """
        Process a single frame through the InspireFace pipeline.

        Flow:
            detect+track → quality filter → liveness (once/track) →
            extract feature (once/track) → match DB

        Args:
            frame: BGR image
            frame_idx: frame number in video

        Returns:
            FrameResult with all detections and attendance records
        """
        start_time = time.time()
        metrics = PipelineMetrics()
        records = []

        # ----- Step 1: Face Detection + Tracking -----
        t0 = time.time()
        detections = self.backend.detect(frame)
        metrics.detection_ms = (time.time() - t0) * 1000
        metrics.faces_detected = len(detections)
        metrics.tracks_active = len(detections)

        if not detections:
            metrics.total_ms = (time.time() - start_time) * 1000
            return FrameResult(
                frame_idx=frame_idx,
                detections=[],
                records=[],
                processing_time_ms=metrics.total_ms,
                metrics=metrics,
            )

        # ----- Step 2: Run pipeline (liveness + quality) in batch -----
        t0 = time.time()
        pipeline_results = self.backend.run_pipeline(frame, detections)
        metrics.quality_ms = (time.time() - t0) * 1000

        # ----- Process each detected face -----
        for i, det in enumerate(detections):
            pipe_result = pipeline_results[i] if i < len(pipeline_results) else {}
            track_id = det.track_id

            # Step 3: Quality Filter (using InspireFace pose + quality)
            if self.quality_filter_enabled:
                # Check face size
                w = det.bbox[2] - det.bbox[0]
                h = det.bbox[3] - det.bbox[1]
                if min(w, h) < self.min_face_size:
                    metrics.faces_filtered += 1
                    logger.debug(
                        f"Frame {frame_idx}: Track {track_id} filtered (too_small)"
                    )
                    continue

                # Check face angle using InspireFace pose
                if abs(det.yaw) > self.max_yaw or abs(det.pitch) > self.max_pitch:
                    metrics.faces_filtered += 1
                    logger.debug(
                        f"Frame {frame_idx}: Track {track_id} filtered "
                        f"(bad_angle: yaw={det.yaw:.1f}, pitch={det.pitch:.1f})"
                    )
                    continue

                # Check quality confidence
                quality_conf = pipe_result.get("quality_confidence", 1.0)
                if quality_conf < self.backend.quality_threshold:
                    metrics.faces_filtered += 1
                    logger.debug(
                        f"Frame {frame_idx}: Track {track_id} filtered "
                        f"(low_quality: {quality_conf:.2f})"
                    )
                    continue

            # Step 4: Check embedding cache (skip if already recognized)
            if track_id >= 0 and self._embed_cache.has(track_id):
                cached = self._embed_cache.get(track_id)
                metrics.tracks_cached += 1

                spoof_state = self._track_spoof_state.get(track_id, {})
                record = AttendanceRecord(
                    person_id=cached.person_id,
                    name=cached.person_name,
                    similarity=cached.similarity,
                    is_real=spoof_state.get("is_real", True),
                    spoof_confidence=spoof_state.get("confidence", 1.0),
                    spoof_label=spoof_state.get("label", "real"),
                    timestamp=time.time(),
                    frame_idx=frame_idx,
                    track_id=track_id,
                    bbox=det.bbox,
                )
                records.append(record)
                continue

            # Step 5: Liveness / Anti-Spoofing (once per track)
            is_real = True
            spoof_confidence = 1.0
            spoof_label = "real"

            if self.anti_spoof_enabled:
                spoof_state = self._track_spoof_state.get(track_id, {})

                if not spoof_state.get("checked", False):
                    t0 = time.time()
                    liveness_conf = pipe_result.get("liveness_confidence", 1.0)
                    is_real = liveness_conf >= self.backend.liveness_threshold
                    spoof_confidence = liveness_conf
                    spoof_label = "real" if is_real else "spoof"
                    metrics.antispoof_ms += (time.time() - t0) * 1000

                    # Cache spoof state for this track
                    if track_id >= 0:
                        self._track_spoof_state[track_id] = {
                            "checked": True,
                            "is_real": is_real,
                            "confidence": spoof_confidence,
                            "label": spoof_label,
                        }

                    if not is_real:
                        record = AttendanceRecord(
                            person_id="spoofed",
                            name="Spoofed Face",
                            similarity=0.0,
                            is_real=False,
                            spoof_confidence=spoof_confidence,
                            spoof_label=spoof_label,
                            timestamp=time.time(),
                            frame_idx=frame_idx,
                            track_id=track_id,
                            bbox=det.bbox,
                        )
                        records.append(record)
                        logger.debug(
                            f"Frame {frame_idx}: Track {track_id} "
                            f"spoof detected (conf={spoof_confidence:.2f})"
                        )
                        continue
                elif not spoof_state.get("is_real", True):
                    # Already checked, still spoofed
                    continue
                else:
                    is_real = spoof_state.get("is_real", True)
                    spoof_confidence = spoof_state.get("confidence", 1.0)
                    spoof_label = spoof_state.get("label", "real")

            # Step 6: Feature Extraction (InspireFace handles alignment internally)
            t0 = time.time()
            embedding = self.backend.extract_feature(frame, det)
            metrics.recognition_ms += (time.time() - t0) * 1000
            metrics.tracks_recognized += 1

            # Step 7: Database Matching
            t0 = time.time()
            match = self.database.search(
                embedding, threshold=self.recognition_threshold
            )
            metrics.matching_ms += (time.time() - t0) * 1000

            # Cache the result for this track
            if track_id >= 0:
                self._embed_cache.set(
                    track_id=track_id,
                    embedding=embedding,
                    person_id=match.person_id,
                    person_name=match.name,
                    similarity=match.similarity,
                    matched=match.matched,
                )

            record = AttendanceRecord(
                person_id=match.person_id,
                name=match.name,
                similarity=match.similarity,
                is_real=is_real,
                spoof_confidence=spoof_confidence,
                spoof_label=spoof_label,
                timestamp=time.time(),
                frame_idx=frame_idx,
                track_id=track_id,
                bbox=det.bbox,
            )
            records.append(record)

            if match.matched:
                logger.debug(
                    f"Frame {frame_idx}: Track {track_id} "
                    f"recognized '{match.name}' (sim={match.similarity:.3f})"
                )

        metrics.total_ms = (time.time() - start_time) * 1000

        # Log performance at DEBUG level
        logger.debug(
            f"Frame {frame_idx}: "
            f"det+trk={metrics.detection_ms:.1f}ms, "
            f"pipeline={metrics.quality_ms:.1f}ms, "
            f"rec={metrics.recognition_ms:.1f}ms, "
            f"match={metrics.matching_ms:.1f}ms, "
            f"total={metrics.total_ms:.1f}ms | "
            f"faces={metrics.faces_detected}, "
            f"cached={metrics.tracks_cached}"
        )

        return FrameResult(
            frame_idx=frame_idx,
            detections=detections,
            records=records,
            processing_time_ms=metrics.total_ms,
            metrics=metrics,
        )

    def register_face(
        self,
        image: np.ndarray,
        person_id: str,
        name: str,
    ) -> bool:
        """
        Register a face from an image.

        Args:
            image: BGR image containing one face
            person_id: unique identifier
            name: display name

        Returns:
            True if registration succeeded
        """
        detections = self.backend.detect(image)
        if not detections:
            logger.warning(f"No face detected for registration of '{name}'")
            return False

        # Use the highest-confidence detection
        best = max(detections, key=lambda d: d.score)

        # Extract feature (InspireFace handles alignment internally)
        embedding = self.backend.extract_feature(image, best)

        # Store in database
        return self.database.register(person_id, name, embedding)

    def get_metrics(self) -> PipelineMetrics:
        """Get cumulative pipeline metrics."""
        return self._total_metrics

    def reset(self):
        """Reset caches and tracking state."""
        self._embed_cache.clear()
        self._track_spoof_state.clear()
