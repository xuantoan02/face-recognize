"""
Video processor for attendance tracking.

Reads video files, samples frames, runs the recognition pipeline,
deduplicates attendance records, and exports results.

Optimized with threaded video decoding for improved FPS.
Saves annotated output video with face recognition results.
"""

import csv
import json
import logging
import os
import time
import threading
import queue
from datetime import datetime

import cv2
import numpy as np

from src.pipeline.pipeline import FaceRecognitionPipeline, AttendanceRecord, FrameResult
from src.attendance.attendance_logic import AttendanceTracker

logger = logging.getLogger(__name__)


class VideoDecodeThread(threading.Thread):
    """
    Background thread for video frame decoding.

    Decodes frames ahead of the processing pipeline to maximize
    CPU utilization — the decode thread runs while the main thread
    processes the previous frame.

    Sends ALL frames to the queue for smooth output video.
    Frames that should be processed by the pipeline are marked with
    should_process=True.
    """

    def __init__(
        self,
        video_path: str,
        frame_queue: queue.Queue,
        frame_skip: int = 3,
        max_frame_width: int = 640,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.frame_skip = frame_skip
        self.max_frame_width = max_frame_width
        self.stopped = False
        self.total_frames = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.frame_queue.put(None)
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed
            h, w = frame.shape[:2]
            if w > self.max_frame_width:
                scale = self.max_frame_width / w
                new_w = self.max_frame_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Store frame dimensions (after resize) for video writer
            if frame_idx == 0:
                fh, fw = frame.shape[:2]
                self.frame_width = fw
                self.frame_height = fh

            should_process = (frame_idx % self.frame_skip == 0)

            try:
                self.frame_queue.put(
                    (frame_idx, frame, should_process), timeout=5.0
                )
            except queue.Full:
                if self.stopped:
                    break

            frame_idx += 1

        cap.release()
        self.frame_queue.put(None)  # Sentinel

    def stop(self):
        self.stopped = True


class VideoProcessor:
    """
    Process video files for face recognition attendance.

    Features:
    - Threaded video decoding for improved FPS
    - Configurable frame skipping for performance
    - Frame resizing for weak CPUs
    - Attendance tracking with debounce (via AttendanceTracker)
    - Saves annotated output video with recognition results
    - CSV and JSON export
    """

    def __init__(
        self,
        pipeline: FaceRecognitionPipeline,
        frame_skip: int = 3,
        max_frame_width: int = 640,
        cooldown_seconds: float = 30.0,
        output_dir: str = "output",
        use_threading: bool = True,
        queue_size: int = 10,
    ):
        self.pipeline = pipeline
        self.frame_skip = frame_skip
        self.max_frame_width = max_frame_width
        self.output_dir = output_dir
        self.use_threading = use_threading
        self.queue_size = queue_size

        # Attendance tracker with debounce logic
        self.attendance_tracker = AttendanceTracker(
            cooldown_seconds=cooldown_seconds,
        )

        # Last frame result for drawing on skipped frames
        self._last_result: FrameResult | None = None

    def _update_attendance(self, records: list[AttendanceRecord]):
        """Update attendance using the debounce tracker."""
        for record in records:
            if not record.is_real:
                continue
            if record.person_id == "unknown":
                continue

            self.attendance_tracker.record(
                person_id=record.person_id,
                person_name=record.name,
                similarity=record.similarity,
                is_real=record.is_real,
                spoof_label=record.spoof_label,
                timestamp=record.timestamp,
                frame_idx=record.frame_idx,
            )

    @staticmethod
    def draw_results_on_frame(
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """
        Draw face recognition results on a frame.

        Draws bounding boxes, names, similarity scores, spoof labels,
        and face landmarks.

        Args:
            frame: BGR image (will be copied)
            result: FrameResult from pipeline

        Returns:
            Annotated copy of the frame
        """
        draw = frame.copy()
        h, w = draw.shape[:2]
        scale = max(w, h) / 1000.0
        line_thickness = max(1, int(2 * scale))
        font_scale = max(0.4, 0.5 * scale)

        # Build a lookup from track_id → record for quick matching
        record_by_track: dict[int, AttendanceRecord] = {}
        for rec in result.records:
            if rec.track_id >= 0:
                record_by_track[rec.track_id] = rec

        for i, det in enumerate(result.detections):
            x1, y1, x2, y2 = det.bbox.astype(int)
            track_id = det.track_id

            # Find the matching record
            record = record_by_track.get(track_id)

            if record is None:
                # Detection without a record (filtered by quality, etc.)
                color = (128, 128, 128)  # Gray
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 1)
                continue

            if not record.is_real:
                # Spoofed face — red
                color = (0, 0, 255)
                label = f"SPOOF ({record.spoof_confidence:.2f})"
            elif record.person_id == "unknown":
                # Unknown face — orange
                color = (0, 165, 255)
                label = f"Unknown ({record.similarity:.2f})"
            else:
                # Recognized face — green
                color = (0, 255, 0)
                label = f"{record.name} ({record.similarity:.2f})"

            # Draw bounding box
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, line_thickness)

            # Draw label background + text
            label_with_id = f"[{track_id}] {label}" if track_id >= 0 else label
            (tw, th), _ = cv2.getTextSize(
                label_with_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            label_y = max(y1 - 10, th + 5)
            # Background rectangle for readability
            cv2.rectangle(
                draw,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y + 4),
                color, -1,
            )
            cv2.putText(
                draw, label_with_id, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), line_thickness,
            )

            # Draw landmarks
            if det.landmarks is not None and len(det.landmarks) > 0:
                lm_color = (255, 200, 0) if record.is_real else (128, 128, 128)
                for lx, ly in det.landmarks.astype(int):
                    cv2.circle(draw, (lx, ly), max(2, int(2 * scale)), lm_color, -1)

        return draw

    def _get_output_video_path(self, video_path: str) -> str:
        """Get the output video path based on input video name."""
        os.makedirs(self.output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.output_dir, f"{basename}_annotated.mp4")

    def process_video(self, video_path: str) -> dict:
        """
        Process a video file and generate attendance report.

        Uses threaded decoding when enabled for better performance.
        Saves annotated output video.

        Args:
            video_path: path to video file

        Returns:
            dict with processing stats (includes 'output_video' path)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Reset state
        self.attendance_tracker.reset()
        self.pipeline.reset()
        self._last_result = None

        if self.use_threading:
            return self._process_threaded(video_path)
        else:
            return self._process_sequential(video_path)

    def _process_threaded(self, video_path: str) -> dict:
        """Process video with threaded decoding."""
        frame_queue = queue.Queue(maxsize=self.queue_size)

        # Start decode thread
        decoder = VideoDecodeThread(
            video_path=video_path,
            frame_queue=frame_queue,
            frame_skip=self.frame_skip,
            max_frame_width=self.max_frame_width,
        )
        decoder.start()

        # Wait briefly for metadata
        time.sleep(0.1)

        # Setup video writer
        output_path = self._get_output_video_path(video_path)
        writer = None

        logger.info(
            f"Processing video (threaded): {video_path}\n"
            f"  Frame skip: {self.frame_skip}, "
            f"Max width: {self.max_frame_width}, "
            f"Queue size: {self.queue_size}"
        )

        processed_count = 0
        total_faces = 0
        total_time_ms = 0.0
        start_time = time.time()

        while True:
            item = frame_queue.get()
            if item is None:
                break

            frame_idx, frame, should_process = item

            # Init video writer on first frame
            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = decoder.fps if decoder.fps > 0 else 25.0
                writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
                logger.info(
                    f"  Output video: {output_path} ({fw}x{fh} @ {fps:.1f}fps)"
                )

            if should_process:
                # Run pipeline
                result = self.pipeline.process_frame(frame, frame_idx)
                self._last_result = result

                # Update attendance
                self._update_attendance(result.records)

                total_faces += len(result.detections)
                total_time_ms += result.processing_time_ms
                processed_count += 1

                # Draw results and write frame
                annotated = self.draw_results_on_frame(frame, result)
                writer.write(annotated)

                # Progress logging
                if processed_count % 50 == 0:
                    avg_ms = total_time_ms / processed_count
                    logger.info(
                        f"Progress: frame {frame_idx}/{decoder.total_frames} "
                        f"({100 * frame_idx / max(decoder.total_frames, 1):.1f}%), "
                        f"avg {avg_ms:.1f}ms/frame, "
                        f"faces: {total_faces}"
                    )
            else:
                # Skipped frame — draw last known results for smooth video
                if self._last_result is not None:
                    annotated = self.draw_results_on_frame(frame, self._last_result)
                    writer.write(annotated)
                else:
                    writer.write(frame)

        decoder.stop()
        decoder.join(timeout=2.0)

        if writer is not None:
            writer.release()

        total_elapsed = time.time() - start_time

        stats = {
            "video_path": video_path,
            "output_video": output_path,
            "total_frames": decoder.total_frames,
            "processed_frames": processed_count,
            "frame_skip": self.frame_skip,
            "total_faces_detected": total_faces,
            "unique_persons": self.attendance_tracker.get_count(),
            "total_time_seconds": round(total_elapsed, 2),
            "avg_frame_time_ms": round(total_time_ms / max(processed_count, 1), 2),
            "effective_fps": round(processed_count / max(total_elapsed, 0.001), 2),
            "threading": True,
        }

        self._log_summary(stats, total_elapsed, processed_count, total_faces)
        return stats

    def _process_sequential(self, video_path: str) -> dict:
        """Process video sequentially (non-threaded fallback)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Processing video (sequential): {video_path}\n"
            f"  Frames: {total_frames}, FPS: {fps:.1f}, "
            f"Resolution: {width}x{height}\n"
            f"  Frame skip: {self.frame_skip}, "
            f"Max width: {self.max_frame_width}"
        )

        # Setup video writer
        output_path = self._get_output_video_path(video_path)
        writer = None

        frame_idx = 0
        processed_count = 0
        total_faces = 0
        total_time_ms = 0.0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for performance
            h, w = frame.shape[:2]
            if w > self.max_frame_width:
                scale = self.max_frame_width / w
                new_w = self.max_frame_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Init video writer on first frame
            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = fps if fps > 0 else 25.0
                writer = cv2.VideoWriter(output_path, fourcc, out_fps, (fw, fh))
                logger.info(
                    f"  Output video: {output_path} ({fw}x{fh} @ {out_fps:.1f}fps)"
                )

            if frame_idx % self.frame_skip == 0:
                # Run pipeline
                result = self.pipeline.process_frame(frame, frame_idx)
                self._last_result = result

                # Update attendance
                self._update_attendance(result.records)

                total_faces += len(result.detections)
                total_time_ms += result.processing_time_ms
                processed_count += 1

                # Draw results and write frame
                annotated = self.draw_results_on_frame(frame, result)
                writer.write(annotated)

                # Progress logging
                if processed_count % 50 == 0:
                    avg_ms = total_time_ms / processed_count
                    logger.info(
                        f"Progress: frame {frame_idx}/{total_frames} "
                        f"({100 * frame_idx / max(total_frames, 1):.1f}%), "
                        f"avg {avg_ms:.1f}ms/frame, "
                        f"faces: {total_faces}"
                    )
            else:
                # Skipped frame — draw last known results for smooth video
                if self._last_result is not None:
                    annotated = self.draw_results_on_frame(frame, self._last_result)
                    writer.write(annotated)
                else:
                    writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        total_elapsed = time.time() - start_time

        stats = {
            "video_path": video_path,
            "output_video": output_path,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "frame_skip": self.frame_skip,
            "total_faces_detected": total_faces,
            "unique_persons": self.attendance_tracker.get_count(),
            "total_time_seconds": round(total_elapsed, 2),
            "avg_frame_time_ms": round(total_time_ms / max(processed_count, 1), 2),
            "effective_fps": round(processed_count / max(total_elapsed, 0.001), 2),
            "threading": False,
        }

        self._log_summary(stats, total_elapsed, processed_count, total_faces)
        return stats

    def _log_summary(
        self,
        stats: dict,
        total_elapsed: float,
        processed_count: int,
        total_faces: int,
    ):
        """Log processing summary."""
        logger.info(
            f"\nProcessing complete:\n"
            f"  Processed {processed_count}/{stats['total_frames']} frames "
            f"in {total_elapsed:.1f}s\n"
            f"  Avg {stats['avg_frame_time_ms']:.1f}ms/frame "
            f"({stats['effective_fps']:.1f} FPS)\n"
            f"  Total faces detected: {total_faces}\n"
            f"  Unique persons: {self.attendance_tracker.get_count()}\n"
            f"  Output video: {stats.get('output_video', 'N/A')}"
        )

    def get_attendance(self) -> list[dict]:
        """Get current attendance records as list of dicts."""
        records = []
        for entry in self.attendance_tracker.get_records():
            records.append({
                "person_id": entry.person_id,
                "name": entry.person_name,
                "similarity": round(entry.similarity, 4),
                "is_real": entry.is_real,
                "spoof_label": entry.spoof_label,
                "timestamp": datetime.fromtimestamp(
                    entry.timestamp
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "frame_idx": entry.frame_idx,
            })
        return records

    def export_csv(self, filename: str | None = None) -> str:
        """Export attendance to CSV file."""
        os.makedirs(self.output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attendance_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        records = self.get_attendance()

        if not records:
            logger.warning("No attendance records to export")
            return filepath

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Exported {len(records)} records to {filepath}")
        return filepath

    def export_json(self, filename: str | None = None) -> str:
        """Export attendance to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attendance_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)
        records = self.get_attendance()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"attendance": records, "total": len(records)},
                f, indent=2, ensure_ascii=False,
            )

        logger.info(f"Exported {len(records)} records to {filepath}")
        return filepath
