"""
Attendance tracking with debounce logic.

Ensures each person is only logged once per configurable time window,
preventing duplicate attendance records when a person is visible
across many frames.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttendanceEntry:
    """Single attendance record."""
    person_id: str
    person_name: str
    similarity: float
    is_real: bool
    spoof_label: str
    timestamp: float
    frame_idx: int


class AttendanceTracker:
    """
    Attendance tracker with debounce/cooldown logic.

    Records attendance only on first detection, then ignores the
    same person for cooldown_seconds. This prevents duplicate
    logging when someone is visible for extended periods.
    """

    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown_seconds = cooldown_seconds
        self._records: dict[str, AttendanceEntry] = {}
        self._last_seen: dict[str, float] = {}

        logger.info(f"AttendanceTracker: cooldown={cooldown_seconds}s")

    def should_record(self, person_id: str, timestamp: float) -> bool:
        """
        Check if a person should be recorded at this timestamp.

        Args:
            person_id: unique person identifier
            timestamp: current timestamp (seconds)

        Returns:
            True if this detection should be recorded
        """
        if person_id in self._last_seen:
            elapsed = timestamp - self._last_seen[person_id]
            if elapsed < self.cooldown_seconds:
                return False
        return True

    def record(
        self,
        person_id: str,
        person_name: str,
        similarity: float,
        is_real: bool,
        spoof_label: str,
        timestamp: float,
        frame_idx: int,
    ) -> bool:
        """
        Record attendance for a person (with debounce check).

        Returns:
            True if recorded, False if within cooldown
        """
        if not self.should_record(person_id, timestamp):
            return False

        entry = AttendanceEntry(
            person_id=person_id,
            person_name=person_name,
            similarity=similarity,
            is_real=is_real,
            spoof_label=spoof_label,
            timestamp=timestamp,
            frame_idx=frame_idx,
        )

        self._records[person_id] = entry
        self._last_seen[person_id] = timestamp

        logger.info(
            f"Attendance: {person_name} (ID: {person_id}) "
            f"at frame {frame_idx}, similarity={similarity:.3f}"
        )
        return True

    def get_records(self) -> list[AttendanceEntry]:
        """Get all attendance records."""
        return list(self._records.values())

    def get_count(self) -> int:
        """Get number of unique persons recorded."""
        return len(self._records)

    def reset(self):
        """Reset all attendance records."""
        self._records.clear()
        self._last_seen.clear()
