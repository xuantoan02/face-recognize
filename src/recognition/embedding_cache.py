"""
Embedding cache for track-based face recognition.

Caches recognition results per track_id so that embedding extraction
and database matching only run once per tracked face.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedIdentity:
    """Cached recognition result for a track."""
    embedding: np.ndarray
    person_id: str
    person_name: str
    similarity: float
    matched: bool


class EmbeddingCache:
    """
    Track-based embedding cache.

    Maps track_id → CachedIdentity so recognition only runs once
    per new tracked face.
    """

    def __init__(self, max_size: int = 200):
        self._cache: dict[int, CachedIdentity] = {}
        self._max_size = max_size

    def has(self, track_id: int) -> bool:
        """Check if track has cached recognition result."""
        return track_id in self._cache

    def get(self, track_id: int) -> CachedIdentity | None:
        """Get cached result for a track."""
        return self._cache.get(track_id)

    def set(
        self,
        track_id: int,
        embedding: np.ndarray,
        person_id: str = "unknown",
        person_name: str = "Unknown",
        similarity: float = 0.0,
        matched: bool = False,
    ):
        """Cache recognition result for a track."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._max_size and track_id not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[track_id] = CachedIdentity(
            embedding=embedding,
            person_id=person_id,
            person_name=person_name,
            similarity=similarity,
            matched=matched,
        )
        logger.debug(
            f"Cached embedding for track {track_id}: "
            f"person={person_id}, sim={similarity:.3f}"
        )

    def remove(self, track_id: int):
        """Remove a track from cache."""
        self._cache.pop(track_id, None)

    def clear(self):
        """Clear all cached embeddings."""
        self._cache.clear()

    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
