"""
Face embedding database with HNSWlib vector search.

Uses HNSWlib for fast approximate nearest neighbor search (O(log n)),
replacing brute-force cosine similarity (O(n)).
Falls back to brute-force if hnswlib is not installed.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import hnswlib, fall back to brute-force
try:
    import hnswlib
    HAS_HNSWLIB = True
    logger.info("Using HNSWlib for vector search")
except ImportError:
    HAS_HNSWLIB = False
    logger.warning("hnswlib not installed, using brute-force search (pip install hnswlib)")


@dataclass
class MatchResult:
    """Database match result."""
    person_id: str
    name: str
    similarity: float
    matched: bool


class FaceDatabase:
    """
    Face embedding database with scalable vector search.

    Uses HNSWlib index for fast O(log n) search when available,
    with fallback to brute-force cosine similarity.

    Stores multiple embeddings per person for robust matching.
    """

    def __init__(
        self,
        db_path: str = "data/database/face_db.npz",
        max_embeddings_per_person: int = 5,
        embedding_dim: int = 512,
        hnsw_ef_construction: int = 200,
        hnsw_M: int = 16,
    ):
        self.db_path = db_path
        self.max_embeddings_per_person = max_embeddings_per_person
        self.embedding_dim = embedding_dim
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_M = hnsw_M

        # Internal storage: {person_id: {"name": str, "embeddings": np.ndarray}}
        self._db: dict[str, dict] = {}

        # Metadata file alongside .npz
        self._meta_path = db_path.replace(".npz", "_meta.json")

        # HNSW index
        self._index = None
        self._index_labels: list[tuple[str, int]] = []  # [(person_id, emb_idx), ...]

        # Load existing database
        self._load()

    def _load(self):
        """Load database from disk."""
        if not os.path.exists(self.db_path):
            logger.info(f"No existing DB at {self.db_path}, starting fresh.")
            return

        try:
            data = np.load(self.db_path, allow_pickle=True)
            meta = {}
            if os.path.exists(self._meta_path):
                with open(self._meta_path, "r") as f:
                    meta = json.load(f)

            for key in data.files:
                person_id = key
                embeddings = data[key]
                name = meta.get(person_id, {}).get("name", person_id)
                self._db[person_id] = {
                    "name": name,
                    "embeddings": embeddings,
                }

            logger.info(f"Loaded {len(self._db)} persons from {self.db_path}")

            # Rebuild HNSW index
            self._rebuild_index()

        except Exception as e:
            logger.error(f"Failed to load DB: {e}")

    def _save(self):
        """Save database to disk."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        # Save embeddings
        save_dict = {}
        for person_id, info in self._db.items():
            save_dict[person_id] = info["embeddings"]
        np.savez(self.db_path, **save_dict)

        # Save metadata
        meta = {}
        for person_id, info in self._db.items():
            meta[person_id] = {"name": info["name"]}
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved {len(self._db)} persons to {self.db_path}")

    def _rebuild_index(self):
        """Rebuild HNSW index from all stored embeddings."""
        if not HAS_HNSWLIB or not self._db:
            self._index = None
            self._index_labels = []
            return

        # Collect all embeddings
        all_embeddings = []
        labels = []
        for person_id, info in self._db.items():
            for i, emb in enumerate(info["embeddings"]):
                all_embeddings.append(emb)
                labels.append((person_id, i))

        if not all_embeddings:
            self._index = None
            self._index_labels = []
            return

        data = np.array(all_embeddings, dtype=np.float32)
        dim = data.shape[1]
        num_elements = len(data)

        # Create HNSW index with inner product (cosine sim for L2-normalized vectors)
        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(
            max_elements=max(num_elements * 2, 100),
            ef_construction=self.hnsw_ef_construction,
            M=self.hnsw_M,
        )
        index.set_ef(50)  # search-time ef

        index.add_items(data, list(range(num_elements)))

        self._index = index
        self._index_labels = labels

        logger.debug(
            f"HNSW index rebuilt: {num_elements} embeddings, dim={dim}"
        )

    def register(
        self,
        person_id: str,
        name: str,
        embeddings: np.ndarray,
    ) -> bool:
        """
        Register a person with their face embeddings.

        Args:
            person_id: unique identifier
            name: display name
            embeddings: (N, dim) or (dim,) array of face embeddings

        Returns:
            True if registered successfully
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if person_id in self._db:
            # Append new embeddings (up to max)
            existing = self._db[person_id]["embeddings"]
            combined = np.vstack([existing, embeddings])
            if len(combined) > self.max_embeddings_per_person:
                combined = combined[-self.max_embeddings_per_person:]
            self._db[person_id]["embeddings"] = combined
            self._db[person_id]["name"] = name
            logger.info(f"Updated person '{name}' ({person_id}): {len(combined)} embeddings")
        else:
            if len(embeddings) > self.max_embeddings_per_person:
                embeddings = embeddings[:self.max_embeddings_per_person]
            self._db[person_id] = {
                "name": name,
                "embeddings": embeddings,
            }
            logger.info(f"Registered person '{name}' ({person_id}): {len(embeddings)} embeddings")

        self._save()
        self._rebuild_index()
        return True

    def search(
        self,
        embedding: np.ndarray,
        threshold: float = 0.4,
        top_k: int = 1,
    ) -> MatchResult:
        """
        Search for the closest matching person.

        Uses HNSW index for fast search if available,
        falls back to brute-force cosine similarity.

        Args:
            embedding: (dim,) query embedding (L2-normalized)
            threshold: minimum similarity for a match
            top_k: number of neighbors to search

        Returns:
            MatchResult with best match info
        """
        if not self._db:
            return MatchResult(
                person_id="unknown", name="Unknown",
                similarity=0.0, matched=False,
            )

        # HNSW search
        if self._index is not None and HAS_HNSWLIB:
            return self._search_hnsw(embedding, threshold, top_k)

        # Fallback: brute-force
        return self._search_brute_force(embedding, threshold)

    def _search_hnsw(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
    ) -> MatchResult:
        """Search using HNSW index."""
        query = embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k * 5, len(self._index_labels))  # search more to find best person

        labels, distances = self._index.knn_query(query, k=k)

        best_sim = -1.0
        best_id = "unknown"
        best_name = "Unknown"

        for idx, dist in zip(labels[0], distances[0]):
            if idx < len(self._index_labels):
                person_id, _ = self._index_labels[idx]
                # hnswlib 'ip' returns d = 1 - inner_product
                # So similarity = 1 - distance for L2-normalized vectors
                sim = float(1.0 - dist)
                if sim > best_sim:
                    best_sim = sim
                    best_id = person_id
                    best_name = self._db[person_id]["name"]

        matched = best_sim >= threshold

        return MatchResult(
            person_id=best_id if matched else "unknown",
            name=best_name if matched else "Unknown",
            similarity=best_sim,
            matched=matched,
        )

    def _search_brute_force(
        self,
        embedding: np.ndarray,
        threshold: float,
    ) -> MatchResult:
        """Brute-force search using cosine similarity."""
        best_sim = -1.0
        best_id = "unknown"
        best_name = "Unknown"

        for person_id, info in self._db.items():
            for ref_emb in info["embeddings"]:
                sim = float(np.dot(embedding.flatten(), ref_emb.flatten()))
                if sim > best_sim:
                    best_sim = sim
                    best_id = person_id
                    best_name = info["name"]

        matched = best_sim >= threshold

        return MatchResult(
            person_id=best_id if matched else "unknown",
            name=best_name if matched else "Unknown",
            similarity=best_sim,
            matched=matched,
        )

    def remove(self, person_id: str) -> bool:
        """Remove a person from the database."""
        if person_id in self._db:
            del self._db[person_id]
            self._save()
            self._rebuild_index()
            logger.info(f"Removed person {person_id}")
            return True
        return False

    def list_all(self) -> list[dict]:
        """List all registered persons."""
        result = []
        for person_id, info in self._db.items():
            result.append({
                "person_id": person_id,
                "name": info["name"],
                "num_embeddings": len(info["embeddings"]),
            })
        return result

    def size(self) -> int:
        """Return number of registered persons."""
        return len(self._db)
