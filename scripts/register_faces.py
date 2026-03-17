"""
Register faces from image files into the face database.

Usage:
    python scripts/register_faces.py --input data/register/ --config config/default.yaml

Directory structure:
    data/register/
    ├── nguyen_van_a/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── tran_thi_b/
    │   ├── photo1.png
    │   └── photo2.png
    └── ...

Each subfolder name becomes the person's name.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inspireface_backend.isf_session import InspireFaceBackend
from src.database.vector_db import FaceDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def register_from_directory(
    input_dir: str,
    config_path: str = "config/default.yaml",
):
    """Register all faces from a directory of person folders."""

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    isf_config = config.get("inspireface", {})
    features = isf_config.get("features", {})

    # Initialize InspireFace backend
    backend = InspireFaceBackend(
        model_pack=isf_config.get("model_pack", "Pikachu"),
        detect_mode="always_detect",  # Use always_detect for registration (no tracking needed)
        detect_pixel_level=isf_config.get("detect_pixel_level", 320),
        max_detect_num=5,
        enable_recognition=features.get("recognition", True),
        enable_liveness=False,   # Not needed for registration
        enable_quality=False,    # Not needed for registration
        confidence_threshold=isf_config.get("confidence_threshold", 0.5),
    )

    db_config = config.get("database", {})
    database = FaceDatabase(
        db_path=db_config.get("path", "data/database/face_db.npz"),
        max_embeddings_per_person=db_config.get("max_embeddings_per_person", 5),
    )

    # Scan input directory
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    person_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    if not person_dirs:
        logger.error(f"No person subdirectories found in {input_dir}")
        return

    logger.info(f"Found {len(person_dirs)} persons to register")

    total_registered = 0
    total_failed = 0

    for person_name in person_dirs:
        person_dir = os.path.join(input_dir, person_name)

        # Get image files
        image_files = sorted([
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            logger.warning(f"No images found for '{person_name}'")
            continue

        # Generate person_id from folder name
        person_id = person_name.lower().replace(" ", "_")
        display_name = person_name.replace("_", " ").title()

        logger.info(f"Registering '{display_name}' from {len(image_files)} images...")

        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"  Cannot read: {img_file}")
                continue

            # Detect face using InspireFace
            detections = backend.detect(image)
            if not detections:
                logger.warning(f"  No face found in: {img_file}")
                total_failed += 1
                continue

            # Use best detection
            best = max(detections, key=lambda d: d.score)

            # Extract feature (InspireFace handles alignment internally)
            emb = backend.extract_feature(image, best)
            embeddings.append(emb)

            logger.info(f"  ✓ {img_file} (score={best.score:.3f})")

        if embeddings:
            emb_array = np.array(embeddings)
            database.register(person_id, display_name, emb_array)
            total_registered += 1
            logger.info(
                f"  → Registered '{display_name}' with {len(embeddings)} embeddings"
            )
        else:
            logger.warning(f"  → No valid face found for '{display_name}'")
            total_failed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Registration complete:")
    print(f"  ✓ Registered: {total_registered} persons")
    print(f"  ✗ Failed: {total_failed}")
    print(f"  Database size: {database.size()} persons")
    print(f"  Database path: {database.db_path}")
    print("=" * 50)

    # List all registered persons
    print("\nRegistered persons:")
    for person in database.list_all():
        print(f"  - {person['name']} (ID: {person['person_id']}, "
              f"embeddings: {person['num_embeddings']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register faces from image directory"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input directory with person subdirectories"
    )
    parser.add_argument(
        "--config", "-c", default="config/default.yaml",
        help="Config file path"
    )
    args = parser.parse_args()

    register_from_directory(args.input, args.config)
