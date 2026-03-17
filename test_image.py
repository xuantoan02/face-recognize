"""
Test the InspireFace pipeline with a single image.

Usage:
    python test_image.py <image_path>
    python test_image.py <image_path> --save output.jpg
"""

import argparse
import sys
import time

import cv2
import numpy as np
import yaml

from src.inspireface_backend.isf_session import InspireFaceBackend
from src.database.vector_db import FaceDatabase


def main():
    parser = argparse.ArgumentParser(description="Test InspireFace pipeline on an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save", "-s", default=None, help="Save annotated image to path")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="Config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    isf_config = config.get("inspireface", {})
    features = isf_config.get("features", {})

    # Initialize backend
    print("🔧 Initializing InspireFace backend...")
    backend = InspireFaceBackend(
        model_pack=isf_config.get("model_pack", "Pikachu"),
        detect_mode="always_detect",
        detect_pixel_level=isf_config.get("detect_pixel_level", 320),
        max_detect_num=isf_config.get("max_detect_num", 20),
        enable_recognition=features.get("recognition", True),
        enable_liveness=features.get("liveness", True),
        enable_quality=features.get("quality", True),
        confidence_threshold=isf_config.get("confidence_threshold", 0.5),
        liveness_threshold=isf_config.get("liveness_threshold", 0.5),
        quality_threshold=isf_config.get("quality_threshold", 0.3),
    )

    # Load database (for recognition matching)
    db_config = config.get("database", {})
    database = FaceDatabase(
        db_path=db_config.get("path", "data/database/face_db.npz"),
        max_embeddings_per_person=db_config.get("max_embeddings_per_person", 5),
    )
    recognition_threshold = config.get("recognition", {}).get("similarity_threshold", 0.4)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Cannot read image: {args.image}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"📷 Image: {args.image} ({w}x{h})")

    # === Step 1: Detection ===
    total_start = time.time()
    t0 = time.time()
    detections = backend.detect(image)
    det_ms = (time.time() - t0) * 1000
    print(f"\n🔍 Detection: {len(detections)} faces found ({det_ms:.1f}ms)")

    if not detections:
        print("⚠ No faces detected.")
        sys.exit(0)

    # === Step 2: Pipeline (liveness + quality) ===
    t0 = time.time()
    pipeline_results = backend.run_pipeline(image, detections)
    pipe_ms = (time.time() - t0) * 1000
    print(f"🧪 Pipeline (liveness+quality): {pipe_ms:.1f}ms")

    # === Step 3: Anti-spoof check + Feature extraction + DB match ===
    liveness_threshold = isf_config.get("liveness_threshold", 0.5)
    quality_threshold = isf_config.get("quality_threshold", 0.3)

    draw = image.copy()
    scale = max(w, h) / 1000.0
    line_thickness = max(1, int(2 * scale))
    font_scale = max(0.4, 0.5 * scale)

    real_count = 0
    fake_count = 0
    matched_count = 0

    print(f"\n{'=' * 60}")
    for i, det in enumerate(detections):
        pipe = pipeline_results[i] if i < len(pipeline_results) else {}

        # Liveness & quality
        liveness_conf = pipe.get("liveness_confidence", -1)
        quality_conf = pipe.get("quality_confidence", -1)
        is_real = liveness_conf >= liveness_threshold
        quality_ok = quality_conf >= quality_threshold

        if is_real:
            real_count += 1
        else:
            fake_count += 1

        # Print results
        status = "✅ REAL" if is_real else "🚫 FAKE/SPOOF"
        print(f"\n👤 Face {i}: {status}")
        print(f"   📦 BBox: [{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f}]")
        print(f"   🎯 Detection confidence: {det.score:.3f}")
        print(f"   🧭 Pose: roll={det.roll:.1f}°, yaw={det.yaw:.1f}°, pitch={det.pitch:.1f}°")
        print(f"   🛡️  Liveness: {liveness_conf:.3f} {'✅' if is_real else '🚫'} (threshold={liveness_threshold})")
        print(f"   ✨ Quality: {quality_conf:.3f} {'✅' if quality_ok else '⚠️'} (threshold={quality_threshold})")

        # Draw bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)

        if not is_real:
            # FAKE — red box, skip recognition (like the pipeline does)
            color = (0, 0, 255)  # Red
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, line_thickness)
            label = f"SPOOF ({liveness_conf:.2f})"
            label_y = max(y1 - 10, 20)
            cv2.putText(draw, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_thickness)
            print(f"   ⏭️  Skipped recognition (spoofed face)")
        else:
            # REAL — extract features + match DB
            t0 = time.time()
            embedding = backend.extract_feature(image, det)
            feat_ms = (time.time() - t0) * 1000

            match = database.search(embedding, threshold=recognition_threshold)

            print(f"   🧬 Embedding: shape={embedding.shape}, extraction={feat_ms:.1f}ms")

            if match.matched:
                print(f"   ✅ Match: {match.name} (ID: {match.person_id}, sim={match.similarity:.3f})")
                color = (0, 255, 0)  # Green
                label = f"{match.name} ({match.similarity:.2f})"
                matched_count += 1
            else:
                print(f"   ❓ No match (best_sim={match.similarity:.3f})")
                color = (0, 165, 255)  # Orange
                label = f"Unknown ({match.similarity:.2f})"

            cv2.rectangle(draw, (x1, y1), (x2, y2), color, line_thickness)
            label_y = max(y1 - 10, 20)
            cv2.putText(draw, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_thickness)

        # Draw landmarks
        if det.landmarks is not None:
            for lx, ly in det.landmarks.astype(int):
                lm_color = (255, 200, 0) if is_real else (128, 128, 128)
                cv2.circle(draw, (lx, ly), max(2, int(2 * scale)), lm_color, -1)

    print(f"\n{'=' * 60}")
    print(f"📊 Summary:")
    print(f"   Total faces: {len(detections)}")
    print(f"   ✅ Real:     {real_count}")
    print(f"   🚫 Spoofed:  {fake_count}")
    print(f"   🔗 Matched:  {matched_count}")
    total_ms = (time.time() - total_start) * 1000
    print(f"   📁 Database: {database.size()} registered persons")
    print(f"   ⏱️  Total time: {total_ms:.1f}ms")

    # Save annotated image
    save_path = args.save or "output/test_result.jpg"
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, draw)
    print(f"\n💾 Annotated image saved: {save_path}")


if __name__ == "__main__":
    main()
