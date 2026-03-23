"""
Test the InspireFace pipeline with a single image.

Usage:
    # Default (InspireFace backend for recognition)
    python test_image.py <image_path>

    # Use FaceRecognizer (MobileFaceNet preprocessing)
    python test_image.py <image_path> --recognizer face --rec-model models/recognition.onnx

    # Use ONNXRecognizer (testmodel.py preprocessing)
    python test_image.py <image_path> --recognizer onnx --rec-model models/recognition.onnx
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

from src.inspireface_backend.isf_session import InspireFaceBackend
from src.database.vector_db import FaceDatabase
from src.recognition import FaceRecognizer, ONNXRecognizer


def _get_providers(device: str) -> list[str]:
    if device == "gpu":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_recognizer(recognizer_type, model_path, num_threads=2, providers=None):
    """
    Build a recognizer instance based on the chosen type.

    Args:
        recognizer_type: 'inspireface', 'face', or 'onnx'
        model_path: Path to .onnx model file (required for 'face' / 'onnx')
        num_threads: ONNX Runtime intra-op threads
        providers: ONNX Runtime execution providers

    Returns:
        FaceRecognizer | ONNXRecognizer | None
    """
    if recognizer_type == "inspireface":
        return None  # use backend.extract_feature()

    if not model_path:
        print("❌ --rec-model is required when --recognizer is 'face' or 'onnx'")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"❌ Recognition model not found: {model_path}")
        sys.exit(1)

    if recognizer_type == "face":
        return FaceRecognizer(model_path=model_path, num_threads=num_threads, providers=providers)
    elif recognizer_type == "onnx":
        return ONNXRecognizer(model_path=model_path, num_threads=num_threads, providers=providers)
    else:
        print(f"❌ Unknown recognizer type: {recognizer_type}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test InspireFace pipeline on an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save", "-s", default=None, help="Save annotated image to path")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="Config file")
    parser.add_argument(
        "--recognizer", "-r",
        choices=["inspireface", "face", "onnx"],
        default="inspireface",
        help="Recognizer to use: inspireface (default), face (FaceRecognizer), onnx (ONNXRecognizer)",
    )
    parser.add_argument("--rec-model", default=None, help="Path to .onnx recognition model (for face/onnx)")
    parser.add_argument("--device", "-d", choices=["cpu", "gpu"], default="gpu",
                        help="Thiết bị inference: cpu hoặc gpu (CUDA)")
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

    # Build recognizer
    providers = _get_providers(args.device)
    recognizer = build_recognizer(args.recognizer, args.rec_model, providers=providers)
    rec_label = args.recognizer.upper()
    print(f"🧠 Recognizer: {rec_label}" + (f" ({args.rec_model})" if args.rec_model else ""))

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

            if recognizer is not None:
                # Use FaceRecognizer or ONNXRecognizer
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size == 0:
                    print(f"   ⚠️  Empty face crop, skipping")
                    continue
                embedding = recognizer.get_embedding(face_crop)
            else:
                # Use InspireFace backend
                embedding = backend.extract_feature(image, det)

            feat_ms = (time.time() - t0) * 1000

            match = database.search(embedding, threshold=recognition_threshold)

            print(f"   🧬 Embedding: shape={embedding.shape}, extraction={feat_ms:.1f}ms (via {rec_label})")

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
    print(f"   Recognizer:  {rec_label}")
    print(f"   Total faces: {len(detections)}")
    print(f"   ✅ Real:     {real_count}")
    print(f"   🚫 Spoofed:  {fake_count}")
    print(f"   🔗 Matched:  {matched_count}")
    total_ms = (time.time() - total_start) * 1000
    print(f"   📁 Database: {database.size()} registered persons")
    print(f"   ⏱️  Total time: {total_ms:.1f}ms")

    # Save annotated image
    save_path = args.save or "output/test_result.jpg"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, draw)
    print(f"\n💾 Annotated image saved: {save_path}")


if __name__ == "__main__":
    main()
