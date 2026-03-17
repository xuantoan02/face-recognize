"""
Face Recognition Attendance System — Main CLI Entry Point.

Commands:
    process   - Process a video file for attendance
    register  - Register faces from images
    list      - List registered persons
    remove    - Remove a person from database

Usage:
    python main.py process --video data/videos/sample.mp4
    python main.py register --input data/register/
    python main.py list
    python main.py remove --person-id nguyen_van_a
"""

import argparse
import logging
import os
import sys

import yaml

from src.inspireface_backend.isf_session import InspireFaceBackend
from src.database.vector_db import FaceDatabase
from src.pipeline.pipeline import FaceRecognitionPipeline
from src.video.video_processor import VideoProcessor


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict) -> tuple[FaceRecognitionPipeline, dict]:
    """
    Build the full face recognition pipeline from config.

    Returns:
        (pipeline, config)
    """
    isf_config = config.get("inspireface", {})
    features = isf_config.get("features", {})

    # InspireFace Backend (replaces detector, aligner, anti-spoof, recognizer, tracker)
    backend = InspireFaceBackend(
        model_pack=isf_config.get("model_pack", "Pikachu"),
        detect_mode=isf_config.get("detect_mode", "light_track"),
        detect_pixel_level=isf_config.get("detect_pixel_level", 320),
        max_detect_num=isf_config.get("max_detect_num", 20),
        enable_recognition=features.get("recognition", True),
        enable_liveness=features.get("liveness", True),
        enable_quality=features.get("quality", True),
        enable_mask_detect=features.get("mask_detect", False),
        enable_interaction=features.get("interaction", False),
        confidence_threshold=isf_config.get("confidence_threshold", 0.5),
        liveness_threshold=isf_config.get("liveness_threshold", 0.5),
        quality_threshold=isf_config.get("quality_threshold", 0.3),
    )

    # Face Database (HNSWlib) — unchanged
    db_config = config.get("database", {})
    recognition_config = config.get("recognition", {})
    database = FaceDatabase(
        db_path=db_config.get("path", "data/database/face_db.npz"),
        max_embeddings_per_person=db_config.get("max_embeddings_per_person", 5),
        embedding_dim=recognition_config.get("embedding_dim", 512),
        hnsw_ef_construction=db_config.get("hnsw_ef_construction", 200),
        hnsw_M=db_config.get("hnsw_M", 16),
    )

    # Quality filter settings
    quality_config = config.get("quality", {})

    # Pipeline
    pipeline = FaceRecognitionPipeline(
        backend=backend,
        database=database,
        recognition_threshold=recognition_config.get("similarity_threshold", 0.4),
        anti_spoof_enabled=config.get("anti_spoof", {}).get("enabled", True),
        quality_filter_enabled=quality_config.get("enabled", True),
        max_yaw=quality_config.get("max_yaw", 45.0),
        max_pitch=quality_config.get("max_pitch", 35.0),
        min_face_size=quality_config.get("min_face_size", 80),
    )

    return pipeline, config


def cmd_process(args, config: dict):
    """Process a video file for attendance."""
    logger = logging.getLogger("main.process")

    pipeline, config = build_pipeline(config)

    video_config = config.get("video", {})
    attendance_config = config.get("attendance", {})

    processor = VideoProcessor(
        pipeline=pipeline,
        frame_skip=video_config.get("frame_skip", 3),
        max_frame_width=video_config.get("max_frame_width", 640),
        cooldown_seconds=attendance_config.get("cooldown_seconds", 30.0),
        output_dir=attendance_config.get("output_dir", "output"),
        use_threading=video_config.get("use_threading", True),
        queue_size=video_config.get("queue_size", 10),
    )

    # Process video
    stats = processor.process_video(args.video)

    # Export results
    export_formats = attendance_config.get("export_formats", ["csv", "json"])

    output_basename = os.path.splitext(os.path.basename(args.video))[0]

    if "csv" in export_formats:
        csv_path = processor.export_csv(f"{output_basename}_attendance.csv")
        print(f"\n📄 CSV exported: {csv_path}")

    if "json" in export_formats:
        json_path = processor.export_json(f"{output_basename}_attendance.json")
        print(f"📄 JSON exported: {json_path}")

    # Print attendance summary
    attendance = processor.get_attendance()
    if attendance:
        print(f"\n{'=' * 60}")
        print(f"📋 Attendance Summary ({len(attendance)} persons)")
        print(f"{'=' * 60}")
        for record in attendance:
            print(
                f"  ✓ {record['name']} "
                f"(similarity: {record['similarity']:.3f}, "
                f"time: {record['timestamp']})"
            )
    else:
        print("\n⚠ No persons recognized in the video.")
        print("  Make sure to register faces first: python main.py register --input <dir>")

    print(f"\n⏱ Performance: {stats['effective_fps']:.1f} FPS "
          f"({stats['avg_frame_time_ms']:.1f} ms/frame)")


def cmd_register(args, config: dict):
    """Register faces from images."""
    from scripts.register_faces import register_from_directory
    register_from_directory(args.input, args.config)


def cmd_list(args, config: dict):
    """List registered persons."""
    db_config = config.get("database", {})
    database = FaceDatabase(
        db_path=db_config.get("path", "data/database/face_db.npz"),
    )

    persons = database.list_all()
    if not persons:
        print("No persons registered.")
        print("Use: python main.py register --input <dir>")
        return

    print(f"\n{'=' * 50}")
    print(f"Registered Persons ({len(persons)})")
    print(f"{'=' * 50}")
    for p in persons:
        print(f"  • {p['name']} (ID: {p['person_id']}, "
              f"embeddings: {p['num_embeddings']})")
    print(f"\nDatabase: {database.db_path}")


def cmd_remove(args, config: dict):
    """Remove a person from database."""
    db_config = config.get("database", {})
    database = FaceDatabase(
        db_path=db_config.get("path", "data/database/face_db.npz"),
    )

    if database.remove(args.person_id):
        print(f"✓ Removed person: {args.person_id}")
    else:
        print(f"✗ Person not found: {args.person_id}")
        persons = database.list_all()
        if persons:
            print("\nAvailable persons:")
            for p in persons:
                print(f"  • {p['person_id']} ({p['name']})")


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Attendance System (InspireFace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process --video data/videos/meeting.mp4
  python main.py register --input data/register/
  python main.py list
  python main.py remove --person-id john_doe
        """,
    )
    parser.add_argument(
        "--config", "-c", default="config/default.yaml",
        help="Configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    proc_parser = subparsers.add_parser("process", help="Process a video file")
    proc_parser.add_argument(
        "--video", "-v", required=True,
        help="Path to input video file",
    )

    # Register command
    reg_parser = subparsers.add_parser("register", help="Register faces from images")
    reg_parser.add_argument(
        "--input", "-i", required=True,
        help="Directory with person subdirectories",
    )

    # List command
    subparsers.add_parser("list", help="List registered persons")

    # Remove command
    rem_parser = subparsers.add_parser("remove", help="Remove a person")
    rem_parser.add_argument(
        "--person-id", required=True,
        help="Person ID to remove",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.log_level)
    config = load_config(args.config)

    commands = {
        "process": cmd_process,
        "register": cmd_register,
        "list": cmd_list,
        "remove": cmd_remove,
    }

    commands[args.command](args, config)


if __name__ == "__main__":
    main()
