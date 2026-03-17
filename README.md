# Face Recognition Attendance System

Production-grade face recognition pipeline for enterprise attendance, optimized for **weak CPU** machines.

## Pipeline

```
Video → Frame Sampling → BlazeFace (detect) → Align → MiniFASNetV2 (anti-spoof) → MobileFaceNet (recognize) → DB Match → CSV/JSON
```

| Model | Task | Input Size | ~Size |
|---|---|---|---|
| BlazeFace | Face Detection | 128×128 | 0.5 MB |
| MobileFaceNet | Face Recognition | 112×112 | 4.5 MB |
| MiniFASNetV2 | Anti-Spoofing | 80×80 | 1.5 MB |
| MiniFASNetV2-SE | Anti-Spoofing | 80×80 | 0.6 MB |

All models run via **ONNX Runtime CPU** — no GPU required.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ONNX models
python scripts/download_models.py

# 3. Register faces
#    Place images in: data/register/<person_name>/img1.jpg
python main.py register --input data/register/

# 4. Process a video
python main.py process --video data/videos/sample.mp4
```

## Commands

| Command | Description |
|---|---|
| `python main.py process --video <path>` | Process video, generate attendance |
| `python main.py register --input <dir>` | Register faces from image folders |
| `python main.py list` | List registered persons |
| `python main.py remove --person-id <id>` | Remove a person |

## Registration Directory Structure

```
data/register/
├── nguyen_van_a/
│   ├── img1.jpg
│   └── img2.jpg
├── tran_thi_b/
│   └── photo.png
└── ...
```

## Configuration

All parameters are in `config/default.yaml`:

- **Detection**: confidence threshold, NMS, max faces
- **Anti-spoofing**: enable/disable, threshold, multi-scale crops
- **Recognition**: similarity threshold, embedding dimension
- **Video**: frame skip, max width, cooldown
- **Performance**: ONNX thread count

## Output

Attendance results are exported to `output/` as:
- `{video_name}_attendance.csv`
- `{video_name}_attendance.json`

## Project Structure

```
face-recognize/
├── config/default.yaml       # Configuration
├── models/                   # ONNX models (gitignored)
├── data/
│   ├── database/             # Face embeddings (.npz)
│   ├── register/             # Face images for registration
│   └── videos/               # Input videos
├── src/
│   ├── detector.py           # BlazeFace face detection
│   ├── aligner.py            # 5-point face alignment
│   ├── anti_spoof.py         # MiniFASNetV2 liveness check
│   ├── recognizer.py         # MobileFaceNet embeddings
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── face_database.py      # Embedding store + matching
│   ├── video_processor.py    # Video reader + attendance
│   └── utils.py              # Image processing helpers
├── scripts/
│   ├── download_models.py    # Download ONNX models
│   └── register_faces.py     # Batch face registration
├── main.py                   # CLI entry point
└── requirements.txt
```

## Performance

Optimized for weak CPUs:
- Frame skipping (configurable, default: every 3rd frame)
- Frame resizing (max 640px width)
- Lightweight models (<10 MB total)
- Anti-spoof before recognition (skip expensive embedding for fakes)
- Configurable ONNX thread count
