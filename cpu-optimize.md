# Face Recognition Attendance System (CPU Optimized)

Production-grade face recognition attendance system designed to run on **weak CPUs with ONNX Runtime only (no GPU)**.

The system processes **video input** and outputs **attendance records (CSV / JSON)**.

The architecture is optimized for **low compute usage, high stability, and scalability to thousands of employees**.

---

# Key Features

* CPU-only inference (ONNX Runtime)
* High-speed pipeline using **face tracking**
* Anti-spoof protection
* Optimized for **weak CPUs**
* Supports **10k+ employee databases**
* Modular production architecture
* CLI tools for enrollment and processing
* Output attendance in CSV/JSON

---

# Optimized Pipeline

The naive pipeline would run recognition for every frame, which is extremely inefficient.

This project uses a **track-based pipeline** to minimize compute.

```
Video
 ↓
Frame Sampling
 ↓
Face Detection
 ↓
Face Tracking
 ↓
Face Alignment
 ↓
Face Quality Filter
 ↓
Anti-Spoof (once per track)
 ↓
Face Embedding (once per track)
 ↓
Vector Search (HNSW)
 ↓
Attendance Logic
 ↓
CSV / JSON
```

### Why tracking?

Without tracking:

```
30 FPS video
→ 30 detections
→ 30 recognitions
```

With tracking:

```
30 FPS video
→ 3–5 detections
→ 1 recognition per person
```

CPU load drops **80–90%**.

---

# Models Used

All models are **ONNX and CPU-friendly**.

| Task           | Model                             | Notes                         |
| -------------- | --------------------------------- | ----------------------------- |
| Face Detection | SCRFD-500M / RetinaFace-MobileNet | Fast and accurate             |
| Face Alignment | 5-point affine                    | ArcFace standard              |
| Anti-Spoof     | MiniFASNetV2                      | Detect photo / screen attacks |
| Embedding      | MobileFaceNet / ArcFace Mobile    | 128-512D embeddings           |
| Vector Search  | HNSWlib                           | Fast large-scale matching     |

---

# Project Structure

```
face-recognize/
│
├── config/
│   └── default.yaml
│
├── models/
│   ├── detector.onnx
│   ├── recognizer.onnx
│   └── antispoof.onnx
│
├── data/
│   ├── database/
│   │   └── embeddings.db
│   └── videos/
│
├── output/
│   ├── attendance.csv
│   └── attendance.json
│
├── src/
│
│   ├── pipeline/
│   │   └── pipeline.py
│
│   ├── detectors/
│   │   └── face_detector.py
│
│   ├── trackers/
│   │   └── face_tracker.py
│
│   ├── alignment/
│   │   └── aligner.py
│
│   ├── recognition/
│   │   ├── recognizer.py
│   │   └── embedding_cache.py
│
│   ├── antispoof/
│   │   └── anti_spoof.py
│
│   ├── database/
│   │   └── vector_db.py
│
│   ├── video/
│   │   └── video_processor.py
│
│   ├── attendance/
│   │   └── attendance_logic.py
│
│   └── utils/
│       └── image_utils.py
│
├── scripts/
│   ├── download_models.py
│   └── register_faces.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Major Architecture Improvements

Compared to the original design, the following improvements were added to maximize performance.

---

# 1. Face Tracking (Major Performance Gain)

Original design:

```
detect → align → spoof → embed
```

for every frame.

Improved design:

```
detect → track → recognize once
```

Tracking allows the system to reuse identity across frames.

Recommended trackers:

```
SORT
ByteTrack
KCF
```

Default recommendation:

```
SORT (CPU friendly)
```

---

# 2. Recognition Cache

Each tracked face stores recognition results.

```
track_id → embedding
track_id → identity
```

Recognition runs **only once per track**.

---

# 3. Anti-Spoof Optimization

Anti-spoof is expensive.

Instead of running every frame:

```
run once per track
```

Example logic:

```
if track not verified:
    run anti-spoof
else:
    skip
```

---

# 4. Face Quality Filter

Recognition should run only on **good face images**.

Filters:

```
min_face_size
blur threshold
face angle
```

Example rule:

```
if face_size < 80px:
    skip recognition
```

This improves both **accuracy and performance**.

---

# 5. Vector Database (Scalable Matching)

Original plan used `.npz`.

This works for small databases but scales poorly.

New system uses:

```
HNSWlib
```

Advantages:

* logarithmic search complexity
* handles 100k embeddings
* extremely fast CPU search

Search time:

```
~0.1 ms per query
```

---

# 6. Quantized ONNX Models

All models should be converted to **INT8**.

Benefits:

```
2× speed increase
smaller memory footprint
```

Quantization tool:

```
onnxruntime.quantization
```

---

# 7. Threaded Pipeline

Video processing is separated into stages.

```
Thread 1 → video decoding
Thread 2 → detection
Thread 3 → recognition
Thread 4 → database search
```

Queues:

```
frame_queue
face_queue
embedding_queue
```

This increases FPS by **2–3×**.

---

# 8. Attendance Debounce Logic

Without debounce, the same person may be logged multiple times.

Correct behavior:

```
log first detection
ignore duplicates for N minutes
```

Example rule:

```
if last_seen < 5 minutes:
    ignore
```

---

# 9. CPU Optimization Settings

ONNX Runtime should use optimized session settings.

Example:

```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
```

This improves performance by **20–30%**.

---

# Configuration Example

```
detector_model: models/detector.onnx
recognizer_model: models/recognizer.onnx
antispoof_model: models/antispoof.onnx

frame_skip: 5
min_face_size: 80

recognition_threshold: 0.45
spoof_threshold: 0.8

attendance_cooldown_minutes: 5
```

---
