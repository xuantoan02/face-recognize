# Face Recognition Attendance System

## Technical Review & Optimization Report

## 1. System Overview

Current pipeline:

Video → Frame Sampling → BlazeFace → Align → MiniFASNetV2 →
MobileFaceNet → Match DB → Attendance

Goal of the system:

-   Run on **weak CPU machines**
-   Use **ONNX Runtime only**
-   Process **video input**
-   Export **attendance CSV/JSON**

The architecture is modular and good for MVP, but several issues prevent
it from reaching **production-grade performance**.

------------------------------------------------------------------------

# 2. Missing Face Tracking

## Problem

The system processes each frame independently:

detect → align → spoof → embed → match

Example:

30 FPS video\
→ 30 detections\
→ 30 recognitions

Even when the same person appears across many frames.

## Impact

-   Very high CPU usage
-   Unnecessary repeated inference
-   Low FPS

## Optimization

Add a tracking layer:

Video\
→ Detect\
→ Track\
→ Recognize once per track

Recommended trackers:

-   SORT
-   ByteTrack
-   KCF

Expected improvement:

CPU usage ↓ 70--90%\
FPS ↑ 2--4×

------------------------------------------------------------------------

# 3. Anti-Spoof Runs Too Frequently

## Problem

Anti-spoof currently runs **every frame**.

Example:

1 person appears in 100 frames\
→ 100 anti-spoof inferences

## Optimization

Run anti-spoof **once per track**.

Example logic:

    if track_id not verified:
        run anti_spoof
    else:
        skip

Expected improvement:

Anti-spoof compute ↓ \~90%

------------------------------------------------------------------------

# 4. Recognition Runs Too Often

## Problem

MobileFaceNet embeddings are generated for each frame.

But embeddings for the same face are nearly identical across frames.

## Optimization

Cache embeddings by track.

Example:

track_id → embedding\
track_id → identity

Recognition runs only once per tracked face.

------------------------------------------------------------------------

# 5. Detector Choice Not Optimal

Current detector:

BlazeFace

Pros: - Very fast

Cons: - Medium accuracy - Poor with small faces - Less robust with pose
variation

## Recommended Detectors

-   SCRFD‑500M
-   RetinaFace MobileNet

These models offer better detection accuracy with similar CPU cost.

------------------------------------------------------------------------

# 6. Face Database Not Scalable

Current storage:

    npz file
    cosine similarity brute force

Problem:

For 10k employees:

Matching complexity = **O(n)**

Latency grows with database size.

## Optimization

Use vector search index.

Recommended:

-   HNSWlib
-   Faiss
-   Annoy

Best option for CPU:

**HNSWlib**

Search complexity:

O(log n)

------------------------------------------------------------------------

# 7. Face Enrollment Not Robust

Current approach:

Average embedding per person.

Problem:

Face embeddings vary due to:

-   lighting
-   pose
-   expression

## Optimization

Store **multiple embeddings per person**.

Example:

person_id - embedding_1 - embedding_2 - embedding_3

Matching uses **maximum similarity**.

This improves recognition accuracy significantly.

------------------------------------------------------------------------

# 8. Missing Face Quality Filter

Recognition should only run on good images.

Currently missing checks:

-   blur
-   face size
-   face angle

## Optimization

Add filters such as:

    min_face_size
    blur threshold
    yaw/pitch threshold

Example rule:

    if face_size < 80px:
        skip recognition

Benefits:

-   faster pipeline
-   higher recognition accuracy

------------------------------------------------------------------------

# 9. Synchronous Pipeline

Current pipeline runs in a single thread.

Example:

video read → detect → align → spoof → embed → match

## Problem

CPU utilization is not optimal.

## Optimization

Use threaded pipeline:

Thread 1 → video decode\
Thread 2 → detection\
Thread 3 → recognition\
Thread 4 → database search

Expected improvement:

FPS increase **2--3×**

------------------------------------------------------------------------

# 10. ONNX Runtime Not Optimized

Default ONNX runtime settings are not optimal.

## Optimization

Enable graph optimization and thread settings.

Example:

``` python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
```

Performance gain:

\~20--30% faster inference.

------------------------------------------------------------------------

# 11. Missing Model Quantization

ONNX models can be quantized to **INT8**.

Benefits:

-   \~2× faster inference
-   reduced memory usage

Tool:

    onnxruntime.quantization

This is important for weak CPU systems.

------------------------------------------------------------------------

# 12. Attendance Duplicate Logging

Current system logs attendance when a face is detected.

Problem:

A person appearing for several minutes may be logged multiple times.

## Optimization

Add debounce logic.

Example:

    if last_seen < 5 minutes:
        ignore

Ensures one attendance record per time window.

------------------------------------------------------------------------

# 13. Missing Logging & Metrics

Production systems require monitoring.

Important metrics:

-   FPS
-   detection latency
-   recognition latency
-   spoof detection rate

Logging required:

-   pipeline logs
-   error logs
-   performance logs

This is critical for debugging production deployments.

------------------------------------------------------------------------

# 14. Recommended Top Optimizations

Top improvements with highest impact:

1.  Add **face tracking**
2.  Run **anti-spoof once per track**
3.  Cache **embeddings per track**
4.  Use **vector database (HNSWlib)**
5.  Replace **BlazeFace with SCRFD**
6.  Add **face quality filters**
7.  Use **multi-thread pipeline**
8.  Enable **ONNX runtime optimizations**
9.  Apply **INT8 quantization**
10. Add **attendance debounce logic**

------------------------------------------------------------------------

# 15. Expected Performance After Optimization

Example hardware:

Intel i5 (4 cores)

Before optimization:

3--5 FPS

After optimization:

12--20 FPS

------------------------------------------------------------------------

# Conclusion

The current repository has:

-   clean modular architecture
-   good foundation for MVP

However several improvements are required for production readiness.

Key focus areas:

-   pipeline efficiency
-   recognition reuse
-   scalable database
-   concurrency
-   model optimization

Applying the proposed optimizations can:

-   reduce CPU usage by \~80%
-   increase FPS by 3--5×
-   improve scalability for thousands of employees
