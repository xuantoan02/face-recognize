# Face Recognition Optimization Task

## Planning
- [x] Read review report & cpu-optimize docs
- [x] Read all existing source files  
- [x] Write implementation plan
- [ ] Get user approval

## Restructure Code (New Package Layout)
- [ ] Create new sub-package directories under `src/`
- [ ] Move & refactor [detector.py](file:///home/toan-dx/AI/face-recognize/src/detector.py) → `src/detectors/face_detector.py`
- [ ] Move [aligner.py](file:///home/toan-dx/AI/face-recognize/src/aligner.py) → `src/alignment/aligner.py`
- [ ] Move [anti_spoof.py](file:///home/toan-dx/AI/face-recognize/src/anti_spoof.py) → `src/antispoof/anti_spoof.py`
- [ ] Move [recognizer.py](file:///home/toan-dx/AI/face-recognize/src/recognizer.py) → `src/recognition/recognizer.py`
- [ ] Create `src/recognition/embedding_cache.py` (NEW)
- [ ] Create `src/trackers/face_tracker.py` (NEW)
- [ ] Move [face_database.py](file:///home/toan-dx/AI/face-recognize/src/face_database.py) → `src/database/vector_db.py` (refactor to HNSWlib)
- [ ] Move [video_processor.py](file:///home/toan-dx/AI/face-recognize/src/video_processor.py) → `src/video/video_processor.py`
- [ ] Refactor [pipeline.py](file:///home/toan-dx/AI/face-recognize/src/pipeline.py) → `src/pipeline/pipeline.py`
- [ ] Move [utils.py](file:///home/toan-dx/AI/face-recognize/src/utils.py) → `src/utils/image_utils.py`
- [ ] Create `src/attendance/attendance_logic.py` (NEW)
- [ ] Create `src/quality/face_quality.py` (NEW)
- [ ] Add [__init__.py](file:///home/toan-dx/AI/face-recognize/src/__init__.py) for all new packages

## Implement Optimizations
- [ ] Face tracking (SORT algorithm)
- [ ] Anti-spoof once per track
- [ ] Embedding cache per track
- [ ] Face quality filter (blur, size, angle)
- [ ] HNSWlib vector database
- [ ] ONNX runtime optimizations
- [ ] Attendance debounce logic
- [ ] Threaded pipeline (video decode + detection + recognition)
- [ ] Performance logging/metrics

## Update Config & Dependencies
- [ ] Update [config/default.yaml](file:///home/toan-dx/AI/face-recognize/config/default.yaml)
- [ ] Update [requirements.txt](file:///home/toan-dx/AI/face-recognize/requirements.txt)
- [ ] Update [main.py](file:///home/toan-dx/AI/face-recognize/main.py) imports
- [ ] Update [scripts/register_faces.py](file:///home/toan-dx/AI/face-recognize/scripts/register_faces.py) imports
- [ ] Update [scripts/download_models.py](file:///home/toan-dx/AI/face-recognize/scripts/download_models.py) for SCRFD

## Verification
- [ ] Run import test
- [ ] Verify structure matches cpu-optimize spec
