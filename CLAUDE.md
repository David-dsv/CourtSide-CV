# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CourtSide-CV is a deep learning pipeline for tennis video analysis. It analyzes tennis match footage to provide players with actionable insights on their strengths and weaknesses. The pipeline chains object detection, multi-object tracking, ball trajectory analysis, and shot classification to produce annotated video with per-player statistics.

A fine-tuned YOLOv11 model is hosted on HuggingFace: `Davidsv/CourtSide-Computer-Vision-v1` (10 classes: racket, tennis_ball, court zones, net). A YOLO26 single-class ball model is trained locally: `training/yolo26_ball_best.pt`.

## Analysis Goals

The core purpose is to help players analyze their game. The pipeline must detect and evaluate:

### Shot Classification
- **Forehand** and **Backhand** only (no slice, volley, smash for now)
- Classification is based on which side of the player's body the racket/arm moves during the stroke relative to ball contact

### Shot Quality Metrics
Each shot receives a quality score (0-100) computed from:
- **Ball speed**: estimated from ball trajectory displacement between frames around the hit (pixels/frame → approximate km/h using court dimensions as reference)
- **Ball depth**: where the ball lands relative to the opponent's baseline — deep shots (landing near baseline) score higher than short shots (landing near service line or net)

### Match Events
- **Unforced errors (fautes directes)**: ball goes into net or out of bounds after a player's shot
- **Winners (coups gagnants)**: ball lands in court and opponent fails to reach it (ball bounces twice or exits play without opponent contact)

### Per-Player Stats Output
For each player, the pipeline produces:
- Total forehand count, average quality, speed distribution
- Total backhand count, average quality, speed distribution
- Winner count (forehand/backhand breakdown)
- Unforced error count (forehand/backhand breakdown)
- Shot depth distribution (deep / mid / short)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the annotated pipeline (primary entry point)
python run_pipeline_8s.py path/to/video.mp4 -s 60 -d 8 --device cpu

# Run the full modular pipeline
python main.py --video path/to/video.mp4 --yolo-version yolov11

# Process a segment of video
python main.py --video path/to/video.mp4 --start-time 60 --duration 120

# Run the standalone ball-tracking demo
python courtside_cv.py --video path/to/video.mp4 --start 0:00 --end 0:30

# Train custom YOLO model
python training/train_yolov11.py
python training/train_combined.py
```

There are no automated tests in this repository.

## Architecture

### Three Entry Points

1. **`run_pipeline_8s.py`** — Primary annotated pipeline (2-pass):
   - Pass 1: Ball detection (YOLO26 fine-tuned) → Kalman filter tracking → static false positive filtering → cubic spline interpolation → bounce detection
   - Pass 2: Pose estimation (YOLOv8-pose) → 2-player filtering (central horizontal band scoring) → skeleton drawing → ball trail → bounce markers → court overlay
   - Player selection: scores each detected person by horizontal centrality (20%-80% band hard filter) + bounding box area. Fallback to generic YOLO person detection if pose model finds <2 players.
   - Outputs annotated video with skeletons, player bounding boxes, ball trail, bounce markers

2. **`main.py`** — Full 5-step modular pipeline orchestrated by `TennisAnalysisPipeline`:
   - Detection → Tracking → Clip extraction (optional) → Action recognition → Video annotation
   - Uses `TennisYOLODetector` (dual-model) + ByteTrack tracking
   - Currently uses **geometric action detection** by default

3. **`courtside_cv.py`** — Standalone demo with ByteTrack ball tracking + YOLOv8 pose + interactive keyboard controls

### Detection Strategy

`TennisYOLODetector` (`models/yolo_detector.py`) combines two models:
- **Fine-tuned ball model** (`training/yolo26_ball_best.pt` — YOLO26, single class: tennis_ball). Falls back to HuggingFace multi-class model (`Davidsv/CourtSide-Computer-Vision-v1`) if not found.
- **Generic YOLO** (`yolov8m.pt`): detects persons (COCO class 0).

Custom model class IDs are mapped to COCO IDs internally (tennis_ball→32, person stays 0).

### Ball Tracking (run_pipeline_8s.py)

- `BallKalmanFilter`: constant-velocity Kalman filter (state: [x, y, vx, vy])
- Adaptive gating: gate grows with missed frames to handle occlusions
- Static false positive filtering: detections appearing at same location in >8% of frames are rejected (court logos, marks)
- Post-processing: cubic spline interpolation fills gaps up to 0.4s
- Bounce detection: vertical velocity sign change (vy going down→up) with minimum y-position threshold

### Player Detection & Tracking (run_pipeline_8s.py)

- Pose model (`yolov8m-pose.pt`, configurable via `--pose-model`) at high resolution (up to 1920px)
- Scoring system filters ball boys/umpires: horizontal centrality (70%) + area (30%), hard reject outside 20-80% horizontal band
- `PlayerTracker`: 2-slot IoU-based tracker for consistent player IDs across frames
- Fallback: if pose model finds <2 players, generic YOLO person detection fills the gap (far player often too small for pose)

### Pipeline Data Flow

Detection results and tracking results are passed as `List[Dict]` between stages. Each dict has `frame_id`, `timestamp`, and either `detections` (boxes/scores/class_ids/class_names) or `tracks` (boxes/track_ids/scores/class_ids).

### Key Modules

- **`models/`** — `yolo_detector.py` (dual-model detection), `tracker.py` (ByteTrack wrapper via roboflow/trackers)
- **`inference/`** — Pipeline steps: `detect.py`, `track.py`, `predict_actions.py`, `annotate_video.py`, `extract_clips.py`
- **`utils/`** — `video_utils.py` (VideoReader/VideoWriter), `drawing.py` (OpenCV visualization), `file_utils.py` (YAML/JSON/CSV I/O), `preprocessing.py` (X3D/SlowFast preprocessing)
- **`training/`** — Dataset preparation and YOLO training scripts

### Configuration

All pipeline parameters are in `config.yaml`. Set `paths.yolo_weights` to a local `.pt` path to skip HuggingFace download, or leave as `null` to auto-download.

## Design Principles

- **Zero hardcoding**: The pipeline must work on any tennis match video extract — any resolution, camera angle, broadcast style, court surface, or players. No magic numbers tied to a specific video. All thresholds must be derived from frame dimensions, fps, or detected features (court zones, player positions). When a detection model fails (e.g. no court zones found), use geometric fallbacks based on standard broadcast tennis proportions, not pixel values from a test video.
- **Generalization over precision**: Prefer robust heuristics that work across many videos over precise values tuned to one clip. Every constant should be expressed as a ratio of frame size, fps, or detected geometry.

## Important Conventions

- The codebase is bilingual: comments and UI strings mix French and English.
- COCO class IDs are used throughout the pipeline: `0` = person, `32` = sports_ball.
- Tracking filters to max 2 person tracks (the 2 tennis players).
- Video annotation copies the audio track from the source video using ffmpeg subprocess calls.
- Model weight files (`.pt`, `.pth`) are gitignored. The fine-tuned ball model is at `training/yolo26_ball_best.pt`, auto-downloaded from HuggingFace if missing.
- Shot quality is the primary metric: computed from ball speed (trajectory displacement) and depth (landing position relative to baseline). Deep + fast = high quality.
