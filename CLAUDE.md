# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Companion file.** This file describes **how the code works** (architecture, modules, conventions). Project **direction, vision, and settled decisions** live in `PROJET.md` — read it too. On *product intent*, `PROJET.md` wins; on *code detail*, this file wins. Don't duplicate `PROJET.md` content here.

## Project Overview

CourtSide-CV is a deep learning pipeline for tennis video analysis, targeting **amateur players filming on a phone** (see `PROJET.md` for the product vision). The pipeline chains ball detection, multi-object tracking, ball trajectory analysis, pose estimation, and bounce/depth analytics to produce annotated video with statistics.

Two models are involved:
- **Ball detection — the only fine-tuned model.** YOLO26 single-class (`tennis_ball`), trained locally: `training/yolo26_ball_best.pt`. This is the one model that must be fine-tuned (the ball is too small/fast for a generic model). Auto-downloaded from HuggingFace if missing.
- **Court zones (multi-class)** — `Davidsv/CourtSide-Computer-Vision-v1` on HuggingFace (10 classes: racket, tennis_ball, court zones, net). Used for the px→meter scale and as a ball fallback.
- **Players & pose — off-the-shelf, never retrained.** YOLOv8-pose and generic YOLOv8 (COCO `person`).

## Canonical Entry Point

**`run_pipeline_8s.py` is the single entry point to maintain.** All active work happens here.

`main.py` and the ML paths it references (X3D/SlowFast action recognition in `models/action_classifier.py`, the ResNet pose stub in `models/pose_estimator.py`, and `utils/preprocessing.py`) are **legacy / dead code** — never trained, not wired into the canonical pipeline. Don't build on them; they're slated for removal. `courtside_cv.py` is a standalone demo, not a product path.

## Feature Status

Each feature of the canonical pipeline (`run_pipeline_8s.py`) and its current state:

| Feature | Status | Notes |
|---|---|---|
| Ball detection (YOLO26) | ✅ Done | Fine-tuned single-class model, works well on broadcast footage |
| Ball tracking (Kalman filter) | ✅ Done | Adaptive gating, static FP filtering, spline interpolation |
| Player detection & tracking | ✅ Done | Pose model + fallback YOLO, 2-slot IoU tracker |
| Skeleton drawing (pose) | ✅ Done | YOLOv8-pose, high-res up to 1920px |
| Player bounding boxes | ✅ Done | Color-coded P1/P2, centrality scoring |
| Ball trail visualization | ✅ Done | Fading trail with configurable length |
| Ball speed (km/h) | ✅ Done | Kalman velocity → px/m scale → km/h HUD. Accuracy depends on homography (see Known Debt). |
| Court zone detection | ✅ Done | HuggingFace model, 8 zone classes |
| Court zone overlay | ✅ Done | Semi-transparent colored rectangles |
| **Bounce detection** | 🔧 In progress | Vertical velocity sign change — **needs accuracy work, current priority** |
| Bounce depth classification | 🔧 In progress | deep/mid/short — depends on bounce detection accuracy |
| Shot quality score | ⏸️ Later | Q = speed_norm + depth_norm — revisit after bounce detection is solid |
| Forehand/backhand classification | ⏸️ Later | Geometric detector exists but rudimentary, ML model not trained |

### Current Priority
**Bounce detection accuracy** in `run_pipeline_8s.py` — the `detect_bounces_from_trajectory()` function. It detects vertical velocity sign changes but produces false positives (player hits misidentified as bounces) and misses real bounces. It must be reliable before any downstream feature (depth, quality, stats) can work.

> ⚠️ Before tuning the bounce algorithm: there is no ground-truth set to measure against, so changes can't currently be judged objectively. Establishing a bounce ground truth (videos with manually-annotated bounce frames/positions) is a prerequisite for this work.

### Out of scope (parking)
These were considered but are **not on the current roadmap** — don't implement unless explicitly reprioritized: per-player shot attribution, winners detection, unforced-error detection, per-player stats aggregation. Kept here only so the intent isn't lost.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the annotated pipeline (canonical entry point)
python run_pipeline_8s.py path/to/video.mp4 -s 60 -d 8 --device cpu
# (on Apple Silicon, --device mps is available; CUDA is not)
```

There are no automated tests in this repository.

## Architecture (run_pipeline_8s.py)

Two-pass annotated pipeline. Input: a video file + time range (`-s` start, `-d` duration). Output: `data/output/<name>_annotated.mp4`.

- **Pass 1 — Ball:** detection (YOLO26 fine-tuned) → Kalman filter tracking → static false-positive filtering → cubic spline interpolation → bounce detection → speed (km/h).
- **Pass 2 — Annotation:** pose estimation (YOLOv8-pose) → 2-player filtering (central horizontal band scoring) → skeleton drawing → player bounding boxes → ball trail → bounce markers → court overlay → speed HUD.
- **Player selection:** scores each detected person by horizontal centrality (hard filter to the 20%-80% band) + bounding box area. Falls back to generic YOLO person detection if the pose model finds <2 players (far player often too small for pose).

### Detection Strategy

`TennisYOLODetector` (`models/yolo_detector.py`) combines two models:
- **Fine-tuned ball model** (`training/yolo26_ball_best.pt` — YOLO26, single class: tennis_ball). Falls back to the HuggingFace multi-class model if not found.
- **Generic YOLO** (`yolov8m.pt`): detects persons (COCO class 0).

Custom model class IDs are mapped to COCO IDs internally (tennis_ball→32, person stays 0).

### Ball Tracking

- `BallKalmanFilter`: constant-velocity Kalman filter (state: [x, y, vx, vy])
- Adaptive gating: gate grows with missed frames to handle occlusions
- Static false-positive filtering: detections appearing at the same location in >8% of frames are rejected (court logos, marks)
- Post-processing: cubic spline interpolation fills gaps up to 0.4s
- Bounce detection: vertical velocity sign change (vy down→up) with a minimum y-position threshold

### Player Detection & Tracking

- Pose model (`yolov8m-pose.pt`, configurable via `--pose-model`) at high resolution (up to 1920px)
- Scoring filters ball boys / umpires: horizontal centrality (70%) + area (30%), hard reject outside the 20-80% horizontal band
- `PlayerTracker`: 2-slot IoU-based tracker for consistent player IDs across frames

### Key Modules

- **`models/`** — `yolo_detector.py` (dual-model detection), `tracker.py` (ByteTrack wrapper). *Note: `action_classifier.py` and `pose_estimator.py` are dead code (see Canonical Entry Point).*
- **`utils/`** — `video_utils.py` (VideoReader/VideoWriter), `drawing.py` (OpenCV visualization), `file_utils.py` (YAML/JSON/CSV I/O)
- **`training/`** — Dataset preparation and YOLO training scripts

### Configuration

Pipeline parameters live in `config.yaml` (primarily used by the legacy `main.py`). Set `paths.yolo_weights` to a local `.pt` path to skip the HuggingFace download, or leave as `null` to auto-download.

## Design Principles

- **Zero hardcoding**: The pipeline must work on any tennis match video — any resolution, camera angle, broadcast style, court surface, or players. No magic numbers tied to a specific video. All thresholds must be derived from frame dimensions, fps, or detected features (court zones, player positions). When a detection model fails (e.g. no court zones found), use geometric fallbacks based on standard tennis proportions, not pixel values from a test video.
- **Generalization over precision**: Prefer robust heuristics that work across many videos over precise values tuned to one clip. Every constant should be expressed as a ratio of frame size, fps, or detected geometry.

## Known Debt (technical guardrails)

Concrete items to fix; these contradict the design principles above or block reliability.

- 🔴 **Exposed secret:** a Roboflow API key was committed in clear in `training/train_roboflow_yolo26.py`. Revoke it and move to an environment variable; ensure `.env` is gitignored.
- 🟠 **Hardcoded values (violate "zero hardcoding"):** player names ("Alcaraz"/"Sinner") mapped to track IDs in `inference/annotate_video.py`; FPS hardcoded to 30 in `inference/predict_actions.py` and `inference/extract_clips.py`. The canonical `run_pipeline_8s.py` correctly uses `reader.fps` — align the rest.
- 🟠 **Homography is the reliability bottleneck:** all metric stats (km/h, depth) rely on a px→meter scale from court lines. The current geometric fallback ("court = 58% of frame height") won't hold on varied amateur framings. This is the #1 risk per `PROJET.md`.
- 🟡 **Fragile environment:** Python 3.14, no lockfile, `install.sh` references absent files (`requirements-minimal.txt`, `QUICKSTART.md`). Reproducibility is weak.
- 🟡 **No automated tests** — risky for a geometric algorithm tuned continuously (bounces). See the ground-truth note above.
- 🟡 **Stray files:** pip-redirect artifacts (`=0.25.0`, `=2.2.0`, `=8.5`) at the repo root, to delete.

## Workflow

- **Push after every change**: after each commit, push to GitHub immediately (`git push`). Never leave commits local-only.

## Important Conventions

- The codebase is bilingual: comments and UI strings mix French and English.
- COCO class IDs are used throughout: `0` = person, `32` = sports_ball.
- Tracking filters to max 2 person tracks (the 2 tennis players).
- Model weight files (`.pt`, `.pth`) are gitignored. The fine-tuned ball model is at `training/yolo26_ball_best.pt`, auto-downloaded from HuggingFace if missing.
- Note: `run_pipeline_8s.py`'s VideoWriter does **not** re-add the audio track (unlike the `courtside_cv.py` demo, which re-injects audio via ffmpeg).