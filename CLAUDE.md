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
| Player detection & tracking | ✅ Done | Top-down pose + side-of-net player selection (`vision/pose.py`) |
| **Locked P1/P2 skeletons** | ✅ Done | `vision/player_track.py` `TwoPlayerTracker`: persistent P1(far)/P2(near) identity via dynamic median net-divider + IoU/centroid association + **gap-fill (zero dropped frames)** + temporal smoothing. Pass 1.5 computes pose once, reused by shots + Pass 2. Distinct colors. Unit-tested (`tests/test_player_track.py`). |
| Skeleton drawing (pose) | ✅ Done | YOLOv8-pose default; **RTMW whole-body optional** (`--pose-backend rtmw`, `vision/pose_rtmw.py`): 133 kpts incl. hands/feet, CPU-only, wins on the tiny far player (17/17 vs ~15/17). |
| Player bounding boxes | ✅ Done | **OFF by default** (user feedback); `--player-boxes` to re-enable. |
| Ball trail visualization | ✅ Done | Cinematic glow comet trail via `vision/fx.py` (`--no-fx` for the plain fading trail). |
| Ball speed (km/h) | ✅ Done | Court-homography → real meters (perspective-correct) when court is localizable, scalar px/m fallback otherwise. `--homography`. Spikes >280 km/h clamped. |
| Court zone detection | ✅ Done | HuggingFace model, 8 zone classes |
| Court zone overlay | ✅ Done | Semi-transparent colored rectangles |
| Court homography (px↔m) | ✅ Done (broadcast) | `models/court_detector.py`: 14 keypoints → ITF meters → `cv2.findHomography`. High confidence on standard broadcast (felix-style oblique angles fall back, see Known Debt). |
| **Court homography (oblique/amateur)** | ✅ Done (semi-auto) | `models/court_calibrator.py`: semi-auto **4-corner homography** for static cameras where the keypoint model fails. Calibrate a clip ONCE via `tools/calibrate_court.py` (click court landmarks) → JSON sidecar (`calibrations/<name>.court.json` or `<video>.court.json`) auto-loaded by the pipeline. felix calibrated → `speed_source=homography(high)` on the oblique angle. `--no-calibration` to ignore. |
| **Cinematic FX** | ✅ Done | `vision/fx.py`: additive bloom, comet trail, glassmorphism HUD/panels, shockwave (bounce), broadcast speed gauge, live stat card, INFERNO placement heatmap, glow marker. Wired into Pass 2 (`--no-fx` to disable). Pure cv2+numpy+PIL, no GPU. |
| **2D minimap (COURT RADAR)** | ✅ Done | `vision/minimap.py`: premium glassmorphism top-down radar — glow ball trajectory, depth-coded bounce dots + placement heat, P1/P2 player pucks, metric-exact vs `~` approximate flag. Projects via the (calibration) homography. `--no-minimap` to disable. |
| Ball tracking (WASB heatmap) | ✅ Done (broadcast) | `--ball-tracker wasb`: WASB heatmap-temporal tracker (`models/wasb_ball_tracker.py`, MIT). Dense trajectory (84% coverage vs Kalman's collapse). **Needs ImageNet normalization** in preprocessing. Default stays `kalman`. |
| **Bounce detection** | ✅ Done (broadcast) | Two detectors in `vision/bounce.py`: curve-fit (kalman path) + robust (WASB path: despike + IRLS + energy hit/bounce test). felix GT F1 **0.83 end-to-end (WASB) / 0.67 (Kalman)**, was 0.30. Both regression-tested. |
| Bounce depth classification | 🔧 In progress | deep/mid/short — accurate via homography meters when available; geometric fallback otherwise. |
| Shot quality score | ⏸️ Later | Q = speed_norm + depth_norm — revisit after bounce detection is solid |
| Forehand/backhand classification | ⏸️ Later | Geometric detector exists but rudimentary, ML model not trained |

### Current Priority
**Oblique/amateur angles now have a metric path via semi-auto calibration.** The oblique-angle homography gap is closed pragmatically: `models/court_calibrator.py` + `tools/calibrate_court.py` let a static-camera clip be calibrated ONCE (click the court corners) → exact metric homography for every frame, unblocking the minimap, real km/h and bounce depth on felix-style angles. This is the deployable SaaS path (the amateur films on a tripod).

The remaining item is **AUTOMATIC** oblique court detection (so the user doesn't have to click): the pre-trained court-keypoint model fires on too few keypoints on low/court-level framings (felix: 3/14). A **2026 SOTA research pass** (see memory `courtside-sota-2026-06`) evaluated RF-DETR keypoint (`RFDETRKeypointPreview`, Apache-2.0, arbitrary skeletons) as the best automatic candidate, but it needs a **cloud GPU fine-tune + a labeled oblique court dataset** and an MPS-inference smoke test first → it's a gated pilot, not a blocker. A classical line+RANSAC detector with warp-back self-scoring (to auto-seed the corners + gate confidence) is the no-GPU interim path.

> ⚠️ WASB lesson (important): pretrained WASB weights need **ImageNet mean/std normalization** in preprocessing. Without it the model returns almost no detections — an easy bug that can make WASB look like a no-go when it actually reaches F1 0.83–0.87 on felix.

> ✅ A bounce ground truth now exists: `tests/fixtures/bounces/felix.bounces.json` (16 bounces) + the evaluator `tools/bounce_eval/eval_bounces.py` + a fast regression test `tests/test_bounce_regression.py` (asserts F1 ≥ 0.72 on a committed detection cache, no GPU/video). Every bounce-algorithm change is now measurable objectively.

### Out of scope (parking)
These were considered but are **not on the current roadmap** — don't implement unless explicitly reprioritized: per-player shot attribution, winners detection, unforced-error detection, per-player stats aggregation. Kept here only so the intent isn't lost.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the annotated pipeline (canonical entry point)
python run_pipeline_8s.py path/to/video.mp4 -s 60 -d 8 --device cpu
# (on Apple Silicon, --device mps is available; CUDA is not)

# With perspective-correct speed/depth via court homography (broadcast angles):
python run_pipeline_8s.py path/to/video.mp4 -s 60 -d 8 --device mps --homography

# With the WASB heatmap ball tracker (best bounce recall; needs the WASB weights):
python run_pipeline_8s.py path/to/video.mp4 -s 60 -d 8 --device mps --ball-tracker wasb

# Export bounce predictions to score against ground truth:
python run_pipeline_8s.py path/to/video.mp4 -s 0 -d 31 --dump-bounces preds.json
python tools/bounce_eval/eval_bounces.py --pred preds.json --truth tests/fixtures/bounces/felix.bounces.json
```

### Tests

```bash
# Fast bounce-accuracy regression (no GPU / no video decode; ~seconds)
python tests/test_bounce_regression.py        # asserts felix bounce F1 >= 0.72
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

- ✅ **(resolved) Exposed secret:** the Roboflow API key is now read from `os.getenv("ROBOFLOW_API_KEY")` in `training/train_roboflow_yolo26.py`. (Rotate the old key if not already done.)
- 🟠 **Hardcoded values (violate "zero hardcoding"):** player names ("Alcaraz"/"Sinner") mapped to track IDs in `inference/annotate_video.py`; FPS hardcoded to 30 in `inference/predict_actions.py` and `inference/extract_clips.py`. The canonical `run_pipeline_8s.py` correctly uses `reader.fps` — align the rest. (These are in legacy/`inference/` paths, not the canonical pipeline.)
- 🟡 **Homography: automatic detection still broadcast-only; oblique has a semi-auto path.** `--homography` (keypoint model) gives perspective-correct meters on standard broadcast (~1% of ITF dims) but fires on <4 points on oblique framings. **Resolved for static cameras** via `models/court_calibrator.py` + `tools/calibrate_court.py` (calibrate once → metric homography sidecar, auto-loaded; felix → `speed_source=homography`). Remaining: a fully *automatic* oblique detector (gated GPU pilot — RF-DETR keypoint / classical line+RANSAC; see `Current Priority`). The scalar fallback still flags `speed_source=scalar` honestly when no homography/calibration is available. ⚠️ On *grazing* oblique angles (camera near-parallel to a sideline, e.g. felix) a 4-corner solve is ill-conditioned in the far field — use more, well-spread points (service lines, net posts) for accuracy.
- ⚠️ **Court-keypoint model license:** `yastrebksv/TennisCourtDetector` ships no LICENSE. Used here for internal evaluation; clarify before redistribution. (WASB, by contrast, is MIT — `vendor/wasb/LICENSE.md`.)
- 🟡 **Fragile environment:** Python 3.14 (beta — `pyexpat`/`gdown`/matplotlib break), no lockfile, `install.sh` references absent files (`requirements-minimal.txt`, `QUICKSTART.md`). Reproducibility is weak.
- ✅ **(resolved) First automated test:** `tests/test_bounce_regression.py` asserts bounce F1 ≥ 0.72 on a committed detection-cache fixture (fast, no GPU/video). A bounce ground truth + evaluator now exist (`tests/fixtures/bounces/`, `tools/bounce_eval/`).
- 🟡 **Stray files:** pip-redirect artifacts (`=0.25.0`, `=2.2.0`, `=8.5`) at the repo root, to delete.

## Workflow

- **Push after every change**: after each commit, push to GitHub immediately (`git push`). Never leave commits local-only.

## Important Conventions

- The codebase is bilingual: comments and UI strings mix French and English.
- COCO class IDs are used throughout: `0` = person, `32` = sports_ball.
- Tracking filters to max 2 person tracks (the 2 tennis players).
- Model weight files (`.pt`, `.pth`) are gitignored. The fine-tuned ball model is at `training/yolo26_ball_best.pt`, auto-downloaded from HuggingFace if missing.
- Note: `run_pipeline_8s.py`'s VideoWriter does **not** re-add the audio track (unlike the `courtside_cv.py` demo, which re-injects audio via ffmpeg).