"""
Optional RTMW whole-body pose backend (rtmlib), drop-in for YOLOv8-pose.

Why: on the tiny far player (a ~30×90px crop) YOLOv8-pose resolves ~15/17 body
keypoints, while RTMW resolves a full 133-keypoint whole-body skeleton (body +
feet + hands) — measured in the pilot. The hands/feet keypoints also unlock
better racket-contact / forehand-backhand cues. RTMW is Apache-2.0 and runs via
onnxruntime; on Apple Silicon the CoreML provider crashes on the bundled YOLOX
detector, so we run pose-only on the CPU provider (we already crop to the player,
so we don't need rtmlib's detector at all).

This backend is OFF by default (slower: ~0.3s/crop on CPU vs ~0.1s for YOLOv8).
Enable with --pose-backend rtmw. It exposes an object that quacks like an
Ultralytics pose result so vision.pose.pose_on_crop works unchanged: calling the
wrapper like `model(crop, conf=, imgsz=, device=, verbose=)` returns
`[result]` where result.keypoints.xy[i] / .conf[i] and result.boxes.xyxy expose
the Ultralytics-compatible numpy-backed interface.

The 17 COCO body keypoints are the first 17 of RTMW's 133 (coco133 layout), so
the pipeline's existing 17-keypoint skeleton drawing maps over with no change;
hands/feet are available as extra channels for downstream shot logic.

Install (project venv, note the expat workaround for this env's Python 3.14):
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/pip install rtmlib onnxruntime
"""
from __future__ import annotations

import numpy as np

# COCO-WholeBody (coco133) channel layout — the body subset we keep is 0..16.
N_BODY = 17


class _Channel:
    """Mimics `result.keypoints` / `result.boxes` — indexable, each item is a
    small object whose .cpu().numpy() returns the underlying array."""
    def __init__(self, arr):
        self._arr = arr  # (N, K, 2) for xy, (N, K) for conf, (N, 4) for boxes

    def __len__(self):
        return len(self._arr)

    @property
    def xy(self):
        return _IndexNumpy(self._arr)

    @property
    def conf(self):
        return _IndexNumpy(self._arr)

    @property
    def xyxy(self):
        return _NumpyLike(self._arr)


class _IndexNumpy:
    """`result.keypoints.xy[i]` → object with .cpu().numpy() → arr[i]."""
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, i):
        return _NumpyLike(self._arr[i])


class _NumpyLike:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Result:
    def __init__(self, kxy, kconf, boxes):
        # kxy: (N,17,2), kconf: (N,17), boxes: (N,4)
        self.keypoints = _KP(kxy, kconf) if len(kxy) else None
        self.boxes = _Channel(boxes) if len(boxes) else None


class _KP:
    def __init__(self, kxy, kconf):
        self._xy = kxy
        self._conf = kconf

    @property
    def xy(self):
        return _IndexNumpy(self._xy)

    @property
    def conf(self):
        return _IndexNumpy(self._conf)


class RTMWPose:
    """Callable RTMW pose wrapper with an Ultralytics-compatible result shape.

    Usage mirrors the YOLO pose model: `rtmw(crop, conf=..., device=..., ...)`
    returns `[result]`. Extra (hands/feet) keypoints are stored on the instance
    after each call as `last_wholebody` for downstream shot logic that wants them.
    """

    def __init__(self, mode="balanced"):
        try:
            from rtmlib import Wholebody
        except Exception as e:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "rtmlib not installed. Install with:\n"
                "  DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib "
                "venv/bin/pip install rtmlib onnxruntime"
            ) from e
        # CPU provider: CoreML/MPS crashes on the bundled YOLOX detector on Apple
        # Silicon. We crop to the player ourselves, so detection cost is moot.
        self._wb = Wholebody(mode=mode, backend="onnxruntime", device="cpu",
                             to_openpose=False)
        self.last_wholebody = None  # (N,133,2),(N,133) of the most recent call

    def __call__(self, crop, conf=0.15, imgsz=None, device=None, verbose=False):
        """Run RTMW on a crop. Returns [Ultralytics-like result] with the 17 COCO
        body keypoints; full 133-kpt whole-body kept on self.last_wholebody."""
        keypoints, scores = self._wb(crop)   # (N,133,2), (N,133)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.ndim != 3 or len(keypoints) == 0:
            self.last_wholebody = None
            return [_Result(np.empty((0, N_BODY, 2)), np.empty((0, N_BODY)),
                            np.empty((0, 4)))]
        self.last_wholebody = (keypoints, scores)
        # body subset (first 17 = COCO body)
        body_xy = keypoints[:, :N_BODY, :].astype(np.float32)
        body_cf = scores[:, :N_BODY].astype(np.float32)
        # per-person bbox from all confident keypoints (rtmlib gives no boxes here)
        boxes = []
        for i in range(len(keypoints)):
            vis = scores[i] > conf
            pts = keypoints[i][vis] if vis.any() else keypoints[i]
            if len(pts) == 0:
                boxes.append([0, 0, 1, 1])
                continue
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        boxes = np.array(boxes, np.float32)
        return [_Result(body_xy, body_cf, boxes)]
