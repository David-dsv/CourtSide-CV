"""
WASB heatmap ball tracker — drop-in replacement for the YOLO+Kalman ball path.

WASB (Tarashima et al., BMVC 2023, "Widely Applicable Strong Baseline for Sports
Ball Detection and Tracking", https://github.com/nttcom/WASB-SBDT, MIT licensed)
is a heatmap-temporal tracker: it takes 3 consecutive frames stacked (9 ch) and
predicts a per-frame heatmap whose peak is the ball. On the tennis benchmark it
reports F1 95.6 — far above what a bbox detector + Kalman achieves on small/fast
balls — and crucially it produces a *smooth, dense* trajectory (the actual
bottleneck for bounce detection) without the static-FP/reinit jitter the Kalman
path fights.

The HRNet architecture is vendored under vendor/wasb/models/hrnet.py (MIT). This
module hardcodes the tennis model config (from the repo's configs/model/wasb.yaml)
so it needs no Hydra/yaml/attrdict, and implements its own lightweight peak
postprocessor (heatmap-weighted centroid of the top blob), returning a confidence
so the caller gets a clean "no ball this frame" signal.

Exposes:
    HeatmapBallTracker(weights_path, device).track_ball(frames) ->
        list[(x, y) | None]   # one entry per input frame, native resolution
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from loguru import logger

# vendored HRNet (MIT, from nttcom/WASB-SBDT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "wasb" / "models"))
import hrnet as _hrnet  # noqa: E402


# Tennis model config, transcribed from WASB-SBDT/src/configs/model/wasb.yaml.
# Kept inline so the tracker has no yaml/attrdict runtime dependency.
WASB_CFG = {
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_height": 288,
    "out_width": 512,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                       "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128],
                       "FUSE_METHOD": "SUM"},
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}

INP_W, INP_H = 512, 288
FRAMES_IN = 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _AttrDict(dict):
    """Dict that also allows attribute access (cfg.MODEL.EXTRA), recursively —
    the vendored HRNet mixes cfg['x'] and cfg.x access."""

    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self[k] = _AttrDict(v) if isinstance(v, dict) else v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class HeatmapBallTracker:
    def __init__(self, weights_path, device="mps", score_threshold=0.5):
        self.device = device
        self.score_threshold = score_threshold
        self.model = _hrnet.HRNet(_AttrDict(WASB_CFG))
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"WASB: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            logger.warning(f"WASB: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        self.model.eval().to(device)
        logger.info(f"WASB heatmap ball tracker loaded ({weights_path}, device={device})")

    @staticmethod
    def _preprocess_stack(frames3):
        """frames3: list of 3 BGR native-res frames -> (1,9,288,512) float tensor.

        Preprocessing matches WASB's training dataloader (this is what makes the
        pretrained tennis weights actually fire): BGR→RGB, /255, then ImageNet
        mean/std normalization. Without the ImageNet normalization the model
        returns almost no detections (the bug that made an earlier integration
        wrongly conclude WASB was a no-go on this footage)."""
        chans = []
        for f in frames3:
            r = cv2.resize(f, (INP_W, INP_H), interpolation=cv2.INTER_LINEAR)
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            r = (r - IMAGENET_MEAN) / IMAGENET_STD
            chans.append(r)
        stacked = np.concatenate(chans, axis=2)              # H,W,9
        t = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).float()  # 1,9,H,W
        return t

    def _peak(self, hm, scale_x, scale_y):
        """heatmap (288,512) -> (x,y) in native res or None. Weighted centroid of
        the brightest blob above threshold."""
        mx = float(hm.max())
        if mx <= self.score_threshold:
            return None
        _, binar = cv2.threshold(hm, self.score_threshold, 1.0, cv2.THRESH_BINARY)
        binar = binar.astype(np.uint8)
        ncc, labels, stats, _ = cv2.connectedComponentsWithStats(binar, connectivity=8)
        if ncc <= 1:
            return None
        # largest blob by area (excluding background label 0)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.where(labels == best)
        w = hm[ys, xs]
        if w.sum() <= 0:
            return None
        cx = float((xs * w).sum() / w.sum())
        cy = float((ys * w).sum() / w.sum())
        return (cx * scale_x, cy * scale_y)

    @torch.no_grad()
    def track_ball(self, frames, batch_size=16, frame_step=1):
        """frames: list of native-res BGR frames. Returns list[(x,y)|None] per frame.

        For frame t, the stack is [t-2*step, t-step, t]; the heatmap for the LAST
        frame in the stack localizes the ball at t. Frames before the window start
        are padded by clamping to 0. `frame_step` lets the 3-frame temporal window
        span more real motion on high-fps video (WASB is trained on ~30fps; at
        60fps use frame_step=2 so 3 frames cover the motion it expects).
        """
        n = len(frames)
        if n == 0:
            return []
        h0, w0 = frames[0].shape[:2]
        scale_x = w0 / INP_W
        scale_y = h0 / INP_H

        centers = [None] * n
        # build all 3-frame stacks with the requested temporal step
        stacks = []
        for t in range(n):
            i0, i1 = max(0, t - 2 * frame_step), max(0, t - frame_step)
            stacks.append([frames[i0], frames[i1], frames[t]])

        for start in range(0, n, batch_size):
            batch = stacks[start:start + batch_size]
            tens = torch.cat([self._preprocess_stack(s) for s in batch], dim=0).to(self.device)
            out = self.model(tens)[0]                 # (B,3,288,512)
            out = torch.sigmoid(out).cpu().numpy()
            for bi in range(out.shape[0]):
                hm = out[bi, FRAMES_IN - 1]           # heatmap for last frame in stack
                centers[start + bi] = self._peak(hm, scale_x, scale_y)
        return centers
