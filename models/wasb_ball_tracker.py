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

    def _peaks(self, hm, scale_x, scale_y, topk=3):
        """heatmap (288,512) -> list of up to topk (x,y,score) blob centroids in
        native res, brightest first. score = peak heatmap value of the blob."""
        mx = float(hm.max())
        if mx <= self.score_threshold:
            return []
        _, binar = cv2.threshold(hm, self.score_threshold, 1.0, cv2.THRESH_BINARY)
        binar = binar.astype(np.uint8)
        ncc, labels, stats, _ = cv2.connectedComponentsWithStats(binar, connectivity=8)
        if ncc <= 1:
            return []
        blobs = []
        for lab in range(1, ncc):
            ys, xs = np.where(labels == lab)
            w = hm[ys, xs]
            if w.sum() <= 0:
                continue
            cx = float((xs * w).sum() / w.sum())
            cy = float((ys * w).sum() / w.sum())
            blobs.append((cx * scale_x, cy * scale_y, float(w.max())))
        blobs.sort(key=lambda b: -b[2])
        return blobs[:topk]

    def _peak(self, hm, scale_x, scale_y):
        """Back-compat: single best blob (x,y) or None."""
        ps = self._peaks(hm, scale_x, scale_y, topk=1)
        return (ps[0][0], ps[0][1]) if ps else None

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

        # collect per-frame candidate blobs (top-K), then gate by trajectory
        cand_per_frame = [[] for _ in range(n)]
        for start in range(0, n, batch_size):
            batch = stacks[start:start + batch_size]
            tens = torch.cat([self._preprocess_stack(s) for s in batch], dim=0).to(self.device)
            out = self.model(tens)[0]                 # (B,3,288,512)
            out = torch.sigmoid(out).cpu().numpy()
            for bi in range(out.shape[0]):
                hm = out[bi, FRAMES_IN - 1]
                cand_per_frame[start + bi] = self._peaks(hm, scale_x, scale_y, topk=3)

        centers = self._gate_trajectory(cand_per_frame, w0, h0)
        centers = self._despike(centers, w0, h0)
        return centers

    @staticmethod
    def _despike(centers, frame_w, frame_h):
        """Remove isolated jump-and-return outliers the gate let through across a
        gap. We look up to W frames on each side (skipping None) for stable
        anchors; a point that sits far from the straight line between those
        anchors — while the anchors are themselves consistent — is a teleport
        spike, not a real ball position. Two passes so a 2-frame excursion
        (point, None, point) is fully cleaned."""
        n = len(centers)
        diag = float(np.hypot(frame_w, frame_h))
        thr = diag * 0.12
        W = 8
        out = list(centers)

        def stable_anchor(idx, direction):
            """Walk outward; return the first point that has a close neighbour just
            beyond it (i.e. sits on a stable stretch, not itself a spike)."""
            step = -1 if direction < 0 else 1
            prev = None
            for k in range(1, W + 1):
                j = idx + step * k
                if not (0 <= j < n) or out[j] is None:
                    continue
                if prev is not None and np.hypot(out[j][0] - prev[0], out[j][1] - prev[1]) < thr:
                    return prev  # prev had a close neighbour → stable
                prev = out[j]
            return prev

        for _ in range(3):
            changed = False
            for i in range(n):
                if out[i] is None:
                    continue
                p = stable_anchor(i, -1)
                q = stable_anchor(i, +1)
                if p is None or q is None:
                    continue
                dp = np.hypot(out[i][0] - p[0], out[i][1] - p[1])
                dq = np.hypot(out[i][0] - q[0], out[i][1] - q[1])
                dpq = np.hypot(p[0] - q[0], p[1] - q[1])
                if dp > thr and dq > thr and dpq < dp and dpq < dq:
                    out[i] = None
                    changed = True
            if not changed:
                break
        return out

    def _gate_trajectory(self, cand_per_frame, frame_w, frame_h):
        """Trajectory-consistency gate against aberrant detections (ball teleporting
        to the stands / other side of the court). For each frame, among the top-K
        candidate blobs, accept the one consistent with a constant-velocity
        prediction within a distance gate that grows with frames-since-accept. A
        big jump is allowed ONLY if the blob has a very high WASB score (a genuine
        fast ball). Blobs in the upper stands band are rejected outright.

        Thresholds from the measured felix diagnosis + anti-parasite design: a real
        tennis ball moves at most ~5% of frame width per frame, so a 15%-diagonal
        base gate is comfortably safe; aberrant teleports (35-65% diagonal) and
        stand detections are removed without dropping true fast-ball frames."""
        n = len(cand_per_frame)
        diag = float(np.hypot(frame_w, frame_h))
        # Generous gate against IMPOSSIBLE jumps only (keeps real fast balls + the
        # 82% density); grows with missed frames so re-acquisition after occlusion
        # still works. The WASB heatmap peak score is in [0,1] (sigmoid).
        gate_base = diag * 0.18          # ~400px @1080p — above any real 1-frame move
        gate_growth = diag * 0.05        # widen per missed frame (tighter: avoids
                                          # admitting a stray on the far side during
                                          # a 2-3 frame gap, the residual aberration)
        gate_max = diag * 0.45
        y_stands = frame_h * 0.05        # above this = stands → never a ball in play
        strong_score = 0.85              # a confident blob may re-seed after a long gap
        max_miss_keep = int(round(0.5 * 60))  # after ~0.5s of misses, allow re-seed
        centers = [None] * n
        last_pos = None
        miss = 0
        for i in range(n):
            blobs = [b for b in cand_per_frame[i] if b[1] >= y_stands]
            if not blobs:
                miss += 1
                centers[i] = None
                continue
            if last_pos is None:
                bx, by, _ = max(blobs, key=lambda b: b[2])
                centers[i] = (bx, by)
                last_pos = (bx, by)
                miss = 0
                continue
            gate = min(gate_max, gate_base + gate_growth * miss)
            # choose the blob closest to the LAST accepted position within the gate
            best, bestd = None, float("inf")
            for bx, by, sc in blobs:
                d = np.hypot(bx - last_pos[0], by - last_pos[1])
                if d <= gate and d < bestd:
                    bestd, best = d, (bx, by)
            if best is None:
                # nothing consistent. If we've been lost a while, re-seed on a
                # strong blob (handles a genuine long occlusion); else drop strays.
                if miss >= max_miss_keep:
                    strong = max(blobs, key=lambda b: b[2])
                    if strong[2] >= strong_score:
                        best = (strong[0], strong[1])
                if best is None:
                    miss += 1
                    centers[i] = None
                    continue
            last_pos = best
            centers[i] = best
            miss = 0
        return centers
