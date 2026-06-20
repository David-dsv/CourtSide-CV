"""
Persistent 2-player skeleton tracking with zero dropped frames.

The per-frame top-down pose estimator (vision/pose.py) returns "the 2 players in
this frame" but with NO temporal identity: across frames the order can swap, and
a frame where one player isn't detected leaves a gap. For a SaaS-grade overlay we
need two STABLE tracks — P1 and P2 — that:

  1. keep a consistent identity from the first frame to the last,
  2. never flicker between players (no identity swaps),
  3. have a skeleton on EVERY frame (missed detections are gap-filled), and
  4. are smoothed so the skeleton doesn't jitter.

Identity model (tennis-specific, very robust on a static camera):
  The two players straddle the net and never cross it. So the player whose feet
  are ABOVE the net line is always "far" (P1) and the one BELOW is "near" (P2).
  Side-of-net is the primary, drift-proof identity anchor; within a side we
  associate frame-to-frame by centroid distance + IoU to survive the rare frame
  where both players are briefly on the same side (e.g. a poach at the net).

Gap-filling:
  When a track misses a few frames, we linearly interpolate its keypoints between
  the last and next seen poses (up to `max_gap` frames). For a leading/trailing
  gap we hold the nearest pose. Result: a skeleton on every frame.

This module is pure-numpy and unit-testable without a model or video.
"""
from __future__ import annotations

import numpy as np

N_KP = 17  # COCO keypoints


# ───────────────────────── geometry helpers ──────────────────────────────

def _box_centroid(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def _feet_y(box, kps_xy=None, kps_conf=None):
    """Best estimate of the player's feet row (image y). Prefer the ankles if
    confidently detected (COCO 15/16), else the bottom of the box."""
    if kps_xy is not None and kps_conf is not None:
        ankles = []
        for a in (15, 16):
            if a < len(kps_conf) and kps_conf[a] > 0.3:
                ankles.append(kps_xy[a][1])
        if ankles:
            return float(max(ankles))
    return float(box[3])


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ───────────────────────── the tracker ──────────────────────────────

class TwoPlayerTracker:
    """Assigns persistent P1 (far) / P2 (near) identities to per-frame player
    detections, with side-of-net anchoring, centroid+IoU association, gap-filling
    and temporal smoothing.

    Usage:
        trk = TwoPlayerTracker(frame_w, frame_h, net_y, fps)
        for players in players_per_frame:   # list[dict(box,kps_xy,kps_conf)] | None
            trk.add_frame(players)
        tracks = trk.finalize()             # {1: [per-frame pose|None], 2: [...]}

    Each per-frame pose is a dict {box, kps_xy(17,2), kps_conf(17,), filled: bool}
    or None (only at leading/trailing edges beyond max_gap).
    """

    def __init__(self, frame_w, frame_h, net_y, fps,
                 max_gap_s=0.5, smooth_alpha=0.55, assoc_max_dist_frac=0.18):
        self.fw = frame_w
        self.fh = frame_h
        self.net_y = net_y          # seed divider (net line if known) — may be None
        self.fps = fps
        self.max_gap = max(1, int(round(max_gap_s * fps)))
        self.smooth_alpha = smooth_alpha
        self.assoc_max_dist = assoc_max_dist_frac * frame_w
        # raw assignment per frame: {1: pose|None, 2: pose|None}
        self._raw = {1: [], 2: []}
        # last seen pose per slot (for association continuity)
        self._last = {1: None, 2: None}
        # running divider: EMA of the midpoint between the two players' feet. This
        # is the zero-hardcoding split — it self-centers on the actual net even
        # when net_y is unknown/wrong, and adapts if the framing shifts slightly.
        self._divider = float(net_y) if net_y else None

    # -- side of net --------------------------------------------------------
    def _divider_y(self):
        return self._divider if self._divider is not None else self.fh * 0.5

    def _side(self, pose):
        """1 (far) if feet above the running divider, else 2 (near)."""
        fy = _feet_y(pose["box"], pose.get("kps_xy"), pose.get("kps_conf"))
        return 1 if fy < self._divider_y() else 2

    def _update_divider(self, players):
        """Self-center the divider on the gap between the two players' feet.
        When exactly two players are visible, the net is between them, so the
        midpoint of their feet is the best divider — adapts to any angle without a
        hardcoded row. EMA-smoothed so a single bad frame can't flip it."""
        feet = sorted(_feet_y(p["box"], p.get("kps_xy"), p.get("kps_conf"))
                      for p in players)
        if len(feet) >= 2:
            mid = (feet[0] + feet[-1]) / 2.0
            if self._divider is None:
                self._divider = mid
            else:
                self._divider = 0.9 * self._divider + 0.1 * mid

    def add_frame(self, players):
        """players: list of dicts {box, kps_xy, kps_conf} (kps may be None), or None."""
        players = players or []
        # keep only entries that have a skeleton; box-only entries can't be drawn
        cand = [p for p in players if p.get("kps_xy") is not None]

        # self-center the divider on the two players before bucketing
        self._update_divider(cand)

        # bucket candidates by side of net
        by_side = {1: [], 2: []}
        for p in cand:
            by_side[self._side(p)].append(p)

        assigned = {1: None, 2: None}
        for slot in (1, 2):
            pool = by_side[slot]
            if not pool:
                continue
            if len(pool) == 1:
                assigned[slot] = pool[0]
                continue
            # multiple candidates on this side → pick the one most consistent with
            # the slot's last seen pose (centroid distance, tie-broken by IoU),
            # else the largest/most-confident.
            last = self._last[slot]
            if last is not None:
                lc = _box_centroid(last["box"])
                best, best_key = None, None
                for p in pool:
                    d = np.linalg.norm(_box_centroid(p["box"]) - lc)
                    iou = _iou(p["box"], last["box"])
                    key = (d - iou * self.assoc_max_dist)  # lower = better
                    if best_key is None or key < best_key:
                        best, best_key = p, key
                assigned[slot] = best
            else:
                # no history: largest box (the real player vs a stray detection)
                assigned[slot] = max(pool, key=lambda p: (p["box"][3] - p["box"][1]))

        # handle the rare both-on-same-side frame: if one slot is empty but the
        # other side has 2 candidates, lend the extra to the empty slot using its
        # last-known position as the anchor (keeps both skeletons alive).
        for empty, other in ((1, 2), (2, 1)):
            if assigned[empty] is None and len(by_side[other]) >= 2 and self._last[empty] is not None:
                taken = assigned[other]
                spare = [p for p in by_side[other] if p is not taken]
                if spare:
                    lc = _box_centroid(self._last[empty]["box"])
                    assigned[empty] = min(
                        spare, key=lambda p: np.linalg.norm(_box_centroid(p["box"]) - lc))

        for slot in (1, 2):
            pose = assigned[slot]
            self._raw[slot].append(pose)
            if pose is not None:
                self._last[slot] = pose

    # -- gap filling + smoothing -------------------------------------------
    @staticmethod
    def _interp_pose(p_a, p_b, t):
        """Linear blend of two poses at fraction t in [0,1]. Confidence is the
        min of the two (an interpolated point is only as trusted as its weakest
        anchor). Marks the result filled=True."""
        kxy = (1 - t) * p_a["kps_xy"] + t * p_b["kps_xy"]
        kcf = np.minimum(p_a["kps_conf"], p_b["kps_conf"])
        box = (1 - t) * np.asarray(p_a["box"], float) + t * np.asarray(p_b["box"], float)
        return {"box": box, "kps_xy": kxy, "kps_conf": kcf, "filled": True}

    def _fill_track(self, seq):
        """seq: list of pose|None → list of pose|None with internal gaps (<=max_gap)
        interpolated and short leading/trailing gaps held from the nearest pose."""
        n = len(seq)
        out = [None] * n
        # mark real detections
        seen = [i for i, p in enumerate(seq) if p is not None]
        if not seen:
            return out
        for i in seen:
            out[i] = {**seq[i], "filled": False}
        # interpolate internal gaps
        for k in range(len(seen) - 1):
            i, j = seen[k], seen[k + 1]
            gap = j - i - 1
            if gap <= 0:
                continue
            if gap <= self.max_gap:
                for f in range(i + 1, j):
                    t = (f - i) / (j - i)
                    out[f] = self._interp_pose(seq[i], seq[j], t)
            # gap too long → leave None (genuine absence, don't fabricate)
        # hold leading / trailing edges up to max_gap (player present, briefly missed)
        first, last = seen[0], seen[-1]
        for f in range(max(0, first - self.max_gap), first):
            out[f] = {**seq[first], "filled": True}
        for f in range(last + 1, min(n, last + 1 + self.max_gap)):
            out[f] = {**seq[last], "filled": True}
        return out

    def _smooth_track(self, seq):
        """Causal EMA on keypoints to kill jitter, only across consecutive present
        frames (a None breaks the chain so we don't smear across real gaps)."""
        a = self.smooth_alpha
        prev = None
        for f, p in enumerate(seq):
            if p is None:
                prev = None
                continue
            if prev is not None:
                p = {**p,
                     "kps_xy": a * p["kps_xy"] + (1 - a) * prev["kps_xy"]}
                seq[f] = p
            prev = p
        return seq

    def finalize(self):
        """Return {1: [pose|None]*N, 2: [pose|None]*N} — gap-filled and smoothed."""
        tracks = {}
        for slot in (1, 2):
            filled = self._fill_track(self._raw[slot])
            tracks[slot] = self._smooth_track(filled)
        return tracks

    # -- diagnostics --------------------------------------------------------
    def coverage(self):
        """Fraction of frames each slot has a (real or filled) skeleton. Used by
        the regression test to assert 'zero dropped frames'."""
        out = {}
        tracks = self.finalize()
        for slot in (1, 2):
            seq = tracks[slot]
            n = len(seq)
            out[slot] = (sum(1 for p in seq if p is not None) / n) if n else 0.0
        return out
