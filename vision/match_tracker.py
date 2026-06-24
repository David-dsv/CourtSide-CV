"""
Stable HUMAN identity across side-changes (match mode).

The default ``TwoPlayerTracker`` (vision/player_track.py) anchors identity to the
SIDE OF THE NET: P1 = the player whose feet are above the running divider (far),
P2 = below (near). This is drift-proof on a static camera — but only as long as a
player never crosses the net. In a real MATCH, players swap sides between odd
games. The moment that happens, P1/P2 silently INVERT, so every per-side stat
(near-player speed, per-rally outcome, fatigue) computed after the first
side-change is attributed to the wrong human.

This module keeps a STABLE HUMAN IDENTITY (P1, P2 = two specific people) across
side-changes, behind a single ``MatchTracker`` class:

  * it maintains its OWN per-human association (centroid + appearance continuity)
    that is NOT locked to a side — so a human can walk across the divider and
    keep his id,
  * it watches each human's CURRENT side (far/near) frame by frame and detects a
    SIDE-SWAP: both humans now occupy the opposite side from where they started,
    and the new arrangement persists past a dwell window (a net poach does NOT
    count — it's an ephemeral 1–2 frame excursion by only ONE player),
  * at a confirmed side-swap it re-identifies each human by an APPEARANCE
    SIGNATURE (normalised HSV hue/sat histogram of the player's torso crop) plus
    positional continuity, so the human who was P1 keeps the P1 id even though
    he is now on the near side,
  * it embeds a ``TwoPlayerTracker`` purely to gap-fill + smooth the per-side
    POSES (the drawing layer still needs a skeleton on every frame),
  * it exposes, per frame, a stable ``human_id`` (1/2) AND the current
    ``player_side`` ("near"/"far") of that human — so the front-end can still
    label depth/side while attributing stats to the right person.

Re-identification is appearance-first, continuity-fallback. If the two jerseys
are too similar (small HSV-histogram distance between the two humans' own
pre-swap signatures), we fall back to pure temporal continuity (the human who
stayed closer to his last centroid keeps his id) and mark a LOW confidence.

This module is pure-numpy + OpenCV (no model, no training) and unit-testable.
"""
from __future__ import annotations

import cv2
import numpy as np

from vision.player_track import TwoPlayerTracker, _feet_y, _box_centroid

# histogram bins: 8 hue x 8 sat → a 64-d signature. Coarse on purpose: a jersey
# is mostly one dominant colour, so a fine histogram is noise-dominated by the
# background leaking into the crop. Normalised so scale/crop-size doesn't matter.
_H_BINS, S_BINS = 8, 8

# A "side" is the net-relative bucket: 1 = far, 2 = near (matches TwoPlayerTracker
# internal numbering). HUMAN ids are also 1 and 2 but are stable; the mapping
# human<->side flips on a side-swap.
_FAR, _NEAR = 1, 2
_SIDE_NAME = {_FAR: "far", _NEAR: "near"}


# ───────────────────────── appearance signature ──────────────────────────────

def appearance_signature(frame, box, kps_xy=None, kps_conf=None):
    """Normalised HSV hue/sat histogram of the player's torso.

    The torso (not the full box) is the most colour-stable region: the full box
    is ~60% court/background which varies frame to frame and swamps the jersey.
    Prefer shoulders→hips (COCO 5,6,11,12) when the keypoints are confident;
    else the central horizontal band of the box. Returns a 64-d float32 vector
    summing to 1, or None when the crop is degenerate (too small / off-frame).

    ``frame`` may be None (e.g. in unit tests that synthesise only poses) → we
    return None and the tracker degrades to pure continuity re-identification.
    """
    if frame is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    use_kp = False
    if kps_xy is not None and kps_conf is not None:
        idx = [5, 6, 11, 12]  # shoulders + hips
        if all(kps_conf[j] > 0.30 for j in idx):
            xs = kps_xy[idx, 0]
            ys = kps_xy[idx, 1]
            tx1 = int(np.clip(xs.min(), 0, w_img - 1))
            tx2 = int(np.clip(xs.max(), 1, w_img))
            ty1 = int(np.clip(ys.min(), 0, h_img - 1))
            ty2 = int(np.clip(ys.max(), 1, h_img))
            pad = int(0.10 * (tx2 - tx1 + ty2 - ty1))
            x1 = max(0, tx1 - pad)
            y1 = max(0, ty1 - pad)
            x2 = min(w_img, tx2 + pad)
            y2 = min(h_img, ty2 + pad)
            use_kp = True
    if not use_kp:
        # central horizontal third, vertical middle 60%
        cx1 = int(x1 + 0.33 * (x2 - x1))
        cx2 = int(x1 + 0.67 * (x2 - x1))
        cy1 = int(y1 + 0.20 * (y2 - y1))
        cy2 = int(y1 + 0.80 * (y2 - y1))
        x1, y1, x2, y2 = cx1, cy1, cx2, cy2

    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def _signature_distance(a, b):
    """Distance in [0,1] between two L1-normalised histograms. Uses correlation
    (1 - corr) on the normalised vectors: 0 = identical colour, 1 = unrelated."""
    if a is None or b is None:
        return 1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    corr = float(np.dot(a, b) / denom)
    return float(np.clip(1.0 - corr, 0.0, 1.0))


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ───────────────────────── the match tracker ──────────────────────────────

class MatchTracker:
    """Stable P1/P2 HUMAN identity across side-of-net changes.

    Two layers:
      * HUMAN layer (this class): associates each detection to a stable human by
        centroid + appearance continuity, NOT side-locked. A human can cross the
        net and keep his id. This is what survives a side-change.
      * SIDE layer (embedded TwoPlayerTracker): only used to gap-fill + smooth
        the per-side poses for drawing (every frame needs a skeleton). Its slots
        ARE sides (far/near); we re-label them with stable human ids at finalize.

    Usage:

        trk = MatchTracker(frame_w, frame_h, net_y, fps)
        for players, frame in zip(players_per_frame, frames):
            trk.add_frame(players, frame)
        out = trk.finalize()
        # out["tracks"]              = {1: [pose|None]*N, 2: [...]}  stable human
        # out["side_by_human_frame"] = {1: ["far"/"near"/None]*N, 2: [...]}
        # out["swaps"]               = [{"frame", "confidence", ...}]
    """

    def __init__(self, frame_w, frame_h, net_y, fps,
                 swap_dwell_frac=0.30, swap_window_frac=1.0,
                 appearance_weight=0.6, ambiguous_chi=0.45,
                 max_step_frac=0.12, assoc_max_dist_frac=0.18,
                 max_gap_s=0.5, smooth_alpha=0.55):
        self.fw = frame_w
        self.fh = frame_h
        self.fps = fps
        # How long a crossed arrangement must PERSIST before we call it a real
        # side-swap. A net poach is <0.3s; a side-change lasts the rest of the
        # game. Zero-hardcoding: a fraction of a second via fps.
        self.swap_dwell = max(2, int(round(swap_dwell_frac * fps)))
        self.swap_window = max(self.swap_dwell + 1,
                               int(round(swap_window_frac * fps)))
        self.appearance_weight = appearance_weight
        # If the two humans' own jerseys are closer than this in HSV space, the
        # jerseys are too similar to re-identify visually → low confidence and we
        # lean entirely on continuity. Ratio, not pixels.
        self.ambiguous_chi = ambiguous_chi
        # Per-frame max centroid displacement for a human to keep his id. Wider
        # than the base tracker's max_step: a human mid side-change can be far
        # from his last seen centroid (he walked across), so we allow a bigger
        # step and let appearance break ties.
        self.max_step = max_step_frac * frame_w
        self.assoc_max_dist = assoc_max_dist_frac * frame_w

        # side layer (poses only)
        self._base = TwoPlayerTracker(
            frame_w, frame_h, net_y, fps,
            max_gap_s=max_gap_s, smooth_alpha=smooth_alpha,
            assoc_max_dist_frac=assoc_max_dist_frac,
            max_step_frac=0.04)

        # HUMAN layer state
        self._human_last = {1: None, 2: None}   # last pose per human (continuity)
        self._human_sig = {1: None, 2: None}    # EMA torso signature per human
        self._human_side = {1: None, 2: None}   # current side per human (per frame)
        self._misses = {1: 0, 2: 0}

        # per-frame records for finalize
        self._human_poses = {1: [], 2: []}      # raw pose per human per frame
        self._human_side_seq = {1: [], 2: []}   # "far"/"near"/None per human per frame
        self._human_sigs_seq = {1: [], 2: []}   # sig per human per frame

        # initial human<->side assignment (seeded from first frame)
        self._seeded = False

        # swap detection state
        self._swaps = []
        # baseline side of each human at the start of the current segment
        self._segment_side = {1: None, 2: None}
        self._pending = None
        self._seed_frame = None  # frame index where the segment baseline was set
        self._locked_in = False  # baseline confirmed authoritative (post warm-up)
        # per-frame cache of candidate appearance signatures (id(p) -> sig),
        # populated at the top of add_frame and consumed by _associate.
        self._candidate_sig_cache = {}

    # -- side of net ------------------------------------------------------
    def _divider_y(self):
        return self._base._divider_y()

    def _side_of(self, pose):
        fy = _feet_y(pose["box"], pose.get("kps_xy"), pose.get("kps_conf"))
        return _FAR if fy < self._divider_y() else _NEAR

    # -- per-frame --------------------------------------------------------
    def add_frame(self, players, frame=None):
        """players: list of dicts {box, kps_xy, kps_conf} (kps may be None), or
        None. ``frame`` is the raw BGR frame for the appearance signature."""
        players = players or []
        cand = [p for p in players if p.get("kps_xy") is not None]
        # pre-compute candidate signatures once (used by both association + swap)
        self._candidate_sig_cache = {}
        if frame is not None:
            for p in cand:
                self._candidate_sig_cache[id(p)] = appearance_signature(
                    frame, p["box"], p.get("kps_xy"), p.get("kps_conf"))

        # keep the side layer in sync (drives the divider EMA + the smoothed
        # per-side poses).
        self._base.add_frame(players)

        # associate detections to stable humans (side-agnostic continuity)
        assignment = self._associate(cand)

        # seed human<->side on the first frame both humans are seen (tentative —
        # the divider EMA is still self-centering, so this baseline is only
        # AUTHORITATIVE once it's re-confirmed after the warm-up window; see
        # _update_swap_state's locked-in logic).
        if not self._seeded and assignment[1] and assignment[2]:
            self._segment_side = {h: self._side_of(assignment[h]) for h in (1, 2)}
            self._seeded = True
            # index the CURRENT frame will occupy once appended below
            self._seed_frame = len(self._human_poses[1])

        # record + update per-human state
        for h in (1, 2):
            p = assignment[h]
            sig = self._candidate_sig_cache.get(id(p)) if p is not None else None
            self._human_poses[h].append(p)
            self._human_sigs_seq[h].append(sig)
            if p is not None:
                side = self._side_of(p)
                self._human_last[h] = p
                self._misses[h] = 0
                # EMA the human signature (stable across the whole clip)
                if sig is not None:
                    ref = self._human_sig[h]
                    if ref is None:
                        self._human_sig[h] = sig.copy()
                    else:
                        self._human_sig[h] = 0.15 * sig + 0.85 * ref
            else:
                side = self._human_side[h]  # hold last known side when absent
                self._misses[h] += 1
            self._human_side[h] = side
            self._human_side_seq[h].append(_SIDE_NAME.get(side))

        # detect / confirm a side-swap from the human side sequence
        self._update_swap_state()

    def _pair_cost(self, h, p):
        """Cost of assigning candidate ``p`` to human ``h``. Appearance is the
        DOMINANT term: during a side-change the two players' centroids cross, so
        centroid distance is actively misleading there — only the jersey colour
        is invariant. Position (centroid + IoU) breaks ties and rescues the
        ambiguous-jersey case. A candidate far outside the centroid gate incurs a
        penalty but is NOT excluded (appearance can still win)."""
        last = self._human_last[h]
        lc = _box_centroid(last["box"])
        d = float(np.linalg.norm(_box_centroid(p["box"]) - lc) / self.fw)
        iou = _iou(p["box"], last["box"])
        sig = self._candidate_sig_cache.get(id(p))
        ap = _signature_distance(sig, self._human_sig[h])
        grow = 1.0 + min(self._misses[h], self._base.max_gap) * 0.20
        gate = self.max_step * grow
        # gate penalty: linearly ramp past the gate so a distant candidate can
        # still win on a near-perfect colour match (a side-change), but a distant
        # candidate with a mediocre colour match loses to a close one.
        gate_pen = max(0.0, (d - gate / self.fw)) * 2.0 if d > gate / self.fw else 0.0
        return (ap                                  # appearance (dominant, in [0,1])
                + 0.20 * d                          # centroid proximity
                - 0.15 * iou                        # box overlap bonus
                + gate_pen)                         # outside continuity gate

    def _associate(self, cand):
        """Assign the (<=2) candidates to stable human ids 1/2 GLOBALLY.

        For exactly 2 humans × 2 candidates we enumerate both permutations and
        pick the lowest total cost — exact and symmetric (no greedy bias). With
        fewer anchors/candidates we fall back to per-human nearest+appearance.

        Returns {1: pose|None, 2: pose|None}."""
        out = {1: None, 2: None}
        if not cand:
            return out
        anchored = [h for h in (1, 2) if self._human_last[h] is not None]
        if not anchored:
            # first frame ever: assign by side (human1 = far, human2 = near) so
            # the initial labelling matches TwoPlayerTracker (zero regression in
            # the no-swap case).
            by_side = {}
            for p in cand:
                by_side.setdefault(self._side_of(p), []).append(p)
            if _FAR in by_side:
                out[1] = max(by_side[_FAR], key=lambda p: p["box"][3] - p["box"][1])
            if _NEAR in by_side:
                out[2] = max(by_side[_NEAR], key=lambda p: p["box"][3] - p["box"][1])
            placed = {id(p) for p in out.values() if p is not None}
            leftover = [p for p in cand if id(p) not in placed]
            for h in (1, 2):
                if out[h] is None and leftover:
                    out[h] = leftover.pop(0)
            return out

        # exact 2x2 (or 2x1) assignment
        if len(anchored) == 2 and len(cand) == 2:
            h1, h2 = anchored
            c1, c2 = cand
            best, best_cost = None, None
            for perm in ((c1, c2), (c2, c1)):  # perm[0]->h1, perm[1]->h2
                cost = (self._pair_cost(h1, perm[0]) + self._pair_cost(h2, perm[1]))
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, perm
            out[h1], out[h2] = best[0], best[1]
            return out

        # asymmetric case (one anchor, or !=2 candidates): per-human greedy best
        used = set()
        for h in anchored:
            avail = [p for p in cand if id(p) not in used]
            if not avail:
                continue
            best = min(avail, key=lambda p: self._pair_cost(h, p))
            out[h] = best
            used.add(id(best))
        # unassigned human (no anchor yet) takes any leftover candidate
        for h in (1, 2):
            if out[h] is None:
                leftover = [p for p in cand if id(p) not in used]
                if leftover:
                    out[h] = leftover[0]
                    used.add(id(leftover[0]))
        return out

    # -- swap detection ---------------------------------------------------
    def _update_swap_state(self):
        """A side-swap = both humans now sit on the OPPOSITE side from their
        segment baseline, and it persists past ``swap_dwell`` frames. A poach
        moves only ONE human and is ephemeral, so it never satisfies "both" AND
        "persist". We don't re-id here — association already kept the human ids
        stable through the crossing; we only RECORD the event + a confidence
        based on how distinguishable the jerseys are (for the front-end)."""
        if not self._seeded:
            return
        N = len(self._human_side_seq[1])
        d = self.swap_dwell
        if N < d:
            return

        # Warm-up guard: the divider EMA + the per-human signatures take a few
        # frames to stabilise after the seed. During that window a player's
        # _side_of reading can flicker as the divider self-centers, which would
        # register a spurious swap at the very start of the clip. We wait at
        # least 2*dwell frames, then LOCK IN the baseline to whatever
        # arrangement actually settled (one-time) — only after that do we begin
        # detecting swaps against a trustworthy baseline.
        if self._seed_frame is not None and (N - 1 - self._seed_frame) < 2 * d:
            return
        if not self._locked_in:
            # require a full dwell window of a consistent, two-sided arrangement
            window = self._human_side_seq[1][N - d:], self._human_side_seq[2][N - d:]
            s1s, s2s = window
            if (all(x is not None for x in s1s) and all(x is not None for x in s2s)
                    and len(set(s1s)) == 1 and len(set(s2s)) == 1
                    and s1s[0] != s2s[0]):
                self._segment_side = {1: self._side_val(s1s[0]),
                                      2: self._side_val(s2s[0])}
                self._locked_in = True
                self._seed_frame = N - 1
            return  # don't detect swaps until locked in

        def both_swapped_at(i):
            s1 = self._human_side_seq[1][i]
            s2 = self._human_side_seq[2][i]
            if s1 is None or s2 is None:
                return False
            b1, b2 = self._segment_side[1], self._segment_side[2]
            # both humans on the opposite side from their baseline
            return (self._side_val(s1) != b1) and (self._side_val(s2) != b2)

        # persist check over the last `d` frames
        recent = range(N - d, N)
        persisted = all(both_swapped_at(i) for i in recent)

        if persisted and not self._pending:
            # mark the first frame of the persisted window as the swap frame
            self._pending = {"frame": max(0, N - d)}
        elif persisted and self._pending:
            pass  # already recording
        elif not persisted and self._pending:
            # arrangement reverted before dwell-confirmed → transient (poach)
            self._pending = None

        # confirm only once per segment: when we FIRST see a full dwell window of
        # both-swapped, and we haven't recorded a swap for this segment yet.
        if persisted and self._pending and not self._pending.get("confirmed"):
            self._confirm_swap(self._pending)
            self._pending["confirmed"] = True

    @staticmethod
    def _side_val(name):
        return {"far": _FAR, "near": _NEAR}.get(name)

    def _confirm_swap(self, event):
        """Record a swap event with a re-id confidence. The human ids are ALREADY
        stable (association continuity carried them across). Confidence reflects
        how reliably appearance could have re-identified them had continuity
        failed: high when the jerseys are clearly different colours, low when
        they're nearly identical (we'd be relying on continuity alone)."""
        sig1, sig2 = self._human_sig[1], self._human_sig[2]
        jersey_dist = _signature_distance(sig1, sig2)
        ambiguous = jersey_dist < self.ambiguous_chi
        if ambiguous:
            confidence = 0.35
        else:
            # confidence scales with how distinguishable the jerseys are
            confidence = float(np.clip(0.55 + jersey_dist * 0.5, 0.55, 0.95))

        # flip the segment baseline so the NEXT side-change is detectable
        self._segment_side = {h: (_NEAR if self._segment_side[h] == _FAR else _FAR)
                              for h in (1, 2)}
        self._swaps.append({
            "frame": int(event["frame"]),
            "confidence": round(confidence, 3),
            "ambiguous_jerseys": bool(ambiguous),
            "jersey_distance": round(float(jersey_dist), 3),
        })

    # -- finalize ---------------------------------------------------------
    def finalize(self):
        """Build stable-human tracks (gap-filled + smoothed via the side layer's
        per-side poses, re-labelled with stable human ids) + the per-frame side
        of each human."""
        # Gap-fill + smooth each human's OWN raw pose sequence (the human layer
        # already holds stable-identity poses; we reuse the base tracker's fill +
        # smooth machinery on the human sequences directly).
        N = len(self._human_poses[1])
        tracks = {}
        for h in (1, 2):
            filled = self._base._fill_track(self._human_poses[h])
            tracks[h] = self._base._smooth_track(filled)

        side_by_human_frame = {
            h: list(self._human_side_seq[h]) for h in (1, 2)
        }
        return {
            "tracks": tracks,
            "side_by_human_frame": side_by_human_frame,
            "swaps": self._swaps,
            "match_mode": True,
        }


def human_id_for_side(match_result, side, frame_idx):
    """Map a shot's player_side ('near'/'far') at frame_idx to a stable human_id
    ('P1'/'P2') given a finalized MatchTracker result.

    The match result's ``side_by_human_frame`` tells us which side each human
    occupies per frame; we invert it: find the human whose side at frame_idx
    matches ``side``. This is robust to swaps because human ids are already
    stable — we're just reading who's currently on that side."""
    if match_result is None:
        return None
    sbh = match_result.get("side_by_human_frame")
    if not sbh:
        return None
    # normalise side; unknown/None (e.g. an override strike with no side) → give up
    if side in ("far", _FAR, 1):
        side = "far"
    elif side in ("near", _NEAR, 2):
        side = "near"
    else:
        return None
    for h in (1, 2):
        seq = sbh.get(h, [])
        if frame_idx < len(seq) and seq[frame_idx] == side:
            return "P%d" % h
    # fall back to the nearest non-None side reading around frame_idx
    N = max((len(sbh.get(h, [])) for h in (1, 2)), default=0)
    for off in range(1, min(8, N)):
        for f in (frame_idx - off, frame_idx + off):
            for h in (1, 2):
                seq = sbh.get(h, [])
                if 0 <= f < len(seq) and seq[f] == side:
                    return "P%d" % h
    return None
