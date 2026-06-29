"""Audit a LIVE ball-track dump for "teleport-then-frozen" static-FP blobs.

This is the DETERMINISM thermometer for S6. A teleport-then-frozen blob is the
signature of a static false positive (a logo / court mark / corner blob) the ball
track "jumps into" and locks onto: the trajectory ENTERS the cluster by a large
jump (> jump_frac * diag) from the last valid ball position, then STAYS FROZEN
(every center within freeze_radius px of the cluster centroid) for >= min_run
frames. Such a frozen blob, read by vision/events.py `_direction_features`,
fabricates a `vy_flip` -> a MANDATORY (un-skippable) BOUNCE anchor that floods the
alternation DP and makes the event F1 oscillate run-to-run (S3 / hits-overfire).

A REAL ball at an arc apex is ALSO near-frozen, but it is NOT entered by a teleport
(the ball decelerates smoothly into the apex: entry step ~0.02*diag, not ~0.5*diag).
So the discriminator is the CONJUNCTION teleport-entry + frozen, never frozen alone.

Usage:
  python tools/ball_gt/blob_audit.py /tmp/dump.json
  python tools/ball_gt/blob_audit.py /tmp/dump.json --json   # machine-readable

Scale-free: every threshold is a fraction of the frame diagonal / fps, never a
pixel constant tied to one clip.
"""
import argparse
import json
import math
import sys


def find_teleport_frozen_blobs(centers, width, height, fps,
                               jump_frac=0.15, freeze_radius_frac=0.012,
                               min_run_frames=8):
    """Return a list of teleport-then-frozen blobs in a ball-center sequence.

    centers: list of [x,y] or None (one per frame, the RAW pre-spline track).
    A blob = a maximal run of consecutive non-None centers all within
    `freeze_radius` px of the run's first center, of length >= min_run, whose
    ENTRY (the step from the last valid center before the run into the run) is a
    jump of > jump_frac * diag.

    freeze_radius defaults to ~1.2% of diagonal (~26px @ 1080p) so it tracks
    resolution; min_run defaults to 8 frames (the prompt's signature). The
    entry-jump is measured from the last NON-None center before the run
    (skipping None gaps), so a blob entered via a 1-2f gap is still caught.
    """
    diag = math.hypot(width, height)
    jump_thresh = jump_frac * diag
    freeze_radius = freeze_radius_frac * diag
    n = len(centers)

    blobs = []
    i = 0
    last_valid = None          # last non-None center BEFORE the current position
    last_valid_idx = None
    while i < n:
        c = centers[i]
        if c is None:
            i += 1
            continue
        # Start a candidate frozen run anchored at i.
        run_start = i
        anchor = c
        j = i + 1
        # extend while subsequent centers stay within freeze_radius of the anchor
        # centroid (recomputed as a running mean to avoid drift sensitivity).
        sx, sy, cnt = float(c[0]), float(c[1]), 1
        while j < n and centers[j] is not None:
            cx, cy = float(centers[j][0]), float(centers[j][1])
            mx, my = sx / cnt, sy / cnt
            if math.hypot(cx - mx, cy - my) <= freeze_radius:
                sx += cx; sy += cy; cnt += 1
                j += 1
            else:
                break
        run_len = cnt
        centroid = (sx / cnt, sy / cnt)
        # entry jump: from the last valid center (before run_start) into the anchor.
        entry_jump = None
        if last_valid is not None:
            entry_jump = math.hypot(anchor[0] - last_valid[0],
                                    anchor[1] - last_valid[1])
        is_teleport = entry_jump is not None and entry_jump > jump_thresh
        is_frozen = run_len >= min_run_frames
        if is_teleport and is_frozen:
            blobs.append({
                "start": run_start,
                "end": run_start + run_len - 1,
                "run_len": run_len,
                "centroid": [round(centroid[0], 1), round(centroid[1], 1)],
                "entry_jump_px": round(entry_jump, 1),
                "entry_jump_frac": round(entry_jump / diag, 3),
                "entered_from": [round(last_valid[0], 1), round(last_valid[1], 1)],
                "entered_from_idx": last_valid_idx,
            })
            # advance past the whole frozen run
            last_valid = centers[run_start + run_len - 1]
            last_valid_idx = run_start + run_len - 1
            i = run_start + run_len
        else:
            # not a blob — advance one frame, update last_valid
            last_valid = c
            last_valid_idx = i
            i += 1
    return blobs, {"diag": round(diag, 1), "jump_thresh_px": round(jump_thresh, 1),
                   "freeze_radius_px": round(freeze_radius, 1),
                   "min_run_frames": min_run_frames}


def coverage(centers):
    n = len(centers)
    cov = sum(1 for c in centers if c is not None)
    return cov, n, (100.0 * cov / max(1, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="a --dump-ball-track JSON")
    ap.add_argument("--jump-frac", type=float, default=0.15)
    ap.add_argument("--freeze-radius-frac", type=float, default=0.012)
    ap.add_argument("--min-run", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    d = json.load(open(args.dump))
    centers = d["centers"]
    w, h, fps = d["width"], d["height"], d["fps"]
    blobs, meta = find_teleport_frozen_blobs(
        centers, w, h, fps, args.jump_frac, args.freeze_radius_frac, args.min_run)
    cov, n, cov_pct = coverage(centers)

    if args.json:
        print(json.dumps({"dump": args.dump, "coverage_pct": round(cov_pct, 1),
                          "coverage": cov, "n": n, "n_blobs": len(blobs),
                          "blobs": blobs, "meta": meta}))
        return

    print(f"dump: {args.dump}")
    print(f"coverage: {cov}/{n} = {cov_pct:.1f}%")
    print(f"thresholds: jump>{meta['jump_thresh_px']}px "
          f"({args.jump_frac}*diag), freeze<={meta['freeze_radius_px']}px, "
          f"run>={meta['min_run_frames']}f")
    print(f">>> teleport-then-frozen blobs: {len(blobs)}")
    for b in blobs:
        print(f"    f{b['start']}-{b['end']} (len {b['run_len']}) "
              f"@ {b['centroid']}  entered from {b['entered_from']} "
              f"(f{b['entered_from_idx']}) by {b['entry_jump_px']}px "
              f"= {b['entry_jump_frac']}*diag")
    return 0 if not blobs else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
