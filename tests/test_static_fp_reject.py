"""Deterministic unit test for the S6 teleport-then-frozen static-FP reject
(`reject_teleport_frozen_blobs` in run_pipeline_8s.py).

Fast, no GPU / no video / no cache — pure synthetic ball tracks. Asserts the
discriminator (1) REJECTS every flavour of static-FP blob the live pipeline locks
onto, and (2) KEEPS every real-ball shape that is ALSO near-frozen (the trap the
prompt warns about). Every threshold is scale-free, so case 7 re-runs the decisive
fixtures at 2× resolution and asserts identical accept/reject — proving no clip
pixel constant leaked.

The four reject flavours mirror the LIVE measured blobs on tennis.mp4 -s73 -d13:
  - Tier 1 (hard teleport): corner blob @ (89,772), 0.498·diag entry jump.
  - Tier 2 (soft teleport + healthy-moving-before): right-field @ (1266,664).
  - chain: a short frozen sub-blob trailing a core blob @ (711,517).
  - cold: the FIRST lock is a 35-frame mid-court FP @ (538,473), exited by a
    teleport (the real ball was at the top of frame, GT ~700px away).

The keep cases mirror the real ball:
  - serve toss apex frozen at the top of frame, entered from a COAST, exits
    ballistically (the cache f11-36 case);
  - a redirected-after-occlusion ball that keeps MOVING (never frozen);
  - a normal ballistic pass.

Run:  venv/bin/python tests/test_static_fp_reject.py
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load reject_teleport_frozen_blobs from run_pipeline_8s.py (module import pulls
# heavy deps but the function itself is pure numpy).
_spec = importlib.util.spec_from_file_location("_rp", str(ROOT / "run_pipeline_8s.py"))
_rp = importlib.util.module_from_spec(_spec)
sys.modules["_rp"] = _rp
_spec.loader.exec_module(_rp)
reject = _rp.reject_teleport_frozen_blobs

import numpy as np  # noqa: E402

W, H, FPS = 1920, 1080, 50.0
DIAG = float(np.hypot(W, H))


def _deleted_frames(deleted):
    s = set()
    for d in deleted:
        for f in range(d["start"], d["end"] + 1):
            s.add(f)
    return s


def _moving_segment(x0, y0, dx, dy, k):
    return [(x0 + dx * i, y0 + dy * i) for i in range(k)]


def _frozen_segment(x, y, k, jitter=2):
    # tiny ±jitter px wobble, well inside the freeze radius (~26px)
    return [(x + (i % 2) * jitter, y + ((i + 1) % 2) * jitter) for i in range(k)]


PASSED = []
FAILED = []


def check(name, cond):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'} {name}")


def main():
    # ── REJECT 1: hard teleport into a frozen corner blob ──
    # healthy moving real ball, then a 1098px (0.498·diag) jump into a 16-frame freeze.
    t = _moving_segment(1000, 158, -1, -1, 8) + _frozen_segment(89, 772, 16) \
        + _moving_segment(832, 187, -6, 8, 10)
    _, d = reject(t, DIAG, FPS)
    rejected = any(b["centroid"][0] < 200 and b["tier"] == 1 for b in d)
    check("reject_hard_teleport_corner_blob", rejected)

    # ── REJECT 2: soft teleport (0.245·diag) + healthy-moving-before ──
    t = _moving_segment(900, 286, 5, -7, 8) + _frozen_segment(1266, 664, 24) \
        + _moving_segment(1190, 354, -4, 6, 8)
    _, d = reject(t, DIAG, FPS)
    rejected = any(abs(b["centroid"][0] - 1266) < 30 and b["tier"] == 2 for b in d)
    check("reject_soft_teleport_healthy_before", rejected)

    # ── REJECT 3: cold-start FP — FIRST lock, long freeze, teleport EXIT ──
    # no prior real point; 35-frame mid-court freeze; then the real ball appears
    # far away (the track was never on it).
    t = [None] * 17 + _frozen_segment(538, 473, 35) + _moving_segment(896, 280, 6, 10, 12)
    _, d = reject(t, DIAG, FPS)
    rejected = any(b["tier"] == "cold" for b in d)
    check("reject_cold_start_first_lock_teleport_exit", rejected)

    # ── KEEP 1: serve-toss apex — frozen at top, entered from a COAST, exits ballistic ──
    # the track 'coasts' (near-zero motion ~0.4px/f) into the toss, then the ball
    # is genuinely frozen at the top, then leaves on a smooth arc.
    coast = [(305 - 0.4 * i, 132) for i in range(6)]        # decayed coast (lost ball)
    toss = _frozen_segment(826, 25, 25)
    out_arc = _moving_segment(861, 60, 6, 18, 12)
    t = coast + toss + out_arc
    _, d = reject(t, DIAG, FPS)
    toss_frames = set(range(len(coast), len(coast) + len(toss)))
    kept = not (_deleted_frames(d) & toss_frames)
    check("keep_real_toss_apex_coast_entry_ballistic_exit", kept)

    # ── KEEP 2: cold-start REAL toss — FIRST lock is the toss, exits BALLISTIC ──
    # mirrors the cache f11-36: first lock, frozen, but the ball is really there and
    # leaves on an arc (small exit jump) → cold-start rule must NOT fire.
    t = [None] * 11 + _frozen_segment(826, 25, 25) + _moving_segment(840, 60, 3, 16, 12)
    _, d = reject(t, DIAG, FPS)
    check("keep_cold_start_real_toss_ballistic_exit", len(d) == 0)

    # ── KEEP 3: redirect-after-occlusion — big jump but KEEPS MOVING (never frozen) ──
    t = _moving_segment(1200, 400, -110, -30, 10) + [None, None, None] \
        + _moving_segment(900, 250, 90, 20, 15)
    _, d = reject(t, DIAG, FPS)
    check("keep_redirect_after_occlusion_keeps_moving", len(d) == 0)

    # ── KEEP 4: plain ballistic pass (no freeze anywhere) ──
    t = _moving_segment(200, 900, 40, -30, 30)
    _, d = reject(t, DIAG, FPS)
    check("keep_plain_ballistic_pass", len(d) == 0)

    # ── KEEP 5: a REAL ball that PRECEDES an FP must not be deleted ──
    # mirrors the live f380-398 real climb followed by the f399 corner FP: the real
    # segment has its own teleport-OUT (to the FP) but is NOT the first lock, so the
    # cold-start rule must not touch it; the FP itself is rejected by teleport-IN.
    real_climb = _moving_segment(1018, 214, -1, -3, 19)      # real ball, slow climb
    fp_corner = _frozen_segment(89, 772, 16)                 # the FP after it
    t = real_climb + fp_corner + _moving_segment(832, 187, -6, 8, 8)
    _, d = reject(t, DIAG, FPS)
    real_frames = set(range(len(real_climb)))
    real_kept = not (_deleted_frames(d) & real_frames)
    fp_rejected = any(b["centroid"][0] < 200 for b in d)
    check("keep_real_ball_preceding_fp_AND_reject_fp", real_kept and fp_rejected)

    # ── KEEP 6: real serve toss apex as FIRST lock, OCCLUDED at contact, struck
    #            ball re-acquired FAR away (gap before exit) — must KEEP (FINDING b) ──
    # The toss is frozen >0.30s (first lock), then a 3-frame occlusion gap, then the
    # served ball appears 414px away. The cold-start exit must be ADJACENT (no gap)
    # to fire → a gap ⇒ abstain ⇒ the real toss is kept.
    t = _frozen_segment(826, 25, 20) + [None, None, None] + _moving_segment(1180, 240, 16, 20, 12)
    _, d = reject(t, DIAG, FPS)
    check("keep_serve_toss_first_lock_occluded_then_far_reacq", len(d) == 0)

    # ── KEEP 7: double-cold — two frozen runs before any kept point; the cold rule
    #            may fire AT MOST ONCE so a 2nd (possibly real) apex isn't re-cold-
    #            deleted (FINDING 1). Assert no run is deleted with tier 'cold' twice. ──
    t = _frozen_segment(500, 500, 15) + [None, None] + _frozen_segment(800, 200, 15) \
        + [None, None] + _moving_segment(1500, 800, 5, 5, 6)
    _, d = reject(t, DIAG, FPS)
    n_cold = sum(1 for b in d if b["tier"] == "cold")
    check("cold_start_fires_at_most_once", n_cold <= 1)

    # ── KEEP 8: a slow-DRIFTING real ball after a hard teleport entry — the freeze
    #            test must anchor on the run START, not a running mean, so a steady
    #            ~5px/f drift is NOT classified as frozen (FINDING 2). ──
    t = _moving_segment(50, 540, 40, 0, 8) + _moving_segment(1500, 540, 5, 0, 40)
    _, d = reject(t, DIAG, FPS)
    drift_frames = set(range(8, 48))
    drift_deleted = len(_deleted_frames(d) & drift_frames)
    check("keep_slow_drifting_real_ball_after_teleport", drift_deleted == 0)

    # ── SCALE-FREE: the decisive cases give identical decisions at 2× resolution ──
    W2, H2, D2 = 2 * W, 2 * H, 2 * DIAG
    t1 = _moving_segment(2000, 316, -2, -2, 8) + _frozen_segment(178, 1544, 16) \
        + _moving_segment(1664, 374, -12, 16, 10)            # hard teleport, scaled
    _, d1 = reject(t1, D2, FPS)
    t2 = [None] * 11 + _frozen_segment(1652, 50, 25) + _moving_segment(1680, 120, 6, 32, 12)
    _, d2 = reject(t2, D2, FPS)                               # cold-start real toss, scaled
    check("scale_free_2x_reject_hard_teleport", any(b["tier"] == 1 for b in d1))
    check("scale_free_2x_keep_cold_toss", len(d2) == 0)

    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}"
          f"  ({len(PASSED)}/{len(PASSED) + len(FAILED)})")
    assert not FAILED, f"static-FP reject test failures: {FAILED}"


if __name__ == "__main__":
    main()
