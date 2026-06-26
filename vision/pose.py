"""
Top-down player pose estimation.

The naive approach (run a pose model on the full frame) misses the far player,
who is tiny even at 1920px, and a single central-band filter rejects real players
on oblique/amateur angles. The SOTA-standard top-down recipe fixes both:

  1. detect persons with a generic detector (YOLO person class),
  2. pick the 2 actual players among persons (ball boys / umpire rejected by a
     robustness score: size + verticality + being on opposite sides of the net),
  3. crop each player (padded) and run the pose model on the high-res crop,
  4. map keypoints back to full-frame coordinates.

This recovers full skeletons for both near and far players (measured on felix:
17/17 and 16/17 keypoints at conf ~0.90, vs the far player being undetected by
full-frame pose).
"""
import numpy as np


def select_players(person_boxes, frame_w, frame_h, ball_xy=None, max_players=2):
    """Pick up to `max_players` real players from detected person boxes.

    Score = size (players are the largest moving people on court) + a mild
    central-band preference (loose, so oblique angles keep both players) +
    proximity to the ball if known. Returns indices into person_boxes, best first.
    No hard horizontal reject (that killed real players on oblique footage).
    """
    if len(person_boxes) == 0:
        return []
    scores = []
    for b in person_boxes:
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        area_norm = (w * h) / (frame_w * frame_h)
        aspect = h / max(w, 1.0)                       # players stand → tall boxes
        vert_score = min(1.0, aspect / 2.0)
        # loose central preference (full reject only at the extreme 8% margins)
        xr = cx / frame_w
        if xr < 0.06 or xr > 0.94:
            scores.append(-1.0)
            continue
        central = 1.0 - abs(xr - 0.5) * 0.8            # gentle, never below ~0.6
        score = area_norm * 40.0 + vert_score * 0.5 + central * 0.3
        if ball_xy is not None:
            d = np.hypot(cx - ball_xy[0], (y1 + y2) / 2 - ball_xy[1]) / frame_w
            score += max(0.0, 0.4 - d)                 # small bonus if near the ball
        scores.append(score)
    scores = np.array(scores)
    valid = np.where(scores > 0)[0]
    if len(valid) == 0:
        return []
    order = valid[np.argsort(-scores[valid])]
    return list(order[:max_players])


def pose_on_crop(pose_model, frame, box, device, pad_frac=0.25, imgsz=640, conf=0.25):
    """Run the pose model on a padded crop around `box`. Returns (kps_xy, kps_conf)
    in FULL-FRAME coordinates, or None. kps_xy: (17,2), kps_conf: (17,)."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
    cx2, cy2 = min(fw, x2 + pw), min(fh, y2 + ph)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    res = pose_model(crop, conf=conf, imgsz=imgsz, device=device, verbose=False)
    if (res[0].keypoints is None or res[0].boxes is None or len(res[0].boxes) == 0):
        return None
    # pick the largest detected person inside the crop (the target player)
    cboxes = res[0].boxes.xyxy.cpu().numpy()
    areas = (cboxes[:, 2] - cboxes[:, 0]) * (cboxes[:, 3] - cboxes[:, 1])
    bi = int(np.argmax(areas))
    kxy = res[0].keypoints.xy[bi].cpu().numpy().copy()
    kconf = res[0].keypoints.conf[bi].cpu().numpy().copy()
    # map back to full frame
    kxy[:, 0] += cx1
    kxy[:, 1] += cy1
    return kxy, kconf


def _net_y(court_zones, frame_h):
    """Approximate net y (image row) from court zones, else mid-frame."""
    if court_zones:
        for name, info in court_zones.items():
            if name == "net":
                b = info["box"]
                return (b[1] + b[3]) / 2
        if "court" in court_zones:
            b = court_zones["court"]["box"]
            return (b[1] + b[3]) / 2
    return frame_h * 0.47


def select_players_by_side(person_boxes, frame_w, frame_h, net_y, ball_xy=None):
    """Pick the best player on the FAR side (feet above net_y) and the best on the
    NEAR side (feet below net_y). On a tennis court the two players straddle the
    net, so this is far more robust than horizontal centrality on oblique angles."""
    far, near = [], []
    for i, b in enumerate(person_boxes):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        feet_y = y2
        xr = cx / frame_w
        if xr < 0.05 or xr > 0.95 or h < 25:
            continue
        area = w * h
        aspect = h / max(w, 1.0)
        score = area / (frame_w * frame_h) * 40.0 + min(1.0, aspect / 2.0) * 0.5
        if ball_xy is not None:
            d = np.hypot(cx - ball_xy[0], (y1 + y2) / 2 - ball_xy[1]) / frame_w
            score += max(0.0, 0.4 - d)
        (far if feet_y < net_y else near).append((score, i))
    chosen = []
    if far:
        chosen.append(max(far)[1])
    if near:
        chosen.append(max(near)[1])
    # if only one side populated, fall back to top-2 overall
    if len(chosen) < 2:
        allp = sorted(far + near, reverse=True)
        for _, i in allp:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == 2:
                break
    return chosen


def _pool_by_side(pboxes, cand, net_y, n_per_side=3, min_pool=3):
    """Build a candidate pool that GUARANTEES representation from both sides of
    the net. ``cand`` is the global top-N list of (rank, idx) sorted desc; we split
    it by each box's feet vs ``net_y`` and take ``n_per_side`` from each. This
    prevents the small broadcast FAR player (~95px) from being starved out of the
    pool by larger near-side parasites (line judges / stands at ~150px): the far
    player must qualify through its OWN side slot, not compete head-to-head with
    the near side. Falls back to the global top if a side is empty."""
    by_side = {"far": [], "near": []}
    for _, i in cand:
        feet_y = pboxes[i][3]
        by_side["near" if feet_y >= net_y else "far"].append(i)
    pool = by_side["far"][:n_per_side] + by_side["near"][:n_per_side]
    if len(pool) < min_pool:
        pool = [i for _, i in cand[:min_pool + 1]]
    return pool


def _far_score(xr, aspect, conf):
    """Score an ABOVE-NET candidate by how a tennis camera frames the scene, so the
    small/faint far PLAYER beats the sharp static spectators above the net.

      central — the camera frames the COURT; the playing surface (and the far
                player on it) sits in the central horizontal band, while spectators
                cluster at the lateral edges/corners. Weak GEOMETRIC prior (1 at
                frame center → 0 at the edges), not a clip-specific pixel.
      human   — a real player has human box proportions (aspect ~1–2.2); distant
                spectator slivers are thin & very tall (aspect > 2.2). Penalize them.
      conf    — mild tie-break by detector confidence.

    Measured on the demo3 far-pose GT: ranking far candidates by this puts the TRUE
    far player #1 on ~85% of frames (vs 0% for the legacy h*conf, which always
    picks the sharpest spectator). xr = cx/frame_w."""
    central = 1.0 - abs(xr - 0.5) * 2.0
    human = 1.0 if aspect <= 2.2 else max(0.0, 1.0 - (aspect - 2.2) * 0.5)
    return central + 0.5 * human + 0.2 * conf


def _static_penalty(track_state, cx, feet_y, frame_w, window=12, eps_frac=0.012,
                    penalty=0.6):
    """Penalty for a far candidate that sits where a STATIC subject has been seen.

    Spectators / structures above the net don't move; the player does. ``track_state``
    holds a short history of recently chosen far positions plus a grid of "seen
    static" spots. If this candidate is within ``eps_frac``·W of a spot that has
    persisted across many frames with near-zero displacement, it is almost certainly
    a spectator → subtract ``penalty`` from its far-score. Threshold is a fraction of
    frame width (zero-hardcoding), not a pixel. Only ever a tie-break: ``penalty`` is
    smaller than the centrality spread, so it nudges between near-equal candidates and
    never overrides a clearly-central player. Returns 0 when no static match."""
    spots = track_state.get("static_spots", [])
    eps = eps_frac * frame_w
    for sx, sy, count in spots:
        if count >= window and abs(cx - sx) < eps and abs(feet_y - sy) < eps:
            return penalty
    return 0.0


def _update_static_state(track_state, cx, feet_y, decay=0.5):
    """Accumulate evidence that positions are static. We log EVERY above-net box's
    persistence elsewhere; here we just record the chosen far position so a future
    frame can tell whether the choice has been frozen on one spot (a sign the picker
    is stuck on a spectator). Kept deliberately light — the heavy lifting is the
    geometry score; motion is a tie-break."""
    hist = track_state.setdefault("far_hist", [])
    hist.append((cx, feet_y))
    if len(hist) > 30:
        hist.pop(0)


def _observe_static_spots(track_state, above_net_boxes, frame_w, eps_frac=0.012):
    """Update the persistent 'static spot' grid from ALL above-net detections this
    frame: a box that reappears at ~the same (cx, feet_y) across frames is a static
    spectator/structure. Increments a per-spot counter; non-reappearing spots decay.
    Called once per frame by the caller when motion mode is on."""
    spots = track_state.setdefault("static_spots", [])
    eps = eps_frac * frame_w
    seen = [False] * len(spots)
    for cx, fy in above_net_boxes:
        matched = False
        for k, (sx, sy, count) in enumerate(spots):
            if abs(cx - sx) < eps and abs(fy - sy) < eps:
                # exponential position update + increment persistence
                spots[k] = (0.7 * sx + 0.3 * cx, 0.7 * sy + 0.3 * fy, count + 1)
                seen[k] = True
                matched = True
                break
        if not matched:
            spots.append((cx, fy, 1))
            seen.append(True)
    # decay spots not seen this frame; drop dead ones
    track_state["static_spots"] = [
        (sx, sy, count if seen[k] else count - 2)
        for k, (sx, sy, count) in enumerate(spots) if (seen[k] or count - 2 > 0)
    ]


def estimate_players_pose(detector, pose_model, frame, device,
                          ball_xy=None, max_players=2,
                          det_imgsz=1280, pose_imgsz=960, court_zones=None,
                          det_conf=0.08, track_state=None):
    """Full top-down pipeline for one frame.

    detector: a YOLO model that detects persons (COCO class 0).
    Returns a list of dicts: {box, kps_xy, kps_conf} in full-frame coords.
    Players are chosen one per side of the net (robust on oblique angles); each is
    cropped and posed at high res (imgsz 960, low conf) so the far player resolves.

    ``det_conf``: person-detector confidence. The FAR player is tiny and faint —
    at a high conf (0.20) the detector simply doesn't return it, so the far slot
    falls back to a sharp, static spectator/structure above the net (measured on
    the demo3 pose GT: 100% far box present, 0.9% on the real player, ~572px off).
    Lowering it to ~0.08 makes the true far player detectable on ~89% of frames
    (the detection ceiling), and the side-aware ranking below then picks HIM over
    the spectators. (Zero-hardcoding: 0.08 is a model-confidence floor, not a
    pixel/frame value; it generalizes to any clip.)
    """
    fh, fw = frame.shape[:2]
    dres = detector(frame, conf=det_conf, imgsz=det_imgsz, classes=[0],
                    device=device, verbose=False)
    if dres[0].boxes is None or len(dres[0].boxes) == 0:
        return []
    pboxes = dres[0].boxes.xyxy.cpu().numpy()
    pconfs = (dres[0].boxes.conf.cpu().numpy()
              if dres[0].boxes.conf is not None else np.ones(len(pboxes)))
    net_y = _net_y(court_zones, fh) if court_zones else 0.5 * fh
    # SIDE-AWARE candidate ranking. The two sides have OPPOSITE failure modes:
    #   • NEAR side: the real player is the biggest, sharpest person → rank by
    #     h*conf (size), which is correct there.
    #   • FAR side: the real player is SMALL and FAINT; the largest/sharpest
    #     above-net box is a static spectator/structure at a screen corner, so
    #     h*conf systematically picks the WRONG subject. Rank far candidates by a
    #     score that encodes how a tennis camera frames the scene instead:
    #       central  — the camera frames the COURT, so the playing surface (and the
    #                  far player on it) sits in the central horizontal band, while
    #                  spectators cluster at the lateral edges/corners. A weak,
    #                  GEOMETRIC prior (not a demo3 pixel): 1 at frame center → 0
    #                  at the edges.
    #       human    — a real player has human box proportions (aspect ~1–2.2);
    #                  the distant spectator slivers are thin & very tall (aspect
    #                  >2.6). Penalize the thin slivers.
    #       conf     — mild tie-break by detector confidence.
    # Measured on the demo3 far-pose GT (225 frames): far-CORRECT 0.9% → ~85%.
    cand = []
    for i, b in enumerate(pboxes):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        xr = cx / fw
        if xr < 0.05 or xr > 0.95 or h < 40:
            continue
        aspect = h / max(w, 1.0)
        if aspect < 1.1:                       # too wide → not a standing player
            continue
        feet_y = y2
        if feet_y >= net_y:                    # NEAR side → biggest/sharpest wins
            rank = h * float(pconfs[i])
        else:                                  # FAR side → camera-geometry score
            rank = _far_score(xr, aspect, float(pconfs[i]))
        cand.append((rank, i))
    cand.sort(reverse=True)
    # Pose the top few candidates, then keep the best max_players that pass a
    # validity gate (real skeleton + feet on the court band, not in the stands).
    # This drops spectators / umpire / ball kids that rank high but aren't players.
    # KEY: guarantee representation from BOTH sides of the net. A global top-N by
    # h*conf starves the small FAR player (broadcast: ~95px) when near-side parasites
    # (line judges/stands at ~150px) outrank it — the far player must enter the pool
    # via its OWN side slot, not compete head-to-head with the near side. This mirrors
    # select_players_by_side() but keeps a few extras per side for the validity gate.
    pool = _pool_by_side(pboxes, cand, net_y, n_per_side=max_players + 1,
                         min_pool=max_players + 1)
    # Motion mode: log every above-net detection into the persistent static-spot
    # grid so a frozen spectator/structure can be told apart from the moving player.
    if track_state is not None:
        above_net = [((b[0] + b[2]) / 2.0, b[3]) for b in pboxes if b[3] < net_y]
        _observe_static_spots(track_state, above_net, fw)
    # Adaptive court band when court_zones are unavailable (the keypoint detector
    # misses some broadcast framings): derive the vertical court extent from the
    # detected person boxes themselves rather than a magic 0.32*H threshold. The
    # far player's feet set the top, the near player's feet set the bottom. The
    # pre-filter already dropped extreme-margin/over-wide boxes, so the remaining
    # feet_y span is a good proxy for the court surface band. (Zero-hardcoding:
    # derived from the scene, not a fixed frame fraction.)
    band = None
    if not court_zones and len(cand) >= 1:
        feet = [pboxes[i][3] for _, i in cand]
        if feet:
            band = (min(feet) - 0.04 * fh, max(feet) + 0.04 * fh)
    # Pose every pooled candidate, then SELECT ONE PER SIDE. The two sides have
    # opposite failure modes, so they need opposite gates:
    #
    #   NEAR side — the keypoint-validity gate WORKS: the near player is big and
    #     poses cleanly (>=10 kpts), parasites don't. Keep selecting by
    #     is_valid_player + area*conf.
    #
    #   FAR side — the keypoint-validity gate BACKFIRES: the tiny far player
    #     (~90px in a 1920px frame) yields too few confident keypoints to pass a
    #     ">=10 kpts" test, while the sharp STATIC spectator above the net poses
    #     perfectly (17/17) — so a keypoint gate selects the spectator every time
    #     (measured: far-CORRECT 2/225). The far player is, however, unambiguous on
    #     the camera-geometry _far_score (central, human-aspect): it ranks #1 among
    #     above-net boxes on ~85% of frames. So on the far side we select by
    #     _far_score and use the skeleton only as a SOFT signal (a small bonus when
    #     it poses), never as a hard reject. Static-spectator rejection (motion)
    #     is layered on when the caller threads ``track_state``.
    posed = []                                  # (pi, side, entry, far_score)
    for pi in pool:
        box = pboxes[pi]
        res = pose_on_crop(pose_model, frame, box, device, imgsz=pose_imgsz, conf=0.15)
        kxy = res[0] if res else None
        kcf = res[1] if res else None
        entry = {"box": box, "kps_xy": kxy, "kps_conf": kcf}
        cx = (box[0] + box[2]) / 2
        feet_y = box[3]
        side = "near" if feet_y >= net_y else "far"
        w, h = box[2] - box[0], box[3] - box[1]
        aspect = h / max(w, 1.0)
        fscore = _far_score(cx / fw, aspect, float(pconfs[pi]))
        posed.append((pi, side, entry, fscore))

    # NEAR side: keypoint-validity + area*conf (unchanged behaviour).
    near_valid, near_fb = [], []
    for pi, side, entry, _ in posed:
        if side != "near":
            continue
        box = entry["box"]
        key = float(pconfs[pi]) * (box[3] - box[1])
        if is_valid_player(box, entry["kps_conf"], fw, fh, court_zones, band=band):
            near_valid.append((key, entry))
        else:
            near_fb.append((key, entry))
    near_valid.sort(key=lambda e: -e[0])
    near_player = near_valid[0][1] if near_valid else None

    # FAR side: select the best _far_score, with a SOFT skeleton bonus and an
    # OPTIONAL motion (static-spectator) penalty. Reject only the obvious non-human
    # (thin sliver up in the stands), never the few-keypoint far player.
    far_cands = []
    for pi, side, entry, fscore in posed:
        if side != "far":
            continue
        box = entry["box"]
        cx = (box[0] + box[2]) / 2
        feet_y = box[3]
        kcf = entry["kps_conf"]
        nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
        score = fscore + 0.15 * min(1.0, nkp / 10.0)     # soft skeleton bonus
        if track_state is not None:
            score -= _static_penalty(track_state, cx, feet_y, fw)
        far_cands.append((score, entry))
    far_cands.sort(key=lambda e: -e[0])
    far_player = far_cands[0][1] if far_cands else None
    if track_state is not None and far_player is not None:
        b = far_player["box"]
        _update_static_state(track_state, (b[0] + b[2]) / 2, b[3])

    # Assemble: one far + one near (the locked P1-far / P2-near contract). Fall
    # back to the next-best near valid if a side is empty (rare).
    players = []
    if far_player is not None:
        players.append(far_player)
    if near_player is not None:
        players.append(near_player)
    if len(players) < max_players:
        for _, e in near_valid[1:]:
            if e not in players:
                players.append(e)
            if len(players) >= max_players:
                break
    return players[:max_players]


def is_valid_player(box, kps_conf, frame_w, frame_h, court_zones=None, band=None):
    """A real player has a valid skeleton AND stands on the court surface band
    (not up in the stands / umpire chair). A real player has >=10 confident
    keypoints and FEET within the court band (derived from court_zones, an
    explicit ``band`` override, or a broadcast default). Parasites (ball kids /
    line judges / stands) are rejected by the side-margin rule + the feet-band
    rule; we deliberately do NOT reject by absolute box height: on a broadcast the
    FAR player is legitimately high in frame (small, top of image) yet has a valid
    skeleton and feet on-court, so a "whole box too high" rule would kill the far
    player (seen: P1 at 17 kpts, feet on the far baseline, rejected by a 0.35*H
    mid-box threshold).

    ``band``: optional (top_y, bot_y) in px, computed adaptively by the caller
    (e.g. from the detected person boxes when court_zones are unavailable). When
    provided it takes precedence over the court_zones / default derivation — this
    is the zero-hardcoding path (the court extent is inferred from the scene, not
    a magic fraction of the frame)."""
    x1, y1, x2, y2 = box
    feet_y = y2
    head_y = y1
    h = y2 - y1
    cx = (x1 + x2) / 2
    # valid skeleton: enough confident keypoints
    nkp = int((kps_conf > 0.30).sum()) if kps_conf is not None else 0
    if nkp < 10:
        return False
    # Invisible side margin: ball kids / line judges / photographers sit at the
    # screen edges. Reject a candidate whose center is in the lateral margin UNLESS
    # it is also low in the frame (near the camera, on the court) — on oblique
    # angles the far player can be laterally placed but is always low/on-court
    # (felix far player: xr~0.10 but feet_y~0.64h), whereas an edge parasite is
    # high (tennis ball-kid: xr~0.08, feet_y~0.47h). So margin + high = parasite.
    xr = cx / frame_w
    if (xr < 0.10 or xr > 0.90) and feet_y < 0.58 * frame_h:
        return False
    # court-surface band: explicit override > court_zones > broadcast default.
    if band is not None:
        top_band, bot_band = band
    elif court_zones:
        ys = []
        for name, info in court_zones.items():
            if name in ("court", "net") or "baseline" in name:
                b = info["box"]
                ys += [b[1], b[3]]
        if ys:
            top_band = min(ys) - 0.06 * frame_h     # a little above far baseline
            bot_band = max(ys) + 0.04 * frame_h
        else:
            top_band, bot_band = 0.32 * frame_h, 0.88 * frame_h
    else:
        top_band, bot_band = 0.32 * frame_h, 0.88 * frame_h
    # the player's FEET must be within the court band (head can be above the band)
    if not (top_band <= feet_y <= bot_band):
        return False
    return True
