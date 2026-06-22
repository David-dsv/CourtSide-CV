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


def estimate_players_pose(detector, pose_model, frame, device,
                          ball_xy=None, max_players=2,
                          det_imgsz=1280, pose_imgsz=960, court_zones=None):
    """Full top-down pipeline for one frame.

    detector: a YOLO model that detects persons (COCO class 0).
    Returns a list of dicts: {box, kps_xy, kps_conf} in full-frame coords.
    Players are chosen one per side of the net (robust on oblique angles); each is
    cropped and posed at high res (imgsz 960, low conf) so the far player resolves.
    """
    fh, fw = frame.shape[:2]
    dres = detector(frame, conf=0.20, imgsz=det_imgsz, classes=[0],
                    device=device, verbose=False)
    if dres[0].boxes is None or len(dres[0].boxes) == 0:
        return []
    pboxes = dres[0].boxes.xyxy.cpu().numpy()
    pconfs = (dres[0].boxes.conf.cpu().numpy()
              if dres[0].boxes.conf is not None else np.ones(len(pboxes)))
    # The 2 players are the 2 largest, most-confident persons on court — far
    # taller/surer than spectators and ball kids (measured on felix: players
    # h=150/69 conf~0.87; spectators h<48 conf<0.45). Rank by height*confidence,
    # keep a vertical-aspect sanity filter, drop the extreme side margins.
    cand = []
    for i, b in enumerate(pboxes):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        xr = (x1 + x2) / 2 / fw
        if xr < 0.05 or xr > 0.95 or h < 40:
            continue
        aspect = h / max(w, 1.0)
        if aspect < 1.1:                       # too wide → not a standing player
            continue
        rank = h * float(pconfs[i])
        cand.append((rank, i))
    cand.sort(reverse=True)
    # Pose the top few candidates, then keep the best max_players that pass a
    # validity gate (real skeleton + feet on the court band, not in the stands).
    # This drops spectators / umpire / ball kids that rank high but aren't players.
    pool = [i for _, i in cand[:max_players + 2]]
    valid, fallback = [], []
    for pi in pool:
        box = pboxes[pi]
        res = pose_on_crop(pose_model, frame, box, device, imgsz=pose_imgsz, conf=0.15)
        kxy = res[0] if res else None
        kcf = res[1] if res else None
        entry = {"box": box, "kps_xy": kxy, "kps_conf": kcf}
        if is_valid_player(box, kcf, fw, fh, court_zones):
            valid.append((float(pconfs[pi]) * (box[3] - box[1]), entry))
        else:
            fallback.append((float(pconfs[pi]) * (box[3] - box[1]), entry))
    valid.sort(key=lambda e: -e[0])
    players = [e for _, e in valid[:max_players]]
    # Do NOT top up from fallback with clear parasites: it is better to show one
    # real player than to draw a spectator/umpire. Only top up with a fallback
    # candidate that has SOME skeleton and is not high in the frame.
    if len(players) < max_players:
        fallback.sort(key=lambda e: -e[0])
        for _, e in fallback:
            kcf = e["kps_conf"]
            nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
            b = e["box"]
            ycenter = (b[1] + b[3]) / 2 / fh
            xr = (b[0] + b[2]) / 2 / fw
            feet_yr = b[3] / fh
            # same invisible side-margin rule as is_valid_player: an edge candidate
            # that isn't low/on-court is a parasite (ball kid / line judge).
            edge_parasite = (xr < 0.10 or xr > 0.90) and feet_yr < 0.58
            if nkp >= 6 and ycenter >= 0.38 and not edge_parasite:
                players.append(e)
            if len(players) >= max_players:
                break
    return players


def is_valid_player(box, kps_conf, frame_w, frame_h, court_zones=None):
    """A real player has a valid skeleton AND stands on the court surface band
    (not up in the stands / umpire chair). Tuned from the felix diagnosis: real
    players have >=10 valid keypoints and box-center y_ratio in ~[0.38, 0.78];
    parasites are high in frame (y<0.38) or have no pose. The far player is small
    but DOES have a valid skeleton, so size is not used to reject."""
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
    # court-surface band: derive from court zones if available, else broadcast default
    if court_zones:
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
    # reject clearly-in-the-stands (whole box high in frame)
    if (head_y + feet_y) / 2 < 0.35 * frame_h:
        return False
    return True
