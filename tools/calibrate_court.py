"""
Interactive court calibration — click the court landmarks on a video's first frame
to produce a metric homography sidecar (<video>.court.json).

This is the deployable, generalize-to-any-clip version of the felix calibration:
for a static-camera clip (a phone on a tripod — the amateur target) the court
homography is constant, so the user marks the court ONCE and every frame gets a
perspective-correct minimap, real km/h, and bounce depth.

Usage
-----
    python tools/calibrate_court.py path/to/video.mp4
    python tools/calibrate_court.py path/to/video.mp4 --frame 120   # pick a clearer frame

Controls
--------
    • The tool prompts for each landmark in turn (the panel lists what to click).
    • LEFT-CLICK to place the current landmark; it advances to the next.
    • 'u'   undo the last point
    • 's'   skip the current landmark (place only what you can see clearly)
    • 'z'   zoom toggle (2× around the cursor for precise corner clicks)
    • ENTER  finish — needs >=4 points; solves + reports RMS, saves the sidecar
    • 'q' / ESC  quit without saving

You only NEED the 4 outer doubles corners (far/near × left/right baseline), but
more well-spread points (service lines, net posts) give a better-conditioned,
more accurate homography — especially on grazing/oblique angles.
"""
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.court_calibrator import (  # noqa: E402
    WORLD_LANDMARKS, homography_from_points, save_calibration, img_to_world)

# Order the user is prompted in: the 4 corners first (the essential quad), then
# the higher-value mid-court anchors, then the rest.
PROMPT_ORDER = [
    ("far_bl_left",   "FAR baseline × LEFT doubles sideline (top-left court corner)"),
    ("far_bl_right",  "FAR baseline × RIGHT doubles sideline (top-right court corner)"),
    ("near_bl_right", "NEAR baseline × RIGHT doubles sideline (bottom-right, closest)"),
    ("near_bl_left",  "NEAR baseline × LEFT doubles sideline (bottom-left, closest)"),
    ("net_left",      "NET × LEFT doubles sideline (left net post)"),
    ("net_right",     "NET × RIGHT doubles sideline (right net post)"),
    ("far_svc_left",  "FAR service line × LEFT singles sideline"),
    ("far_svc_right", "FAR service line × RIGHT singles sideline"),
    ("near_svc_left", "NEAR service line × LEFT singles sideline"),
    ("near_svc_right","NEAR service line × RIGHT singles sideline"),
    ("far_svc_center","FAR service line × CENTER service line (the T, far)"),
    ("near_svc_center","NEAR service line × CENTER service line (the T, near)"),
]

ACCENT = (90, 200, 255)
DONE_COL = (90, 230, 120)
PANEL_BG = (30, 28, 26)


def _draw_panel(canvas, idx, points):
    """Side panel listing landmarks: done (green), current (accent), pending (grey)."""
    h, w = canvas.shape[:2]
    pw = 430
    x0 = w - pw
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.putText(canvas, "COURT CALIBRATION", (x0 + 16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(points)} placed  (need >=4)", (x0 + 16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    y = 92
    for i, (name, desc) in enumerate(PROMPT_ORDER):
        placed = name in points
        if placed:
            col, mark = DONE_COL, "[x]"
        elif i == idx:
            col, mark = ACCENT, ">> "
        else:
            col, mark = (150, 150, 150), "[ ]"
        cv2.putText(canvas, f"{mark} {name}", (x0 + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        y += 22
    y += 10
    for line in ["LEFT-CLICK place   u undo   s skip",
                 "z zoom   ENTER finish   q quit"]:
        cv2.putText(canvas, line, (x0 + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        y += 22
    # current target description, big, bottom-left
    if idx < len(PROMPT_ORDER):
        cv2.putText(canvas, "Click: " + PROMPT_ORDER[idx][1], (20, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2, cv2.LINE_AA)


def _draw_points(canvas, points):
    for name, (x, y) in points.items():
        cv2.drawMarker(canvas, (int(x), int(y)), DONE_COL, cv2.MARKER_CROSS, 16, 2)
        cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, name.replace("_", " "), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, DONE_COL, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Interactive court calibration")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--frame", type=int, default=0,
                    help="Frame index to calibrate on (pick one with unobstructed lines)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        sys.exit(1)
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read frame")
        sys.exit(1)
    H0, W0 = frame.shape[:2]

    state = {"idx": 0, "points": {}, "cursor": (0, 0), "zoom": False}

    def advance():
        # move to the next not-yet-placed landmark
        while state["idx"] < len(PROMPT_ORDER) and PROMPT_ORDER[state["idx"]][0] in state["points"]:
            state["idx"] += 1

    def on_mouse(event, x, y, flags, _):
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and state["idx"] < len(PROMPT_ORDER):
            name = PROMPT_ORDER[state["idx"]][0]
            state["points"][name] = (float(x), float(y))
            state["idx"] += 1
            advance()

    win = "Court Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, W0), min(900, H0))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        _draw_points(canvas, state["points"])
        _draw_panel(canvas, state["idx"], state["points"])
        # zoom inset for precise clicks
        if state["zoom"]:
            cx, cy = state["cursor"]
            r = 60
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(W0, cx + r), min(H0, cy + r)
            if x2 > x1 and y2 > y1:
                inset = cv2.resize(frame[y1:y2, x1:x2], (240, 240),
                                   interpolation=cv2.INTER_NEAREST)
                cv2.line(inset, (120, 0), (120, 240), (0, 255, 255), 1)
                cv2.line(inset, (0, 120), (240, 120), (0, 255, 255), 1)
                canvas[10:250, 10:250] = inset
                cv2.rectangle(canvas, (10, 10), (250, 250), (255, 255, 255), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            print("Quit without saving.")
            break
        elif key == ord('u') and state["points"]:
            last = list(state["points"].keys())[-1]
            del state["points"][last]
            # rewind the prompt to the undone landmark so re-clicking refills it
            state["idx"] = next(i for i, (n, _) in enumerate(PROMPT_ORDER) if n == last)
        elif key == ord('s') and state["idx"] < len(PROMPT_ORDER):
            state["idx"] += 1
            advance()
        elif key == ord('z'):
            state["zoom"] = not state["zoom"]
        elif key in (13, 10):  # ENTER
            if len(state["points"]) < 4:
                print(f"Need >=4 points, have {len(state['points'])}. Keep clicking.")
                continue
            try:
                homo = homography_from_points(state["points"])
            except Exception as e:
                print(f"Homography failed: {e}. Add/adjust points.")
                continue
            print(f"\nHomography solved: {homo['n_used']} points, "
                  f"RMS reprojection = {homo['rms_px']:.2f}px")
            # quick sanity: project the 4 image corners to meters
            for name in ("far_bl_left", "near_bl_right"):
                if name in state["points"]:
                    x, y = state["points"][name]
                    Xm, Ym = img_to_world(homo["H"], x, y)
                    print(f"  {name} → ({Xm:+.2f}, {Ym:+.2f}) m")
            path = save_calibration(args.video, state["points"], frame_wh=(W0, H0))
            print(f"Saved calibration → {path}")
            print("The pipeline will now auto-load it (metric homography enabled).")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
