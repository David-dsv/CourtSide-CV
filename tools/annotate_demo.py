"""
Interactive demo annotator — mark the true players (P1 far / P2 near) and the
real / false bounces on a rendered demo clip, producing a ground-truth sidecar
that run_pipeline_8s.py can consume to force the annotation on this clip.

Schema (tennis_demo3_annotations.json)
--------------------------------------
{
  "video": "data/output/tennis_demo3.mp4",
  "source_video": "tennis.mp4",
  "source_start_sec": 73.0,      # -s used when rendering the demo
  "source_fps": 50,              # REAL fps of the source (auto-read if --source-fps matches)
  "fps": 50,                     # demo clip fps (read from the clip)
  "frame_offset_source": 3650,   # source_start_sec * source_fps  (= 73 * 50)
  "players": {
      "p1_far":  {"demo_frame": 0, "source_frame": 3650, "xy": [x, y]},
      "p2_near": {"demo_frame": 0, "source_frame": 3650, "xy": [x, y]}
  },
  "bounces_true":  [ {"demo_frame": 62, "source_frame": 3712, "xy": [x, y]}, ... ],
  "bounces_false": [ {"demo_frame": 12, "source_frame": 3662, "xy": [x, y]}, ... ],
  "strikes": [ {"demo_frame": 50, "source_frame": 3700, "xy": [x, y], "type": "forehand"}, ... ]
}

Usage
-----
    venv/bin/python tools/annotate_demo.py data/output/tennis_demo3.mp4 \\
        --source-start 73 --start-on-first-frame

  Resume / add only strikes (players + bounces already saved):
    venv/bin/python tools/annotate_demo.py data/output/tennis_demo3.mp4 \\
        --source-start 73 --start-phase strikes \\
        --load data/output/tennis_demo3_annotations.json

  (`--load` alone auto-advances to the first incomplete phase; `--start-phase`
   forces it. Prior phases are validated so you can't land in strikes without
   players/bounces.)

Controls
--------
  Global navigation (all phases):
    LEFT / RIGHT step one frame back / forward (pauses playback)
    UP / DOWN    jump 1 second back / forward
    , / .        same as LEFT / RIGHT (frame step, handy fallback)
    SPACE        play / pause (in BOUNCES / STRIKES)

  Phase 1/3 — PLAYERS (start):
    LEFT-CLICK   place P1 (far player) — then P2 (near player)
    u            undo last click
    ENTER        confirm players, switch to BOUNCES

  Phase 2/3 — BOUNCES:
    LEFT-CLICK   add a TRUE bounce (green square)
    RIGHT-CLICK  mark a FALSE bounce to suppress (red cross)
    u            undo last action
    ENTER        confirm, switch to STRIKES

  Phase 3/3 — STRIKES (racket→ball contacts):
    f            set active strike type to FOREHAND (cyan circle)
    b            set active strike type to BACKHAND (purple diamond)
    LEFT-CLICK   drop a strike of the active type
    u            undo last strike
    ENTER / q    finish & save
"""
import sys
import argparse
import json
from pathlib import Path

import cv2

PANEL_BG = (30, 28, 26)
ACCENT = (90, 200, 255)
P1_COL = (180, 90, 40)      # orange-ish (far)
P2_COL = (40, 140, 255)     # blue-ish (near)
TRUE_COL = (90, 230, 120)
FALSE_COL = (60, 60, 230)
FORE_COL = (120, 230, 255)  # cyan-ish  — forehand strike (circle)
BACK_COL = (200, 110, 255)  # purple-ish — backhand strike (diamond)


def fmt_time(frame_abs, fps):
    return f"f={frame_abs}  t={frame_abs / fps:.2f}s"


def draw_panel(canvas, phase, players, b_true, b_false, strikes, strike_type):
    h, w = canvas.shape[:2]
    pw = 460
    x0 = w - pw
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

    cv2.putText(canvas, "DEMO ANNOTATOR", (x0 + 16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def put(s, yy, scale=0.44, col=(200, 200, 200), thickness=1):
        cv2.putText(canvas, s, (x0 + 16, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness, cv2.LINE_AA)
        return yy + (24 if scale >= 0.5 else 22)

    y = 66
    if phase == "players":
        cv2.putText(canvas, "PHASE 1/3 — PLAYERS", (x0 + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
        y += 26
        y = put(f"P1 (far) :  {'set' if 'p1_far' in players else '>> CLICK (left)'}", y, 0.5, (230, 230, 230))
        y = put(f"P2 (near):  {'set' if 'p2_near' in players else '>> CLICK (left)'}", y, 0.5, (230, 230, 230))
        y += 10
        for s in ["LEFT-CLICK place", "LEFT/RIGHT frame step",
                  "UP/DOWN 1-sec jump", "u undo", "ENTER confirm & continue"]:
            y = put(s, y)
    elif phase == "bounces":
        cv2.putText(canvas, "PHASE 2/3 — BOUNCES", (x0 + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
        y += 26
        y = put(f"true  : {len(b_true)}", y, 0.5, TRUE_COL)
        y = put(f"false : {len(b_false)}", y, 0.5, FALSE_COL)
        y += 8
        for s in ["SPACE play/pause", "LEFT/RIGHT frame step",
                  "UP/DOWN 1-sec jump", "LEFT-CLICK  true bounce",
                  "RIGHT-CLICK false bounce", "u undo", "ENTER confirm & continue"]:
            y = put(s, y)
    else:  # strikes
        cv2.putText(canvas, "PHASE 3/3 — STRIKES", (x0 + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
        y += 26
        stype_col = FORE_COL if strike_type == "forehand" else BACK_COL
        y = put(f"active : {strike_type.upper()}", y, 0.5, stype_col)
        n_fore = sum(1 for s in strikes if s["type"] == "forehand")
        n_back = sum(1 for s in strikes if s["type"] == "backhand")
        y = put(f"forehand : {n_fore}", y, 0.5, FORE_COL)
        y = put(f"backhand : {n_back}", y, 0.5, BACK_COL)
        y += 8
        for s in ["SPACE play/pause", "LEFT/RIGHT frame step",
                  "UP/DOWN 1-sec jump", "f  forehand mode",
                  "b  backhand mode", "LEFT-CLICK drop strike",
                  "u undo", "ENTER/q finish"]:
            y = put(s, y)

    return x0


def draw_markers(canvas, players, b_true, b_false, strikes):
    for key, col in (("p1_far", P1_COL), ("p2_near", P2_COL)):
        if key in players:
            x, y = players[key]["xy"]
            cv2.drawMarker(canvas, (int(x), int(y)), col, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(canvas, (int(x), int(y)), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "P1 far" if key == "p1_far" else "P2 near",
                        (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for b in b_true:
        x, y = b["xy"]
        cv2.drawMarker(canvas, (int(x), int(y)), TRUE_COL, cv2.MARKER_SQUARE, 18, 2)
    for b in b_false:
        x, y = b["xy"]
        cv2.drawMarker(canvas, (int(x), int(y)), FALSE_COL, cv2.MARKER_TILTED_CROSS, 18, 2)
    # strikes: forehand = filled-ish ring (cyan), backhand = diamond (purple)
    for s in strikes:
        x, y = int(s["xy"][0]), int(s["xy"][1])
        col = FORE_COL if s["type"] == "forehand" else BACK_COL
        if s["type"] == "forehand":
            cv2.circle(canvas, (x, y), 14, col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 4, col, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(canvas, (x, y), col, cv2.MARKER_DIAMOND, 20, 2)


def main():
    ap = argparse.ArgumentParser(description="Interactive demo annotator")
    ap.add_argument("demo", help="Rendered demo clip (e.g. data/output/tennis_demo3.mp4)")
    ap.add_argument("--source-start", type=float, default=73.0,
                    help="-s (seconds) used to render the demo from the source video")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="fps of the source video (used to map demo_frame -> source_frame). "
                         "Default: the rendered clip's own fps (the clip is encoded at the "
                         "source fps, so they match).")
    ap.add_argument("--source-video", default=None,
                    help="Optional source video path to record in the sidecar")
    ap.add_argument("--start-on-first-frame", action="store_true",
                    help="Start paused on frame 0 (default: play immediately).")
    ap.add_argument("--load", default=None,
                    help="Load an existing annotations sidecar (JSON) to resume from: "
                         "players + bounces + strikes are restored, then you can edit/add.")
    ap.add_argument("--start-phase", choices=["players", "bounces", "strikes"], default=None,
                    help="Skip ahead to a phase (requires the prior phases to be complete, "
                         "e.g. --start-phase strikes to add strikes only).")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.demo)
    if not cap.isOpened():
        print(f"Could not open {args.demo}")
        sys.exit(1)
    clip_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = clip_fps or 30.0
    # The rendered clip is encoded at the source fps, so source_fps = clip fps
    # unless the caller overrides it (e.g. to reconcile a re-encoded clip).
    source_fps = args.source_fps if args.source_fps is not None else (clip_fps or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_offset_source = int(round(args.source_start * source_fps))

    state = {
        "phase": "players",
        "players": {},
        "b_true": [],
        "b_false": [],
        "strikes": [],
        "strike_type": "forehand",   # active racket-strike type ("forehand"/"backhand")
        "cur": 0,
        "paused": True,
        "frame": None,
        "need_read": True,
    }

    # --- Resume from an existing sidecar (--load) ---
    if args.load:
        with open(args.load) as f:
            prev = json.load(f)
        state["players"] = prev.get("players", {})
        state["b_true"] = prev.get("bounces_true", [])
        state["b_false"] = prev.get("bounces_false", [])
        state["strikes"] = prev.get("strikes", [])
        print(f"Loaded {args.load}")
        print(f"  players       : {len(state['players'])}/2")
        print(f"  bounces_true  : {len(state['b_true'])}")
        print(f"  bounces_false : {len(state['b_false'])}")
        print(f"  strikes       : {len(state['strikes'])}")

    # --- Skip ahead to a phase (--start-phase) ---
    # Validate prerequisites so we don't land in a phase with missing data.
    target_phase = args.start_phase
    if target_phase is None and args.load:
        # Auto-advance to the first incomplete phase when resuming.
        if len(state["players"]) >= 2 and (state["b_true"] or state["b_false"]):
            target_phase = "strikes"
        elif len(state["players"]) >= 2:
            target_phase = "bounces"
    if target_phase == "bounces" and len(state["players"]) < 2:
        print("--start-phase bounces needs both players annotated first.")
        sys.exit(1)
    if target_phase == "strikes":
        if len(state["players"]) < 2:
            print("--start-phase strikes needs both players annotated first.")
            sys.exit(1)
        if not (state["b_true"] or state["b_false"]):
            print("WARNING: starting strikes with no bounces annotated yet.")
    if target_phase:
        state["phase"] = target_phase

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            return None
        return fr

    def seek(delta):
        """Move by delta frames; clamps to [0, n_frames-1] and pauses playback."""
        nxt = max(0, min(n_frames - 1, state["cur"] + delta))
        if nxt == state["cur"]:
            return
        fr = read_frame(nxt)
        if fr is None:
            return
        state["frame"] = fr
        state["cur"] = nxt
        state["paused"] = True

    state["frame"] = read_frame(0)
    if state["frame"] is None:
        print("Could not read first frame")
        sys.exit(1)
    # Start paused when resuming into an annotation phase (bounces/strikes),
    # unless the caller explicitly asked to play immediately.
    start_paused = args.start_on_first_frame or state["phase"] in ("bounces", "strikes")
    state["paused"] = start_paused

    def next_player_slot():
        return None if len(state["players"]) >= 2 else ("p1_far" if "p1_far" not in state["players"] else "p2_near")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["phase"] == "players":
                slot = next_player_slot()
                if slot is None:
                    return
                state["players"][slot] = {
                    "demo_frame": int(state["cur"]),
                    "source_frame": int(state["cur"] + frame_offset_source),
                    "xy": [float(x), float(y)],
                }
            elif state["phase"] == "bounces":
                # true bounce
                if not state["paused"]:
                    return  # ignore clicks while playing for accuracy
                state["b_true"].append({
                    "demo_frame": int(state["cur"]),
                    "source_frame": int(state["cur"] + frame_offset_source),
                    "xy": [float(x), float(y)],
                })
            elif state["phase"] == "strikes":
                if not state["paused"]:
                    return
                state["strikes"].append({
                    "demo_frame": int(state["cur"]),
                    "source_frame": int(state["cur"] + frame_offset_source),
                    "xy": [float(x), float(y)],
                    "type": state["strike_type"],
                })
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["phase"] == "bounces" and state["paused"]:
                state["b_false"].append({
                    "demo_frame": int(state["cur"]),
                    "source_frame": int(state["cur"] + frame_offset_source),
                    "xy": [float(x), float(y)],
                })

    win = "Demo Annotator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, W), min(900, H))
    cv2.setMouseCallback(win, on_mouse)

    playing_msg = ""

    while True:
        # advance frame if playing (bounces + strikes phases are navigable/playable)
        if not state["paused"] and state["phase"] in ("bounces", "strikes"):
            nxt = state["cur"] + 1
            if nxt >= n_frames:
                state["paused"] = True
            else:
                fr = read_frame(nxt)
                if fr is None:
                    state["paused"] = True
                else:
                    state["frame"] = fr
                    state["cur"] = nxt

        canvas = state["frame"].copy()
        draw_markers(canvas, state["players"], state["b_true"], state["b_false"], state["strikes"])
        panel_x = draw_panel(canvas, state["phase"], state["players"],
                             state["b_true"], state["b_false"], state["strikes"], state["strike_type"])

        # time readout (bottom-left)
        src_frame = int(state["cur"] + frame_offset_source)
        ts = (f"demo f={state['cur']}/{n_frames - 1}   "
              f"source f={src_frame}   "
              f"demo t={state['cur'] / fps:.2f}s   "
              f"source t={src_frame / source_fps:.2f}s")
        cv2.rectangle(canvas, (0, H - 30), (panel_x, H), (0, 0, 0), -1)
        cv2.putText(canvas, ts, (12, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # play/pause indicator
        ind = "PLAYING" if not state["paused"] else "PAUSED"
        ind_col = (90, 230, 120) if not state["paused"] else (90, 200, 255)
        cv2.putText(canvas, ind, (panel_x - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ind_col, 2, cv2.LINE_AA)
        if playing_msg:
            cv2.putText(canvas, playing_msg, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        # waitKeyEx returns the full keycode incl. arrows (which are > 0xFF).
        # macOS Cocoa reports arrows as 63232-63235; X11 as 65361-65364. Cover both.
        full_key = cv2.waitKeyEx(1 if not state["paused"] else 20)
        key = full_key & 0xFF

        ARROW_STEP = {
            65361: -1, 63234: -1,   # LEFT  → −1 frame
            65363: +1, 63232: +1,   # RIGHT → +1 frame
        }
        ARROW_SEC = {
            65362: +1, 63230: +1,   # UP    → +1 second
            65364: -1, 63231: -1,   # DOWN  → −1 second
        }
        if full_key in ARROW_STEP:
            seek(ARROW_STEP[full_key])
        elif full_key in ARROW_SEC:
            seek(ARROW_SEC[full_key] * int(fps))
        elif key == ord(','):
            seek(-1)
        elif key == ord('.'):
            seek(+1)

        if key == ord('q') or key == 27:
            if state["phase"] == "players":
                print("Quit during players phase — nothing saved.")
                break
            else:
                playing_msg = "Saving…"
                save_sidecar(args, fps, source_fps, n_frames, W, H, frame_offset_source, state)
                break
        elif key == ord('u'):
            if state["phase"] == "players":
                if state["players"]:
                    last_key = list(state["players"].keys())[-1]
                    del state["players"][last_key]
            elif state["phase"] == "bounces":
                # undo the most recent bounce action
                if state["b_false"] and state["b_true"] and \
                        state["b_false"][-1]["demo_frame"] >= state["b_true"][-1]["demo_frame"]:
                    state["b_false"].pop()
                elif state["b_true"]:
                    state["b_true"].pop()
                elif state["b_false"]:
                    state["b_false"].pop()
            elif state["phase"] == "strikes":
                if state["strikes"]:
                    state["strikes"].pop()
        elif key == ord('f'):
            if state["phase"] == "strikes":
                state["strike_type"] = "forehand"
                playing_msg = "Strike mode: FOREHAND"
        elif key == ord('b'):
            if state["phase"] == "strikes":
                state["strike_type"] = "backhand"
                playing_msg = "Strike mode: BACKHAND"
        elif key in (13, 10):
            if state["phase"] == "players":
                if len(state["players"]) < 2:
                    playing_msg = "Need both P1 and P2 — keep clicking."
                else:
                    state["phase"] = "bounces"
                    state["paused"] = False
                    playing_msg = "Players locked. Annotate bounces — SPACE to pause."
            elif state["phase"] == "bounces":
                state["phase"] = "strikes"
                state["strike_type"] = "forehand"
                playing_msg = "Bounces done. Annotate strikes — f/b to set type."
            else:  # strikes
                save_sidecar(args, fps, source_fps, n_frames, W, H, frame_offset_source, state)
                break
        elif key == ord(' '):
            if state["phase"] in ("bounces", "strikes"):
                state["paused"] = not state["paused"]
                playing_msg = ""

    cap.release()
    cv2.destroyAllWindows()


def save_sidecar(args, fps, source_fps, n_frames, W, H, frame_offset_source, state):
    out = {
        "video": str(Path(args.demo).resolve()),
        "source_video": args.source_video or "",
        "source_start_sec": args.source_start,
        "source_fps": source_fps,
        "fps": fps,
        "n_frames": n_frames,
        "frame_wh": [W, H],
        "frame_offset_source": frame_offset_source,
        "players": state["players"],
        "bounces_true": state["b_true"],
        "bounces_false": state["b_false"],
        "strikes": state["strikes"],
    }
    demo_path = Path(args.demo)
    out_path = demo_path.with_name(demo_path.stem + "_annotations.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"\nSaved → {out_path}")
    print(f"  P1 far  : {'set' if 'p1_far' in state['players'] else 'MISSING'}")
    print(f"  P2 near : {'set' if 'p2_near' in state['players'] else 'MISSING'}")
    print(f"  bounces_true  : {len(state['b_true'])}")
    print(f"  bounces_false : {len(state['b_false'])}")
    n_fore = sum(1 for s in state["strikes"] if s["type"] == "forehand")
    n_back = sum(1 for s in state["strikes"] if s["type"] == "backhand")
    print(f"  strikes       : {len(state['strikes'])}  (forehand {n_fore} / backhand {n_back})")


if __name__ == "__main__":
    main()
