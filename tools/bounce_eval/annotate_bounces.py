"""
Bounce annotation helper — step through a clip frame by frame and mark which
frames contain a real ball bounce. Writes a ground-truth JSON file in the
shared bounce format (see bounce_io.py).

This is a local dev tool: a simple OpenCV window. Reuses utils.video_utils
.VideoReader for all video access (seek/read/fps) rather than reinventing it.

Usage
-----
    python tools/bounce_eval/annotate_bounces.py data/clips/match1.mp4

    # only a slice of the video — e.g. the first 30 seconds
    python tools/bounce_eval/annotate_bounces.py data/clips/match1.mp4 -s 0 -d 30

    # a window in the middle (10s -> 40s), via duration or explicit end
    python tools/bounce_eval/annotate_bounces.py data/clips/match1.mp4 -s 10 -d 30
    python tools/bounce_eval/annotate_bounces.py data/clips/match1.mp4 -s 10 -e 40

    # custom output path (default: tests/fixtures/bounces/<clip>.bounces.json)
    python tools/bounce_eval/annotate_bounces.py data/clips/match1.mp4 \\
        -o tests/fixtures/bounces/match1.bounces.json

Re-running with an existing output file loads it so you can resume / edit.
Frame numbers in the JSON are ABSOLUTE in the source video regardless of the
window; frame_offset records the window start.

Controls (also drawn on-screen)
-------------------------------
    d / RIGHT     next frame              a / LEFT     previous frame
    f             jump +10 frames         b            jump -10 frames
    g             jump to frame (type in terminal)
    SPACE         mark current frame as a bounce
    click         (after marking) set ball x,y for the current frame's bounce
    1 / 2 / 3     set depth of current frame's bounce: deep / mid / short
    x             REMOVE the bounce at (or nearest to) the current frame
    u             undo the most recent add
    s             save        q / ESC   quit (prompts to save if unsaved)

Removal is position-independent: navigate back to any marked frame (the HUD
lists nearby marks) and press `x` to delete it — no need to restart.
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.video_utils import VideoReader  # noqa: E402
from tools.bounce_eval.bounce_io import (  # noqa: E402
    make_bounce, save_bounce_file, load_bounce_file, DEPTH_LABELS,
)

DEPTH_BY_KEY = {ord("1"): "deep", ord("2"): "mid", ord("3"): "short"}
DEPTH_COLOR = {"deep": (0, 220, 0), "mid": (0, 200, 220), "short": (0, 120, 255)}


def parse_args():
    p = argparse.ArgumentParser(description="Annotate ball bounces in a clip.")
    p.add_argument("video", help="Path to the video clip")
    p.add_argument("-o", "--output", default=None,
                   help="Output JSON (default: tests/fixtures/bounces/<clip>.bounces.json)")
    p.add_argument("--annotator", default="", help="Annotator name to record")
    p.add_argument("-s", "--start", type=float, default=0.0,
                   help="Start time in seconds (annotate only from here)")
    p.add_argument("-d", "--duration", type=float, default=None,
                   help="Window length in seconds from --start (default: to end of video)")
    p.add_argument("-e", "--end", type=float, default=None,
                   help="End time in seconds (alternative to --duration)")
    return p.parse_args()


def default_output(video: str) -> str:
    stem = Path(video).stem
    root = Path(__file__).resolve().parents[2]
    return str(root / "tests" / "fixtures" / "bounces" / f"{stem}.bounces.json")


def nearest_mark(marks: dict, frame: int):
    """Return the frame key of the mark nearest to `frame`, or None."""
    if not marks:
        return None
    return min(marks, key=lambda f: abs(f - frame))


def draw_hud(frame, idx, window, marks, status):
    start_frame, last_frame = window
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 88), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    marked_here = idx in marks
    head_color = (0, 255, 0) if marked_here else (255, 255, 255)
    cv2.putText(frame, f"frame {idx}  [{start_frame}-{last_frame}]   bounces: {len(marks)}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2, cv2.LINE_AA)

    if marked_here:
        b = marks[idx]
        depth = b.get("depth", "-")
        pos = f"({b['x']},{b['y']})" if "x" in b else "(no pos)"
        cv2.putText(frame, f"BOUNCE here  depth={depth}  {pos}",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    DEPTH_COLOR.get(depth, (0, 255, 0)), 2, cv2.LINE_AA)

    # nearby marks, so the user can find a mislabeled frame to remove
    near = sorted(marks, key=lambda f: abs(f - idx))[:5]
    near = sorted(near)
    if near:
        label = "  ".join(("[%d]" % f) if f == idx else str(f) for f in near)
        cv2.putText(frame, f"marks near: {label}", (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, status, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # draw the ball position if set for this frame
    if marked_here and "x" in marks[idx]:
        b = marks[idx]
        cv2.circle(frame, (b["x"], b["y"]), 8, (0, 255, 0), 2)
    return frame


def main():
    args = parse_args()
    out_path = args.output or default_output(args.video)

    reader = VideoReader(args.video)
    total = reader.frame_count
    fps = reader.fps

    # ─── Window: bound navigation to [start_frame, end_frame). Frame numbers
    #     written to the JSON stay ABSOLUTE in the source video; frame_offset
    #     records where the window starts so predictions align directly. ───
    start_frame = max(0, min(int(args.start * fps), total - 1))
    if args.duration is not None:
        end_frame = min(start_frame + int(args.duration * fps), total)
    elif args.end is not None:
        end_frame = min(int(args.end * fps), total)
    else:
        end_frame = total
    end_frame = max(end_frame, start_frame + 1)  # always at least one frame
    last_frame = end_frame - 1                    # last navigable index (inclusive)

    # marks: {absolute_frame: bounce_record}
    marks = {}
    frame_offset = start_frame
    annotator = args.annotator
    if Path(out_path).exists():
        doc = load_bounce_file(out_path)
        annotator = annotator or doc.get("annotator", "")
        for b in doc["bounces"]:
            marks[b["frame"]] = dict(b)
        prev_off = doc.get("frame_offset", 0)
        print(f"Loaded {len(marks)} existing bounces from {out_path}")
        if prev_off != frame_offset:
            print(f"  note: existing file frame_offset={prev_off}, this session uses "
                  f"{frame_offset} (--start {args.start:g}s). Frames stay absolute; "
                  f"offset will be rewritten on save.")

    add_history = []   # frames added this session, for undo
    dirty = False
    idx = start_frame
    if args.start or args.duration is not None or args.end is not None:
        print(f"Window: frames {start_frame}..{last_frame} "
              f"({args.start:g}s -> {end_frame/fps:.1f}s @ {fps:.0f}fps)")

    win = "Bounce annotator — press q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, mx, my, flags, _param):
        nonlocal dirty
        if event == cv2.EVENT_LBUTTONDOWN and idx in marks:
            marks[idx]["x"], marks[idx]["y"] = int(mx), int(my)
            dirty = True

    cv2.setMouseCallback(win, on_mouse)

    def save_now():
        save_bounce_file(out_path, video=args.video, fps=fps,
                         bounces=list(marks.values()), frame_offset=frame_offset,
                         annotator=annotator)
        print(f"Saved {len(marks)} bounces -> {out_path}")

    def show(status=""):
        reader.seek(idx)
        ret, frame = reader.read_frame()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = draw_hud(frame, idx, (start_frame, last_frame), marks, status)
        cv2.imshow(win, frame)

    status = "ready"
    show(status)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q / ESC
            if dirty:
                print("Unsaved changes. Press s to save, or q again to discard.")
                if (cv2.waitKey(0) & 0xFF) == ord("s"):
                    save_now()
                    dirty = False
            break

        elif key in (ord("d"), 83):   # next  (83 = right arrow on many builds)
            idx = min(last_frame, idx + 1); status = ""
        elif key in (ord("a"), 81):   # prev  (81 = left arrow)
            idx = max(start_frame, idx - 1); status = ""
        elif key == ord("f"):
            idx = min(last_frame, idx + 10); status = "+10"
        elif key == ord("b"):
            idx = max(start_frame, idx - 10); status = "-10"
        elif key == ord("g"):
            try:
                target = int(input(f"Jump to frame [{start_frame}-{last_frame}]: ").strip())
                idx = max(start_frame, min(last_frame, target)); status = f"jumped to {idx}"
            except (ValueError, EOFError):
                status = "bad frame number"

        elif key == ord(" "):         # mark bounce
            if idx not in marks:
                marks[idx] = make_bounce(idx)
                add_history.append(idx)
                dirty = True
                status = f"marked bounce @ {idx} (click to set pos, 1/2/3 for depth)"
            else:
                status = f"already marked @ {idx}"

        elif key in DEPTH_BY_KEY:     # set depth
            if idx in marks:
                marks[idx]["depth"] = DEPTH_BY_KEY[key]
                dirty = True
                status = f"depth={DEPTH_BY_KEY[key]} @ {idx}"
            else:
                status = "no bounce here to tag — press SPACE first"

        elif key == ord("x"):         # remove bounce at/near current frame
            target = idx if idx in marks else nearest_mark(marks, idx)
            if target is not None:
                del marks[target]
                if target in add_history:
                    add_history.remove(target)
                dirty = True
                status = f"removed bounce @ {target}"
            else:
                status = "no bounces to remove"

        elif key == ord("u"):         # undo last add
            if add_history:
                last = add_history.pop()
                marks.pop(last, None)
                dirty = True
                status = f"undid add @ {last}"
            else:
                status = "nothing to undo"

        elif key == ord("s"):
            save_now()
            dirty = False
            status = f"saved {len(marks)} bounces -> {out_path}"

        show(status)

    reader.release()
    cv2.destroyAllWindows()
    print(f"Done. {len(marks)} bounces in {out_path}")


if __name__ == "__main__":
    main()
