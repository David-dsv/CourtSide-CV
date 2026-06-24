#!/usr/bin/env python3
"""
Feature 1 — Slow-motion replay of a single exchange.

Extracts a frame segment [start_frame, end_frame] from an *annotated* video and
re-plays it in slow motion, re-injecting the audio slowed by the same factor.

    python tools/make_replay.py data/output/tennis_8s_annotated.mp4 \
        --start-frame 121 --end-frame 414 --fps 50 --slowmo 0.25

All lengths are derived from `fps` and frame indices — no magic numbers tied to a
specific clip. Extraction and re-encoding are delegated to ffmpeg (subprocess);
this script only computes frame→time bounds and builds the command.

Output: data/output/<name>_replay_<start_frame>.mp4
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a project dep, but stay runnable
    import logging
    logging.basicConfig(level=logging.INFO, format="{message}")
    class _L:
        info = staticmethod(lambda *a, **k: logging.info(*a))
        warning = staticmethod(lambda *a, **k: logging.warning(*a))
        error = staticmethod(lambda *a, **k: logging.error(*a))
        success = staticmethod(lambda *a, **k: logging.info("✅ " + (a[0] if a else "")))
    logger = _L()


# ─── ffmpeg helpers ───────────────────────────────────────────────────────────

def _require_ffmpeg() -> str:
    """Return the ffmpeg executable path, or exit with a clear error."""
    ff = shutil.which("ffmpeg")
    if not ff:
        logger.error(
            "ffmpeg is not installed or not on PATH. Install it "
            "(e.g. `brew install ffmpeg`) and retry."
        )
        sys.exit(1)
    return ff


def probe_streams(video_path: Path) -> dict:
    """Return {n_frames, fps, duration_s, has_audio, w, h} via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # Fall back to a best-effort probe via ffmpeg itself is messy; require ffprobe.
        logger.error("ffprobe is not installed or not on PATH (ships with ffmpeg).")
        sys.exit(1)

    def _show(stream_sel: str, entries: str) -> str:
        """Query one stream; return '' (not an error) if the stream is absent."""
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", stream_sel,
             "-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True,
        )
        return r.stdout.strip()

    fps_str = _show("v:0", "stream=r_frame_rate")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except ValueError:
        fps = 0.0
    n_frames = int(_show("v:0", "stream=nb_frames") or 0)
    w = int(_show("v:0", "stream=width") or 0)
    h = int(_show("v:0", "stream=height") or 0)
    # has_audio: a non-empty codec_name on the first audio stream. _show returns ''
    # when there is no audio stream, so this is naturally false.
    has_audio = bool(_show("a:0", "stream=codec_name"))
    duration = (n_frames / fps) if (fps > 0 and n_frames > 0) else 0.0
    return {"n_frames": n_frames, "fps": fps, "duration_s": duration,
            "has_audio": has_audio, "w": w, "h": h}


def atempo_chain_for(slowmo: float) -> list[str]:
    """
    Build the `atempo` filter chain to slow audio by `1/slowmo`.

    ffmpeg's single atempo accepts tempo factors in [0.5, 100.0]. To get a *slower*
    playback we need a tempo < 1.0, but the lower bound is 0.5, so factors slower
    than 2× (slowmo < 0.5) must be chained: each atempo halves the tempo.
    e.g. slowmo=0.25 (4× slow) -> tempo 0.5 -> ['atempo=0.5','atempo=0.5'].

    Returns the list of filter tokens (joined by the caller with commas).
    """
    if slowmo <= 0 or slowmo > 1.0:
        raise ValueError(f"--slowmo must be in (0, 1.0]; got {slowmo}")
    tempo = slowmo  # playback tempo = slowmo (0.25 tempo = 4× slower)
    chain: list[str] = []
    remaining = tempo
    while remaining < 0.5 - 1e-9:
        chain.append("atempo=0.5")
        remaining /= 0.5
    # Trim trailing zeros so 0.5 -> "0.5", 0.8 -> "0.8", not "0.500000".
    chain.append(f"atempo={remaining:g}")
    return chain


# ─── core ─────────────────────────────────────────────────────────────────────

def make_replay(
    annotated_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    slowmo: float,
    out_path: Path | None = None,
) -> Path:
    """
    Produce a slow-motion replay of [start_frame, end_frame] from `annotated_video`.

    Args:
        annotated_video: path to the annotated source video.
        start_frame, end_frame: absolute frame indices (inclusive). Clamped to the
            video's real frame range; out-of-range values are handled gracefully.
        fps: source fps (used to convert frames <-> seconds). Should match the
            annotated video's fps; we trust the caller (the pipeline passes reader.fps).
        slowmo: playback speed factor in (0, 1.0]. 0.25 = quarter speed (4× slow).
        out_path: explicit output path. If None, derives
            data/output/<stem>_replay_<start_frame>.mp4.

    Returns the output path.
    """
    ff = _require_ffmpeg()
    if not annotated_video.exists():
        logger.error(f"Annotated video not found: {annotated_video}")
        sys.exit(1)

    info = probe_streams(annotated_video)
    real_fps = info["fps"] or fps
    n_frames = info["n_frames"]

    # ─── Clamp frame bounds to the video's real range ───
    lo, hi = 0, max(0, n_frames - 1)
    s = max(lo, min(int(start_frame), hi))
    e = max(lo, min(int(end_frame), hi))
    if e < s:  # swapped: normalize
        s, e = e, s
    if e == s:
        logger.error(
            f"Empty segment after clamping (start={start_frame}, end={end_frame}, "
            f"video has {n_frames} frames). Nothing to replay."
        )
        sys.exit(1)
    if (s, e) != (start_frame, end_frame):
        logger.warning(
            f"Frame range [{start_frame},{end_frame}] was out of bounds "
            f"(video: {n_frames} frames); clamped to [{s},{e}]."
        )

    # Frames -> seconds (derived from fps; zero hardcoding).
    start_s = s / real_fps
    dur_frames = e - s + 1
    dur_s = dur_frames / real_fps

    # ─── Output path ───
    if out_path is None:
        out_dir = Path("data/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{annotated_video.stem}_replay_{s}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Build the ffmpeg command ───
    # Slow-mo recipe (option ORDER matters — all input opts come BEFORE -i):
    #   - `-ss` (input) seeks the source to the start frame.
    #   - `-t` (input) limits the SOURCE segment to `dur_s`. Placed before -i it caps
    #     the input; placed after -i it caps the *output* duration and silently cancels
    #     the slowdown. Both must precede -i.
    #   - `setpts=(1/slowmo)*PTS` stretches video PTS so playback is `1/slowmo`× longer.
    #   - `atempo` slows audio by `slowmo` (chained to respect the [0.5,100] range).
    setpts = f"setpts={1.0 / slowmo:.6f}*PTS"
    input_opts = ["-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}",
                  "-i", str(annotated_video)]

    if info["has_audio"]:
        # Slow audio with chained atempo (handles slowmo < 0.5).
        achain = ",".join(atempo_chain_for(slowmo))
        cmd = [
            ff, "-y", "-loglevel", "error",
            *input_opts,
            "-filter:v", setpts,
            "-filter:a", achain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        logger.warning(
            f"No audio stream in {annotated_video.name}; producing a silent replay."
        )
        cmd = [
            ff, "-y", "-loglevel", "error",
            *input_opts,
            "-filter:v", setpts,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-an",  # no audio
            "-movflags", "+faststart",
            str(out_path),
        ]

    logger.info(
        f"Replay: frames [{s},{e}] ({dur_frames} f @ {real_fps:.2f}fps = {dur_s:.2f}s) "
        f"× slowmo {slowmo} -> {dur_s / slowmo:.2f}s output"
    )
    subprocess.run(cmd, check=True)

    # ─── Verify output ───
    out_info = probe_streams(out_path)
    logger.success(
        f"Replay written: {out_path} | "
        f"{out_info['w']}x{out_info['h']} | {out_info['fps']:.2f}fps | "
        f"{out_info['n_frames']} frames | {out_info['duration_s']:.2f}s | "
        f"audio={'yes' if out_info['has_audio'] else 'no'}"
    )
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slow-motion replay of a frame segment from an annotated video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("video", type=Path,
                    help="Path to the *annotated* source video (e.g. ..._annotated.mp4).")
    ap.add_argument("--start-frame", type=int, required=True,
                    help="Start frame (absolute index, inclusive).")
    ap.add_argument("--end-frame", type=int, required=True,
                    help="End frame (absolute index, inclusive).")
    ap.add_argument("--fps", type=float, default=None,
                    help="Source fps. If omitted, read from the video via ffprobe.")
    ap.add_argument("--slowmo", type=float, default=0.25,
                    help="Playback speed factor in (0, 1.0]. 0.25 = 4× slow.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: data/output/<stem>_replay_<start>.mp4).")
    args = ap.parse_args()

    # Resolve fps: caller-provided wins, else probe.
    fps = args.fps
    if fps is None:
        fps = probe_streams(args.video)["fps"]
        if fps <= 0:
            logger.error("Could not determine fps; pass --fps explicitly.")
            sys.exit(1)
        logger.info(f"Probed fps from video: {fps:.3f}")

    if not (0 < args.slowmo <= 1.0):
        logger.error(f"--slowmo must be in (0, 1.0]; got {args.slowmo}")
        sys.exit(1)

    make_replay(args.video, args.start_frame, args.end_frame, fps, args.slowmo, args.out)


if __name__ == "__main__":
    main()
