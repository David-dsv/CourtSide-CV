#!/usr/bin/env python3
"""
Feature 2 — 60-second highlight reel from an annotated clip's stats.

Reads a `<name>_stats.json` (produced by run_pipeline_8s.py), picks the best
moments with a simple, defensible scoring rule, then concatenates the matching
video segments from the *annotated* video into a single ≤60s reel with short
text cards ("Meilleur point", "Frappe du match", "À revoir", ...) between them.

    python tools/make_highlight_reel.py \
        --stats data/output/tennis_8s_annotated_stats.json \
        --video data/output/tennis_8s_annotated.mp4 \
        --max-duration 60

Scoring (zero magic numbers — all thresholds are ratios of fps or seconds):
  - top N rallies by mean quality of their *near-side* shots,
  - top M "best shots" by speed × quality,
  - top K "points to review" by lowest quality.

If the stats JSON has no `rallies` (the pipeline only emits them when shots were
detected), rallies are derived from shots using a >2.5s gap rule (same rule the
pipeline itself uses), so the reel still works.

All bounds are in frames; all durations in seconds; everything is derived from
`fps`. Extraction/concat is delegated to ffmpeg (subprocess); cv2/PIL only draw
the text cards.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Make `tools.make_replay` importable whether this file is run as a script
# (no installed package) or imported. We add the repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO, format="{message}")
    class _L:
        info = staticmethod(lambda *a, **k: logging.info(*a))
        warning = staticmethod(lambda *a, **k: logging.warning(*a))
        error = staticmethod(lambda *a, **k: logging.error(*a))
        success = staticmethod(lambda *a, **k: logging.info("✅ " + (a[0] if a else "")))
    logger = _L()

from tools.make_replay import probe_streams, _require_ffmpeg  # reuse helpers


# ─── data ─────────────────────────────────────────────────────────────────────

@dataclass
class Moment:
    """A single highlight: a frame window [start,end] from the annotated video."""
    kind: str            # "rally" | "best_shot" | "review"
    label: str           # card text, e.g. "Meilleur point"
    start_frame: int
    end_frame: int
    score: float
    note: str = ""       # human-readable detail for the CR / debug

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            self.start_frame, self.end_frame = self.end_frame, self.start_frame


# ─── ffmpeg / probe helpers (module-local) ────────────────────────────────────

def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command, raising with stderr on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("ffmpeg failed:\n" + "\n".join(cmd))
        logger.error(r.stderr[-2000:])
        raise RuntimeError("ffmpeg command failed (see above).")


def _render_card(label: str, subtitle: str, w: int, h: int) -> np.ndarray:
    """
    Render a broadcast-style text card (cv2 + PIL for crisp type).
    Returns a BGR frame of size (w, h).
    """
    # Dark glassmorphism background with a subtle gradient + accent bar.
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (12, 12, 18)  # near-black, slightly blue
    # vertical gradient toward slightly lighter at center
    for y in range(h):
        t = abs(y - h / 2) / (h / 2)
        bg[y, :] = (int(12 + 8 * (1 - t)), int(12 + 8 * (1 - t)), int(18 + 10 * (1 - t)))
    # accent bar
    bar_h = max(4, int(h * 0.012))
    cv2.rectangle(bg, (0, h // 2 + int(h * 0.08)), (w, h // 2 + int(h * 0.08) + bar_h),
                  (180, 90, 40), -1)  # warm accent (BGR)

    # Use PIL for anti-aliased text if available; fall back to cv2.
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        # Scale font sizes to frame height (zero hardcoding).
        title_px = max(20, int(h * 0.085))
        sub_px = max(14, int(h * 0.035))
        font_title = _load_pil_font(title_px)
        font_sub = _load_pil_font(sub_px)
        # center text
        tw, th = draw.textlength(label, font=font_title), title_px
        draw.text(((w - tw) / 2, h / 2 - th - int(h * 0.02)), label,
                  font=font_title, fill=(255, 255, 255))
        if subtitle:
            sw = draw.textlength(subtitle, font=font_sub)
            draw.text(((w - sw) / 2, h / 2 + int(h * 0.12)), subtitle,
                      font=font_sub, fill=(190, 190, 200))
        bg = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # cv2 fallback
        title_px = max(20, int(h * 0.08))
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, title_px / 30, 2)
        cv2.putText(bg, label, ((w - tw) // 2, h // 2),
                    font, title_px / 30, (255, 255, 255), 2, cv2.LINE_AA)
    return bg


def _load_pil_font(px: int):
    """Load a PIL TrueType font at `px`, trying common system fonts."""
    from PIL import ImageFont
    for candidate in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",  # CJK fallback
    ]:
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), px)
            except Exception:
                continue
    return ImageFont.load_default()


def _write_card_clip(out_path: Path, label: str, subtitle: str,
                     w: int, h: int, fps: float, seconds: float) -> None:
    """Render a text card to an mp4 of exactly `seconds` at the reel's fps/size."""
    ff = _require_ffmpeg()
    frame = _render_card(label, subtitle, w, h)
    n_frames = max(1, int(round(seconds * fps)))
    # Pipe the same frame `n_frames` times via rawvideo into ffmpeg.
    raw = frame.tobytes()
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",  # read frames from stdin
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    r = subprocess.run(cmd, input=raw * n_frames, capture_output=True)
    if r.returncode != 0:
        logger.error(r.stderr.decode(errors="replace")[-1500:])
        raise RuntimeError(f"card render failed for '{label}'")


def _extract_segment(video: Path, start_frame: int, end_frame: int,
                     fps: float, out_path: Path) -> None:
    """Extract [start_frame, end_frame] (inclusive) from `video` to mp4 (no audio)."""
    ff = _require_ffmpeg()
    start_s = start_frame / fps
    dur_frames = end_frame - start_frame + 1
    dur_s = dur_frames / fps
    # -ss and -t both as INPUT options (see make_replay for why ordering matters).
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}", "-i", str(video),
        "-an",  # segments carry no audio; audio added at the final mux
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", f"{fps}",  # normalize fps so concat is clean
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run_ffmpeg(cmd)


# ─── scoring / selection ─────────────────────────────────────────────────────

def _derive_rallies(shots: list[dict], bounces: list[dict], fps: float) -> list[dict]:
    """
    Re-derive rallies from shots using the pipeline's own >2.5s gap rule, when the
    stats JSON has no usable `rallies`. Returns the same schema as the pipeline.
    """
    if not shots:
        return []
    gap_frames = 2.5 * fps
    sf = sorted(int(s["frame"]) for s in shots)
    bf = sorted(int(b["frame"]) for b in bounces) if bounces else []
    grouped: list[list[int]] = []
    cur = [sf[0]]
    for f in sf[1:]:
        if f - cur[-1] > gap_frames:
            grouped.append(cur)
            cur = []
        cur.append(f)
    grouped.append(cur)
    rallies = []
    for g in grouped:
        start, end = g[0], g[-1]
        bframes = [b for b in bf if start <= b <= end + gap_frames]
        end = max(end, max(bframes, default=end))
        rallies.append({
            "start_frame": start, "end_frame": end,
            "shot_frames": g, "bounce_frames": bframes, "n_shots": len(g),
        })
    return rallies


def _rally_mean_quality(rally: dict, shot_by_frame: dict[int, dict]) -> float:
    """Mean quality of a rally's NEAR-side shots (near = the filming player).
    Falls back to all shots if none are near-side. Returns 0.0 for empty rallies."""
    q: list[float] = []
    for f in rally.get("shot_frames", []):
        s = shot_by_frame.get(int(f))
        if not s:
            continue
        if s.get("player_side") == "near":
            q.append(float(s.get("quality", 0)))
    if not q:  # fallback: any shot in the rally
        for f in rally.get("shot_frames", []):
            s = shot_by_frame.get(int(f))
            if s:
                q.append(float(s.get("quality", 0)))
    return sum(q) / len(q) if q else 0.0


def select_moments(
    stats: dict,
    *,
    top_rallies: int = 5,
    top_best_shots: int = 3,
    top_review: int = 2,
    lead_s: float = 1.5,
) -> list[Moment]:
    """
    Pick the highlight moments. Windows:
      - rally:   [rally.start_frame, rally.end_frame] (the whole exchange)
      - shot:    [shot_frame - lead_s*fps, next_bounce_frame]  (lead-in + the bounce)
        where next_bounce_frame is the first bounce at/after the shot (else shot+0.6s).
    Returns moments in CHRONOLOGICAL order, de-duplicated against overlaps.
    """
    fps = float(stats.get("fps", 0) or 0)
    if fps <= 0:
        raise ValueError("stats.json has no usable `fps`; cannot compute windows.")

    shots = stats.get("shots", []) or []
    bounces = stats.get("bounces", []) or []
    shot_by_frame = {int(s["frame"]): s for s in shots}
    bounce_frames = sorted(int(b["frame"]) for b in bounces)
    lead = lead_s * fps

    rallies = stats.get("rallies") or []
    if not rallies:
        rallies = _derive_rallies(shots, bounces, fps)
        if rallies:
            logger.info(f"stats.json had no rallies; derived {len(rallies)} from shots.")

    moments: list[Moment] = []

    # ── Top rallies by mean near-side shot quality ──
    scored_rallies = sorted(
        ((_rally_mean_quality(r, shot_by_frame), r) for r in rallies if r.get("n_shots", 0) >= 1),
        key=lambda t: t[0], reverse=True,
    )
    for i, (q, r) in enumerate(scored_rallies[:top_rallies]):
        sf, ef = int(r["start_frame"]), int(r["end_frame"])
        if ef <= sf:
            continue
        moments.append(Moment(
            kind="rally", label="Meilleur point",
            start_frame=sf, end_frame=ef, score=q,
            note=f"rally {r.get('n_shots')} shots, mean near-Q={q:.0f}",
        ))

    # ── Top "best shots" by speed × quality (both already in the JSON) ──
    def _speed_quality(s: dict) -> float:
        return float(s.get("speed_kmh", 0)) * float(s.get("quality", 0))

    best = sorted(shots, key=_speed_quality, reverse=True)[:top_best_shots]
    for s in best:
        sf = int(s["frame"])
        # next bounce at/after the shot; else a short tail
        nxt = next((b for b in bounce_frames if b >= sf), None)
        ef = nxt if (nxt is not None and nxt > sf) else int(sf + 0.6 * fps)
        start = max(0, int(sf - lead))
        moments.append(Moment(
            kind="best_shot", label="Frappe du match",
            start_frame=start, end_frame=ef, score=_speed_quality(s),
            note=f"{s.get('stroke','coup')} {s.get('speed_kmh',0):.0f}km/h Q={s.get('quality',0)}",
        ))

    # ── Top "points to review" by lowest quality (need real shots) ──
    review = sorted(shots, key=lambda s: float(s.get("quality", 0)))[:top_review]
    for s in review:
        sf = int(s["frame"])
        nxt = next((b for b in bounce_frames if b >= sf), None)
        ef = nxt if (nxt is not None and nxt > sf) else int(sf + 0.6 * fps)
        start = max(0, int(sf - lead))
        moments.append(Moment(
            kind="review", label="À revoir",
            start_frame=start, end_frame=ef, score=-float(s.get("quality", 0)),
            note=f"Q={s.get('quality',0)} {s.get('stroke','coup')}",
        ))

    # ── Chronological order, then de-duplicate overlapping windows ──
    moments.sort(key=lambda m: (m.start_frame, m.end_frame))
    moments = _dedupe_overlaps(moments)
    return moments


def _dedupe_overlaps(moments: list[Moment]) -> list[Moment]:
    """
    Merge moments whose windows overlap heavily (>50% of the smaller), keeping the
    higher-scoring one's label but the union of the frame range. This prevents the
    reel from showing the same exchange three times (a best shot that is also the
    top rally's climax).
    """
    if not moments:
        return []
    out: list[Moment] = [moments[0]]
    for m in moments[1:]:
        prev = out[-1]
        ov = _overlap(prev, m)
        smaller = min(prev.end_frame - prev.start_frame, m.end_frame - m.start_frame, 1)
        if smaller > 0 and ov / smaller > 0.5:
            # overlap: merge — keep the higher score's label, widen the range
            keep = prev if prev.score >= m.score else m
            merged = Moment(
                kind=keep.kind, label=keep.label,
                start_frame=min(prev.start_frame, m.start_frame),
                end_frame=max(prev.end_frame, m.end_frame),
                score=max(prev.score, m.score),
                note=f"{prev.note} + {m.note}",
            )
            out[-1] = merged
        else:
            out.append(m)
    return out


def _overlap(a: Moment, b: Moment) -> int:
    return max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame))


# ─── assembly ─────────────────────────────────────────────────────────────────

def build_reel(
    stats_path: Path,
    video: Path,
    out_path: Optional[Path] = None,
    *,
    max_duration: float = 60.0,
    card_seconds: float = 1.2,
    top_rallies: int = 5,
    top_best_shots: int = 3,
    top_review: int = 2,
    audio_source: Optional[Path] = None,
) -> Path:
    """
    Build the highlight reel. Returns the output path.

    Args:
        stats_path: the *_stats.json from run_pipeline_8s.py.
        video: the matching annotated video.
        out_path: explicit output; default data/output/<stem>_reel.mp4.
        max_duration: cap the final reel at this many seconds (truncates from the end).
        card_seconds: how long each text card holds.
        audio_source: audio track to mux in (default: the annotated `video` itself,
            which usually has none → silent reel; pass the raw source clip for sound).
    """
    ff = _require_ffmpeg()
    stats = json.loads(stats_path.read_text())
    fps = float(stats.get("fps", 0) or 0)
    if fps <= 0:
        logger.error(f"{stats_path} has no usable fps.")
        sys.exit(1)

    info = probe_streams(video)
    w, h = info["w"], info["h"]
    if not (w and h):
        logger.error(f"Could not read dimensions from {video}.")
        sys.exit(1)

    moments = select_moments(
        stats, top_rallies=top_rallies,
        top_best_shots=top_best_shots, top_review=top_review,
    )
    if not moments:
        logger.error("No highlight moments could be selected (need shots/bounces in stats).")
        sys.exit(1)

    # ─── Budget the moments to fit max_duration ───
    # Each moment contributes (segment_s + card_s). We greedily add moments in
    # chronological order until the budget is exhausted, then trim the last.
    card_s = card_seconds
    chosen: list[Moment] = []
    used = 0.0
    for m in moments:
        seg_s = (m.end_frame - m.start_frame + 1) / fps
        cost = seg_s + (card_s if chosen else 0.0)  # card before each segment except the first
        if used + cost > max_duration and chosen:
            # try to fit a trimmed tail of this moment in the remaining budget
            remaining = max_duration - used
            if remaining > (card_s + 0.5):  # only if there's room for card + ~0.5s
                tail_frames = int((remaining - card_s) * fps)
                m2 = Moment(m.kind, m.label, m.start_frame,
                            m.end_frame, m.score, m.note)
                m2.end_frame = min(m2.end_frame, m2.start_frame + max(1, tail_frames))
                chosen.append(m2)
                used += (m2.end_frame - m2.start_frame + 1) / fps + card_s
            logger.info(
                f"Hit {max_duration:.0f}s budget after {len(chosen)} moments; "
                f"skipping the rest."
            )
            break
        chosen.append(m)
        used += cost
    logger.info(
        f"Selected {len(chosen)}/{len(moments)} moments -> "
        f"~{sum((m.end_frame-m.start_frame+1)/fps for m in chosen):.1f}s of clips "
        f"+ {len(chosen)*card_s:.1f}s of cards = ~{used:.1f}s total"
    )

    # ─── Build per-segment files + a concat list ───
    tmpdir = Path(tempfile.mkdtemp(prefix="reel_"))
    concat_lines: list[str] = []
    try:
        for idx, m in enumerate(chosen):
            # card (skip the very first card for a faster cold-open)
            if idx > 0:
                card_path = tmpdir / f"card_{idx:02d}.mp4"
                _write_card_clip(card_path, m.label, m.note, w, h, fps, card_s)
                concat_lines.append(f"file '{card_path}'")
            # segment
            seg_path = tmpdir / f"seg_{idx:02d}.mp4"
            _extract_segment(video, m.start_frame, m.end_frame, fps, seg_path)
            concat_lines.append(f"file '{seg_path}'")
            logger.info(f"  [{idx}] {m.label:16s} frames [{m.start_frame},{m.end_frame}] "
                        f"({(m.end_frame-m.start_frame+1)/fps:.1f}s) — {m.note}")

        # ─── Concat (demuxer; segments share fps/size/pixfmt) ───
        concat_path = tmpdir / "concat.txt"
        concat_path.write_text("\n".join(concat_lines) + "\n")
        silent_reel = tmpdir / "reel_silent.mp4"
        cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(concat_path), "-c", "copy", str(silent_reel)]
        _run_ffmpeg(cmd)

        # ─── Mux audio (silent if unavailable) ───
        if out_path is None:
            out_dir = Path("data/output"); out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video.stem}_reel.mp4"

        a_src = audio_source or video
        a_info = probe_streams(a_src) if Path(a_src).exists() else {"has_audio": False}
        reel_dur = probe_streams(silent_reel)["duration_s"]

        if a_info.get("has_audio"):
            # Take audio from the start of the source, trimmed to the reel duration.
            cmd = [ff, "-y", "-loglevel", "error",
                   "-i", str(silent_reel),
                   "-ss", "0", "-t", f"{reel_dur:.3f}", "-i", str(a_src),
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                   "-shortest", "-movflags", "+faststart", str(out_path)]
            try:
                _run_ffmpeg(cmd)
                logger.info(f"Muxed audio from {a_src.name}.")
            except RuntimeError:
                logger.warning("Audio mux failed; writing a silent reel.")
                _mux_silent(silent_reel, out_path)
        else:
            logger.warning(
                f"No audio in {a_src.name}; writing a silent reel "
                "(pass --audio-source <clip> for sound)."
            )
            _mux_silent(silent_reel, out_path)

        # ─── Verify ───
        oi = probe_streams(out_path)
        logger.success(
            f"Reel written: {out_path} | {oi['w']}x{oi['h']} | "
            f"{oi['duration_s']:.2f}s | {oi['n_frames']} frames | "
            f"audio={'yes' if oi['has_audio'] else 'no'}"
        )
        return out_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _mux_silent(silent_reel: Path, out_path: Path) -> None:
    """Copy the silent concat out to the final path (or add a silent atrack)."""
    ff = _require_ffmpeg()
    # Add an explicit silent audio track so players don't complain, then we're done.
    dur = probe_streams(silent_reel)["duration_s"]
    cmd = [ff, "-y", "-loglevel", "error",
           "-i", str(silent_reel),
           "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    _run_ffmpeg(cmd)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a ≤60s highlight reel from an annotated clip's stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stats", type=Path, required=True,
                    help="Path to <name>_stats.json (from run_pipeline_8s.py).")
    ap.add_argument("--video", type=Path, required=True,
                    help="Path to the matching annotated video (<name>_annotated.mp4).")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: data/output/<video_stem>_reel.mp4).")
    ap.add_argument("--max-duration", type=float, default=60.0,
                    help="Hard cap on reel length in seconds.")
    ap.add_argument("--card-seconds", type=float, default=1.2,
                    help="How long each text card holds.")
    ap.add_argument("--top-rallies", type=int, default=5)
    ap.add_argument("--top-best-shots", type=int, default=3)
    ap.add_argument("--top-review", type=int, default=2)
    ap.add_argument("--audio-source", type=Path, default=None,
                    help="Audio track to mux in (default: the annotated video, usually silent).")
    args = ap.parse_args()

    build_reel(
        args.stats, args.video, args.out,
        max_duration=args.max_duration, card_seconds=args.card_seconds,
        top_rallies=args.top_rallies, top_best_shots=args.top_best_shots,
        top_review=args.top_review, audio_source=args.audio_source,
    )


if __name__ == "__main__":
    main()
