"use client";

import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { useFrame } from "@/components/video/video-frame-context";
import { frameToTimecode } from "@/lib/format";
import { Pause, Play, SkipBack, SkipForward, Repeat, Clapperboard, Pencil } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Project } from "@/lib/types";

/**
 * Video scrubber. Two display modes, toggled by the "Vidéo réelle / Mock" switch:
 *  - "real" (default): a real <video> element served from /annotated/<id>_annotated.mp4,
 *    synced to the shared VideoFrameContext (frame ↔ currentTime).
 *  - "mock": the original SVG court preview (the previous behaviour).
 *
 * Both modes honor the shared VideoFrameContext: the slider, play/pause, and the
 * optional playbackRange loop (used by the highlight "Replay" button).
 *
 * Playback range (loop): an optional `[start, end]` window can be supplied two
 * ways — via the `playbackRange` prop (controlled) or via the shared
 * `VideoFrameContext`. When active, play loops inside the window instead of the
 * full range. Pass null / omit to clear.
 */
export function VideoScrubber({
  project,
  playbackRange: playbackRangeProp,
}: {
  project: Project;
  /**
   * Optional loop window `[start, end]`. When set, mirrors it into the shared
   * context so the loop survives re-renders and any "play" toggle honors it.
   * Pass null to clear. Left unspecified → the context range (if any) is used.
   */
  playbackRange?: [number, number] | null;
}) {
  const { frame, setFrame, frameRange, frameCount, playbackRange, setPlaybackRange, playing, setPlaying, clip } = useFrame();
  const rafRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fps = project.stats.fps;
  const [realVideo, setRealVideo] = useState(true);

  // Keep the context's range in sync with the controlled prop, if provided.
  useEffect(() => {
    if (playbackRangeProp !== undefined) {
      setPlaybackRange(playbackRangeProp);
    }
  }, [playbackRangeProp, setPlaybackRange]);

  const [loopStart, loopEnd] = playbackRange ?? frameRange;

  useEffect(() => {
    if (!playing) {
      if (rafRef.current) clearInterval(rafRef.current);
      return;
    }
    rafRef.current = setInterval(() => {
      setFrame((prev) => {
        const next = prev + Math.max(1, Math.round(fps / 20));
        // Loop inside the active window, or stop at the full-range end.
        if (next > loopEnd) {
          if (playbackRange) return loopStart;
          setPlaying(false);
          return loopEnd;
        }
        return next;
      });
    }, 50);
    return () => {
      if (rafRef.current) clearInterval(rafRef.current);
    };
  }, [playing, setFrame, setPlaying, playbackRange, loopStart, loopEnd, fps]);

  const pct = frameCount > 0 ? ((frame - frameRange[0]) / frameCount) * 100 : 0;
  const looping = playbackRange !== null;

  return (
    <div className="panel-raised overflow-hidden rounded-2xl">
      {/* poster area */}
      <div className="relative aspect-video w-full court-bg">
        {realVideo ? (
          <RealVideo project={project} />
        ) : (
          /* fake court + ball overlay */
          <CourtPosterMock project={project} pct={pct} />
        )}
        <div className={`absolute left-3 top-3 rounded-md bg-black/50 px-2 py-1 font-mono text-xs backdrop-blur ${realVideo ? "text-court-green" : "text-ball"}`}>
          {realVideo ? "● VIDÉO RÉELLE" : "● MOCK PREVIEW"}
        </div>
        <div className="absolute right-3 top-3 flex items-center gap-1.5">
          {clip.hasCuts && (
            <span
              className="rounded-md bg-clay/25 px-2 py-1 font-mono text-xs text-clay backdrop-blur"
              title="Vidéo annotée raccourcie (frames hors-terrain coupées). Le timecode reste celui de la vidéo d'origine."
            >
              ✂ coupée
            </span>
          )}
          <span className="rounded-md bg-black/50 px-2 py-1 font-mono text-xs text-white backdrop-blur">
            {frameToTimecode(frame, fps)} · f{frame}
          </span>
        </div>
        {looping && (
          <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-md bg-court-green/15 px-2 py-1 font-mono text-xs text-court-green backdrop-blur">
            ↻ rejoue {frameToTimecode(loopStart, fps)} – {frameToTimecode(loopEnd, fps)}
          </div>
        )}
      </div>

      {/* controls */}
      <div className="flex flex-col gap-3 p-3">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setFrame(Math.max(frameRange[0], frame - Math.round(fps * 5)))}
            title="Reculer 5s"
          >
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button
            variant="default"
            size="icon"
            className="h-9 w-9 bg-ball text-coal-950 hover:bg-ball/90"
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setFrame(Math.min(frameRange[1], frame + Math.round(fps * 5)))}
            title="Avancer 5s"
          >
            <SkipForward className="h-4 w-4" />
          </Button>
          {looping && (
            <Button
              variant="secondary"
              size="sm"
              className="h-8 gap-1.5 px-2 text-xs"
              onClick={() => setPlaybackRange(null)}
              title="Lecture normale (sortir du replay)"
              aria-pressed
            >
              <Repeat className="h-3.5 w-3.5" />
              Sortir du replay
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            className="ml-auto h-8 gap-1.5 px-2 text-xs"
            onClick={() => setRealVideo((v) => !v)}
            title={realVideo ? "Basculer vers le mock SVG" : "Basculer vers la vidéo réelle"}
            aria-pressed={realVideo}
          >
            {realVideo ? <Pencil className="h-3.5 w-3.5" /> : <Clapperboard className="h-3.5 w-3.5" />}
            {realVideo ? "Mock" : "Vidéo"}
          </Button>
          <span className="font-mono text-xs text-muted-foreground">
            {frameToTimecode(frame, fps)} / {frameToTimecode(frameRange[1], fps)}
          </span>
        </div>
        <Slider
          value={frame}
          min={frameRange[0]}
          max={frameRange[1]}
          step={1}
          onValueChange={(v: number | readonly number[]) => setFrame(Array.isArray(v) ? v[0] : v)}
          className="[&_[role=slider]]:bg-court-green [&_[role=slider]]:border-0"
        />
      </div>
    </div>
  );
}

/**
 * Real annotated <video>, synced to the shared frame context.
 *
 * Two-way binding:
 *  - context.frame → video.currentTime (when the user scrubs / jumps via stats).
 *  - video.timeupdate → context.frame (when the video plays natively).
 * Native play() / pause() / loop honor the context's `playing` + `playbackRange`.
 */
function RealVideo({ project }: { project: Project }) {
  const { frame, setFrame, fps, playing, playbackRange, clip } = useFrame();
  const videoRef = useRef<HTMLVideoElement>(null);
  // Avoid fighting the user's native seeking: ignore timeupdate while we drive.
  const driving = useRef(false);

  // Source: the real annotated sample served from /public/annotated/.
  // The annotated clip may be SHORTER than the source (cut frames). currentTime
  // lives in ANNOTATED time; every `frame` the app passes is in ORIGINAL frame
  // numbers — `clip` converts between the two (identity when uncut).
  const src = `/annotated/${project.id}_annotated.mp4`;

  // frame (original) → currentTime (annotated). Seeks the video.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const target = clip.toAnnotated(frame) / fps;
    if (Math.abs(v.currentTime - target) > 0.06) {
      driving.current = true;
      v.currentTime = target;
    }
  }, [frame, fps, clip]);

  // playing → video.play()/pause().
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (playing) {
      v.play().catch(() => {/* autoplay may be blocked until first interaction */});
    } else {
      v.pause();
    }
  }, [playing]);

  // currentTime (annotated) → frame (original). Native playback progress.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => {
      if (driving.current) { driving.current = false; return; }
      setFrame(clip.toOriginal(Math.round(v.currentTime * fps)));
    };
    v.addEventListener("timeupdate", onTime);
    return () => v.removeEventListener("timeupdate", onTime);
  }, [setFrame, fps, clip]);

  // Loop window: when playbackRange is active, clamp + loop inside it. The
  // window is in original frames → convert each bound to annotated time.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!playbackRange) return;
    const onTime = () => {
      const start = clip.toAnnotated(playbackRange[0]) / fps;
      const end = clip.toAnnotated(playbackRange[1]) / fps;
      if (v.currentTime < start - 0.1) v.currentTime = start;
      else if (v.currentTime >= end) v.currentTime = start;
    };
    v.addEventListener("timeupdate", onTime);
    return () => v.removeEventListener("timeupdate", onTime);
  }, [playbackRange, fps, clip]);

  return (
    <video
      ref={videoRef}
      src={src}
      className="h-full w-full object-contain"
      playsInline
      muted
      preload="auto"
      onError={(e) => console.warn("annotated video missing:", src, e)}
    />
  );
}

/** A stylized fake court frame with a moving ball marker driven by pct. */
function CourtPosterMock({ project, pct }: { project: Project; pct: number }) {
  // place bounces along the timeline as faint dots
  const bounces = project.stats.bounces;
  const fps = project.stats.fps;
  return (
    <svg viewBox="0 0 1920 1080" className="h-full w-full" preserveAspectRatio="xMidYMid slice">
      <defs>
        <linearGradient id="poster-bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#211a13" />
          <stop offset="100%" stopColor="#14110c" />
        </linearGradient>
      </defs>
      <rect width="1920" height="1080" fill="url(#poster-bg)" />
      {/* perspective court trapezoid */}
      <polygon
        points="640,300 1280,300 1620,980 300,980"
        fill="rgba(226,96,58,0.05)"
        stroke="rgba(239,231,214,0.22)"
        strokeWidth="2"
      />
      <line x1="960" y1="300" x2="960" y2="980" stroke="rgba(239,231,214,0.15)" strokeWidth="2" />
      <line x1="500" y1="640" x2="1420" y2="640" stroke="rgba(239,231,214,0.15)" strokeWidth="2" />
      {/* bounce markers */}
      {bounces.map((b) => {
        const bpct = (b.frame - project.stats.frame_range[0]) / (project.stats.frame_range[1] - project.stats.frame_range[0]);
        const visible = Math.abs(bpct * 100 - pct) < 2;
        const color = b.depth === "deep" ? "#e2603a" : b.depth === "mid" ? "#f2b441" : "#d8f64a";
        return (
          <circle
            key={b.frame}
            cx={b.x}
            cy={b.y}
            r={visible ? 26 : 7}
            fill={color}
            opacity={visible ? 0.5 : 0.25}
          />
        );
      })}
      {/* current ball position glow */}
      {(() => {
        const f = Math.round(project.stats.frame_range[0] + (pct / 100) * (project.stats.frame_range[1] - project.stats.frame_range[0]));
        // nearest trajectory point
        const traj = project.trajectory ?? [];
        let near = traj[0];
        let bd = Infinity;
        for (const p of traj) {
          const d = Math.abs(p.frame - f);
          if (d < bd) {
            bd = d;
            near = p;
          }
        }
        if (!near) return null;
        return (
          <g>
            <circle cx={near.x} cy={near.y} r={44} fill="#d8f64a" opacity={0.2} />
            <circle cx={near.x} cy={near.y} r={12} fill="#d8f64a" />
          </g>
        );
      })()}
      <text x="960" y="1040" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="22" fontFamily="monospace">
        {project.video} · {fps.toFixed(0)} fps · preview mock
      </text>
    </svg>
  );
}
