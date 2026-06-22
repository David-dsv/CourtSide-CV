"use client";

import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { useFrame } from "@/components/video/video-frame-context";
import { frameToTimecode } from "@/lib/format";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Project } from "@/lib/types";

/**
 * Mock video scrubber — no real video this sprint. A poster placeholder + a
 * frame slider bound to the shared VideoFrameContext. The whole dashboard
 * reacts to frame changes (frame-as-truth).
 */
export function VideoScrubber({ project }: { project: Project }) {
  const { frame, setFrame, frameRange, frameCount } = useFrame();
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fps = project.stats.fps;

  useEffect(() => {
    if (!playing) {
      if (rafRef.current) clearInterval(rafRef.current);
      return;
    }
    rafRef.current = setInterval(() => {
      setFrame((prev) => {
        const next = Math.min(prev + Math.max(1, Math.round(fps / 20)), frameRange[1]);
        if (next >= frameRange[1]) setPlaying(false);
        return next;
      });
    }, 50);
    return () => {
      if (rafRef.current) clearInterval(rafRef.current);
    };
  }, [playing, setFrame, frameRange, fps]);

  const pct = frameCount > 0 ? ((frame - frameRange[0]) / frameCount) * 100 : 0;

  return (
    <div className="panel-raised overflow-hidden rounded-2xl">
      {/* poster area */}
      <div className="relative aspect-video w-full court-bg">
        {/* fake court + ball overlay */}
        <CourtPosterMock project={project} pct={pct} />
        <div className="absolute left-3 top-3 rounded-md bg-black/50 px-2 py-1 font-mono text-xs text-ball backdrop-blur">
          ● MOCK PREVIEW
        </div>
        <div className="absolute right-3 top-3 rounded-md bg-black/50 px-2 py-1 font-mono text-xs text-white backdrop-blur">
          {frameToTimecode(frame, fps)} · f{frame}
        </div>
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
          <span className="ml-1 font-mono text-xs text-muted-foreground">
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
