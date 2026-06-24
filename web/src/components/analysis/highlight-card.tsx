"use client";

import { useFrame } from "@/components/video/video-frame-context";
import { GlassCard } from "@/components/core/glass-card";
import { highlightMeta } from "@/lib/analysis/highlights";
import { cn } from "@/lib/utils";
import type { Highlight } from "@/lib/types";
import { frameToTimecode } from "@/lib/format";
import { Button } from "@/components/ui/button";
import { Trophy, AlertCircle, Activity, Award, Zap, Play, Repeat } from "lucide-react";

const ICON = {
  best_point: Trophy,
  worst_point: AlertCircle,
  longest: Activity,
  winner: Award,
  top_shot: Zap,
} as const;

const ACCENT_DOT: Record<string, string> = {
  green: "bg-court-green",
  red: "bg-clay",
  cyan: "bg-court-cyan",
  amber: "bg-court-amber",
};

const ACCENT_TEXT: Record<string, string> = {
  green: "text-court-green",
  red: "text-clay",
  cyan: "text-court-cyan",
  amber: "text-court-amber",
};

/**
 * A single highlight card. Clicking the card jumps the scrubber to the
 * defining frame (frame-as-truth). The secondary "Rejouer" button plays the
 * highlight's window as a loop in the VideoScrubber instead of a single jump.
 *
 * The card is a `div[role=button]` rather than a `<button>` so the Replay
 * control can be a nested `<button>` (valid HTML — buttons can't nest).
 */
export function HighlightCard({ highlight, fps }: { highlight: Highlight; fps: number }) {
  const { setFrame, setPlaybackRange, setPlaying, playing, playbackRange } = useFrame();
  const meta = highlightMeta(highlight.type);
  const Icon = ICON[highlight.type];
  const window = highlight.window;
  const hasWindow = !!window && window[1] > window[0];
  const isLoopingThis =
    hasWindow && playing && playbackRange?.[0] === window![0] && playbackRange?.[1] === window![1];

  const replay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window) {
      // No usable window → degrade to a plain jump.
      setFrame(highlight.frame);
      return;
    }
    if (isLoopingThis) {
      // Toggle off: exit the loop.
      setPlaybackRange(null);
      setPlaying(false);
      return;
    }
    setFrame(window[0]);
    setPlaybackRange(window);
    setPlaying(true);
  };

  return (
    <GlassCard
      glow={meta.accent === "green" ? "green" : "none"}
      role="button"
      tabIndex={0}
      onClick={() => setFrame(highlight.frame)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          setFrame(highlight.frame);
        }
      }}
      title={`Aller au frame ${highlight.frame}`}
      className="group h-full cursor-pointer transition-transform group-hover:-translate-y-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 rounded-2xl"
    >
      <div className="mb-2 flex items-center gap-2">
        <Icon className={cn("h-4 w-4", ACCENT_TEXT[meta.accent])} />
        <span className={cn("h-1.5 w-1.5 rounded-full", ACCENT_DOT[meta.accent])} />
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {meta.label}
        </span>
      </div>
      <h4 className="text-sm font-semibold leading-tight">{highlight.title}</h4>
      <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{highlight.reason}</p>

      {highlight.chips && highlight.chips.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {highlight.chips.map((c) => (
            <span
              key={c.label}
              className="rounded-md glass px-2 py-0.5 text-[11px] tabular-nums"
            >
              <span className="text-muted-foreground">{c.label} </span>
              <span className="font-medium text-foreground">{c.value}</span>
            </span>
          ))}
        </div>
      )}

      <div className="mt-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1 text-[11px] font-mono text-muted-foreground transition-colors group-hover:text-court-green">
          <Zap className="h-3 w-3" />
          {frameToTimecode(highlight.frame, fps)}
        </div>
        {hasWindow && (
          <Button
            type="button"
            variant={isLoopingThis ? "default" : "secondary"}
            size="sm"
            className="h-7 gap-1.5 px-2 text-[11px]"
            onClick={replay}
            title={
              isLoopingThis
                ? "Arrêter le replay"
                : `Rejouer ${frameToTimecode(window![0], fps)} → ${frameToTimecode(window![1], fps)}`
            }
            aria-pressed={isLoopingThis}
          >
            {isLoopingThis ? <Repeat className="h-3 w-3" /> : <Play className="h-3 w-3" />}
            {isLoopingThis ? "En cours" : "Rejouer"}
          </Button>
        )}
      </div>
    </GlassCard>
  );
}
