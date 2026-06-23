"use client";

import { useFrame } from "@/components/video/video-frame-context";
import { GlassCard } from "@/components/core/glass-card";
import { highlightMeta } from "@/lib/analysis/highlights";
import { cn } from "@/lib/utils";
import type { Highlight } from "@/lib/types";
import { frameToTimecode } from "@/lib/format";
import { Trophy, AlertCircle, Activity, Award, Zap } from "lucide-react";

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
 * A single highlight card. Clickable → jumps the scrubber to the defining
 * frame (with a short replay lead-in window).
 */
export function HighlightCard({ highlight, fps }: { highlight: Highlight; fps: number }) {
  const { setFrame } = useFrame();
  const meta = highlightMeta(highlight.type);
  const Icon = ICON[highlight.type];

  return (
    <button
      type="button"
      onClick={() => setFrame(highlight.frame)}
      className="group block w-full text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 rounded-2xl"
      title={`Aller au frame ${highlight.frame}`}
    >
      <GlassCard
        glow={meta.accent === "green" ? "green" : "none"}
        className="h-full transition-transform group-hover:-translate-y-1"
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

        <div className="mt-3 flex items-center gap-1 text-[11px] font-mono text-muted-foreground transition-colors group-hover:text-court-green">
          <Zap className="h-3 w-3" />
          {frameToTimecode(highlight.frame, fps)}
        </div>
      </GlassCard>
    </button>
  );
}
