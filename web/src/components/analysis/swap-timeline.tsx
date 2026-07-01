"use client";

import { GlassCard } from "@/components/core/glass-card";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { frameToTimecode } from "@/lib/format";
import type { MatchInfo } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ArrowLeftRight, AlertTriangle } from "lucide-react";

/**
 * Side-change timeline (match-mode) — the match.side_swaps the tracker
 * detected, laid on a thin rail scaled to the clip. Ambiguous re-ids
 * (identical kits) are flagged with a warning icon + tooltip instead of being
 * silently trusted — the honest-confidence house rule.
 */
export function SwapTimeline({
  match,
  frameRange,
  fps,
  className,
}: {
  match: MatchInfo | undefined | null;
  frameRange: [number, number];
  fps: number;
  className?: string;
}) {
  const swaps = match?.side_swaps ?? [];
  if (!swaps.length) return null;
  const [f0, f1] = frameRange;
  const span = Math.max(1, f1 - f0);
  const ambiguous = swaps.filter((s) => s.ambiguous_jerseys).length;

  return (
    <GlassCard className={cn("flex flex-col gap-3 p-4", className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="eyebrow inline-flex items-center gap-1.5 !text-muted-foreground !tracking-[0.18em]">
          <ArrowLeftRight className="h-3.5 w-3.5" />
          Changements de côté
        </span>
        <span className="font-mono text-[11px] text-muted-foreground">
          {swaps.length} détecté{swaps.length > 1 ? "s" : ""}
          {ambiguous > 0 && ` · ${ambiguous} incertain${ambiguous > 1 ? "s" : ""}`}
        </span>
      </div>

      <div className="relative h-8">
        {/* rail */}
        <div
          className="absolute left-0 right-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full"
          style={{ background: "var(--viz-bar-track)" }}
        />
        {swaps.map((s, i) => {
          const pct = ((s.frame - f0) / span) * 100;
          return (
            <span
              key={`${s.frame}-${i}`}
              className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `clamp(10px, ${pct}%, calc(100% - 10px))` }}
            >
              <FrameJumpLink
                frame={s.frame}
                title={`${frameToTimecode(s.frame - f0, fps)} — confiance ${(s.confidence * 100).toFixed(0)} %${s.ambiguous_jerseys ? " · maillots similaires, re-id incertaine" : ""}`}
              >
                <span
                  className={cn(
                    "flex h-5 w-5 items-center justify-center rounded-full border-2 bg-background transition-transform hover:scale-125",
                    s.ambiguous_jerseys
                      ? "border-court-amber text-court-amber"
                      : "border-foreground/50 text-foreground/70",
                  )}
                >
                  {s.ambiguous_jerseys ? (
                    <AlertTriangle className="h-2.5 w-2.5" />
                  ) : (
                    <ArrowLeftRight className="h-2.5 w-2.5" />
                  )}
                </span>
              </FrameJumpLink>
            </span>
          );
        })}
      </div>

      <p className="text-xs text-muted-foreground">
        L&apos;identité P1/P2 reste stable à travers ces changements (re-identification
        par maillot) — les statistiques par joueur restent attribuées à la bonne personne.
      </p>
    </GlassCard>
  );
}
