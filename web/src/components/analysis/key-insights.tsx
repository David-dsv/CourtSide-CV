"use client";

import { GlassCard } from "@/components/core/glass-card";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { deriveKeyInsights } from "@/lib/analysis/insights";
import type { PipelineStats } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Lightbulb, TrendingUp, AlertTriangle, Info } from "lucide-react";

/**
 * "Ce que la vidéo dit de ton jeu" — the ranked, data-grounded coach strip.
 * Every card is a claim + the numbers behind it + a jump to the exact moment
 * (frame-as-truth). Rules only fire on strong signals (see insights.ts), so an
 * empty strip means the data is genuinely unremarkable — we then render nothing
 * rather than filler.
 */
export function KeyInsights({
  stats,
  className,
}: {
  stats: PipelineStats;
  className?: string;
}) {
  const insights = deriveKeyInsights(stats);
  if (!insights.length) return null;

  return (
    <section className={cn("flex flex-col gap-3", className)}>
      <div className="flex items-center gap-2">
        <Lightbulb className="h-4 w-4 text-court-amber" />
        <h2 className="font-display text-sm font-semibold tracking-tight">
          Ce que la vidéo dit de ton jeu
        </h2>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {insights.map((ins) => {
          const icon =
            ins.tone === "good" ? (
              <TrendingUp className="h-4 w-4 text-court-green-deep dark:text-ball" />
            ) : ins.tone === "warn" ? (
              <AlertTriangle className="h-4 w-4 text-court-amber" />
            ) : (
              <Info className="h-4 w-4 text-muted-foreground" />
            );
          const card = (
            <GlassCard
              className={cn(
                "flex h-full flex-col gap-2 p-4 transition-transform duration-300",
                ins.frame !== null && "cursor-pointer hover:-translate-y-1",
              )}
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 shrink-0">{icon}</span>
                <p className="text-sm font-medium leading-snug text-foreground">{ins.title}</p>
              </div>
              <p className="pl-6 font-mono text-[11px] leading-relaxed text-muted-foreground">
                {ins.detail}
              </p>
              {ins.frame !== null && (
                <span className="mt-auto pl-6 font-mono text-[10px] text-muted-foreground/70">
                  cliquer = voir le moment
                </span>
              )}
            </GlassCard>
          );
          return ins.frame !== null ? (
            <FrameJumpLink key={ins.id} frame={ins.frame} className="block h-full">
              {card}
            </FrameJumpLink>
          ) : (
            <div key={ins.id}>{card}</div>
          );
        })}
      </div>
    </section>
  );
}
