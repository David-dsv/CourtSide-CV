"use client";

import { GlassCard } from "@/components/core/glass-card";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { deriveFatigue } from "@/lib/analysis/insights";
import { fmtKmh } from "@/lib/format";
import type { FatigueInfo } from "@/lib/types";
import { cn } from "@/lib/utils";
import { BatteryLow, BatteryFull, TrendingDown, TrendingUp, ArrowRight } from "lucide-react";

/**
 * Real fatigue card, straight from the pipeline block (vision/fatigue.py) —
 * replaces the momentum-proxy stub. Hero number = the first→last-third speed
 * drop; two same-hue mini bars carry the two averages (one measure, two
 * moments — direct-labeled, no legend needed). Sample-size confidence is shown
 * honestly (the product never hides "trop peu de frappes").
 */
export function FatigueCard({
  fatigue,
  className,
}: {
  fatigue: FatigueInfo | undefined;
  className?: string;
}) {
  const f = deriveFatigue(fatigue);
  if (!f) return null;

  const max = Math.max(f.firstThird ?? 0, f.lastThird ?? 0, 1);
  const drop = f.dropPct;
  const slowing = drop !== null && drop >= 3;
  const improving = drop !== null && drop <= -3;

  return (
    <GlassCard className={cn("flex flex-col gap-4 p-4", className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="eyebrow !text-muted-foreground !tracking-[0.18em]">
          Condition physique
        </span>
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px]",
            f.usable
              ? "border-border text-muted-foreground"
              : "border-court-amber/40 text-court-amber",
          )}
          title={
            f.usable
              ? `Estimé sur ${f.nShots} frappes côté joueur`
              : "Trop peu de frappes pour une lecture fiable"
          }
        >
          {f.usable ? `${f.nShots} frappes` : "échantillon court"}
        </span>
      </div>

      {!f.usable ? (
        <p className="text-sm text-muted-foreground">
          Pas assez de frappes pour mesurer une tendance — filme une session plus longue
          pour activer le suivi de fatigue.
        </p>
      ) : (
        <>
          <div className="flex items-center gap-3">
            {f.fatigued || slowing ? (
              <BatteryLow className="h-8 w-8 text-court-amber" strokeWidth={1.5} />
            ) : (
              <BatteryFull className="h-8 w-8 text-court-green-deep dark:text-ball" strokeWidth={1.5} />
            )}
            <div className="flex items-baseline gap-2">
              <span className="stat-numeral text-4xl">
                {drop !== null ? `${drop > 0 ? "−" : "+"}${Math.abs(drop).toFixed(0)} %` : "—"}
              </span>
              <span className="inline-flex items-center gap-1 font-mono text-xs text-muted-foreground">
                {slowing ? (
                  <TrendingDown className="h-3.5 w-3.5 text-court-amber" />
                ) : improving ? (
                  <TrendingUp className="h-3.5 w-3.5 text-court-green-deep dark:text-ball" />
                ) : (
                  <ArrowRight className="h-3.5 w-3.5" />
                )}
                vitesse fin de session
              </span>
            </div>
          </div>

          {/* two same-measure bars: first vs last third, direct-labeled */}
          <div className="flex flex-col gap-2">
            {[
              { label: "Premier tiers", v: f.firstThird! },
              { label: "Dernier tiers", v: f.lastThird! },
            ].map(({ label, v }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="w-24 shrink-0 text-[11px] text-muted-foreground">{label}</span>
                <div className="h-2 flex-1 rounded-[4px]" style={{ background: "var(--viz-bar-track)" }}>
                  <div
                    className="h-full rounded-[4px] bg-foreground/45"
                    style={{ width: `${(v / max) * 100}%` }}
                  />
                </div>
                <span className="stat-numeral w-20 shrink-0 text-right text-sm">
                  {fmtKmh(v, 0)} <span className="font-mono text-[10px] text-muted-foreground">km/h</span>
                </span>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[11px] text-muted-foreground">
              {f.slopeKmhPerMin !== null &&
                `${f.slopeKmhPerMin > 0 ? "+" : ""}${f.slopeKmhPerMin.toFixed(1)} km/h par minute`}
            </span>
            {f.sampleFrame !== null && (
              <FrameJumpLink frame={f.sampleFrame} className="font-mono text-[11px] text-muted-foreground hover:text-foreground">
                voir le moment →
              </FrameJumpLink>
            )}
          </div>
        </>
      )}
    </GlassCard>
  );
}
