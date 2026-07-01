"use client";

import { GlassCard } from "@/components/core/glass-card";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { deriveStrokeStats } from "@/lib/analysis/insights";
import { fmtKmh } from "@/lib/format";
import type { PipelineStats } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Coup droit vs revers — paired horizontal bars on ONE axis (km/h), entity
 * colors from the validated --viz-stroke-* steps (same hue families as the
 * shot glyphs). Each bar is direct-labeled (name + value + count), so
 * identity never rides on color alone; quality rides as small secondary text,
 * never a second axis.
 */
export function StrokeCompare({
  stats,
  className,
}: {
  stats: PipelineStats;
  className?: string;
}) {
  const rows = deriveStrokeStats(stats).filter((s) => s.count > 0 && s.avgSpeed !== null);
  if (rows.length < 2) return null;
  const max = Math.max(...rows.map((r) => r.maxSpeed ?? r.avgSpeed!), 1);
  const gap = rows[0].avgSpeed! - rows[1].avgSpeed!;

  return (
    <GlassCard className={cn("flex flex-col gap-4 p-4", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="eyebrow !text-muted-foreground !tracking-[0.18em]">
          Coup droit vs revers
        </span>
        <span className="font-mono text-[11px] text-muted-foreground">vitesse moyenne</span>
      </div>

      <div className="flex flex-col gap-3">
        {rows.map((r) => {
          const color =
            r.stroke === "forehand" ? "var(--viz-stroke-fh)" : "var(--viz-stroke-bh)";
          const bar = (
            <div className="flex items-center gap-2">
              <span className="w-24 shrink-0 text-xs text-foreground">
                {r.stroke === "forehand" ? "Coup droit" : "Revers"}
                <span className="ml-1 font-mono text-[10px] text-muted-foreground">×{r.count}</span>
              </span>
              <div className="h-2.5 flex-1 rounded-[4px]" style={{ background: "var(--viz-bar-track)" }}>
                <div
                  className="h-full rounded-[4px] transition-[filter] hover:brightness-110"
                  style={{ width: `${(r.avgSpeed! / max) * 100}%`, background: color }}
                />
              </div>
              <span className="stat-numeral w-24 shrink-0 text-right text-base">
                {fmtKmh(r.avgSpeed!, 0)}
                <span className="ml-1 font-mono text-[10px] text-muted-foreground">km/h</span>
              </span>
            </div>
          );
          return (
            <div key={r.stroke} className="flex flex-col gap-0.5">
              {r.bestFrame !== null ? (
                <FrameJumpLink frame={r.bestFrame} title="Voir la frappe la plus rapide">
                  {bar}
                </FrameJumpLink>
              ) : (
                bar
              )}
              <span className="pl-26 font-mono text-[10px] text-muted-foreground">
                {r.maxSpeed !== null && `pointe ${fmtKmh(r.maxSpeed, 0)} km/h`}
                {r.avgQuality !== null && ` · qualité ${Math.round(r.avgQuality)}/100`}
              </span>
            </div>
          );
        })}
      </div>

      {Math.abs(gap) >= 3 && (
        <p className="text-xs text-muted-foreground">
          Ton {gap > 0 ? "revers" : "coup droit"} est en moyenne{" "}
          <span className="stat-numeral text-foreground">{Math.abs(gap).toFixed(0)} km/h</span>{" "}
          plus lent — un axe de travail concret.
        </p>
      )}
    </GlassCard>
  );
}
