"use client";

import { GlassCard } from "@/components/core/glass-card";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { deriveOutcomeSummary } from "@/lib/analysis/insights";
import type { Rally, RallyOutcome } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Trophy, Minus, ShieldAlert, XCircle } from "lucide-react";
import type { ReactNode } from "react";

/**
 * Rally-outcome mix — a single 100 % stacked bar (parts-of-whole) + status
 * legend. Status semantics (winner = good … unforced = serious) use the
 * validated per-theme --viz-outcome-* steps; every segment also carries an
 * icon + label + count chip (identity is never color-alone) and 2 px surface
 * gaps separate segments.
 */

const META: Record<
  RallyOutcome,
  { label: string; varName: string; icon: ReactNode; hint?: string }
> = {
  winner: {
    label: "Points gagnants",
    varName: "var(--viz-outcome-winner)",
    icon: <Trophy className="h-3.5 w-3.5" />,
  },
  neutral: {
    label: "Neutres",
    varName: "var(--viz-outcome-neutral)",
    icon: <Minus className="h-3.5 w-3.5" />,
  },
  forced_error: {
    label: "Fautes provoquées",
    varName: "var(--viz-outcome-forced)",
    icon: <ShieldAlert className="h-3.5 w-3.5" />,
    hint: "Confiance basse — l'adversaire a probablement forcé la faute",
  },
  unforced_error: {
    label: "Fautes directes",
    varName: "var(--viz-outcome-unforced)",
    icon: <XCircle className="h-3.5 w-3.5" />,
  },
};

export function OutcomeBar({
  rallies,
  className,
}: {
  rallies: Rally[] | undefined;
  className?: string;
}) {
  const summary = deriveOutcomeSummary(rallies);
  if (summary.total === 0) return null;
  const visible = summary.slices.filter((s) => s.count > 0);

  return (
    <GlassCard className={cn("flex flex-col gap-4 p-4", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="eyebrow !text-muted-foreground !tracking-[0.18em]">
          Issue des échanges
        </span>
        <span className="font-mono text-[11px] text-muted-foreground">
          {summary.total} échange{summary.total > 1 ? "s" : ""} classé{summary.total > 1 ? "s" : ""}
        </span>
      </div>

      {/* 100 % stacked bar — thin mark, 2px gaps, 4px rounded ends */}
      <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-[4px]">
        {visible.map((s) => (
          <div
            key={s.outcome}
            title={`${META[s.outcome].label} — ${s.count}`}
            className="h-full transition-[filter] hover:brightness-110"
            style={{
              width: `${(s.count / summary.total) * 100}%`,
              background: META[s.outcome].varName,
            }}
          />
        ))}
      </div>

      {/* legend chips: icon + label + count (never color-alone) */}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {visible.map((s) => {
          const chip = (
            <span
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
              title={META[s.outcome].hint}
            >
              <span style={{ color: META[s.outcome].varName }}>{META[s.outcome].icon}</span>
              <span>{META[s.outcome].label}</span>
              <span className="stat-numeral text-sm text-foreground">{s.count}</span>
              {s.outcome === "forced_error" && (
                <span className="font-mono text-[10px] text-muted-foreground/70">?</span>
              )}
            </span>
          );
          return s.firstFrame !== null ? (
            <FrameJumpLink key={s.outcome} frame={s.firstFrame} title="Voir le premier">
              {chip}
            </FrameJumpLink>
          ) : (
            <span key={s.outcome}>{chip}</span>
          );
        })}
      </div>

      {summary.unforcedShare !== null && (
        <p className="text-xs text-muted-foreground">
          {Math.round(summary.unforcedShare * 100)} % des points décisifs sont des fautes
          directes — le ratio à faire baisser en priorité.
        </p>
      )}
    </GlassCard>
  );
}
