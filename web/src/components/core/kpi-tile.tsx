"use client";

import { GlassCard } from "@/components/core/glass-card";
import { ConfidenceBadge, type ConfidenceTier } from "@/components/core/confidence-badge";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

type Accent = "green" | "cyan" | "amber" | "red" | "none";

const accentText: Record<Accent, string> = {
  green: "text-ball",
  cyan: "text-court-cyan",
  amber: "text-court-amber",
  red: "text-clay",
  none: "text-foreground",
};

export function KpiTile({
  label,
  value,
  unit,
  source,
  confidence,
  tier,
  frame,
  accent = "none",
  sub,
  icon,
  className,
}: {
  label: string;
  value: string | number;
  unit?: string;
  source?: import("@/lib/types").SpeedSource | string;
  confidence?: import("@/lib/types").HomographyConfidence;
  tier?: ConfidenceTier;
  frame?: number;
  accent?: Accent;
  sub?: ReactNode;
  icon?: ReactNode;
  className?: string;
}) {
  const valueBlock = (
    <div className="flex items-baseline gap-1.5">
      <span className={cn("stat-numeral text-4xl", accentText[accent])}>{value}</span>
      {unit && <span className="font-mono text-xs text-muted-foreground">{unit}</span>}
    </div>
  );

  return (
    <GlassCard
      className={cn(
        "group flex flex-col gap-3 p-4",
        frame !== undefined && "cursor-pointer transition-transform duration-300 hover:-translate-y-1",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="eyebrow !text-muted-foreground !tracking-[0.18em]">{label}</span>
        {icon && (
          <span className="text-muted-foreground transition-colors group-hover:text-ball">{icon}</span>
        )}
      </div>
      {frame !== undefined ? (
        <FrameJumpLink frame={frame} className="inline-flex w-fit">
          {valueBlock}
        </FrameJumpLink>
      ) : (
        valueBlock
      )}
      <div className="mt-auto flex items-center justify-between gap-2 pt-1">
        {sub && <span className="font-mono text-[11px] text-muted-foreground">{sub}</span>}
        {(source || confidence || tier) && (
          <ConfidenceBadge source={source} confidence={confidence} tier={tier} showTooltip />
        )}
      </div>
    </GlassCard>
  );
}
