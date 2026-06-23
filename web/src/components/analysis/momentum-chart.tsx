"use client";

import { useFrame } from "@/components/video/video-frame-context";
import type { MomentumBucket } from "@/lib/types";
import { frameToTimecode } from "@/lib/format";

/**
 * Momentum timeline — a centered zero-baseline sparkline. Green bars above the
 * line = good stretches, clay bars below = poor stretches. Clickable buckets
 * jump the scrubber to that moment.
 */
export function MomentumChart({
  buckets,
  fps,
}: {
  buckets: MomentumBucket[];
  fps: number;
}) {
  const { setFrame } = useFrame();
  if (buckets.length === 0) {
    return <div className="text-sm text-muted-foreground">Pas assez de frappes pour le momentum.</div>;
  }

  const w = 600;
  const h = 110;
  const pad = 8;
  const midY = h / 2;
  const maxAbs = Math.max(1, ...buckets.map((b) => Math.abs(b.score)));
  const barW = (w - pad * 2) / buckets.length;

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
        {/* zero baseline */}
        <line x1={pad} y1={midY} x2={w - pad} y2={midY} stroke="currentColor" strokeWidth={1} className="text-foreground/20" />
        {buckets.map((b, i) => {
          const x = pad + i * barW;
          const barH = (Math.abs(b.score) / maxAbs) * (midY - pad);
          const isGood = b.score > 0;
          const isNeutral = b.score === 0;
          const y = isGood ? midY - barH : midY;
          const fill = isNeutral ? "#5b6b6e" : isGood ? "#d8f64a" : "#e2603a";
          return (
            <g key={i} className="cursor-pointer" onClick={() => setFrame(b.frame)}>
              {/* invisible hit area spanning full column */}
              <rect x={x} y={0} width={Math.max(barW - 1, 1)} height={h} fill="transparent" />
              <rect
                x={x}
                y={y}
                width={Math.max(barW - 2, 1)}
                height={Math.max(barH, isNeutral ? 1.5 : 1)}
                rx={1.5}
                fill={fill}
                opacity={isNeutral ? 0.4 : 0.85}
              >
                <title>{`${frameToTimecode(b.frame, fps)} — score ${b.score > 0 ? "+" : ""}${b.score}`}</title>
              </rect>
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex justify-between text-xs text-muted-foreground">
        <span>Début</span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-court-green" /> en forme
          <span className="ml-2 h-2 w-2 rounded-sm bg-clay" /> à travailler
        </span>
        <span>Fin</span>
      </div>
    </div>
  );
}
