"use client";

import { useMemo } from "react";
import { pxToMeters, projectMeters, onRadar } from "@/lib/court/projection";
import { COURT_LINES, NET_LINE, toPx, PLAYER_HEX } from "@/lib/court/court-geometry";
import { ConfidenceBadge } from "@/components/core/confidence-badge";
import { depthHex } from "@/lib/format";
import type {
  Bounce,
  HomographyConfidence,
  PlayerPosition,
  TrajectoryPoint,
} from "@/lib/types";
import { cn } from "@/lib/utils";

export interface CourtMinimapProps {
  bounces: Bounce[];
  trajectory?: TrajectoryPoint[];
  players?: PlayerPosition[];
  confidence?: HomographyConfidence;
  source?: string;
  /** show the ball trajectory comet */
  showTrajectory?: boolean;
  /** show player pucks */
  showPlayers?: boolean;
  /** current frame (for animating ball position + pucks) */
  frame?: number;
  /** on bounce click (frame-as-truth) */
  onBounceClick?: (frame: number) => void;
  className?: string;
  /** aspect: true = portrait court (default), false = compact */
  height?: number;
}

const VIEW_W = 360;
const VIEW_H = 760;

export function CourtMinimap({
  bounces,
  trajectory,
  players,
  confidence,
  source,
  showTrajectory = true,
  showPlayers = true,
  frame,
  onBounceClick,
  className,
}: CourtMinimapProps) {
  const hasH = source?.startsWith("homography");

  // project bounces to court meters then to tile px
  const bounceTiles = useMemo(() => {
    return bounces
      .map((b) => {
        const { Xm, Ym } = pxToMeters(b.x, b.y, undefined, 1920, 1080);
        return { bounce: b, Xm, Ym };
      })
      .filter((p) => onRadar(p.Xm, p.Ym));
    // pxToMeters uses the geometric fallback here (H wiring is a follow-up);
    // see lib/mock/fixtures.ts for the per-project H used in production.
  }, [bounces]);

  const trajTiles = useMemo(() => {
    if (!showTrajectory || !trajectory) return [] as { x: number; y: number; frame: number }[];
    return trajectory.map((p) => {
      const { Xm, Ym } = pxToMeters(p.x, p.y, undefined, 1920, 1080);
      const [tx, ty] = projectMeters(Xm, Ym, VIEW_W, VIEW_H);
      return { x: tx, y: ty, frame: p.frame };
    });
  }, [trajectory, showTrajectory]);

  const playerTiles = useMemo(() => {
    if (!showPlayers || !players) return [];
    // pick the player position closest to current frame (or last known)
    const f = frame ?? 0;
    const byPlayer: Record<number, PlayerPosition | undefined> = {};
    for (const p of players) {
      const cur = byPlayer[p.id];
      if (!cur || (Math.abs(p.frame - f) < Math.abs(cur.frame - f) && p.frame <= f + 30)) {
        byPlayer[p.id] = p;
      }
    }
    return Object.values(byPlayer).filter(Boolean).map((p) => {
      const { Xm, Ym } = pxToMeters(p!.x, p!.y, undefined, 1920, 1080);
      const [tx, ty] = projectMeters(Xm, Ym, VIEW_W, VIEW_H);
      return { id: p!.id, x: tx, y: ty };
    });
  }, [players, showPlayers, frame]);

  return (
    <div
      className={cn(
        "relative court-bg overflow-hidden rounded-2xl border border-white/10",
        className,
      )}
      style={{ aspectRatio: `${VIEW_W} / ${VIEW_H}` }}
    >
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className="h-full w-full"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id="court-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2a1a12" />
            <stop offset="100%" stopColor="#1a120c" />
          </linearGradient>
          <linearGradient id="trail-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#d8f64a" stopOpacity={0} />
            <stop offset="100%" stopColor="#d8f64a" stopOpacity={0.95} />
          </linearGradient>
          <radialGradient id="ball-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#d8f64a" stopOpacity={0.95} />
            <stop offset="100%" stopColor="#d8f64a" stopOpacity={0} />
          </radialGradient>
        </defs>

        {/* court surface */}
        <rect x="0" y="0" width={VIEW_W} height={VIEW_H} fill="url(#court-grad)" />

        {/* court lines */}
        {COURT_LINES.map((ln, i) => {
          const { x1, y1, x2, y2 } = toPx(ln, VIEW_W, VIEW_H);
          return (
            <line
              key={i}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#ebf0f5"
              strokeWidth={ln.soft ? 1 : 1.6}
              strokeOpacity={ln.soft ? 0.45 : 0.85}
            />
          );
        })}

        {/* net */}
        {(() => {
          const n = toPx(NET_LINE, VIEW_W, VIEW_H);
          return (
            <g>
              <line
                x1={n.x1}
                y1={n.y1}
                x2={n.x2}
                y2={n.y2}
                stroke="#f5faff"
                strokeWidth={2.5}
                strokeDasharray="4 3"
              />
            </g>
          );
        })()}

        {/* trajectory comet */}
        {trajTiles.length > 1 && (
          <polyline
            points={trajTiles.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke="url(#trail-grad)"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.8}
          />
        )}

        {/* current ball position */}
        {showTrajectory && frame !== undefined && trajTiles.length > 0 && (
          <CurrentBall trajTiles={trajTiles} frame={frame} />
        )}

        {/* bounces */}
        {bounceTiles.map(({ bounce, Xm, Ym }) => {
          const [tx, ty] = projectMeters(Xm, Ym, VIEW_W, VIEW_H);
          const color = depthHex(bounce.depth);
          return (
            <g
              key={bounce.frame}
              onClick={() => onBounceClick?.(bounce.frame)}
              className={onBounceClick ? "cursor-pointer" : undefined}
            >
              <circle cx={tx} cy={ty} r={9} fill={color} opacity={0.18} />
              <circle cx={tx} cy={ty} r={4.5} fill={color} stroke="#0b0a08" strokeWidth={1} />
            </g>
          );
        })}

        {/* player pucks */}
        {playerTiles.map((p) => (
          <g key={p.id}>
            <circle cx={p.x} cy={p.y} r={11} fill={PLAYER_HEX[p.id as 1 | 2]} opacity={0.18} />
            <circle cx={p.x} cy={p.y} r={6} fill={PLAYER_HEX[p.id as 1 | 2]} stroke="#0b0a08" strokeWidth={1.2} />
            <text
              x={p.x}
              y={p.y + 2}
              textAnchor="middle"
              className="fill-[#0b0a08] font-semibold"
              fontSize={7}
            >
              P{p.id}
            </text>
          </g>
        ))}
      </svg>

      {/* confidence badge overlay */}
      {(source || confidence) && (
        <div className="absolute right-2 top-2">
          <ConfidenceBadge source={source as never} confidence={confidence} showTooltip />
        </div>
      )}

      {/* depth legend */}
      <div className="absolute bottom-2 left-2 flex items-center gap-3 rounded-lg glass px-2.5 py-1.5 text-[10px]">
        <LegendDot color={depthHex("deep")} label="Profond" />
        <LegendDot color={depthHex("mid")} label="Moyen" />
        <LegendDot color={depthHex("short")} label="Court" />
      </div>

      {hasH && (
        <div className="absolute bottom-2 right-2 rounded-lg glass px-2.5 py-1.5 text-[10px] text-muted-foreground">
          métrique exact
        </div>
      )}
    </div>
  );
}

function CurrentBall({
  trajTiles,
  frame,
}: {
  trajTiles: { x: number; y: number; frame: number }[];
  frame: number;
}) {
  // find closest trajectory point to current frame
  let closest = trajTiles[0];
  let bestDist = Infinity;
  for (const p of trajTiles) {
    const d = Math.abs(p.frame - frame);
    if (d < bestDist) {
      bestDist = d;
      closest = p;
    }
  }
  return (
    <g>
      <circle cx={closest.x} cy={closest.y} r={14} fill="url(#ball-glow)" />
      <circle cx={closest.x} cy={closest.y} r={4} fill="#d8f64a" stroke="#0b0a08" strokeWidth={1} />
    </g>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1 text-muted-foreground">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
