"use client";

import { useEffect, useMemo, useRef } from "react";
import { pxToMeters, projectMeters, depthFromMeters } from "@/lib/court/projection";
import type { Bounce, PipelineStats, Stroke, Depth } from "@/lib/types";

/**
 * Filter modes for the placement heatmap.
 * - `all`              : every bounce
 * - `forehand|backhand`: bounces that are the *result* of a user shot of that
 *                        stroke (we join shots→next-bounce; only user/near shots)
 * - `deep|short`       : bounces by their own depth
 */
export type HeatmapFilter =
  | "all"
  | "forehand"
  | "backhand"
  | "deep"
  | "mid"
  | "short";

const INFERNO: [number, number, number][] = [
  [0, 0, 4],
  [27, 12, 65],
  [74, 12, 107],
  [120, 28, 109],
  [165, 44, 96],
  [207, 68, 70],
  [237, 105, 37],
  [251, 155, 6],
  [247, 209, 61],
  [252, 255, 164],
];

function infernoColor(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (INFERNO.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = INFERNO[i];
  const b = INFERNO[Math.min(i + 1, INFERNO.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/**
 * Placement heatmap — Canvas KDE over projected bounce points, INFERNO ramp,
 * additive blend. Port of vision/fx.py placement_heatmap intent.
 *
 * Pass `stats` + `filter` to slice the placement by stroke (forehand/backhand)
 * or by depth. When omitted, falls back to the raw `bounces` prop (back-compat).
 */
export function PlacementHeatmap({
  bounces,
  stats,
  filter = "all",
  width = 360,
  height = 760,
  H,
  frameW = 1920,
  frameH = 1080,
  className,
}: {
  bounces: Bounce[];
  stats?: PipelineStats;
  filter?: HeatmapFilter;
  width?: number;
  height?: number;
  /** optional court homography 3x3 (image px → court meters). Metric-exact when
   *  present; geometric fallback otherwise. Keep this in sync with the CourtMinimap
   *  on the same page so the heatmap and the radar dots overlap. */
  H?: number[][];
  frameW?: number;
  frameH?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  // Resolve which bounces to plot under the active filter.
  const filtered = useMemo<Bounce[]>(() => {
    if (!stats || filter === "all") return bounces;
    const sortedShots = [...stats.shots].sort((a, b) => a.frame - b.frame);
    const sortedBounces = [...bounces].sort((a, b) => a.frame - b.frame);

    if (filter === "deep" || filter === "mid" || filter === "short") {
      // reclassify from the projected |Ym| — same source of truth as the dot
      // color on the minimap, so the heatmap and the radar agree.
      const want = filter as Depth;
      return bounces.filter((b) => {
        const { Ym } = pxToMeters(b.x, b.y, H, frameW, frameH);
        return depthFromMeters(Ym) === want;
      });
    }
    // stroke filter: bounces that are the result of a user (near) shot of this stroke
    const stroke = filter as Stroke;
    const resultFrames = new Set<number>();
    for (const s of sortedShots) {
      if (s.player_side !== "near" || s.stroke !== stroke) continue;
      const nb = sortedBounces.find((b) => b.frame > s.frame);
      if (nb) resultFrames.add(nb.frame);
    }
    return bounces.filter((b) => resultFrames.has(b.frame));
  }, [bounces, stats, filter, H, frameW, frameH]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // project bounces to court meters → tile px
    const pts = filtered
      .map((b) => {
        const { Xm, Ym } = pxToMeters(b.x, b.y, H, frameW, frameH);
        return projectMeters(Xm, Ym, width, height);
      })
      .filter(([x, y]) => x >= 0 && x <= width && y >= 0 && y <= height);

    if (pts.length === 0) return;

    // build density via additive gaussian splats
    const sigma = Math.max(width, height) * 0.06;
    ctx.globalCompositeOperation = "lighter";

    // render to an offscreen low-res buffer for performance
    const scale = 4;
    const loW = Math.ceil(width / scale);
    const loH = Math.ceil(height / scale);
    const off = document.createElement("canvas");
    off.width = loW;
    off.height = loH;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(loW, loH);

    // accumulate density
    const density = new Float32Array(loW * loH);
    const r = Math.ceil(sigma / scale);
    for (const [px, py] of pts) {
      const cx = px / scale;
      const cy = py / scale;
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const x = Math.round(cx + dx);
          const y = Math.round(cy + dy);
          if (x < 0 || y < 0 || x >= loW || y >= loH) continue;
          const d2 = dx * dx + dy * dy;
          const g = Math.exp(-d2 / (2 * (sigma / scale) ** 2));
          density[y * loW + x] += g;
        }
      }
    }
    const max = Math.max(...density, 1e-6);
    for (let i = 0; i < density.length; i++) {
      const t = density[i] / max;
      const [r0, g0, b0] = infernoColor(t);
      const a = Math.pow(t, 0.7) * 235;
      img.data[i * 4 + 0] = r0;
      img.data[i * 4 + 1] = g0;
      img.data[i * 4 + 2] = b0;
      img.data[i * 4 + 3] = a;
    }
    octx.putImageData(img, 0, 0);

    ctx.globalCompositeOperation = "source-over";
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.85;
    ctx.drawImage(off, 0, 0, width, height);
    ctx.globalAlpha = 1;
  }, [filtered, width, height, H, frameW, frameH]);

  return (
    <canvas
      ref={ref}
      style={{ width, height }}
      className={className}
      aria-label="Heatmap de placement des balles"
    />
  );
}
