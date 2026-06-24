/**
 * Court projection — faithful TypeScript port of vision/minimap.py.
 * Court coordinate system: origin at court center, +X right, +Y toward the far
 * baseline (half-length 11.885 m), doubles half-width 5.485 m.
 */
import type { Depth } from "@/lib/types";

export const COURT = {
  DOUBLES_HALF_W: 5.485,
  SINGLES_HALF_W: 4.115,
  HALF_LEN: 11.885,
  SERVICE_FROM_NET: 6.4,
  PAD_M: 0.8,
} as const;

/** Court meters → tile pixel. +Y (far baseline) maps to TOP, +X to the right. */
export function projectMeters(
  Xm: number,
  Ym: number,
  w: number,
  h: number,
): [number, number] {
  const spanX = COURT.DOUBLES_HALF_W + COURT.PAD_M;
  const spanY = COURT.HALF_LEN + COURT.PAD_M;
  const px = w / 2 + (Xm / spanX) * (w / 2);
  const py = h / 2 - (Ym / spanY) * (h / 2);
  return [px, py];
}

/** Apply a 3x3 homography (row-major) to a point. Returns [x', y', w']. */
function applyHomography(H: number[][], x: number, y: number): [number, number] {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  const xp = (H[0][0] * x + H[0][1] * y + H[0][2]) / (w || 1e-9);
  const yp = (H[1][0] * x + H[1][1] * y + H[1][2]) / (w || 1e-9);
  return [xp, yp];
}

/**
 * Frame pixel → court meters.
 * - If `H` (3x3, image→court meters) is provided → exact homography.
 * - Otherwise → geometric fallback (port of minimap.py `project_to_court`):
 *   maps the frame vertical band [top..bot] to [far..near] baseline with a
 *   perspective widening factor. Returns metric-approximate coordinates.
 *
 * `top`/`bot` are normalized (0..1) frame-y of the far/near baselines; when
 * omitted they default to a reasonable band (1/3 .. 5/6 of the frame).
 */
export function pxToMeters(
  x: number,
  y: number,
  H?: number[][],
  frameW = 1920,
  frameH = 1080,
  top = 0.32,
  bot = 0.86,
): { Xm: number; Ym: number; exact: boolean } {
  if (H) {
    const [Xm, Ym] = applyHomography(H, x, y);
    return { Xm, Ym, exact: true };
  }

  // geometric fallback
  const yr = y / frameH;
  const xr = x / frameW - 0.5;
  const depth = (yr - top) / (bot - top + 1e-9); // 0 = far baseline, 1 = near
  const Ym = COURT.HALF_LEN - depth * (2 * COURT.HALF_LEN); // +HALF_LEN(far) → -HALF_LEN(near)
  const persp = 0.55 + 0.9 * depth; // near rows span more width
  let Xm = (xr / 0.5) * COURT.DOUBLES_HALF_W * persp;
  Xm = Math.max(
    -COURT.DOUBLES_HALF_W - COURT.PAD_M,
    Math.min(COURT.DOUBLES_HALF_W + COURT.PAD_M, Xm),
  );
  return { Xm, Ym, exact: false };
}

export function onRadar(Xm: number, Ym: number): boolean {
  return (
    Math.abs(Xm) <= COURT.DOUBLES_HALF_W + COURT.PAD_M + 2.5 &&
    Math.abs(Ym) <= COURT.HALF_LEN + COURT.PAD_M + 3.0
  );
}

/**
 * Depth band from the projected court coordinate. The geometric position is the
 * single source of truth for what is shown ON THE MAP: the dot color and its
 * vertical placement always agree because both derive from |Ym| (distance from
 * the net at Ym=0).
 *
 * ITF-derived bands (court half-length 11.885 m, service line 6.4 m from net):
 *   short : |Ym| < 4 m   — between net and mid-service area
 *   mid   : 4 ≤ |Ym| < 7 m — service box region
 *   deep  : |Ym| ≥ 7 m   — past the service line, toward the baseline
 *
 * NOTE: this intentionally ignores `bounce.depth` from the pipeline `_stats.json`
 * for MAP rendering (that label is kept for lists/stats). The Python
 * `classify_bounce_depth` will be aligned to the same |Ym| bands in a separate
 * pass; until then the frontend reclassifies from the projected position so the
 * demo is internally consistent. See PROJET.md / courtside-felix-grazing-angle.
 */
export function depthFromMeters(Ym: number): Depth {
  const d = Math.abs(Ym);
  if (d < 4) return "short";
  if (d < 7) return "mid";
  return "deep";
}

/**
 * Clamp a projected court point to the pad-expanded court rectangle. Used for
 * synthetic puck positions (derived from shot pixels) that can fall just outside
 * the calibrated court region on grazing clips — we'd rather show the puck on
 * the edge than off the map.
 */
export function clampToCourt(Xm: number, Ym: number): { Xm: number; Ym: number } {
  return {
    Xm: Math.max(
      -(COURT.DOUBLES_HALF_W + COURT.PAD_M),
      Math.min(COURT.DOUBLES_HALF_W + COURT.PAD_M, Xm),
    ),
    Ym: Math.max(
      -(COURT.HALF_LEN + COURT.PAD_M),
      Math.min(COURT.HALF_LEN + COURT.PAD_M, Ym),
    ),
  };
}
