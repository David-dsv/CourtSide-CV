/**
 * Court line geometry for the SVG minimap — port of vision/minimap.py
 * render_court_base(). All coordinates in court meters, projected to tile
 * pixels via projectMeters() at draw time.
 */
import { COURT, projectMeters } from "@/lib/court/projection";

export interface Line {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  soft?: boolean;
}

export const COURT_LINES: Line[] = [
  // doubles rectangle (baselines + sidelines)
  { x1: -COURT.DOUBLES_HALF_W, y1: +COURT.HALF_LEN, x2: +COURT.DOUBLES_HALF_W, y2: +COURT.HALF_LEN },
  { x1: -COURT.DOUBLES_HALF_W, y1: -COURT.HALF_LEN, x2: +COURT.DOUBLES_HALF_W, y2: -COURT.HALF_LEN },
  { x1: -COURT.DOUBLES_HALF_W, y1: +COURT.HALF_LEN, x2: -COURT.DOUBLES_HALF_W, y2: -COURT.HALF_LEN },
  { x1: +COURT.DOUBLES_HALF_W, y1: +COURT.HALF_LEN, x2: +COURT.DOUBLES_HALF_W, y2: -COURT.HALF_LEN },
  // singles sidelines
  { x1: -COURT.SINGLES_HALF_W, y1: +COURT.HALF_LEN, x2: -COURT.SINGLES_HALF_W, y2: -COURT.HALF_LEN, soft: true },
  { x1: +COURT.SINGLES_HALF_W, y1: +COURT.HALF_LEN, x2: +COURT.SINGLES_HALF_W, y2: -COURT.HALF_LEN, soft: true },
  // service lines
  { x1: -COURT.SINGLES_HALF_W, y1: +COURT.SERVICE_FROM_NET, x2: +COURT.SINGLES_HALF_W, y2: +COURT.SERVICE_FROM_NET, soft: true },
  { x1: -COURT.SINGLES_HALF_W, y1: -COURT.SERVICE_FROM_NET, x2: +COURT.SINGLES_HALF_W, y2: -COURT.SERVICE_FROM_NET, soft: true },
  // center service line
  { x1: 0, y1: +COURT.SERVICE_FROM_NET, x2: 0, y2: -COURT.SERVICE_FROM_NET, soft: true },
  // center marks on baselines
  { x1: 0, y1: +COURT.HALF_LEN, x2: 0, y2: +COURT.HALF_LEN - 0.4, soft: true },
  { x1: 0, y1: -COURT.HALF_LEN, x2: 0, y2: -COURT.HALF_LEN + 0.4, soft: true },
];

/** Net endpoints (slightly wider than doubles). */
export const NET_LINE: Line = {
  x1: -COURT.DOUBLES_HALF_W - 0.25,
  y1: 0,
  x2: +COURT.DOUBLES_HALF_W + 0.25,
  y2: 0,
};

export function toPx(line: Line, w: number, h: number) {
  const [x1, y1] = projectMeters(line.x1, line.y1, w, h);
  const [x2, y2] = projectMeters(line.x2, line.y2, w, h);
  return { x1, y1, x2, y2 };
}

// Player puck colors (RGB) — port of run_pipeline_8s.PLAYER_COLORS (BGR→RGB)
export const PLAYER_HEX: Record<1 | 2, string> = {
  1: "#00bfff", // cyan (near)
  2: "#ff9c32", // orange (far)
};
