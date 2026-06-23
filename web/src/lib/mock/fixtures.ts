/**
 * Mock fixtures — built from the REAL pipeline outputs in ./raw/*.json
 * (copied from data/output/). Only synthetic additions are the ball trajectory
 * (interpolated between bounces) and the player puck positions + a plausible
 * court homography H for the homography(high|low) cases. felix (scalar/none)
 * stays on the honest geometric fallback (no H).
 */
import type { Project, PipelineStats, TrajectoryPoint, PlayerPosition } from "@/lib/types";
import felixRaw from "./raw/felix.json";
import tennisRaw from "./raw/tennis.json";
import sotaRaw from "./raw/sota.json";

const felixStats = felixRaw as unknown as PipelineStats;
const tennisStats = tennisRaw as unknown as PipelineStats;
const sotaStats = sotaRaw as unknown as PipelineStats;

/** Build a synthetic ball trajectory (px) interpolating between bounces. */
function synthTrajectory(stats: PipelineStats): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [];
  const bounces = [...stats.bounces].sort((a, b) => a.frame - b.frame);
  if (bounces.length === 0) return pts;
  for (let i = 0; i < bounces.length; i++) {
    const cur = bounces[i];
    pts.push({ x: cur.x, y: cur.y, frame: cur.frame });
    if (i < bounces.length - 1) {
      const next = bounces[i + 1];
      const steps = 6;
      for (let s = 1; s < steps; s++) {
        const t = s / steps;
        // add a slight arc (parabola) to mimic ball flight
        const x = cur.x + (next.x - cur.x) * t;
        const baseY = cur.y + (next.y - cur.y) * t;
        const arc = -Math.sin(t * Math.PI) * 60; // upward arc in px
        pts.push({ x, y: baseY + arc, frame: Math.round(cur.frame + (next.frame - cur.frame) * t) });
      }
    }
  }
  return pts;
}

/** Derive player puck positions from shots (near/far side), sampled along frames. */
function synthPlayers(stats: PipelineStats): PlayerPosition[] {
  const out: PlayerPosition[] = [];
  const nearShots = stats.shots.filter((s) => s.player_side === "near");
  const farShots = stats.shots.filter((s) => s.player_side === "far");
  for (const s of nearShots) {
    out.push({ id: 1, x: s.x, y: s.y + 90, frame: s.frame });
  }
  for (const s of farShots) {
    out.push({ id: 2, x: s.x, y: s.y - 60, frame: s.frame });
  }
  // ensure at least a default position per player at frame 0
  if (!out.some((p) => p.id === 1))
    out.push({ id: 1, x: 960, y: 800, frame: 0 });
  if (!out.some((p) => p.id === 2))
    out.push({ id: 2, x: 960, y: 400, frame: 0 });
  return out;
}

/**
 * Plausible homography H (image px → court meters), row-major 3x3.
 * For a standard behind-baseline camera these map roughly:
 *   image center → court center (0,0), image-top → far baseline (+Y).
 * These are illustrative (the real ones come from court_calibrator.py).
 */
function H_standard(): number[][] {
  // affine-ish, scaled so that frame corners map sensibly to court meters
  return [
    [0.012, 0.0, -11.0],
    [0.0, -0.022, 13.0],
    [0.0, 0.0, 1.0],
  ];
}

function H_grazing(): number[][] {
  // weaker conditioning for the low-confidence grazing case
  return [
    [0.011, 0.002, -10.5],
    [0.001, -0.020, 12.0],
    [0.0, 0.0, 1.0],
  ];
}

export const felixProject: Project = {
  id: "felix",
  name: "Felix — angle grazing",
  video: "felix.mp4",
  createdAt: "2026-06-12T18:30:00Z",
  status: "ready",
  stats: felixStats,
  trajectory: synthTrajectory(felixStats),
  players: synthPlayers(felixStats),
  // no H → honest geometric fallback (~approx) — felix is the pathological grazing case
  posterFrame: felixStats.bounces[0]?.frame ?? 0,
};

export const tennisProject: Project = {
  id: "tennis-demo",
  name: "Match démo — angle standard",
  video: "tennis_1min.mp4",
  createdAt: "2026-06-18T14:10:00Z",
  status: "ready",
  stats: tennisStats,
  trajectory: synthTrajectory(tennisStats),
  players: synthPlayers(tennisStats),
  H: H_standard(),
  posterFrame: tennisStats.bounces[0]?.frame ?? 0,
};

export const sotaProject: Project = {
  id: "felix-sota",
  name: "Felix — calibration 4-coins",
  video: "felix.mp4",
  createdAt: "2026-06-19T09:45:00Z",
  status: "ready",
  stats: sotaStats,
  trajectory: synthTrajectory(sotaStats),
  players: synthPlayers(sotaStats),
  H: H_grazing(),
  posterFrame: sotaStats.bounces[0]?.frame ?? 0,
};

export const mockProjects: Project[] = [tennisProject, sotaProject, felixProject];
