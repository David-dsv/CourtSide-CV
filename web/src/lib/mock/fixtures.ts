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
import tennisFullRaw from "./raw/tennis-full.json";

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
 * Mock court homography H (image px → court meters), row-major 3x3.
 *
 * ⚠️ These are CALIBRATED AFFINE APPROXIMATIONS for the demo, not real
 * projective homographies. They were fit so that every bounce pixel in the
 * mock `_stats.json` lands inside the court rectangle with a depth ordering
 * that respects the pipeline's `depth` label (deep → toward a baseline,
 * short → near the net). An affine is used deliberately: grazing/oblique
 * clips (felix) are ill-conditioned for a 4-corner projective solve (see
 * `models/court_calibrator.py` + memory courtside-felix-grazing-angle), so a
 * projective H fitted to noisy bounce centroids extrapolates wildly for points
 * off the centroid — an affine cannot.
 *
 * In production the REAL projective homography comes from `court_calibrator.py`
 * (4+ clicked court corners → cv2.findHomography) and is stored on `Project.H`;
 * these affines only stand in for the mock fixtures. Direction: image center →
 * court center (0,0), image-top (small y) → far side (+Y), image-bottom → near
 * side (−Y), image-left/right → −/+X.
 *
 * Calibration source: the bounce (x,y,depth) triples in src/lib/mock/raw/*.json
 * (real pipeline outputs). Constants verified to map 100% of bounces in-court
 * (|Xm| ≤ doubles-half-width+pad, |Ym| ≤ half-length+pad).
 */
function H_standard(): number[][] {
  // Broadcast/standard angle (tennis-demo). Calibrated on tennis.json (9 bounces).
  // ax = 3.7 / max|x-dev| (maps the spread to singles half-width), Ym spans +8..−2.
  return [
    [0.009840425531914894, 0.0, -9.446808510638299],
    [0.0, -0.021367521367521368, 14.858974358974361],
    [0.0, 0.0, 1.0],
  ];
}

function H_grazing(): number[][] {
  // Grazing/oblique angle (sota, same felix.mp4 geometry). Calibrated on sota.json
  // (20 bounces). Tighter x-scale than broadcast (perspective compression) and a
  // steeper y-scale (the visible half-court is foreshortened).
  return [
    [0.005417276720351391, 0.0, -5.200585651537335],
    [0.0, -0.05714285714285715, 37.31428571428572],
    [0.0, 0.0, 1.0],
  ];
}

export const felixProject: Project = {
  id: "felix",
  name: "Felix — angle grazing",
  video: "felix.mp4",
  createdAt: "2026-06-12T18:30:00Z",
  status: "ready",
  matchType: "training",
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
  matchType: "training",
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
  matchType: "training",
  stats: sotaStats,
  trajectory: synthTrajectory(sotaStats),
  players: synthPlayers(sotaStats),
  H: H_grazing(),
  posterFrame: sotaStats.bounces[0]?.frame ?? 0,
};

/**
 * tennis-full — la vidéo tennis COMPLÈTE en mode "match" (les joueurs changent
 * de côté entre jeux). Démo de la logique side-swap (gérée côté backend,
 * feat/match-mode).
 *
 * Stats RÉELLES : généré par `run_pipeline_8s.py tennis.mp4 -s 0 -d 235
 * --match-mode --device cpu` sur les 11 750 frames (235 s @ 50 fps). Le JSON
 * brut est web/src/lib/mock/raw/tennis-full.json (copié de data/output/). Il
 * porte les champs match-mode : `match_mode: true`, `match.side_swaps` (19
 * changements de côté détectés), et un `human_id` stable sur chaque shot/rally.
 * Ces champs ne sont pas encore lus par le frontend (le type PipelineStats ne
 * les déclare pas → tolérés via le cast `as unknown`), mais la source de stats
 * est désormais réelle (38 bounces / 149 shots / 15 rallies), plus le clone du
 * clip court. La vidéo annotée réelle est
 * web/public/annotated/tennis-full_annotated.mp4.
 */
const tennisFullStats = tennisFullRaw as unknown as PipelineStats;

export const tennisFullProject: Project = {
  id: "tennis-full",
  name: "Match démo — tennis complet",
  video: "tennis.mp4",
  createdAt: "2026-06-24T10:00:00Z",
  status: "ready",
  matchType: "match",
  stats: tennisFullStats,
  trajectory: synthTrajectory(tennisFullStats),
  players: synthPlayers(tennisFullStats),
  H: H_standard(),
  posterFrame: tennisFullStats.bounces[0]?.frame ?? 0,
};

export const mockProjects: Project[] = [tennisProject, tennisFullProject, sotaProject, felixProject];
