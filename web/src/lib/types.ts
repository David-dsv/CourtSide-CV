/**
 * Shared types — derived 1:1 from the CourtSide pipeline stats JSON
 * (run_pipeline_8s.py lines 1058-1089). Keeping these faithful means the mock
 * fixtures can be imported straight from data/output/*.json.
 */

/**
 * Type d'analyse choisi par l'utilisateur à l'upload.
 * - "training" : caméra fixe, un joueur qui ne change PAS de côté. Le verrou
 *   P1/P2 fixe actuel marche parfaitement — c'est le cœur de cible CourtSide
 *   (amateur sur trépied).
 * - "match" : vraie vidéo de match, les joueurs changent de côté entre jeux.
 *   La ré-identité joueur (side-swap) est gérée côté backend (feat/match-mode) ;
 *   tant qu'elle n'est pas branchée, les stats par joueur (vitesse, fatigue)
 *   restent approximatives → l'UI affiche un avertissement (MatchModeNotice).
 */
export type MatchType = "training" | "match";

export type Depth = "deep" | "mid" | "short";
export type Stroke = "forehand" | "backhand";
/**
 * The stroke value a single shot may carry. The geometric classifier emits
 * "unknown" when it can't decide (real `_stats.json` shots routinely have it,
 * e.g. raw/tennis-full.json). Kept separate from {@link Stroke} so the
 * forehand/backhand-only aggregates (Summary.strokes) stay exhaustive.
 */
export type ShotStroke = Stroke | "unknown";
export type PlayerSide = "near" | "far";
export type HomographyConfidence = "high" | "low" | "fallback" | "none";
export type SpeedSource = "scalar" | `homography(${"high" | "low"})`;

export interface Bounce {
  frame: number;
  x: number;
  y: number;
  depth: Depth;
  speed_kmh: number;
  quality: number;
}

export interface Shot {
  frame: number;
  x: number;
  y: number;
  player_side: PlayerSide;
  stroke: ShotStroke;
  quality: number;
  speed_kmh: number;
  /** stable human identity across side-changes (match-mode). Absent in
   *  training mode / legacy fixtures. See {@link MatchInfo}. */
  human_id?: string;
}

export interface Summary {
  total_bounces: number;
  depth: Record<Depth, number>;
  speed_kmh: { avg: number; max: number; min: number };
  quality: { avg: number; max: number; high_count: number };
  strokes: Record<Stroke, number>;
}

/**
 * A rally — a contiguous sequence of play grouped by temporal gap between
 * shots. Derived from shots[] (gap > RALLY_GAP_FRAMES_FRACTION * fps breaks a
 * rally). Optional in the pipeline JSON (legacy files omit it); the front-end
 * derives it client-side as a fallback (see deriveRallies).
 */
export interface Rally {
  /** inclusive frame of the first event (bounce or shot) */
  start_frame: number;
  /** inclusive frame of the last event */
  end_frame: number;
  /** shot frames that belong to this rally, ascending */
  shot_frames: number[];
  /** bounce frames that belong to this rally, ascending */
  bounce_frames: number[];
  /** number of shots — the natural "length" of a rally */
  n_shots: number;
  /** server-computed geometric outcome (vision/rally_outcome.py). Absent on
   * legacy fixtures → the front-end may derive a fallback winner heuristic. */
  outcome?: RallyOutcomeInfo;
  /** stable human identity of the player who controlled the rally (match-mode).
   *  Absent in training mode / legacy fixtures. */
  human_id?: string;
}

/**
 * Per-rally geometric outcome, computed by vision/rally_outcome.py and emitted
 * on each rally in the pipeline _stats.json. This is the SINGLE source of truth
 * for winner/forced/unforced — the front-end reads it instead of re-deriving.
 *
 * Confidence note: `forced_error` is geometrically unreliable (can't tell
 * "player overwhelmed" from "bad shot"). The UI treats it as low-confidence
 * (grouped with neutral / distinct badge). `winner`, `unforced_error`, and
 * `neutral` are the solid labels.
 */
export type RallyOutcome = "winner" | "forced_error" | "unforced_error" | "neutral";

export interface RallyOutcomeInfo {
  outcome: RallyOutcome;
  /** frame to point the UI at (the defining shot/bounce) */
  defining_frame: number;
  /** human-readable justification (FR/EN), directly displayable */
  reason: string;
}

/** Highlight categories surfaced in the post-analysis summary. */
export type HighlightType =
  | "best_point" // best-played rally by the user
  | "worst_point" // lowest-quality / error rally
  | "longest" // most shots
  | "winner" // likely point-ending winning shot
  | "top_shot"; // single best stroke of the match

export interface Highlight {
  type: HighlightType;
  title: string;
  reason: string;
  /** frame the player should jump to (usually the defining shot) */
  frame: number;
  /** optional window [start, end] for replay */
  window?: [number, number];
  /** quick stat chips shown on the card */
  chips?: { label: string; value: string }[];
}

/**
 * Momentum timeline — net performance sampled on a fixed time grid.
 * Each bucket holds a signed score (positive = playing well) so the sparkline
 * can show runs of good/bad play.
 */
export interface MomentumBucket {
  frame: number;
  score: number;
}

/**
 * One detected side-change (match-mode). Emitted by the MatchTracker
 * (feat/match-mode) on `match.side_swaps`. `ambiguous_jerseys` flags the case
 * the re-id can't resolve (identical kits) — the UI treats those as
 * low-confidence.
 */
export interface SideSwap {
  frame: number;
  confidence: number;
  ambiguous_jerseys: boolean;
  jersey_distance: number;
}

/** Match-mode summary block (run_pipeline_8s.py --match-mode). */
export interface MatchInfo {
  side_swaps: SideSwap[];
  n_swaps: number;
}

/**
 * Per-player fatigue block (vision/fatigue.py). The label is keyed on SPEED
 * decay only (not shot quality) — see memory courtside-fatigue-rally-outcome.
 * Typed 1:1 on the pipeline JSON (`stats_fatigue` in run_pipeline_8s.py);
 * unknown extra fields stay tolerated for forward-compat.
 */
export interface FatigueInfo {
  /** near-player shot-speed trend over the clip, km/h per minute (negative = slowing). */
  slope_kmh_per_min?: number;
  slope_quality_per_min?: number;
  /** average shot speed over the first / last third of the clip (km/h). */
  first_third_avg_speed?: number;
  last_third_avg_speed?: number;
  /** relative speed drop first→last third (0.12 = −12 %). Negative = got faster. */
  drop_pct?: number;
  fatigued?: boolean;
  /** sample-size confidence emitted by the pipeline ("none" = too few shots). */
  confidence?: string;
  /** representative late-match frame (jump target for "voir le moment"). */
  sample_frame?: number;
  n_near_shots?: number;
  [k: string]: unknown;
}

export interface PipelineStats {
  video: string;
  fps: number;
  frame_range: [number, number];
  speed_source: SpeedSource;
  homography_confidence: HomographyConfidence;
  /**
   * 3x3 image-px → court-meters homography (row-major; origin at court CENTER,
   * +Y toward the far baseline — the exact convention projection.ts applies).
   * Emitted by run_pipeline_8s.py when a court homography/calibration exists;
   * null/absent otherwise. When present the minimap is metric-exact and no
   * synthetic H is needed.
   */
  homography_H?: number[][] | null;
  summary: Summary;
  bounces: Bounce[];
  shots: Shot[];
  /** optional server-computed rallies (run_pipeline_8s.py). Absent on legacy files. */
  rallies?: Rally[];
  /** true when the clip was analyzed in match-mode (--match-mode). */
  match_mode?: boolean;
  /** match-mode side-change summary (present when match_mode). */
  match?: MatchInfo;
  /** optional per-player fatigue block (vision/fatigue.py). */
  fatigue?: FatigueInfo;
}

/**
 * Cut-timeline index — sidecar emitted next to the annotated video by the
 * shot-guard cut pass (feat/shot-guard-cut, S2). The annotated video is SHORTER
 * than the original (non-court frames removed), so frame N in the annotated clip
 * is NOT frame N in the source. This maps between the two.
 *
 * Served at `/annotated/<id>_clipindex.json`. When absent the front-end falls
 * back to the identity mapping (no cut) — see `lib/video/clip-index.ts`.
 */
export interface ClipIndex {
  fps: number;
  /** total frame count of the ORIGINAL (uncut) video. */
  original_total: number;
  /** total frame count of the ANNOTATED (cut) video. */
  annotated_total: number;
  /** kept_original_frames[k] = original frame index of the k-th annotated frame. */
  kept_original_frames: number[];
  /** removed spans in ORIGINAL frame indices, inclusive [start, end]. */
  cut_spans: [number, number][];
  /** original frame index the analysis window started at (== frame_range[0]). */
  start_frame: number;
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  frame: number;
}

export interface PlayerPosition {
  id: 1 | 2;
  x: number;
  y: number;
  frame: number;
}

export interface Project {
  id: string;
  name: string;
  video: string;
  createdAt: string;
  status: "ready";
  /** training = caméra fixe (défaut) | match = side-swap entre jeux */
  matchType: MatchType;
  stats: PipelineStats;
  /** synthetic ball trajectory (px), for the minimap comet trail */
  trajectory?: TrajectoryPoint[];
  /** optional player puck positions (px) per frame-snapshot */
  players?: PlayerPosition[];
  /** optional court homography 3x3 (row-major). When absent, minimap falls back geometrically with `~`. */
  H?: number[][];
  /** representative annotated frame path (mock poster) */
  posterFrame?: number;
  /** optional cut-timeline index (annotated↔original frame mapping). In
   *  production this is fetched from `/annotated/<id>_clipindex.json`; the mock
   *  fixtures may inline one for testing the realignment. */
  clipIndex?: ClipIndex;
}
