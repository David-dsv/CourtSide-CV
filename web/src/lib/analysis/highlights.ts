/**
 * Rally segmentation + highlight + momentum derivation.
 *
 * Everything here is a PURE function of the pipeline stats JSON (bounces +
 * shots). No LLM, no hallucination: every highlight points to a real frame,
 * every rally is a real temporal cluster of shots.
 *
 * The pipeline may compute rallies server-side (run_pipeline_8s.py emits
 * `rallies[]`); when absent we derive them here so legacy fixtures keep
 * working. Highlights + momentum are always derived client-side because they
 * are presentation concerns (and cheap).
 */
import type {
  Bounce,
  Highlight,
  HighlightType,
  MomentumBucket,
  PipelineStats,
  Rally,
  Shot,
} from "@/lib/types";

/** A rally breaks when the gap between consecutive shots exceeds this many seconds. */
const RALLY_GAP_SEC = 2.5;
/** Bucket size for the momentum timeline, in seconds. */
const MOMENTUM_BUCKET_SEC = 10;
/** Frames of context shown before a highlight jump (replay lead-in). */
const HIGHLIGHT_LEAD_FRAMES_SEC = 1.5;

export interface ShotWithBounce extends Shot {
  /** the next bounce after this shot (the shot's "result"), if any */
  resultBounce?: Bounce;
}

/** Group shots into rallies by inter-shot gap. Falls back to bounces if no shots. */
export function deriveRallies(stats: PipelineStats): Rally[] {
  const { fps, shots, bounces } = stats;
  const gapFrames = RALLY_GAP_SEC * fps;

  // Build a unified, frame-sorted event list of shot frames.
  const shotFrames = [...shots].map((s) => s.frame).sort((a, b) => a - b);
  const bounceFrames = [...bounces].map((b) => b.frame).sort((a, b) => a - b);

  if (shotFrames.length === 0) return [];

  const groups: number[][] = [];
  let cur: number[] = [shotFrames[0]];
  for (let i = 1; i < shotFrames.length; i++) {
    if (shotFrames[i] - shotFrames[i - 1] > gapFrames) {
      groups.push(cur);
      cur = [];
    }
    cur.push(shotFrames[i]);
  }
  groups.push(cur);

  const rallies = groups.map((gf) => {
    const start = gf[0];
    const end = gf[gf.length - 1];
    const bf = bounceFrames.filter((f) => f >= start && f <= end + gapFrames);
    return {
      start_frame: start,
      end_frame: Math.max(end, bf[bf.length - 1] ?? end),
      shot_frames: gf,
      bounce_frames: bf,
      n_shots: gf.length,
    };
  });

  return mergeSoloRallies(rallies);
}

/**
 * Re-attribute any 1-shot rally ("solo") into its nearest neighbour (prev or
 * next), absorbing its bounces. A solo arises when a shot sits >gap from both
 * neighbours — i.e. a genuine isolated strike — but surfacing it as its own
 * point is noise in the highlights. Merging keeps the break detection honest
 * (the gap was real) while avoiding single-shot "rallies" in the summary.
 * If the solo is equidistant from prev/next it goes to the previous (chronological).
 */
function mergeSoloRallies(rallies: Rally[]): Rally[] {
  if (rallies.length <= 1) return rallies;
  // Work on mutable copies.
  const out = rallies.map((r) => ({
    ...r,
    shot_frames: [...r.shot_frames],
    bounce_frames: [...r.bounce_frames],
  }));
  for (let i = 0; i < out.length; i++) {
    const r = out[i];
    if (r.n_shots !== 1) continue;
    const prev = out[i - 1];
    const next = out[i + 1];
    const gapPrev = prev ? r.start_frame - prev.end_frame : Infinity;
    const gapNext = next ? next.start_frame - r.end_frame : Infinity;
    // Absorb into whichever neighbour is closer.
    if (gapPrev <= gapNext && prev) {
      prev.shot_frames.push(...r.shot_frames);
      prev.bounce_frames.push(...r.bounce_frames);
      prev.end_frame = Math.max(prev.end_frame, r.end_frame);
      prev.n_shots += r.n_shots;
      out.splice(i, 1);
      i--; // re-examine this index (shifted)
    } else if (next) {
      next.shot_frames = [...r.shot_frames, ...next.shot_frames];
      next.bounce_frames = [...r.bounce_frames, ...next.bounce_frames];
      next.start_frame = Math.min(next.start_frame, r.start_frame);
      next.n_shots += r.n_shots;
      out.splice(i, 1);
      i--;
    }
    // If neither neighbour exists (shouldn't happen with length>1), leave it.
  }
  return out;
}

/** Attach each shot to the bounce that follows it (the shot's result). */
export function shotsWithResults(stats: PipelineStats): ShotWithBounce[] {
  const sortedBounces = [...stats.bounces].sort((a, b) => a.frame - b.frame);
  return [...stats.shots]
    .sort((a, b) => a.frame - b.frame)
    .map((s) => {
      const resultBounce = sortedBounces.find((b) => b.frame > s.frame);
      return { ...s, resultBounce };
    });
}

/** Attach each shot to its enclosing rally (by frame), if any. */
function shotToRally(rallies: Rally[]): Map<number, Rally> {
  const m = new Map<number, Rally>();
  for (const r of rallies) for (const f of r.shot_frames) m.set(f, r);
  return m;
}

/** Average quality of the user's (near-side) shots in a rally. */
function userQuality(rally: Rally, stats: PipelineStats): number | null {
  const userShots = rally.shot_frames
    .map((f) => stats.shots.find((s) => s.frame === f && s.player_side === "near"))
    .filter((s): s is Shot => Boolean(s));
  if (userShots.length === 0) return null;
  return userShots.reduce((a, s) => a + s.quality, 0) / userShots.length;
}

/**
 * Compute the highlights. Deterministic, picks at most one per category (with
 * ties broken by frame). Returns an empty array if there's nothing to show.
 */
export function deriveHighlights(stats: PipelineStats): Highlight[] {
  const rallies = deriveRallies(stats);
  const shotRally = shotToRally(rallies);
  const out: Highlight[] = [];
  const fps = stats.fps;
  const lead = Math.round(HIGHLIGHT_LEAD_FRAMES_SEC * fps);

  // ── Top single shot: max(speed * quality) ──
  const sortedShots = [...stats.shots].sort((a, b) => a.frame - b.frame);
  if (sortedShots.length > 0) {
    let best = sortedShots[0];
    let bestScore = -1;
    for (const s of sortedShots) {
      const score = s.speed_kmh * (s.quality / 100);
      if (score > bestScore) {
        bestScore = score;
        best = s;
      }
    }
    out.push({
      type: "top_shot",
      title: "Frappe du match",
      reason: `Meilleur compromis vitesse × qualité : ${Math.round(best.speed_kmh)} km/h, qualité ${best.quality}.`,
      frame: best.frame,
      window: [Math.max(0, best.frame - lead), best.frame],
      chips: [
        { label: "Vitesse", value: `${Math.round(best.speed_kmh)} km/h` },
        { label: "Qualité", value: `${best.quality}/100` },
      ],
    });
  }

  // ── Longest rally ──
  if (rallies.length > 0) {
    const longest = [...rallies].sort((a, b) => b.n_shots - a.n_shots || a.start_frame - b.start_frame)[0];
    out.push({
      type: "longest",
      title: "Échange le plus long",
      reason: `${longest.n_shots} frappes en ${((longest.end_frame - longest.start_frame) / fps).toFixed(1)} s.`,
      frame: longest.shot_frames[Math.floor(longest.shot_frames.length / 2)] ?? longest.start_frame,
      window: [longest.start_frame, longest.end_frame],
      chips: [
        { label: "Frappes", value: `${longest.n_shots}` },
        { label: "Durée", value: `${((longest.end_frame - longest.start_frame) / fps).toFixed(1)} s` },
      ],
    });
  }

  // ── Best & worst points (by user quality), need user shots ──
  const ralliesWithUser = rallies
    .map((r) => ({ r, q: userQuality(r, stats) }))
    .filter((x): x is { r: Rally; q: number } => x.q !== null);

  if (ralliesWithUser.length > 0) {
    const ranked = [...ralliesWithUser].sort((a, b) => b.q - a.q || a.r.start_frame - b.r.start_frame);
    const best = ranked[0];
    const worst = ranked[ranked.length - 1];

    // only emit best/worst if they're actually distinct rallies
    if (best.r !== worst.r) {
      out.push({
        type: "best_point",
        title: "Meilleur point",
        reason: `Qualité moyenne ${Math.round(best.q)}/100 sur vos frappes de cet échange.`,
        frame: best.r.shot_frames[Math.floor(best.r.shot_frames.length / 2)] ?? best.r.start_frame,
        window: [best.r.start_frame, best.r.end_frame],
        chips: [
          { label: "Qualité", value: `${Math.round(best.q)}/100` },
          { label: "Frappes", value: `${best.r.n_shots}` },
        ],
      });
      out.push({
        type: "worst_point",
        title: "Point à revoir",
        reason: `Qualité moyenne ${Math.round(worst.q)}/100 — la frappe à revoir.`,
        frame: worst.r.shot_frames[Math.floor(worst.r.shot_frames.length / 2)] ?? worst.r.start_frame,
        window: [worst.r.start_frame, worst.r.end_frame],
        chips: [{ label: "Qualité", value: `${Math.round(worst.q)}/100` }],
      });
    }
  }

  // ── Winner highlight ──
  // Single source of truth: if the backend classified a rally outcome, use it
  // (vision/rally_outcome.py). Only fall back to the local heuristic when no
  // outcome is present (legacy fixtures). Pick the most recent winner rally.
  const winnerFromBackend = rallies.findLast?.((r) => r.outcome?.outcome === "winner");
  const winnerRally = winnerFromBackend ?? null;

  if (winnerRally?.outcome) {
    const o = winnerRally.outcome;
    out.push({
      type: "winner",
      title: "Coup gagnant",
      reason: o.reason,
      frame: o.defining_frame,
      window: [winnerRally.start_frame, winnerRally.end_frame],
      chips: [{ label: "Échange", value: `${winnerRally.n_shots} frappes` }],
    });
  } else {
    // Fallback heuristic (legacy fixtures without backend outcome): last user
    // shot that is fast AND whose following bounce is deep.
    for (const s of [...stats.shots].sort((a, b) => b.frame - a.frame)) {
      if (s.player_side !== "near") continue;
      const r = shotRally.get(s.frame);
      if (!r) continue;
      const isLast = r.shot_frames[r.shot_frames.length - 1] === s.frame;
      if (!isLast) continue;
      const followingBounce = stats.bounces.find((b) => b.frame > s.frame && b.frame <= r.end_frame + 5 * fps);
      if (!followingBounce) continue;
      if (s.speed_kmh >= 110 && followingBounce.depth === "deep" && s.quality >= 55) {
        out.push({
          type: "winner",
          title: "Coup gagnant probable",
          reason: `Dernière frappe de l'échange à ${Math.round(s.speed_kmh)} km/h, balle profonde (qualité ${s.quality}).`,
          frame: s.frame,
          window: [Math.max(0, s.frame - lead), followingBounce.frame],
          chips: [
            { label: "Vitesse", value: `${Math.round(s.speed_kmh)} km/h` },
            { label: "Profondeur", value: "Profond" },
          ],
        });
        break; // one winner highlight is enough
      }
    }
  }

  // ── Unforced-error highlight (the solid complement to the winner) ──
  // Driven by the backend outcome (the reliable label). We deliberately do NOT
  // surface `forced_error` as a highlight — it's geometrically unreliable
  // (see OUTCOME_CONFIDENCE below). Pick the most recent unforced_error rally.
  const ueRally = rallies.findLast?.((r) => r.outcome?.outcome === "unforced_error");
  if (ueRally?.outcome) {
    const o = ueRally.outcome;
    // Avoid duplicating the worst_point card when it already points at the same rally.
    const worstFrame = out.find((h) => h.type === "worst_point")?.frame;
    const dupesWorst =
      worstFrame !== undefined &&
      ueRally.shot_frames.includes(worstFrame);
    if (!dupesWorst) {
      out.push({
        type: "worst_point",
        title: "Erreur non forcée",
        reason: o.reason,
        frame: o.defining_frame,
        window: [ueRally.start_frame, ueRally.end_frame],
        chips: [{ label: "Échange", value: `${ueRally.n_shots} frappes` }],
      });
    }
  }

  return out;
}

/**
 * Confidence level for each rally outcome label. `forced_error` is the only
 * LOW-confidence label — geometrically we can't reliably distinguish "the
 * player was overwhelmed" from "they misplayed". The UI should treat LOW as
 * grouped-with-neutral or a distinct, de-emphasised badge. The other three are
 * solid (HIGH) and can be surfaced as highlights.
 */
export const OUTCOME_CONFIDENCE: Record<import("@/lib/types").RallyOutcome, "high" | "low"> = {
  winner: "high",
  unforced_error: "high",
  neutral: "high",
  forced_error: "low",
};

/** True for outcome labels that are reliable enough to surface/promote. */
export function isSolidOutcome(o: import("@/lib/types").RallyOutcome): boolean {
  return OUTCOME_CONFIDENCE[o] === "high";
}

/**
 * Momentum timeline: bucket the user's shots by time, score each bucket by
 * the net of (good shots - poor shots). A smoothed signed score per window.
 */
export function deriveMomentum(stats: PipelineStats): MomentumBucket[] {
  const { fps, shots, frame_range } = stats;
  const bucket = MOMENTUM_BUCKET_SEC * fps;
  if (shots.length === 0) return [];

  const end = frame_range[1] || Math.max(...shots.map((s) => s.frame));
  const nBuckets = Math.max(1, Math.ceil((end - frame_range[0]) / bucket));
  const scores = new Array(nBuckets).fill(0);

  for (const s of shots) {
    if (s.player_side !== "near") continue;
    const idx = Math.min(nBuckets - 1, Math.floor((s.frame - frame_range[0]) / bucket));
    // +1 for quality ≥ 55, -1 for < 40 → net "playing well" signal
    scores[idx] += s.quality >= 55 ? 1 : s.quality < 40 ? -1 : 0;
  }

  return scores.map((score, i) => ({
    frame: Math.round(frame_range[0] + i * bucket),
    score,
  }));
}

/** Human label + accent color for a highlight type. */
export function highlightMeta(type: HighlightType): {
  label: string;
  accent: "green" | "red" | "cyan" | "amber";
} {
  switch (type) {
    case "best_point":
      return { label: "Top point", accent: "green" };
    case "worst_point":
      return { label: "À revoir", accent: "red" };
    case "longest":
      return { label: "Longueur", accent: "cyan" };
    case "winner":
      return { label: "Gagnant", accent: "green" };
    case "top_shot":
      return { label: "Frappe", accent: "amber" };
  }
}
