/**
 * Player profile & multi-session trends (MOCK).
 *
 * Synthesizes a "view across matches" for ONE player from the list of their
 * projects. This is the V1 of the multi-match overview: every number is
 * derived deterministically from the real per-project pipeline stats, so the
 * trends shown are genuine aggregates (no hallucination). Only the *fatigue*
 * score is a stub — see TODO below.
 *
 * BACKEND CONTRACT (for the FastAPI service to implement later):
 *   getPlayerProfile(playerId) → PlayerProfile
 * where PlayerProfile maps 1:1 to the types exported here. The frontend then
 * only swaps this mock module for a `fetch` call. The per-project numbers are
 * already real (listProjects → project.stats.summary), so the trends are
 * trustworthy today; the only fabricated field is `fatigue` (TODO).
 */
import { listProjects } from "@/lib/mock/projects";
import { generateCoachInsights, type Insight } from "@/lib/mock/coach-insights";
import { deriveMomentum } from "@/lib/analysis/highlights";
import type { Project } from "@/lib/types";

/** A single per-match metric sample (oldest → newest). */
export interface TrendPoint {
  /** ISO date of the match */
  date: string;
  /** project id — clickable (frame-as-truth) */
  projectId: string;
  /** average shot/bounce speed this match (km/h) */
  speedKmh: number;
  /** average quality this match (/100) */
  quality: number;
  /** share of bounces that landed deep (%) */
  deepPct: number;
  /** derived fatigue score 0..100 (higher = more fatigued). STUB — TODO backend */
  fatigue: number;
}

/** Top strengths / work axes aggregated across matches, each citing the best match. */
export interface AggregatedInsight {
  type: "strength" | "work";
  /** short label */
  label: string;
  /** one-line explanation grounded in the aggregate */
  text: string;
  /** the project that best illustrates it (clickable) */
  projectId: string;
  /** a representative frame inside that project */
  frames: number[];
  /** how many of the matches exhibited this */
  matchCount: number;
}

export interface PlayerProfile {
  /** player display name (mock — single shared persona) */
  player: string;
  /** matches oldest → newest */
  matches: TrendPoint[];
  /** "{sign}{pct}% ce mois" headline, e.g. "augmenté de 7%" — null if <2 matches */
  speedDeltaPct: number | null;
  qualityDeltaPct: number | null;
  /** latest-match averages, for the headline tiles */
  latest: { speedKmh: number; quality: number; deepPct: number; fatigue: number };
  topStrengths: AggregatedInsight[];
  workAxes: AggregatedInsight[];
}

/** Average speed across a match (prefer shot speed, fall back to bounce speed). */
function matchSpeed(p: Project): number {
  const s = p.stats.summary.speed_kmh;
  return s.avg;
}

/** Average quality across a match (bounces quality). */
function matchQuality(p: Project): number {
  return p.stats.summary.quality.avg;
}

/** % of bounces that landed deep this match. */
function matchDeepPct(p: Project): number {
  const d = p.stats.summary.depth;
  const total = d.deep + d.mid + d.short || 1;
  return (d.deep / total) * 100;
}

/**
 * Derived fatigue score (0..100, higher = more fatigued).
 *
 * TODO(backend): this is a PLACEHOLDER. The real fatigue signal will come from
 * the session-B backend (physical-load model: shot-speed decay across the
 * match, rally length, motion intensity). Until that lands we proxy fatigue
 * from the momentum trend of the match: a net-negative momentum in the second
 * half (performance dropping over time) reads as fatigued. This is honest
 * about being a proxy — it's labeled "~proxy" in the UI.
 */
function matchFatigue(p: Project): number {
  const buckets = deriveMomentum(p.stats);
  if (buckets.length < 2) return 30;
  const half = Math.floor(buckets.length / 2);
  const firstAvg = avg(buckets.slice(0, half).map((b) => b.score));
  const secondAvg = avg(buckets.slice(half).map((b) => b.score));
  // drop in performance → fatigue. Map a -2..+2 delta to 0..100.
  const drop = firstAvg - secondAvg; // positive = got worse
  const score = 50 + drop * 20;
  return Math.max(0, Math.min(100, Math.round(score)));
}

function avg(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}

/** Cluster per-match insights by theme → aggregated strengths / work axes. */
function aggregateInsights(
  projects: Project[],
): { strengths: AggregatedInsight[]; work: AggregatedInsight[] } {
  // For each project, get its insights and bucket by a coarse theme key
  // derived from the insight text keywords. Each bucket keeps the strongest
  // (most-cited) representative.
  type Bucket = {
    key: string;
    label: string;
    type: "strength" | "work";
    text: string;
    projectId: string;
    frames: number[];
    count: number;
    // ranking weight = how well it scores
    weight: number;
  };

  const classify = (ins: Insight): { key: string; label: string; type: "strength" | "work" } | null => {
    const t = ins.text.toLowerCase();
    if (ins.type === "weakness" && t.includes("court")) return { key: "short_depth", label: "Profondeur courte", type: "work" };
    if (ins.type === "strength" && t.includes("profond")) return { key: "deep_depth", label: "Profondeur", type: "strength" };
    if (ins.type === "strength" && (t.includes("vitesse") || t.includes("puissance"))) return { key: "power", label: "Puissance", type: "strength" };
    if (ins.type === "recommendation" && t.includes("revers")) return { key: "backhand", label: "Revers", type: "work" };
    if (ins.type === "weakness" && t.includes("qualité")) return { key: "quality_low", label: "Qualité", type: "work" };
    if (ins.type === "strength" && t.includes("qualité")) return { key: "quality_high", label: "Haute qualité", type: "strength" };
    if (ins.type === "strength") return { key: "general_str", label: "Régularité", type: "strength" };
    if (ins.type === "weakness" || ins.type === "recommendation") return { key: "general_work", label: "Constance", type: "work" };
    return null;
  };

  const buckets = new Map<string, Bucket>();
  for (const p of projects) {
    for (const ins of generateCoachInsights(p)) {
      const c = classify(ins);
      if (!c) continue;
      const existing = buckets.get(c.key);
      if (existing) {
        existing.count += 1;
        existing.weight += 1;
      } else {
        buckets.set(c.key, {
          key: c.key,
          label: c.label,
          type: c.type,
          text: ins.text,
          projectId: p.id,
          frames: ins.frames,
          count: 1,
          weight: 1,
        });
      }
    }
  }

  const toOut = (b: Bucket): AggregatedInsight => ({
    type: b.type,
    label: b.label,
    text: b.text,
    projectId: b.projectId,
    frames: b.frames,
    matchCount: b.count,
  });

  const all = [...buckets.values()];
  const strengths = all
    .filter((b) => b.type === "strength")
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3)
    .map(toOut);
  const work = all
    .filter((b) => b.type === "work")
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3)
    .map(toOut);

  return { strengths, work };
}

/**
 * Build the player profile from the mock project list.
 *
 * `forProjectId` (optional) — when called from a project page, scopes the
 * "latest" headline to that match context but still aggregates across ALL
 * the player's matches (the multi-session view is global). Today there is one
 * shared mock persona, so we aggregate all projects.
 */
export function getPlayerProfile(forProjectId?: string): PlayerProfile {
  const all = listProjects();
  // oldest → newest by createdAt
  const projects = [...all].sort(
    (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
  );

  const matches: TrendPoint[] = projects.map((p) => ({
    date: p.createdAt,
    projectId: p.id,
    speedKmh: matchSpeed(p),
    quality: matchQuality(p),
    deepPct: matchDeepPct(p),
    fatigue: matchFatigue(p),
  }));

  const pctDelta = (first: number | undefined, last: number | undefined): number | null => {
    if (first == null || last == null || first === 0 || matches.length < 2) return null;
    return ((last - first) / first) * 100;
  };

  const latestMatch = matches[matches.length - 1];
  const first = matches[0];

  const { strengths, work } = aggregateInsights(projects);

  return {
    player: forProjectId ? (all.find((p) => p.id === forProjectId)?.name.split(" — ")[0] ?? "Vous") : "Vous",
    matches,
    speedDeltaPct: pctDelta(first?.speedKmh, latestMatch?.speedKmh),
    qualityDeltaPct: pctDelta(first?.quality, latestMatch?.quality),
    latest: latestMatch
      ? {
          speedKmh: latestMatch.speedKmh,
          quality: latestMatch.quality,
          deepPct: latestMatch.deepPct,
          fatigue: latestMatch.fatigue,
        }
      : { speedKmh: 0, quality: 0, deepPct: 0, fatigue: 0 },
    topStrengths: strengths,
    workAxes: work,
  };
}
