import type { FatigueInfo, PipelineStats, Rally, RallyOutcome } from "@/lib/types";

/**
 * Post-analysis derivations that read the pipeline JSON as-is (no synthetic
 * data): stroke-type comparison, rally-outcome aggregate, fatigue view and the
 * ranked "key insights" strip. Every insight carries a frame so the UI can
 * keep the frame-as-truth contract (click → jump to the moment).
 */

// ── strokes ──────────────────────────────────────────────────────────

export interface StrokeStat {
  stroke: "forehand" | "backhand";
  count: number;
  avgSpeed: number | null;
  maxSpeed: number | null;
  avgQuality: number | null;
  /** frame of the fastest shot of this stroke (jump target). */
  bestFrame: number | null;
}

export function deriveStrokeStats(stats: PipelineStats): StrokeStat[] {
  return (["forehand", "backhand"] as const).map((stroke) => {
    const shots = stats.shots.filter((s) => s.stroke === stroke);
    const speeds = shots.filter((s) => s.speed_kmh > 0);
    const quals = shots.filter((s) => s.quality > 0);
    const best = speeds.length
      ? speeds.reduce((a, b) => (b.speed_kmh > a.speed_kmh ? b : a))
      : null;
    return {
      stroke,
      count: shots.length,
      avgSpeed: speeds.length
        ? speeds.reduce((a, s) => a + s.speed_kmh, 0) / speeds.length
        : null,
      maxSpeed: best ? best.speed_kmh : null,
      avgQuality: quals.length
        ? quals.reduce((a, s) => a + s.quality, 0) / quals.length
        : null,
      bestFrame: best ? best.frame : null,
    };
  });
}

// ── rally outcomes ───────────────────────────────────────────────────

export interface OutcomeSlice {
  outcome: RallyOutcome;
  count: number;
  /** first rally with this outcome (jump target = its defining frame). */
  firstFrame: number | null;
}

export interface OutcomeSummary {
  slices: OutcomeSlice[];
  /** rallies that carry an outcome label at all. */
  total: number;
  /** unforced errors / (winners + unforced) — the classic "clean tennis" ratio. */
  unforcedShare: number | null;
}

/** Display order = positive → neutral → negative (stacked-bar segment order). */
export const OUTCOME_ORDER: RallyOutcome[] = [
  "winner",
  "neutral",
  "forced_error",
  "unforced_error",
];

export function deriveOutcomeSummary(rallies: Rally[] | undefined): OutcomeSummary {
  const withOutcome = (rallies ?? []).filter((r) => r.outcome);
  const slices = OUTCOME_ORDER.map((outcome) => {
    const matching = withOutcome.filter((r) => r.outcome!.outcome === outcome);
    return {
      outcome,
      count: matching.length,
      firstFrame: matching.length ? matching[0].outcome!.defining_frame : null,
    };
  });
  const winners = slices.find((s) => s.outcome === "winner")!.count;
  const unforced = slices.find((s) => s.outcome === "unforced_error")!.count;
  return {
    slices,
    total: withOutcome.length,
    unforcedShare:
      winners + unforced > 0 ? unforced / (winners + unforced) : null,
  };
}

// ── fatigue ──────────────────────────────────────────────────────────

export interface FatigueView {
  usable: boolean;
  confidence: string;
  firstThird: number | null;
  lastThird: number | null;
  /** signed % (positive = slower at the end). */
  dropPct: number | null;
  slopeKmhPerMin: number | null;
  fatigued: boolean;
  sampleFrame: number | null;
  nShots: number;
}

export function deriveFatigue(f: FatigueInfo | undefined): FatigueView | null {
  if (!f) return null;
  const conf = typeof f.confidence === "string" ? f.confidence : "none";
  const first = typeof f.first_third_avg_speed === "number" ? f.first_third_avg_speed : null;
  const last = typeof f.last_third_avg_speed === "number" ? f.last_third_avg_speed : null;
  return {
    usable: conf !== "none" && first !== null && last !== null,
    confidence: conf,
    firstThird: first,
    lastThird: last,
    dropPct: typeof f.drop_pct === "number" ? f.drop_pct * 100 : null,
    slopeKmhPerMin:
      typeof f.slope_kmh_per_min === "number" ? f.slope_kmh_per_min : null,
    fatigued: f.fatigued === true,
    sampleFrame: typeof f.sample_frame === "number" ? f.sample_frame : null,
    nShots: typeof f.n_near_shots === "number" ? f.n_near_shots : 0,
  };
}

// ── rhythm (inter-shot cadence) ─────────────────────────────────────

/** Median seconds between consecutive shots INSIDE rallies (rally rhythm). */
export function deriveCadenceSec(stats: PipelineStats): number | null {
  const gaps: number[] = [];
  for (const r of stats.rallies ?? []) {
    const fr = r.shot_frames;
    for (let i = 1; i < fr.length; i++) gaps.push((fr[i] - fr[i - 1]) / stats.fps);
  }
  if (!gaps.length) return null;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

// ── key insights (the actionable strip) ─────────────────────────────

export interface KeyInsight {
  id: string;
  /** short claim, FR, coach voice. */
  title: string;
  /** the number(s) behind the claim. */
  detail: string;
  tone: "good" | "warn" | "info";
  frame: number | null;
}

/**
 * Ranked, data-grounded findings. Each rule only fires when its signal is
 * strong enough to be worth a coach's sentence — no filler. Capped at 4.
 */
export function deriveKeyInsights(stats: PipelineStats): KeyInsight[] {
  const out: KeyInsight[] = [];
  const [fh, bh] = deriveStrokeStats(stats);

  // 1 — stroke speed gap (needs both strokes measured, gap ≥ 5 km/h)
  if (fh.avgSpeed !== null && bh.avgSpeed !== null && fh.count >= 2 && bh.count >= 2) {
    const gap = fh.avgSpeed - bh.avgSpeed;
    if (Math.abs(gap) >= 5) {
      const slower = gap > 0 ? "revers" : "coup droit";
      const faster = gap > 0 ? "coup droit" : "revers";
      out.push({
        id: "stroke-gap",
        title: `Ton ${slower} est ${Math.abs(gap).toFixed(0)} km/h plus lent que ton ${faster}`,
        detail: `${fh.avgSpeed.toFixed(0)} km/h en coup droit vs ${bh.avgSpeed.toFixed(0)} km/h en revers`,
        tone: "warn",
        frame: (gap > 0 ? bh : fh).bestFrame,
      });
    }
  }

  // 2 — fatigue (pipeline-flagged, or a ≥ 8 % measured drop)
  const fat = deriveFatigue(stats.fatigue);
  if (fat?.usable && fat.dropPct !== null && (fat.fatigued || fat.dropPct >= 8)) {
    out.push({
      id: "fatigue",
      title: `Baisse de régime en fin de session : −${fat.dropPct.toFixed(0)} % de vitesse`,
      detail: `${fat.firstThird!.toFixed(0)} km/h sur le premier tiers → ${fat.lastThird!.toFixed(0)} km/h sur le dernier`,
      tone: "warn",
      frame: fat.sampleFrame,
    });
  }

  // 3 — unforced-error share (≥ 3 decided rallies to be meaningful)
  const oc = deriveOutcomeSummary(stats.rallies);
  const winners = oc.slices.find((s) => s.outcome === "winner")!;
  const unforced = oc.slices.find((s) => s.outcome === "unforced_error")!;
  if (winners.count + unforced.count >= 3 && oc.unforcedShare !== null) {
    const pct = Math.round(oc.unforcedShare * 100);
    if (pct >= 50) {
      out.push({
        id: "unforced",
        title: `${pct} % de tes points décisifs sont des fautes directes`,
        detail: `${unforced.count} fautes directes pour ${winners.count} points gagnants`,
        tone: "warn",
        frame: unforced.firstFrame,
      });
    } else {
      out.push({
        id: "winners",
        title: `Tu gagnes proprement : ${winners.count} points gagnants pour ${unforced.count} faute${unforced.count > 1 ? "s" : ""} directe${unforced.count > 1 ? "s" : ""}`,
        detail: `${100 - pct} % de tes points décisifs sont des winners`,
        tone: "good",
        frame: winners.firstFrame,
      });
    }
  }

  // 4 — depth bias (short balls dominate → easy attack for the opponent)
  const { deep, mid, short } = stats.summary.depth;
  const totalB = deep + mid + short;
  if (totalB >= 5) {
    const shortPct = Math.round((short / totalB) * 100);
    const deepPct = Math.round((deep / totalB) * 100);
    if (shortPct >= 45) {
      const firstShort = stats.bounces.find((b) => b.depth === "short");
      out.push({
        id: "depth-short",
        title: `${shortPct} % de tes balles retombent courtes`,
        detail: "Une balle courte invite l'adversaire à attaquer — vise la profondeur",
        tone: "warn",
        frame: firstShort ? firstShort.frame : null,
      });
    } else if (deepPct >= 45) {
      const bestDeep = stats.bounces.find((b) => b.depth === "deep");
      out.push({
        id: "depth-deep",
        title: `Belle profondeur : ${deepPct} % de balles profondes`,
        detail: "La profondeur neutralise l'attaque adverse — continue",
        tone: "good",
        frame: bestDeep ? bestDeep.frame : null,
      });
    }
  }

  // 5 — rhythm (only if nothing else fired much; keeps the strip full but honest)
  if (out.length < 3) {
    const cad = deriveCadenceSec(stats);
    if (cad !== null) {
      out.push({
        id: "cadence",
        title: `Rythme d'échange : une frappe toutes les ${cad.toFixed(1)} s`,
        detail: "Cadence médiane entre deux frappes au sein d'un échange",
        tone: "info",
        frame: null,
      });
    }
  }

  return out.slice(0, 4);
}
