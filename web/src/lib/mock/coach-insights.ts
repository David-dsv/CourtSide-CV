/**
 * Mock coach-insight generator — grounded on the real pipeline stats.
 * In production this is replaced by an LLM structured-output call (JSON schema,
 * 1:1 mapping to detected events, frame citations, no hallucination). Here we
 * derive a handful of deterministic insights from summary + bounces + shots so
 * the UI is real and clickable.
 */
import type { Project } from "@/lib/types";

export interface Insight {
  type: "strength" | "weakness" | "recommendation";
  text: string;
  frames: number[];
}

export function generateCoachInsights(project: Project): Insight[] {
  const { summary, bounces, shots } = project.stats;
  const out: Insight[] = [];

  const total = summary.total_bounces || 1;
  const shortPct = (summary.depth.short / total) * 100;
  const deepPct = (summary.depth.deep / total) * 100;

  // depth weakness
  if (shortPct > 50) {
    out.push({
      type: "weakness",
      text: `${Math.round(shortPct)}% de vos balles atterrissent dans la zone courte. Adversaire facile à attaque au filet. Visez plus de profondeur, surtout en revers.`,
      frames: bounces.filter((b) => b.depth === "short").slice(0, 4).map((b) => b.frame),
    });
  }

  // depth strength
  if (deepPct > 30) {
    out.push({
      type: "strength",
      text: `${summary.depth.deep} balles profondes (${Math.round(deepPct)}%) — bon travail pour repousser l'adversaire derrière la ligne de fond.`,
      frames: bounces.filter((b) => b.depth === "deep").slice(0, 4).map((b) => b.frame),
    });
  }

  // speed
  if (summary.speed_kmh.avg > 120) {
    out.push({
      type: "strength",
      text: `Vitesse moyenne élevée (${summary.speed_kmh.avg.toFixed(0)} km/h, max ${summary.speed_kmh.max.toFixed(0)}). Votre puissance offensive est un atout.`,
      frames: shots.slice(0, 3).map((s) => s.frame),
    });
  }

  // forehand vs backhand
  const fh = summary.strokes.forehand;
  const bh = summary.strokes.backhand;
  if (fh > bh * 1.5 && bh > 0) {
    out.push({
      type: "recommendation",
      text: `Vous frappez ${fh} coups droits pour seulement ${bh} revers. L'adversaire va cibler votre revers — travaillez sa profondeur et sa vitesse.`,
      frames: shots.filter((s) => s.stroke === "backhand").slice(0, 3).map((s) => s.frame),
    });
  }

  // quality
  if (summary.quality.avg < 50) {
    out.push({
      type: "weakness",
      text: `Qualité moyenne faible (${Math.round(summary.quality.avg)}/100). Concentrez-vous sur le placement plutôt que la puissance — plusieurs frappes sont sorties du court.`,
      frames: bounces.filter((b) => b.quality < 40).slice(0, 4).map((b) => b.frame),
    });
  } else if (summary.quality.high_count > 0) {
    out.push({
      type: "strength",
      text: `${summary.quality.high_count} frappe(s) de haute qualité (≥70). Reproduisez ces séquences : vitesse + profondeur combinées.`,
      frames: bounces.filter((b) => b.quality >= 70).slice(0, 4).map((b) => b.frame),
    });
  }

  // momentum (naive: cluster of low-quality bounces)
  out.push({
    type: "recommendation",
    text: `Analysez votre momentum : repérez les séquences d'erreurs consécutives pour identifier les moments de perte de concentration.`,
    frames: bounces.slice(-3).map((b) => b.frame),
  });

  return out.slice(0, 6);
}
