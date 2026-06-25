import type { Depth, ShotStroke } from "@/lib/types";

export function fmtKmh(v: number, digits = 1): string {
  return v.toFixed(digits);
}

export function depthLabel(d: Depth): string {
  return d === "deep" ? "Profond" : d === "mid" ? "Moyen" : "Court";
}

export function depthColorClass(d: Depth): string {
  return d === "deep"
    ? "text-clay"
    : d === "mid"
      ? "text-court-amber"
      : "text-ball";
}

export function depthHex(d: Depth): string {
  // warm "broadcast clay" depth palette (deep=clay, mid=amber, short=ball)
  return d === "deep" ? "#e2603a" : d === "mid" ? "#f2b441" : "#d8f64a";
}

export function strokeLabel(s: ShotStroke): string {
  return s === "forehand" ? "Coup droit" : s === "backhand" ? "Revers" : "Frappe";
}

/** Accent hex per stroke — used to keep shots visually distinct from bounces
 *  (bounces use the warm depth palette; shots use cool stroke colors). */
export function strokeHex(s: ShotStroke): string {
  // forehand = cyan, backhand = violet, unknown = neutral slate.
  return s === "forehand" ? "#38bdf8" : s === "backhand" ? "#a78bfa" : "#94a3b8";
}

export function sideLabel(s: "near" | "far"): string {
  return s === "near" ? "Proche" : "Lointain";
}

export function frameToTimecode(frame: number, fps: number): string {
  const totalSec = frame / fps;
  const m = Math.floor(totalSec / 60);
  const s = Math.floor(totalSec % 60);
  const cs = Math.floor((totalSec * 100) % 100);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
}

export function qualityLabel(q: number): { label: string; className: string } {
  if (q >= 80) return { label: "Excellent", className: "text-ball" };
  if (q >= 60) return { label: "Bon", className: "text-court-amber" };
  if (q >= 40) return { label: "Moyen", className: "text-court-amber" };
  return { label: "Faible", className: "text-clay" };
}

export function qualityHex(q: number): string {
  if (q >= 80) return "#d8f64a";
  if (q >= 40) return "#f2b441";
  return "#e2603a";
}
