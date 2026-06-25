/**
 * Cut-timeline realignment — annotated ⇄ original frame mapping.
 *
 * Why this exists (S2 contract): the annotated video the player loads is
 * SHORTER than the source. The shot-guard cut pass (feat/shot-guard-cut) drops
 * non-court frames (close-ups, replays, graphics — ~36% of tennis.mp4), so the
 * k-th frame of the annotated clip corresponds to some *later* frame in the
 * original. If the UI showed "frame 1200" naively, that would point at the
 * 1200-th annotated frame, which is NOT the 1200-th real frame.
 *
 * The sidecar `<id>_clipindex.json` (see {@link ClipIndex}) carries the map:
 *   kept_original_frames[k] === original frame index of annotated frame k.
 *
 * IMPORTANT — which space is which.
 * Everywhere else in the front-end, frames are expressed in ORIGINAL coordinates
 * (the pipeline `_stats.json` reports bounce/shot frames in the source video's
 * numbering, and `frame_range` is `[start_frame, …]` in the source). The ONLY
 * place we need annotated coordinates is when driving the actual `<video>`
 * element (its currentTime is in annotated time). So:
 *   - stats / scrubber / jump links  → ORIGINAL frames
 *   - video.currentTime ↔ frame      → ANNOTATED frames
 * This module is the single converter between the two.
 *
 * When no sidecar is present we use {@link IDENTITY}: original === annotated,
 * so an uncut clip behaves exactly as before this feature landed.
 */
import type { ClipIndex } from "@/lib/types";

/**
 * A ready-to-use mapper. `original` is the source-video frame number; `annotated`
 * is the index into the (possibly shorter) annotated clip. `hasCuts` is false
 * for the identity mapping so callers can cheaply skip cut-aware UI.
 */
export interface ClipMapper {
  readonly hasCuts: boolean;
  readonly fps: number;
  /** [first, last] ORIGINAL frame covered by the annotated clip. */
  readonly originalRange: [number, number];
  /** total annotated frames. */
  readonly annotatedTotal: number;
  /** annotated frame index → original frame number. */
  toOriginal(annotated: number): number;
  /** original frame number → nearest annotated frame index (clamped). For a
   *  frame inside a cut span, returns the next kept annotated frame. */
  toAnnotated(original: number): number;
  /** true when the given ORIGINAL frame was cut out of the annotated clip. */
  isCut(original: number): boolean;
}

/** Identity mapper — used when there is no `_clipindex.json` (uncut clip). */
export function identityMapper(fps: number, frameRange: [number, number]): ClipMapper {
  const [f0, f1] = frameRange;
  return {
    hasCuts: false,
    fps,
    originalRange: [f0, f1],
    annotatedTotal: Math.max(0, f1 - f0),
    // annotated frame index 0 == original f0.
    toOriginal: (a) => f0 + a,
    toAnnotated: (o) => Math.max(0, Math.round(o) - f0),
    isCut: () => false,
  };
}

/**
 * Build a mapper from a parsed {@link ClipIndex}. Defensive: a malformed or
 * empty index degrades to identity over its declared fps/range so the UI never
 * crashes on a bad sidecar.
 */
export function mapperFromClipIndex(ci: ClipIndex): ClipMapper {
  const kept = ci.kept_original_frames;
  if (!Array.isArray(kept) || kept.length === 0) {
    return identityMapper(ci.fps, [ci.start_frame, ci.start_frame + ci.annotated_total]);
  }
  const first = kept[0];
  const last = kept[kept.length - 1];
  // kept is ascending → binary search original→annotated.
  const cutSet = buildCutTester(ci.cut_spans ?? []);

  return {
    hasCuts: (ci.cut_spans?.length ?? 0) > 0,
    fps: ci.fps,
    originalRange: [first, last],
    annotatedTotal: kept.length,
    toOriginal(annotated: number): number {
      if (annotated <= 0) return first;
      if (annotated >= kept.length) return last;
      return kept[Math.round(annotated)];
    },
    toAnnotated(original: number): number {
      const o = Math.round(original);
      if (o <= first) return 0;
      if (o >= last) return kept.length - 1;
      // lower-bound: first kept frame >= o (next visible frame if o is in a cut).
      let lo = 0;
      let hi = kept.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (kept[mid] < o) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    },
    isCut: (original: number) => cutSet(Math.round(original)),
  };
}

/** Compile cut spans into a fast membership test (spans are few + sorted). */
function buildCutTester(spans: [number, number][]): (f: number) => boolean {
  if (spans.length === 0) return () => false;
  const sorted = [...spans].sort((a, b) => a[0] - b[0]);
  return (f: number) => {
    for (const [s, e] of sorted) {
      if (f < s) return false; // sorted → no later span can contain f
      if (f <= e) return true;
    }
    return false;
  };
}

/**
 * Resolve a mapper for a project. Prefers an inlined `project.clipIndex` (mock
 * fixtures); otherwise tries the network sidecar; falls back to identity.
 * Pure-ish: pass the already-fetched index or undefined.
 */
export function resolveMapper(
  fps: number,
  frameRange: [number, number],
  clipIndex?: ClipIndex,
): ClipMapper {
  return clipIndex ? mapperFromClipIndex(clipIndex) : identityMapper(fps, frameRange);
}

/** Public path of the sidecar served alongside the annotated video. */
export function clipIndexUrl(projectId: string): string {
  return `/annotated/${projectId}_clipindex.json`;
}
