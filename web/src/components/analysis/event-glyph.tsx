import { depthHex, strokeHex } from "@/lib/format";
import type { Depth, ShotStroke } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Shot vs bounce — one consistent visual vocabulary across the whole app.
 *
 * The two were getting confused because lists rendered them with the same
 * neutral styling. The rule, now enforced everywhere this glyph is used:
 *   • BOUNCE = a filled DISC in the warm depth palette (clay/amber/ball),
 *     i.e. "where the ball landed". Same colors as the radar dots.
 *   • SHOT   = a hollow CHEVRON/racquet stroke in the cool stroke palette
 *     (forehand=cyan, backhand=violet, unknown=slate), i.e. "a strike".
 *
 * Color is the same source of truth as the minimap (depthHex) and the new
 * strokeHex, so a green disc on the radar reads as the same thing as a green
 * disc in a list. Keep this the single place that decides those shapes/colors.
 */

export function BounceGlyph({ depth, size = 12, className }: { depth: Depth; size?: number; className?: string }) {
  const c = depthHex(depth);
  return (
    <span
      className={cn("inline-block shrink-0 rounded-full ring-1 ring-black/40", className)}
      style={{ width: size, height: size, background: c }}
      aria-hidden
    />
  );
}

export function ShotGlyph({ stroke, size = 13, className }: { stroke: ShotStroke; size?: number; className?: string }) {
  const c = strokeHex(stroke);
  // A small upward chevron — "a strike", visually distinct from a round bounce.
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      className={cn("inline-block shrink-0", className)}
      aria-hidden
    >
      <path
        d="M3 11 L8 5 L13 11"
        fill="none"
        stroke={c}
        strokeWidth={2.2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

type EventGlyphProps =
  | { kind: "bounce"; depth: Depth; size?: number; className?: string }
  | { kind: "shot"; stroke: ShotStroke; size?: number; className?: string };

/** Discriminated convenience wrapper. */
export function EventGlyph(props: EventGlyphProps) {
  if (props.kind === "bounce") {
    return <BounceGlyph depth={props.depth} size={props.size} className={props.className} />;
  }
  return <ShotGlyph stroke={props.stroke} size={props.size} className={props.className} />;
}

/** A compact legend chip pair, reusable in section headers. */
export function ShotBounceLegend({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-3 text-[11px] text-muted-foreground", className)}>
      <span className="flex items-center gap-1.5">
        <ShotGlyph stroke="forehand" size={12} /> Frappe
      </span>
      <span className="flex items-center gap-1.5">
        <BounceGlyph depth="mid" size={10} /> Rebond
      </span>
    </div>
  );
}
