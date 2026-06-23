import { cn } from "@/lib/utils";

/**
 * CourtSide brand mark — an abstract "court tile": a rounded square holding the
 * service-box lines, with a single ball-yellow trajectory arc sweeping across.
 * Reads as court + motion, no literal tennis ball.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("h-7 w-7", className)} aria-hidden>
      {/* court tile */}
      <rect
        x="2.5"
        y="2.5"
        width="27"
        height="27"
        rx="7"
        fill="var(--color-coal-800)"
        stroke="color-mix(in oklab, white 12%, transparent)"
        strokeWidth="1"
      />
      {/* court lines (chalk) */}
      <g stroke="color-mix(in oklab, var(--color-chalk) 45%, transparent)" strokeWidth="1.4" strokeLinecap="round">
        <line x1="16" y1="6" x2="16" y2="26" />
        <line x1="8" y1="16" x2="24" y2="16" />
      </g>
      {/* trajectory arc — the signature */}
      <path
        d="M6 24 Q 16 2, 26 14"
        fill="none"
        stroke="var(--color-ball)"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      {/* impact point */}
      <circle cx="26" cy="14" r="2.6" fill="var(--color-ball)" />
    </svg>
  );
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <LogoMark className="h-7 w-7" />
      <span className="font-display text-lg font-extrabold tracking-tight">
        Court<span className="text-ball">Side</span>
      </span>
    </span>
  );
}
