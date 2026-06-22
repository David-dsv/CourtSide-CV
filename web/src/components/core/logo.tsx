import { cn } from "@/lib/utils";

/** CourtSide brand mark — tennis ball with the court seam curve. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("h-7 w-7", className)} aria-hidden>
      <circle cx="16" cy="16" r="14" fill="var(--color-ball)" />
      <circle cx="16" cy="16" r="14" fill="none" stroke="rgba(0,0,0,0.35)" strokeWidth="0.5" />
      {/* the iconic seam — two opposing curves */}
      <path
        d="M5 7 C 12 13, 12 19, 5 25"
        fill="none"
        stroke="#16150f"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M27 7 C 20 13, 20 19, 27 25"
        fill="none"
        stroke="#16150f"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
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
