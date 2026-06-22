"use client";

import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

export interface HeroDemoVideoProps {
  /** sources are served from /public; mp4 first, webm fallback */
  mp4: string;
  webm?: string;
  poster?: string;
  /** floating chrome label, e.g. "live court radar" */
  label?: string;
  className?: string;
}

/**
 * Autoplaying, looping, muted demo clip for the landing hero — the real annotated
 * pipeline output (locked skeletons, glow ball, court radar). Replaces the static
 * minimap so the hero has motion. Muted + playsInline so it autoplays on every
 * browser/mobile; poster shows instantly while the video buffers (no blank gap).
 */
export function HeroDemoVideo({
  mp4,
  webm,
  poster,
  label = "live court radar",
  className,
}: HeroDemoVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);

  return (
    <div className={cn("panel-raised relative p-3", className)}>
      {/* floating chrome label (matches the previous radar card) */}
      <div className="absolute -top-3 left-4 z-10 inline-flex items-center gap-2 rounded-md border border-white/10 bg-coal-900 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-ball opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-ball" />
        </span>
        {label}
      </div>

      <div className="relative overflow-hidden rounded-lg ring-1 ring-white/8">
        {/* poster fades out once the video can play, so there's never a blank frame */}
        <video
          ref={videoRef}
          className="block h-auto w-full"
          autoPlay
          loop
          muted
          playsInline
          preload="metadata"
          poster={poster}
          onCanPlay={() => setReady(true)}
          aria-label="Démonstration : analyse vidéo de tennis annotée par CourtSide"
        >
          {webm && <source src={webm} type="video/webm" />}
          <source src={mp4} type="video/mp4" />
        </video>

        {/* subtle top/bottom vignette so the floating label + edges read cleanly */}
        <div
          aria-hidden
          className={cn(
            "pointer-events-none absolute inset-0 bg-gradient-to-b from-coal-950/20 via-transparent to-coal-950/30 transition-opacity duration-700",
            ready ? "opacity-100" : "opacity-0",
          )}
        />
      </div>
    </div>
  );
}
