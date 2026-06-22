"use client";

import { useFrame } from "@/components/video/video-frame-context";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

/**
 * Frame-as-truth: any stat / insight / row wrapped by this becomes a clickable
 * element that jumps the shared video scrubber to the given frame.
 */
export function FrameJumpLink({
  frame,
  children,
  className,
  title,
}: {
  frame: number;
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  const { setFrame } = useFrame();
  return (
    <button
      type="button"
      title={title ?? `Aller au frame ${frame}`}
      onClick={() => setFrame(frame)}
      className={cn(
        "cursor-pointer text-left underline-offset-4 decoration-court-green/0 transition-colors",
        "hover:decoration-court-green/60 hover:text-foreground/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 rounded-sm",
        className,
      )}
    >
      {children}
    </button>
  );
}
