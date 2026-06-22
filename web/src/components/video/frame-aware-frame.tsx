"use client";

import { useFrame } from "@/components/video/video-frame-context";
import { frameToTimecode } from "@/lib/format";
import { Badge } from "@/components/ui/badge";

/** Displays the current scrubber frame/timecode — reflects frame-as-truth. */
export function FrameAwareFrame() {
  const { frame, fps } = useFrame();
  return (
    <Badge variant="outline" className="font-mono text-xs">
      {frameToTimecode(frame, fps)} · f{frame}
    </Badge>
  );
}
