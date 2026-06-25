"use client";

import { CourtMinimap, type CourtMinimapProps } from "@/components/court/court-minimap";
import { useFrame } from "@/components/video/video-frame-context";

/**
 * Frame-aware wrapper around {@link CourtMinimap}. Lets server-rendered pages
 * (overview) drop in a LIVE radar — synced to the shared scrubber frame and
 * click-to-seek — without becoming client components themselves.
 *
 * Defaults to `liveBounceOnly` (a single current bounce), matching the backend
 * radar. Pass `liveBounceOnly={false}` for the static "all placements" map.
 */
export function LiveCourtMinimap(
  props: Omit<CourtMinimapProps, "frame" | "onBounceClick">,
) {
  const { frame, setFrame } = useFrame();
  return <CourtMinimap {...props} frame={frame} onBounceClick={setFrame} />;
}
