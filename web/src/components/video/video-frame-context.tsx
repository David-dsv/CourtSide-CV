"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import type { ClipIndex } from "@/lib/types";
import {
  type ClipMapper,
  clipIndexUrl,
  identityMapper,
  mapperFromClipIndex,
} from "@/lib/video/clip-index";

/**
 * An optional playback window. When set, the VideoScrubber loops playback
 * inside [start, end] instead of the full frame range. Cleared by passing
 * null. Used by the highlight "Replay" button to play a segment.
 *
 * Windows, like every frame the app passes around, are in ORIGINAL frame
 * coordinates (see lib/video/clip-index.ts). The video element converts to
 * annotated time via the shared {@link ClipMapper}.
 */
type PlaybackRange = [number, number] | null;

interface VideoFrameCtx {
  frame: number;
  setFrame: Dispatch<SetStateAction<number>>;
  frameRange: [number, number];
  fps: number;
  /** total frames in the analyzed range */
  frameCount: number;
  /** active playback loop window, or null (full-range playback). */
  playbackRange: PlaybackRange;
  /** set/clear the playback loop window. Setting a window also clamps frame into it. */
  setPlaybackRange: (range: PlaybackRange) => void;
  /** whether playback is running. Shared so any control (scrubber, replay) can start/stop it. */
  playing: boolean;
  setPlaying: Dispatch<SetStateAction<boolean>>;
  /**
   * Cut-timeline mapper (annotated ⇄ original). Identity when the clip is uncut
   * or its `_clipindex.json` sidecar hasn't loaded. Consumers that drive the
   * real <video> use it to convert original frames → annotated time and back.
   */
  clip: ClipMapper;
}

const Ctx = createContext<VideoFrameCtx | null>(null);

export function VideoFrameProvider({
  children,
  frameRange,
  fps,
  initial,
  /** project id — used to fetch the cut-timeline sidecar at runtime. */
  projectId,
  /** inlined cut index (mock fixtures). Takes precedence over the fetch. */
  clipIndex: clipIndexProp,
}: {
  children: ReactNode;
  frameRange: [number, number];
  fps: number;
  initial?: number;
  projectId?: string;
  clipIndex?: ClipIndex;
}) {
  const [frame, setFrame] = useState<number>(initial ?? frameRange[0]);
  const [playbackRange, setRange] = useState<PlaybackRange>(null);
  const [playing, setPlaying] = useState(false);
  const [fetchedClip, setFetchedClip] = useState<ClipIndex | null>(null);

  // Fetch the sidecar once (skipped when an inlined index is supplied). A 404 /
  // parse error is non-fatal: we stay on the identity mapping (uncut clip).
  useEffect(() => {
    if (clipIndexProp || !projectId) return;
    let alive = true;
    fetch(clipIndexUrl(projectId))
      .then((r) => (r.ok ? r.json() : null))
      .then((j: ClipIndex | null) => {
        if (alive && j && Array.isArray(j.kept_original_frames)) setFetchedClip(j);
      })
      .catch(() => {/* no sidecar → identity mapping */});
    return () => {
      alive = false;
    };
  }, [projectId, clipIndexProp]);

  const clip = useMemo<ClipMapper>(() => {
    const ci = clipIndexProp ?? fetchedClip;
    return ci ? mapperFromClipIndex(ci) : identityMapper(fps, frameRange);
  }, [clipIndexProp, fetchedClip, fps, frameRange]);

  const setPlaybackRange = useCallback((range: PlaybackRange) => {
    setRange((prev) => {
      // Exiting a loop? Nothing to clamp.
      if (range === null) return null;
      // Entering / moving a loop: snap the frame into the new window so the
      // scrubber doesn't start playing from outside the range.
      const [start, end] = range;
      setFrame((f) => (f < start ? start : f > end ? end : f));
      // Don't emit a no-op state change — keeps the effect dependency stable.
      if (prev && prev[0] === start && prev[1] === end) return prev;
      return range;
    });
  }, []);

  const value = useMemo<VideoFrameCtx>(
    () => ({
      frame,
      setFrame,
      frameRange,
      fps,
      frameCount: frameRange[1] - frameRange[0],
      playbackRange,
      setPlaybackRange,
      playing,
      setPlaying,
      clip,
    }),
    [frame, frameRange, fps, playbackRange, setPlaybackRange, playing, clip],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useFrame(): VideoFrameCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useFrame must be used within VideoFrameProvider");
  return v;
}
