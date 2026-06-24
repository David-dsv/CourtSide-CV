"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";

/**
 * An optional playback window. When set, the VideoScrubber loops playback
 * inside [start, end] instead of the full frame range. Cleared by passing
 * null. Used by the highlight "Replay" button to play a segment.
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
}

const Ctx = createContext<VideoFrameCtx | null>(null);

export function VideoFrameProvider({
  children,
  frameRange,
  fps,
  initial,
}: {
  children: ReactNode;
  frameRange: [number, number];
  fps: number;
  initial?: number;
}) {
  const [frame, setFrame] = useState<number>(initial ?? frameRange[0]);
  const [playbackRange, setRange] = useState<PlaybackRange>(null);
  const [playing, setPlaying] = useState(false);

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
    }),
    [frame, frameRange, fps, playbackRange, setPlaybackRange, playing],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useFrame(): VideoFrameCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useFrame must be used within VideoFrameProvider");
  return v;
}
