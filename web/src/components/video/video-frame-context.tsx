"use client";

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";

interface VideoFrameCtx {
  frame: number;
  setFrame: Dispatch<SetStateAction<number>>;
  frameRange: [number, number];
  fps: number;
  /** total frames in the analyzed range */
  frameCount: number;
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
  const value = useMemo<VideoFrameCtx>(
    () => ({
      frame,
      setFrame,
      frameRange,
      fps,
      frameCount: frameRange[1] - frameRange[0],
    }),
    [frame, frameRange, fps],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useFrame(): VideoFrameCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useFrame must be used within VideoFrameProvider");
  return v;
}
