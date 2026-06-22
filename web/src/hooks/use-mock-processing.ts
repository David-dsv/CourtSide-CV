"use client";

import { useEffect, useRef, useState } from "react";

export type ProcessingState =
  | "idle"
  | "uploading"
  | "processing-pass1"
  | "processing-pass2"
  | "ready";

export interface ProcessingStep {
  id: ProcessingState;
  label: string;
  detail: string;
  durationMs: number;
}

export const STEPS: ProcessingStep[] = [
  { id: "uploading", label: "Upload", detail: "Réception de la vidéo…", durationMs: 1500 },
  { id: "processing-pass1", label: "Pass 1 — Balle", detail: "Calibration court + détection balle + tracking Kalman/WASB…", durationMs: 4000 },
  { id: "processing-pass2", label: "Pass 2 — Analyse", detail: "Rebonds, vitesse, profondeur, qualité, frappes…", durationMs: 3500 },
];

export function useMockProcessing() {
  const [state, setState] = useState<ProcessingState>("idle");
  const [progress, setProgress] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  function start() {
    setState("uploading");
    setProgress(0);
    let stepIdx = 0;
    let stepStart = Date.now();
    let stepElapsed = 0;

    timerRef.current = setInterval(() => {
      const step = STEPS[stepIdx];
      stepElapsed = Date.now() - stepStart;
      const stepPct = Math.min(100, (stepElapsed / step.durationMs) * 100);
      const overall = ((stepIdx + stepPct / 100) / STEPS.length) * 100;
      setProgress(overall);

      if (stepElapsed >= step.durationMs) {
        stepIdx += 1;
        stepStart = Date.now();
        if (stepIdx >= STEPS.length) {
          if (timerRef.current) clearInterval(timerRef.current);
          setProgress(100);
          setState("ready");
        } else {
          setState(STEPS[stepIdx].id);
        }
      }
    }, 60);
  }

  function reset() {
    if (timerRef.current) clearInterval(timerRef.current);
    setState("idle");
    setProgress(0);
  }

  return { state, progress, start, reset };
}
