"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { GlassCard } from "@/components/core/glass-card";
import { ConfidenceBadge } from "@/components/core/confidence-badge";
import { Button } from "@/components/ui/button";
import { useMockProcessing, STEPS } from "@/hooks/use-mock-processing";
import { Upload, FileVideo, CheckCircle2, Loader2, Camera, Ruler, Zap, CircleDot, Users } from "lucide-react";
import { toast } from "sonner";
import type { MatchType } from "@/lib/types";
import { cn } from "@/lib/utils";

const STEP_ICONS = [Upload, Camera, Zap];

type ModeOption = {
  value: MatchType;
  label: string;
  hint: string;
  icon: typeof Camera;
};

const MODES: ModeOption[] = [
  {
    value: "training",
    label: "Entraînement",
    hint: "Caméra fixe, pas de changement de côté.",
    icon: Camera,
  },
  {
    value: "match",
    label: "Match",
    hint: "Les joueurs changent de côté entre jeux (analyse plus longue).",
    icon: Users,
  },
];

export default function NewProjectPage() {
  const router = useRouter();
  const { state, progress, start } = useMockProcessing();
  const [filename, setFilename] = useState<string | null>(null);
  const [matchType, setMatchType] = useState<MatchType>("training");

  function onDemo() {
    setFilename("felix.mp4");
    start();
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setFilename(f.name);
      toast.success(`${f.name} sélectionné · ${matchType === "match" ? "Match" : "Entraînement"}`);
      start();
    }
  }

  if (state === "ready") {
    // pick the felix-sota project (homography(low), the "calibrated" demo)
    return (
      <div className="mx-auto max-w-xl px-4 py-16">
        <GlassCard strong glow="green" className="text-center">
          <CheckCircle2 className="mx-auto mb-4 h-14 w-14 text-court-green" />
          <h2 className="text-xl font-semibold">Analyse terminée !</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            {filename} a été analysé. Voici le résultat.
          </p>
          <div className="mt-4 flex justify-center">
            <ConfidenceBadge source="homography(low)" />
          </div>
          <Button
            className="mt-6 bg-ball text-coal-950 hover:bg-ball/90"
            onClick={() => router.push("/projects/felix-sota/overview")}
          >
            Ouvrir le dashboard
          </Button>
        </GlassCard>
      </div>
    );
  }

  if (state !== "idle") {
    const activeStepIdx = STEPS.findIndex((s) => s.id === state);
    return (
      <div className="mx-auto max-w-xl px-4 py-16">
        <GlassCard strong>
          <div className="mb-6 flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-court-green" />
            <div>
              <h2 className="font-semibold">Analyse en cours…</h2>
              <p className="text-xs text-muted-foreground">{filename}</p>
            </div>
          </div>

          {/* progress bar */}
          <div className="mb-6 h-2 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-court-green to-court-cyan transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* steps */}
          <div className="flex flex-col gap-3">
            {STEPS.map((step, i) => {
              const done = i < activeStepIdx;
              const active = i === activeStepIdx;
              const Icon = STEP_ICONS[i] ?? CircleDot;
              return (
                <div
                  key={step.id}
                  className={`flex items-center gap-3 rounded-lg p-3 transition-colors ${
                    active ? "glass-strong" : done ? "opacity-60" : "opacity-30"
                  }`}
                >
                  <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${active || done ? "bg-court-green/15 text-court-green" : "bg-white/5 text-muted-foreground"}`}>
                    {done ? <CheckCircle2 className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{step.label}</div>
                    <div className="text-xs text-muted-foreground">{step.detail}</div>
                  </div>
                  {active && <Loader2 className="h-4 w-4 animate-spin text-court-green" />}
                </div>
              );
            })}
          </div>
        </GlassCard>
      </div>
    );
  }

  // idle — dropzone
  return (
    <div className="mx-auto max-w-xl px-4 py-16">
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Analyser un match</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Uploadez une vidéo de tennis. Calibre le court une fois, on s&apos;occupe du reste.
        </p>
      </div>

      {/* Sélecteur de type d'analyse — explicite avant la soumission.
          training = caméra fixe (verrou P1/P2 actuel) ; match = side-swap. */}
      <fieldset className="mb-6">
        <legend className="mb-2 text-sm font-medium">Type d&apos;analyse</legend>
        <div className="grid grid-cols-2 gap-3">
          {MODES.map((m) => {
            const Icon = m.icon;
            const selected = matchType === m.value;
            return (
              <button
                key={m.value}
                type="button"
                aria-pressed={selected}
                onClick={() => setMatchType(m.value)}
                className={cn(
                  "flex flex-col items-start gap-2 rounded-xl border p-4 text-left transition-colors",
                  selected
                    ? "border-court-green/60 bg-court-green/10 glow-green"
                    : "border-white/10 bg-white/[0.02] hover:border-court-green/40",
                )}
              >
                <span
                  className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-lg",
                    selected ? "bg-court-green/15 text-court-green" : "bg-white/5 text-muted-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                </span>
                <span className="text-sm font-medium">{m.label}</span>
                <span className="text-xs text-muted-foreground">{m.hint}</span>
              </button>
            );
          })}
        </div>
      </fieldset>

      <label htmlFor="video-upload" className="block cursor-pointer">
        <GlassCard
          strong
          className="flex flex-col items-center justify-center gap-4 border-2 border-dashed border-white/15 py-16 text-center transition-colors hover:border-court-green/50"
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-full glass-strong text-court-green">
            <Upload className="h-6 w-6" />
          </div>
          <div>
            <p className="font-medium">Glissez votre vidéo ici</p>
            <p className="mt-1 text-xs text-muted-foreground">ou cliquez pour parcourir · MP4, MOV</p>
          </div>
          <input id="video-upload" type="file" accept="video/*" className="hidden" onChange={onFile} />
        </GlassCard>
      </label>

      <div className="my-6 flex items-center gap-3 text-xs text-muted-foreground">
        <div className="h-px flex-1 bg-white/10" />
        OU
        <div className="h-px flex-1 bg-white/10" />
      </div>

      <Button variant="outline" className="w-full" onClick={onDemo}>
        <FileVideo className="mr-2 h-4 w-4 text-court-green" />
        Utiliser felix.mp4 (démo)
      </Button>

      <div className="mt-8 flex items-center justify-center gap-2 text-xs text-muted-foreground">
        <Ruler className="h-3.5 w-3.5" />
        Calibration 4-coins semi-auto · homographie métrique
      </div>
    </div>
  );
}
