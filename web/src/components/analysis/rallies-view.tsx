"use client";

import { useMemo, useState } from "react";
import { useFrame } from "@/components/video/video-frame-context";
import { GlassCard } from "@/components/core/glass-card";
import { ShotGlyph, BounceGlyph, ShotBounceLegend } from "@/components/analysis/event-glyph";
import { enrichRallies, outcomeMeta, type EnrichedRally } from "@/lib/analysis/highlights";
import { fmtKmh, frameToTimecode, strokeLabel, depthLabel } from "@/lib/format";
import type { PipelineStats } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ChevronRight, Play, Target, CircleDot, Clock } from "lucide-react";

const ACCENT_BADGE: Record<string, string> = {
  green: "bg-court-green/15 text-court-green",
  red: "bg-clay/15 text-clay",
  amber: "bg-court-amber/15 text-court-amber",
  slate: "bg-foreground/10 text-muted-foreground",
};

/**
 * Per-rally ("ÉCHANGE") view. Each card is one rally; click the header to seek
 * the scrubber to its start. Expand to walk its shot→bounce sequence with the
 * shared shot/bounce glyphs (so the events are never confused with each other,
 * and read identically to the radar + the shots/bounces tables).
 *
 * Rallies come from the backend (`stats.rallies`, the single source of truth)
 * or the client fallback — see `enrichRallies`/`getRallies`.
 */
export function RalliesView({ stats }: { stats: PipelineStats }) {
  const { setFrame, setPlaybackRange, setPlaying } = useFrame();
  const rallies = useMemo(() => enrichRallies(stats), [stats]);
  const isMatch = Boolean(stats.match_mode);

  if (rallies.length === 0) {
    return (
      <GlassCard>
        <p className="text-sm text-muted-foreground">Aucun échange détecté sur ce clip.</p>
      </GlassCard>
    );
  }

  const replay = (r: EnrichedRally) => {
    setFrame(r.rally.start_frame);
    setPlaybackRange([r.rally.start_frame, r.rally.end_frame]);
    setPlaying(true);
  };

  return (
    <div className="flex flex-col gap-3">
      <ShotBounceLegend className="px-1" />
      {rallies.map((r) => (
        <RallyCard key={r.rally.start_frame} r={r} fps={stats.fps} isMatch={isMatch} onSeek={setFrame} onReplay={replay} />
      ))}
    </div>
  );
}

function RallyCard({
  r,
  fps,
  isMatch,
  onSeek,
  onReplay,
}: {
  r: EnrichedRally;
  fps: number;
  isMatch: boolean;
  onSeek: (f: number) => void;
  onReplay: (r: EnrichedRally) => void;
}) {
  const [open, setOpen] = useState(false);
  const o = r.rally.outcome;
  const meta = o ? outcomeMeta(o.outcome) : null;

  return (
    <GlassCard className="p-0 overflow-hidden">
      {/* header — click to expand; the ÉCHANGE label seeks */}
      <div className="flex items-center gap-3 p-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex min-w-0 flex-1 items-center gap-3 text-left"
          aria-expanded={open}
        >
          <ChevronRight className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} />
          <span className="font-display text-sm font-bold tracking-tight">ÉCHANGE {r.index}</span>
          {isMatch && r.rally.human_id && (
            <span className="rounded bg-foreground/10 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              {r.rally.human_id}
            </span>
          )}
          {meta && (
            <span className={cn("rounded-md px-2 py-0.5 text-[11px] font-medium", ACCENT_BADGE[meta.accent])}>
              {meta.label}
            </span>
          )}
        </button>

        <div className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1" title="Frappes">
            <Target className="h-3.5 w-3.5" /> {r.rally.n_shots}
          </span>
          <span className="flex items-center gap-1" title="Rebonds">
            <CircleDot className="h-3.5 w-3.5" /> {r.bounces.length}
          </span>
          <span className="flex items-center gap-1 tabular-nums" title="Durée">
            <Clock className="h-3.5 w-3.5" /> {r.durationSec.toFixed(1)}s
          </span>
          <button
            type="button"
            onClick={() => onSeek(r.rally.start_frame)}
            className="rounded font-mono hover:text-court-green"
            title={`Aller au début (${frameToTimecode(r.rally.start_frame, fps)})`}
          >
            {frameToTimecode(r.rally.start_frame, fps)}
          </button>
          <button
            type="button"
            onClick={() => onReplay(r)}
            className="flex items-center gap-1 rounded-md bg-court-green/15 px-2 py-1 text-court-green hover:bg-court-green/25"
            title="Rejouer cet échange"
          >
            <Play className="h-3 w-3" /> Rejouer
          </button>
        </div>
      </div>

      {/* outcome reason + shot→bounce sequence */}
      {open && (
        <div className="border-t border-foreground/5 px-3 pb-3 pt-2">
          {o?.reason && <p className="mb-2 text-xs italic text-muted-foreground">{o.reason}</p>}
          <div className="flex flex-col divide-y divide-foreground/5">
            {r.shots.map((s) => (
              <div key={s.frame} className="flex items-center gap-2.5 py-2 text-sm">
                <button
                  type="button"
                  onClick={() => onSeek(s.frame)}
                  className="flex items-center gap-2 rounded hover:text-court-green"
                  title={`Frappe — ${frameToTimecode(s.frame, fps)}`}
                >
                  <ShotGlyph stroke={s.stroke} />
                  <span className="font-medium">{strokeLabel(s.stroke)}</span>
                  <span className="text-xs text-muted-foreground">
                    {s.player_side === "near" ? "proche" : "lointain"}
                  </span>
                </button>
                <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {fmtKmh(s.speed_kmh)} km/h · Q{s.quality}
                </span>
                {/* the bounce this shot produced (the shot's "result") */}
                {s.resultBounce && (
                  <button
                    type="button"
                    onClick={() => onSeek(s.resultBounce!.frame)}
                    className="ml-auto flex items-center gap-1.5 rounded text-xs hover:text-court-green"
                    title={`Rebond résultant — ${frameToTimecode(s.resultBounce.frame, fps)}`}
                  >
                    <ChevronRight className="h-3 w-3 text-muted-foreground" />
                    <BounceGlyph depth={s.resultBounce.depth} size={10} />
                    <span className="text-muted-foreground">{depthLabel(s.resultBounce.depth)}</span>
                  </button>
                )}
              </div>
            ))}
            {r.shots.length === 0 && (
              <p className="py-2 text-xs text-muted-foreground">Pas de frappe résolue (rebonds seuls).</p>
            )}
          </div>
        </div>
      )}
    </GlassCard>
  );
}
