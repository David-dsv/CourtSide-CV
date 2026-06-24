import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { VideoScrubber } from "@/components/video/video-scrubber";
import { CourtMinimap } from "@/components/court/court-minimap";
import { KpiTile } from "@/components/core/kpi-tile";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { HighlightCard } from "@/components/analysis/highlight-card";
import { MomentumChart } from "@/components/analysis/momentum-chart";
import { deriveHighlights, deriveMomentum } from "@/lib/analysis/highlights";
import { depthColorClass, depthLabel, fmtKmh, strokeLabel } from "@/lib/format";
import { Gauge, Target, CircleDot, Activity, Zap, Map, TrendingUp, Sparkles } from "lucide-react";
import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ projectId: string }>;
}): Promise<Metadata> {
  const { projectId } = await params;
  const p = getProject(projectId);
  return { title: p ? `${p.name} — CourtSide` : "CourtSide" };
}

export default async function OverviewPage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();

  const { summary, bounces, shots, speed_source, homography_confidence } = project.stats;
  const recentBounces = [...bounces].sort((a, b) => b.frame - a.frame).slice(0, 5);
  const recentShots = [...shots].sort((a, b) => b.frame - a.frame).slice(0, 5);
  const highlights = deriveHighlights(project.stats);
  const momentum = deriveMomentum(project.stats);

  return (
    <div className="flex flex-col gap-6">
      {/* scrubber — scrolls with the page (no longer sticky) */}
      <VideoScrubber project={project} />

      <SectionHeading
        title="Vue d'ensemble"
        description="Résumé du match — cliquez n'importe quelle valeur pour sauter au moment exact."
        icon={<Activity className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      {/* KPI grid */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiTile
          label="Vitesse moyenne"
          value={fmtKmh(summary.speed_kmh.avg)}
          unit="km/h"
          accent="green"
          sub={`max ${fmtKmh(summary.speed_kmh.max)}`}
          source={speed_source}
          confidence={homography_confidence}
          icon={<Gauge className="h-4 w-4" />}
        />
        <KpiTile
          label="Rebonds"
          value={summary.total_bounces}
          accent="cyan"
          sub={`${summary.depth.deep + summary.depth.mid} profonds/moyens`}
          icon={<CircleDot className="h-4 w-4" />}
        />
        <KpiTile
          label="Qualité moyenne"
          value={Math.round(summary.quality.avg)}
          unit="/100"
          accent="amber"
          sub={`${summary.quality.high_count} excellents`}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <KpiTile
          label="Frappes"
          value={shots.length}
          accent="green"
          sub={`${summary.strokes.forehand} CD / ${summary.strokes.backhand} RV`}
          icon={<Target className="h-4 w-4" />}
        />
      </div>

      {/* Momentum timeline */}
      <GlassCard>
        <SectionHeading
          title="Momentum"
          description="Votre rythme au fil du match — cliquez une barre pour y revenir."
          icon={<TrendingUp className="h-4 w-4" />}
          className="mb-4"
        />
        <MomentumChart buckets={momentum} fps={project.stats.fps} />
      </GlassCard>

      {/* Highlights — best/worst points, longest rallies, winners */}
      {highlights.length > 0 && (
        <div>
          <SectionHeading
            title="Faits marquants"
            description="Vos meilleurs coups, points à revoir et échanges les plus longs — cliquez pour rejouer le moment."
            icon={<Sparkles className="h-4 w-4" />}
            className="mb-3"
          />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {highlights.map((h, i) => (
              <HighlightCard key={`${h.type}-${i}`} highlight={h} fps={project.stats.fps} />
            ))}
          </div>
        </div>
      )}

      {/* minimap + speed highlight */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
        <div>
          <SectionHeading title="Court radar" icon={<Map className="h-4 w-4" />} className="mb-3" />
          <CourtMinimap
            bounces={bounces}
            trajectory={project.trajectory}
            players={project.players}
            source={speed_source}
            confidence={homography_confidence}
            H={project.H}
            className="mx-auto max-w-[320px]"
          />
        </div>

        <div className="flex flex-col gap-4">
          <GlassCard>
            <SectionHeading title="Rebonds récents" icon={<CircleDot className="h-4 w-4" />} className="mb-3" />
            <div className="flex flex-col divide-y divide-white/5">
              {recentBounces.map((b) => (
                <div key={b.frame} className="flex items-center gap-3 py-2.5">
                  <FrameJumpLink frame={b.frame} className="font-mono text-xs text-muted-foreground hover:text-court-green">
                    f{b.frame}
                  </FrameJumpLink>
                  <span className={`text-sm font-medium ${depthColorClass(b.depth)}`}>
                    {depthLabel(b.depth)}
                  </span>
                  <span className="ml-auto flex items-center gap-3 text-sm">
                    <span className="flex items-center gap-1 text-muted-foreground">
                      <Zap className="h-3.5 w-3.5" /> {fmtKmh(b.speed_kmh)} km/h
                    </span>
                    <span className="tabular-nums text-muted-foreground">Q{b.quality}</span>
                  </span>
                </div>
              ))}
              {recentBounces.length === 0 && (
                <p className="py-4 text-sm text-muted-foreground">Aucun rebond détecté.</p>
              )}
            </div>
          </GlassCard>

          <GlassCard>
            <SectionHeading title="Frappes récentes" icon={<Target className="h-4 w-4" />} className="mb-3" />
            <div className="flex flex-col divide-y divide-white/5">
              {recentShots.map((s) => (
                <div key={s.frame} className="flex items-center gap-3 py-2.5">
                  <FrameJumpLink frame={s.frame} className="font-mono text-xs text-muted-foreground hover:text-court-green">
                    f{s.frame}
                  </FrameJumpLink>
                  <span className="text-sm font-medium">{strokeLabel(s.stroke)}</span>
                  <span className="text-xs text-muted-foreground">{s.player_side === "near" ? "proche" : "lointain"}</span>
                  <span className="ml-auto flex items-center gap-3 text-sm">
                    <span className="flex items-center gap-1 text-muted-foreground">
                      <Zap className="h-3.5 w-3.5" /> {fmtKmh(s.speed_kmh)} km/h
                    </span>
                    <span className="tabular-nums text-muted-foreground">Q{s.quality}</span>
                  </span>
                </div>
              ))}
              {recentShots.length === 0 && (
                <p className="py-4 text-sm text-muted-foreground">Aucune frappe détectée.</p>
              )}
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
