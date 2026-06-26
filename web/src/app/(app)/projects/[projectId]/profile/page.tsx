import Link from "next/link";
import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { getPlayerProfile, type AggregatedInsight } from "@/lib/mock/player-profile";
import { VideoScrubber } from "@/components/video/video-scrubber";
import { GlassCard } from "@/components/core/glass-card";
import { KpiTile } from "@/components/core/kpi-tile";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { TrendChart } from "@/components/charts/trend-chart";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { fmtKmh } from "@/lib/format";
import { TrendingUp, TrendingDown, Minus, Gauge, Award, Layers, BatteryWarning, Sparkles, ArrowRight } from "lucide-react";
import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ projectId: string }>;
}): Promise<Metadata> {
  const { projectId } = await params;
  const p = getProject(projectId);
  return { title: p ? `${p.name} — Profil · CourtSide` : "CourtSide" };
}

export default async function ProfilePage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();

  const profile = getPlayerProfile(projectId);

  const speedUp = (profile.speedDeltaPct ?? 0) > 1.5;
  const speedDown = (profile.speedDeltaPct ?? 0) < -1.5;
  const DeltaIcon = speedUp ? TrendingUp : speedDown ? TrendingDown : Minus;
  const deltaWord = speedUp ? "augmenté" : speedDown ? "baissé" : "stable";

  return (
    <div className="flex flex-col gap-6">
      {/* scrubber — scrolls with the page (no longer sticky) */}
      <VideoScrubber project={project} />

      <SectionHeading
        title="Profil & tendances"
        description="Vue d'ensemble multi-matchs — cliquez un point du graphique ou une force pour y revenir."
        icon={<Sparkles className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      {/* headline: monthly delta */}
      <GlassCard glow="green">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <DeltaIcon
            className={
              speedUp ? "h-7 w-7 text-ball" : speedDown ? "h-7 w-7 text-clay" : "h-7 w-7 text-muted-foreground"
            }
          />
          <p className="text-lg leading-snug">
            {profile.speedDeltaPct == null ? (
              <>Pas encore assez de matchs pour mesurer une tendance.</>
            ) : (
              <>
                Votre vitesse moyenne a{" "}
                <span className={speedUp ? "text-ball" : speedDown ? "text-clay" : "text-foreground"}>
                  {deltaWord} de {Math.abs(profile.speedDeltaPct).toFixed(0)}%
                </span>{" "}
                sur vos {profile.matches.length} derniers matchs.
              </>
            )}
          </p>
          <Link
            href={`/projects/${projectId}/stats`}
            className="ml-auto inline-flex items-center gap-1 text-sm text-court-green hover:underline"
          >
            Détails du dernier match <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </GlassCard>

      {/* KPI tiles (latest match) */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiTile
          label="Vitesse moyenne"
          value={fmtKmh(profile.latest.speedKmh)}
          unit="km/h"
          accent="green"
          sub={profile.speedDeltaPct == null ? "—" : `${profile.speedDeltaPct >= 0 ? "+" : ""}${profile.speedDeltaPct.toFixed(0)}% vs 1er match`}
          icon={<Gauge className="h-4 w-4" />}
        />
        <KpiTile
          label="Qualité moyenne"
          value={Math.round(profile.latest.quality)}
          unit="/100"
          accent="amber"
          sub={profile.qualityDeltaPct == null ? "—" : `${profile.qualityDeltaPct >= 0 ? "+" : ""}${profile.qualityDeltaPct.toFixed(0)}%`}
          icon={<Award className="h-4 w-4" />}
        />
        <KpiTile
          label="Profondeur"
          value={`${Math.round(profile.latest.deepPct)}%`}
          accent="cyan"
          sub="balles profondes"
          icon={<Layers className="h-4 w-4" />}
        />
        <KpiTile
          label="Fatigue (proxy)"
          value={Math.round(profile.latest.fatigue)}
          unit="/100"
          accent="red"
          sub="~proxy momentum — TODO backend"
          icon={<BatteryWarning className="h-4 w-4" />}
        />
      </div>

      {/* trend chart */}
      <GlassCard>
        <SectionHeading
          title="Évolution"
          description="Vitesse et qualité sur vos derniers matchs — le point blanc est le match courant."
          icon={<TrendingUp className="h-4 w-4" />}
          className="mb-4"
        />
        <TrendChart matches={profile.matches} projectId={projectId} />
      </GlassCard>

      {/* strengths + work axes */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <GlassCard>
          <SectionHeading title="Vos forces" icon={<Award className="h-4 w-4" />} className="mb-4" />
          <InsightList insights={profile.topStrengths} projectId={projectId} accent="green" empty="Pas encore de force récurrente." />
        </GlassCard>
        <GlassCard>
          <SectionHeading title="Axes de travail" icon={<TrendingDown className="h-4 w-4" />} className="mb-4" />
          <InsightList insights={profile.workAxes} projectId={projectId} accent="red" empty="Rien de récurrent à corriger — bon travail !" />
        </GlassCard>
      </div>
    </div>
  );
}

/**
 * A list of aggregated insights. Each row is clickable:
 *  - if it cites the CURRENT project → FrameJumpLink (frame-as-truth, same scrubber)
 *  - if it cites ANOTHER project → Link to that project's overview (navigate)
 * The "see the moment" contract still holds either way.
 */
function InsightList({
  insights,
  projectId,
  accent,
  empty,
}: {
  insights: AggregatedInsight[];
  projectId: string;
  accent: "green" | "red";
  empty: string;
}) {
  if (insights.length === 0) {
    return <p className="py-2 text-sm text-muted-foreground">{empty}</p>;
  }
  const dot = accent === "green" ? "bg-court-green" : "bg-clay";
  return (
    <div className="flex flex-col divide-y divide-foreground/5">
      {insights.map((ins) => {
        const sameProject = ins.projectId === projectId;
        const label = (
          <>
            <span className="font-medium">{ins.label}</span>
            <span className="ml-2 text-[11px] text-muted-foreground">
              {ins.matchCount} match{ins.matchCount > 1 ? "s" : ""}
            </span>
          </>
        );
        return (
          <div key={`${ins.type}-${ins.label}`} className="py-3">
            <div className="mb-1 flex items-center gap-2">
              <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
              {sameProject ? (
                <FrameJumpLink frame={ins.frames[0] ?? 0} className="text-sm hover:text-court-green">
                  {label}
                </FrameJumpLink>
              ) : (
                <Link
                  href={`/projects/${ins.projectId}/overview`}
                  className="text-sm text-court-green hover:underline"
                  title={`Voir le match ${ins.projectId}`}
                >
                  {label} <ArrowRight className="ml-1 inline h-3 w-3" />
                </Link>
              )}
            </div>
            <p className="pl-3.5 text-xs leading-relaxed text-muted-foreground">{ins.text}</p>
          </div>
        );
      })}
    </div>
  );
}
