import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { SpeedGauge, DepthDonut, StrokesBar, QualitySparkline } from "@/components/charts/charts";
import { BarChart3, Gauge, CircleDot, Target, TrendingUp } from "lucide-react";

export default async function StatsPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();
  const { summary, bounces, speed_source } = project.stats;

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Statistiques"
        description="Décomposition offensive, défensive et qualité — niveau broadcast."
        icon={<BarChart3 className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <GlassCard>
          <SectionHeading title="Vitesse" icon={<Gauge className="h-4 w-4" />} className="mb-4" />
          <SpeedGauge summary={summary} />
          <p className="mt-3 text-xs text-muted-foreground">
            Source : {speed_source}. La vitesse est perspective-corrigée quand une homographie est calibrée.
          </p>
        </GlassCard>

        <GlassCard>
          <SectionHeading title="Profondeur des rebonds" icon={<CircleDot className="h-4 w-4" />} className="mb-4" />
          <DepthDonut summary={summary} />
        </GlassCard>

        <GlassCard>
          <SectionHeading title="Répartition des coups" icon={<Target className="h-4 w-4" />} className="mb-4" />
          <StrokesBar summary={summary} />
        </GlassCard>

        <GlassCard>
          <SectionHeading title="Qualité dans le temps" icon={<TrendingUp className="h-4 w-4" />} className="mb-4" />
          <QualitySparkline bounces={bounces} />
        </GlassCard>
      </div>

      <GlassCard>
        <SectionHeading title="Tableau des rebonds" className="mb-3" />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-foreground/10 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-4">Frame</th>
                <th className="py-2 pr-4">Profondeur</th>
                <th className="py-2 pr-4">Vitesse</th>
                <th className="py-2 pr-4">Qualité</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-foreground/5">
              {bounces.map((b) => (
                <tr key={b.frame} className="hover:bg-foreground/5">
                  <td className="py-2 pr-4 font-mono text-xs text-muted-foreground">f{b.frame}</td>
                  <td className="py-2 pr-4 capitalize">{b.depth}</td>
                  <td className="py-2 pr-4 tabular-nums">{b.speed_kmh.toFixed(1)} km/h</td>
                  <td className="py-2 pr-4 tabular-nums">{b.quality}/100</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
}
