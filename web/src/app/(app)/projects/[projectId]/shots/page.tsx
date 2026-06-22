import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { fmtKmh, qualityLabel, sideLabel, strokeLabel } from "@/lib/format";
import { Target, Zap } from "lucide-react";

export default async function ShotsPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();
  const shots = [...project.stats.shots].sort((a, b) => a.frame - b.frame);

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Frappes"
        description={`${shots.length} frappes détectées — cliquez une ligne pour la rejouer.`}
        icon={<Target className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      <GlassCard className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-3 pl-4 pr-4">Frame</th>
                <th className="py-3 pr-4">Coup</th>
                <th className="py-3 pr-4">Côté</th>
                <th className="py-3 pr-4">Vitesse</th>
                <th className="py-3 pr-4">Qualité</th>
                <th className="py-3 pr-4">Position</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {shots.map((s) => {
                const q = qualityLabel(s.quality);
                return (
                  <tr key={s.frame} className="hover:bg-white/5">
                    <td className="py-3 pl-4 pr-4">
                      <FrameJumpLink frame={s.frame} className="font-mono text-xs hover:text-court-green">
                        f{s.frame}
                      </FrameJumpLink>
                    </td>
                    <td className="py-3 pr-4 font-medium">{strokeLabel(s.stroke)}</td>
                    <td className="py-3 pr-4 text-muted-foreground">{sideLabel(s.player_side)}</td>
                    <td className="py-3 pr-4">
                      <span className="flex items-center gap-1 tabular-nums">
                        <Zap className="h-3.5 w-3.5 text-court-amber" />
                        {fmtKmh(s.speed_kmh)} km/h
                      </span>
                    </td>
                    <td className={`py-3 pr-4 tabular-nums ${q.className}`}>
                      {s.quality} <span className="text-xs">({q.label})</span>
                    </td>
                    <td className="py-3 pr-4 font-mono text-xs text-muted-foreground">
                      ({s.x}, {s.y})
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
}
