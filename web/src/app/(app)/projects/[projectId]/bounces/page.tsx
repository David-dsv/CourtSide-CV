import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { LiveCourtMinimap } from "@/components/court/live-court-minimap";
import { EventGlyph } from "@/components/analysis/event-glyph";
import { fmtKmh, depthColorClass, depthHex, depthLabel } from "@/lib/format";
import { CircleDot, Zap, Map } from "lucide-react";
import type { Bounce } from "@/lib/types";

export default async function BouncesPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();
  const bounces = [...project.stats.bounces].sort((a, b) => a.frame - b.frame);
  const [f0, f1] = project.stats.frame_range;
  const span = f1 - f0 || 1;

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Rebonds"
        description={`${bounces.length} rebonds — chronologie et détail.`}
        icon={<CircleDot className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      {/* all-placements radar (the exhaustive view; the live radar elsewhere
          shows only the current bounce) */}
      {bounces.length > 0 && (
        <GlassCard>
          <SectionHeading
            title="Tous les placements"
            description="Chaque rebond projeté sur le court. Cliquez un point pour y revenir."
            icon={<Map className="h-4 w-4" />}
            className="mb-4"
          />
          <LiveCourtMinimap
            bounces={project.stats.bounces}
            trajectory={project.trajectory}
            source={project.stats.speed_source}
            confidence={project.stats.homography_confidence}
            H={project.H}
            showTrajectory={false}
            showPlayers={false}
            liveBounceOnly={false}
            className="mx-auto max-w-[300px]"
          />
        </GlassCard>
      )}

      {/* timeline */}
      <GlassCard>
        <SectionHeading title="Chronologie" className="mb-4" />
        <div className="relative h-16 rounded-lg bg-foreground/10">
          <div className="absolute inset-y-0 left-0 right-0 flex items-center">
            <div className="relative h-full w-full">
              {bounces.map((b) => {
                const pct = ((b.frame - f0) / span) * 100;
                return (
                  <span
                    key={b.frame}
                    className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
                    style={{ left: `clamp(6px, ${pct}%, calc(100% - 6px))` }}
                  >
                    <FrameJumpLink
                      frame={b.frame}
                      title={`f${b.frame} · ${depthLabel(b.depth)} · ${fmtKmh(b.speed_kmh)} km/h`}
                    >
                      <span
                        className="block h-3 w-3 rounded-full ring-2 transition-transform hover:scale-150"
                        style={{
                          background: depthHex(b.depth),
                          // theme-aware outline so the dot reads on both surfaces
                          ["--tw-ring-color" as string]: "var(--svg-outline)",
                        }}
                      />
                    </FrameJumpLink>
                  </span>
                );
              })}
            </div>
          </div>
        </div>
        <div className="mt-2 flex justify-between text-xs text-muted-foreground">
          <span>Début</span>
          <span>Fin</span>
        </div>
      </GlassCard>

      {/* table */}
      <GlassCard className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-foreground/10 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-3 pl-4 pr-4">Frame</th>
                <th className="py-3 pr-4">Profondeur</th>
                <th className="py-3 pr-4">Vitesse</th>
                <th className="py-3 pr-4">Qualité</th>
                <th className="py-3 pr-4">Position</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-foreground/5">
              {bounces.map((b: Bounce) => (
                <tr key={b.frame} className="hover:bg-foreground/5">
                  <td className="py-3 pl-4 pr-4">
                    <FrameJumpLink frame={b.frame} className="font-mono text-xs hover:text-court-green">
                      f{b.frame}
                    </FrameJumpLink>
                  </td>
                  <td className={`py-3 pr-4 font-medium ${depthColorClass(b.depth)}`}>
                    <span className="flex items-center gap-2">
                      <EventGlyph kind="bounce" depth={b.depth} />
                      {depthLabel(b.depth)}
                    </span>
                  </td>
                  <td className="py-3 pr-4">
                    <span className="flex items-center gap-1 tabular-nums">
                      <Zap className="h-3.5 w-3.5 text-court-amber" />
                      {fmtKmh(b.speed_kmh)} km/h
                    </span>
                  </td>
                  <td className="py-3 pr-4 tabular-nums">{b.quality}/100</td>
                  <td className="py-3 pr-4 font-mono text-xs text-muted-foreground">
                    ({b.x}, {b.y})
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
}
