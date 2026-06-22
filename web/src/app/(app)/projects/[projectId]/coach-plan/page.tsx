import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameJumpLink } from "@/components/core/frame-jump-link";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { generateCoachInsights, type Insight } from "@/lib/mock/coach-insights";
import { ClipboardList, Lightbulb, AlertTriangle, TrendingUp } from "lucide-react";

export default async function CoachPlanPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();
  const insights = generateCoachInsights(project);

  const iconFor = (t: Insight["type"]) =>
    t === "strength" ? TrendingUp : t === "weakness" ? AlertTriangle : Lightbulb;
  const colorFor = (t: Insight["type"]) =>
    t === "strength" ? "text-court-green" : t === "weakness" ? "text-court-red" : "text-court-cyan";

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Plan coach"
        description="Diagnostics et recommandations — générés depuis les données du match."
        icon={<ClipboardList className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {insights.map((ins, i) => {
          const Icon = iconFor(ins.type);
          return (
            <GlassCard key={i} glow={ins.type === "strength" ? "green" : "none"}>
              <div className="mb-2 flex items-center gap-2">
                <Icon className={`h-5 w-5 ${colorFor(ins.type)}`} />
                <span className={`text-xs font-medium uppercase tracking-wide ${colorFor(ins.type)}`}>
                  {ins.type === "strength" ? "Point fort" : ins.type === "weakness" ? "À travailler" : "Recommandation"}
                </span>
              </div>
              <p className="text-sm leading-relaxed">{ins.text}</p>
              {ins.frames.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {ins.frames.slice(0, 4).map((f) => (
                    <FrameJumpLink
                      key={f}
                      frame={f}
                      className="rounded-md glass px-2 py-0.5 font-mono text-[11px] text-muted-foreground hover:text-court-green"
                    >
                      f{f}
                    </FrameJumpLink>
                  ))}
                </div>
              )}
            </GlassCard>
          );
        })}
      </div>

      <GlassCard className="text-sm text-muted-foreground">
        <p className="font-medium text-foreground">À propos de ce rapport</p>
        <p className="mt-1">
          Ces insights sont générés à partir des données structurées du pipeline (rebonds, vitesses, profondeur,
          qualité). Chaque recommandation est ancrée à des frames vérifiables. En production, un LLM génère
          ces blocs en structured output, avec benchmarks UTR / Match Charting Project — zéro hallucination.
        </p>
      </GlassCard>
    </div>
  );
}
