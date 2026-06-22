import Link from "next/link";
import { listProjects } from "@/lib/mock/projects";
import { GlassCard } from "@/components/core/glass-card";
import { ConfidenceBadge } from "@/components/core/confidence-badge";
import { CourtMinimap } from "@/components/court/court-minimap";
import { Plus, Calendar, Activity } from "lucide-react";

export default function ProjectsPage() {
  const projects = listProjects();

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Mes matchs</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {projects.length} match{projects.length > 1 ? "s" : ""} analysé{projects.length > 1 ? "s" : ""}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <Link href="/projects/new" className="group">
          <GlassCard className="flex h-full min-h-[280px] flex-col items-center justify-center gap-3 text-muted-foreground transition-colors group-hover:text-foreground">
            <div className="flex h-12 w-12 items-center justify-center rounded-full glass-strong text-court-green">
              <Plus className="h-6 w-6" />
            </div>
            <span className="text-sm font-medium">Analyser un nouveau match</span>
          </GlassCard>
        </Link>

        {projects.map((p) => (
          <Link key={p.id} href={`/projects/${p.id}`} className="group">
            <GlassCard className="flex h-full flex-col gap-3 p-0 transition-transform group-hover:-translate-y-1">
              <div className="relative">
                <CourtMinimap
                  bounces={p.stats.bounces}
                  trajectory={p.trajectory}
                  showPlayers={false}
                  showTrajectory={false}
                  source={p.stats.speed_source}
                  confidence={p.stats.homography_confidence}
                  className="rounded-b-none rounded-t-2xl"
                />
              </div>
              <div className="flex flex-1 flex-col gap-2 px-4 pb-4">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="font-semibold leading-tight">{p.name}</h3>
                  <ConfidenceBadge
                    source={p.stats.speed_source}
                    confidence={p.stats.homography_confidence}
                    showTooltip={false}
                  />
                </div>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Activity className="h-3.5 w-3.5" />
                    {p.stats.summary.total_bounces} rebonds
                  </span>
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3.5 w-3.5" />
                    {new Date(p.createdAt).toLocaleDateString("fr-FR", { day: "numeric", month: "short" })}
                  </span>
                </div>
              </div>
            </GlassCard>
          </Link>
        ))}
      </div>
    </div>
  );
}
