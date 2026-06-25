import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { MatchModeNotice } from "@/components/core/match-mode-notice";
import { RalliesView } from "@/components/analysis/rallies-view";
import { Swords } from "lucide-react";
import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ projectId: string }>;
}): Promise<Metadata> {
  const { projectId } = await params;
  const p = getProject(projectId);
  return { title: p ? `Échanges — ${p.name}` : "Échanges" };
}

export default async function RalliesPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Échanges"
        description="Chaque point décomposé : frappes, rebonds et issue. Cliquez pour rejouer l'échange."
        icon={<Swords className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      {project.matchType === "match" && <MatchModeNotice />}

      <RalliesView stats={project.stats} />
    </div>
  );
}
