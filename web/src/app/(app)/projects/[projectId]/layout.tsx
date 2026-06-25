import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { VideoFrameProvider } from "@/components/video/video-frame-context";
import { ProjectShell } from "@/components/layout/project-shell";

export default async function ProjectLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();

  return (
    <VideoFrameProvider
      frameRange={project.stats.frame_range}
      fps={project.stats.fps}
      initial={project.stats.bounces[0]?.frame ?? project.stats.frame_range[0]}
      projectId={project.id}
      clipIndex={project.clipIndex}
    >
      <ProjectShell project={project}>{children}</ProjectShell>
    </VideoFrameProvider>
  );
}
