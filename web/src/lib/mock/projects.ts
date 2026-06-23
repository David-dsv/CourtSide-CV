import { mockProjects } from "@/lib/mock/fixtures";
import type { Project } from "@/lib/types";

export function listProjects(): Project[] {
  return mockProjects;
}

export function getProject(id: string): Project | undefined {
  return mockProjects.find((p) => p.id === id);
}

export function getProjectOrThrow(id: string): Project {
  const p = getProject(id);
  if (!p) throw new Error(`Project not found: ${id}`);
  return p;
}
