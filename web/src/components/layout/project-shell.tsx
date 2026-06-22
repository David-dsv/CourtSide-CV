"use client";

import Link from "next/link";
import { useState } from "react";
import { ProjectSidebar } from "@/components/layout/project-sidebar";
import { ConfidenceBadge } from "@/components/core/confidence-badge";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { ChevronLeft, Menu } from "lucide-react";
import type { Project } from "@/lib/types";

export function ProjectShell({ project, children }: { project: Project; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  const sidebar = (
    <div className="flex flex-col gap-4">
      <div className="px-2">
        <Link href="/projects" className="mb-3 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
          <ChevronLeft className="h-3 w-3" /> Mes matchs
        </Link>
        <h2 className="font-semibold leading-tight">{project.name}</h2>
        <div className="mt-2">
          <ConfidenceBadge
            source={project.stats.speed_source}
            confidence={project.stats.homography_confidence}
          />
        </div>
      </div>
      <ProjectSidebar projectId={project.id} />
    </div>
  );

  return (
    <div className="mx-auto flex max-w-7xl gap-6 px-4 py-6 sm:px-6">
      {/* desktop sidebar */}
      <aside className="hidden w-56 shrink-0 lg:block">
        <div className="sticky top-20">{sidebar}</div>
      </aside>

      {/* mobile drawer trigger */}
      <div className="lg:hidden">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger
            render={
              <Button variant="outline" size="sm" className="fixed bottom-4 left-4 z-30 gap-2">
                <Menu className="h-4 w-4" /> Navigation
              </Button>
            }
          />
          <SheetContent side="left" className="w-64">
            <div className="mt-4">{sidebar}</div>
          </SheetContent>
        </Sheet>
      </div>

      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
