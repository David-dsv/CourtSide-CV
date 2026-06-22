"use client";

import { use, useState } from "react";
import { getProject } from "@/lib/mock/projects";
import { CourtMinimap } from "@/components/court/court-minimap";
import { PlacementHeatmap } from "@/components/court/placement-heatmap";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { useFrame } from "@/components/video/video-frame-context";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Map } from "lucide-react";

export default function MinimapPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = use(params);
  const project = getProject(projectId);
  const { frame, setFrame } = useFrame();
  const [showTraj, setShowTraj] = useState(true);
  const [showPlayers, setShowPlayers] = useState(true);
  const [showHeat, setShowHeat] = useState(false);

  if (!project) return null;

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Court radar"
        description="Placement, trajectoire et joueurs projetés via l'homographie court→mètres."
        icon={<Map className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
        <div className="relative">
          <CourtMinimap
            bounces={project.stats.bounces}
            trajectory={project.trajectory}
            players={project.players}
            source={project.stats.speed_source}
            confidence={project.stats.homography_confidence}
            showTrajectory={showTraj}
            showPlayers={showPlayers}
            frame={frame}
            onBounceClick={(f) => setFrame(f)}
            className="mx-auto max-w-[440px]"
          />
          {showHeat && (
            <div className="pointer-events-none absolute inset-0 mx-auto max-w-[440px]">
              <PlacementHeatmap bounces={project.stats.bounces} width={360} height={760} className="h-full w-full opacity-70" />
            </div>
          )}
        </div>

        <GlassCard className="flex flex-col gap-5">
          <h3 className="text-sm font-semibold">Affichage</h3>
          <ToggleRow label="Trajectoire balle" checked={showTraj} onChecked={setShowTraj} />
          <ToggleRow label="Joueurs (P1/P2)" checked={showPlayers} onChecked={setShowPlayers} />
          <ToggleRow label="Heatmap placement (INFERNO)" checked={showHeat} onChecked={setShowHeat} />

          <div className="mt-2 rounded-lg glass p-3 text-xs leading-relaxed text-muted-foreground">
            <p className="mb-1 font-medium text-foreground">À propos de cette vue</p>
            <p>
              Chaque rebond est coloré par profondeur (rouge = profond, ambre = moyen, vert = court).
              La trajectoire reconstitue le vol de la balle entre rebonds. Le badge indique la confiance
              de la calibration : {project.stats.speed_source}.
            </p>
          </div>
        </GlassCard>
      </div>
    </div>
  );
}

function ToggleRow({ label, checked, onChecked }: { label: string; checked: boolean; onChecked: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <Label htmlFor={label} className="text-sm">{label}</Label>
      <Switch id={label} checked={checked} onCheckedChange={onChecked} />
    </div>
  );
}
