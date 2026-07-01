import { notFound } from "next/navigation";
import { getProject } from "@/lib/mock/projects";
import { GlassCard } from "@/components/core/glass-card";
import { SectionHeading } from "@/components/core/section-heading";
import { CourtMinimap } from "@/components/court/court-minimap";
import { FrameAwareFrame } from "@/components/video/frame-aware-frame";
import { Users } from "lucide-react";
import { PLAYER_HEX } from "@/lib/court/court-geometry";
import type { Bounce, Shot } from "@/lib/types";

export default async function PlayerComparePage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  const project = getProject(projectId);
  if (!project) notFound();

  const shots = project.stats.shots;
  const near = shots.filter((s: Shot) => s.player_side === "near");
  const far = shots.filter((s: Shot) => s.player_side === "far");

  const { near: nearBounces, far: farBounces } = attributeBounces(
    project.stats.bounces,
    shots,
    project.stats.fps,
  );

  const avgQ = (arr: Shot[]) => (arr.length ? arr.reduce((a, s) => a + s.quality, 0) / arr.length : 0);
  const avgSpd = (arr: Shot[]) => (arr.length ? arr.reduce((a, s) => a + s.speed_kmh, 0) / arr.length : 0);
  const fh = (arr: Shot[]) => arr.filter((s) => s.stroke === "forehand").length;
  const bh = (arr: Shot[]) => arr.filter((s) => s.stroke === "backhand").length;

  const card = (label: string, color: string, sideShots: Shot[], sideBounces: Bounce[]) => (
    <GlassCard>
      <div className="mb-4 flex items-center gap-2">
        <span className="h-3 w-3 rounded-full" style={{ background: color }} />
        <h3 className="font-semibold">{label}</h3>
        <span className="ml-auto text-xs text-muted-foreground">{sideShots.length} frappes</span>
      </div>
      <div className="grid grid-cols-3 gap-3 text-center">
        <Stat label="Vitesse moy." value={`${avgSpd(sideShots).toFixed(0)}`} unit="km/h" />
        <Stat label="Qualité moy." value={`${Math.round(avgQ(sideShots))}`} unit="/100" />
        <Stat label="CD / RV" value={`${fh(sideShots)}/${bh(sideShots)}`} />
      </div>
      <div className="mt-4">
        <CourtMinimap
          bounces={sideBounces}
          showPlayers={false}
          showTrajectory={false}
          source={project.stats.speed_source}
          confidence={project.stats.homography_confidence}
          H={project.H}
          className="mx-auto max-w-[260px]"
        />
      </div>
    </GlassCard>
  );

  return (
    <div className="flex flex-col gap-6">
      <SectionHeading
        title="Comparaison des joueurs"
        description="P1 (proche) vs P2 (lointain) — placement et frappes."
        icon={<Users className="h-4 w-4" />}
        actions={<FrameAwareFrame />}
      />
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        {card("P1 — Proche", PLAYER_HEX[1], near, nearBounces)}
        {card("P2 — Lointain", PLAYER_HEX[2], far, farBounces)}
      </div>
    </div>
  );
}

function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="rounded-lg glass p-3">
      <div className="text-xl font-semibold tabular-nums">
        {value}
        {unit && <span className="ml-1 text-xs text-muted-foreground">{unit}</span>}
      </div>
      <div className="mt-0.5 text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

/**
 * Real attribution: a bounce is the landing of the LAST shot struck before it
 * (same shot→next-bounce pairing as `shotsWithResults` on the shots page), so
 * each player's map shows where THEIR balls landed. A bounce with no shot in
 * the preceding 3 s (serve landings at clip start, stray segments) stays
 * unattributed rather than guessed — replaces the old `i % 2` placeholder.
 */
function attributeBounces(
  bounces: Bounce[],
  shots: Shot[],
  fps: number,
): { near: Bounce[]; far: Bounce[] } {
  const sorted = [...shots].sort((a, b) => a.frame - b.frame);
  const maxGap = 3 * fps;
  const near: Bounce[] = [];
  const far: Bounce[] = [];
  for (const b of bounces) {
    let striker: Shot | null = null;
    for (const s of sorted) {
      if (s.frame >= b.frame) break;
      striker = s;
    }
    if (!striker || b.frame - striker.frame > maxGap) continue;
    (striker.player_side === "near" ? near : far).push(b);
  }
  return { near, far };
}
