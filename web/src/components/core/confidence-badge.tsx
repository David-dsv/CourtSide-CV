import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Check, Minus, TriangleAlert, Waves } from "lucide-react";
import { cn } from "@/lib/utils";
import type { HomographyConfidence, SpeedSource } from "@/lib/types";

/**
 * Calibrated confidence — the central design element of CourtSide.
 * Mirrors the pipeline vocabulary (vision/minimap.py): homography(high) is
 * metric-exact; homography(low) is approximate; scalar is a px/m fallback;
 * fallback/none is flagged `~approx`. The product never lies about precision.
 */
export type ConfidenceTier =
  | "homography-high"
  | "homography-low"
  | "scalar"
  | "approx"
  | "fallback"
  | "none";

export function tierFromSource(
  source?: SpeedSource | string,
  confidence?: HomographyConfidence,
): ConfidenceTier {
  const s = (source ?? "").toString();
  if (s.startsWith("homography")) {
    if (s.includes("high") || confidence === "high") return "homography-high";
    if (s.includes("low") || confidence === "low") return "homography-low";
    return "homography-low";
  }
  if (s === "scalar") return "scalar";
  if (confidence === "fallback") return "fallback";
  if (confidence === "none") return "none";
  return "scalar";
}

const config: Record<
  ConfidenceTier,
  {
    label: string;
    icon: typeof Check;
    className: string;
    explainer: string;
  }
> = {
  "homography-high": {
    label: "homography(high)",
    icon: Check,
    className:
      "border-court-green/40 bg-court-green/15 text-court-green hover:bg-court-green/20",
    explainer:
      "Homographie court→mètres calibrée, haute confiance. Vitesses et profondeurs en mètres réels, perspective-corrigées.",
  },
  "homography-low": {
    label: "homography(low)",
    icon: Waves,
    className:
      "border-court-amber/40 bg-court-amber/15 text-court-amber hover:bg-court-amber/20",
    explainer:
      "Homographie calibrée mais conditionnement géométrique faible (angle grazing). Métriques approchées.",
  },
  scalar: {
    label: "scalar",
    icon: Minus,
    className:
      "border-court-cyan/40 bg-court-cyan/15 text-court-cyan hover:bg-court-cyan/20",
    explainer:
      "Pas d'homographie : vitesse dérivée d'un scalaire px/m basé sur les proportions ITF du court.",
  },
  approx: {
    label: "~approx",
    icon: TriangleAlert,
    className:
      "border-court-red/40 bg-court-red/15 text-court-red hover:bg-court-red/20",
    explainer: "Valeurs approximatives (fallback géométrique). À considérer comme indicatives.",
  },
  fallback: {
    label: "~approx",
    icon: TriangleAlert,
    className:
      "border-court-red/40 bg-court-red/15 text-court-red hover:bg-court-red/20",
    explainer: "Fallback géométrique : homographie indisponible, métriques indicatives.",
  },
  none: {
    label: "~approx",
    icon: TriangleAlert,
    className:
      "border-court-red/40 bg-court-red/15 text-court-red hover:bg-court-red/20",
    explainer: "Aucune calibration disponible : métriques indicatives uniquement.",
  },
};

export function ConfidenceBadge({
  source,
  confidence,
  tier,
  className,
  showTooltip = true,
}: {
  source?: SpeedSource | string;
  confidence?: HomographyConfidence;
  tier?: ConfidenceTier;
  className?: string;
  showTooltip?: boolean;
}) {
  const t = tier ?? tierFromSource(source, confidence);
  const c = config[t];
  const Icon = c.icon;

  const badge = (
    <Badge
      variant="outline"
      className={cn(
        "gap-1 font-mono text-[10px] font-medium uppercase tracking-wide",
        c.className,
        className,
      )}
    >
      <Icon className="h-3 w-3" />
      {c.label}
    </Badge>
  );

  if (!showTooltip) return badge;

  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex" />}>
        {badge}
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs text-xs leading-relaxed">
        {c.explainer}
      </TooltipContent>
    </Tooltip>
  );
}
