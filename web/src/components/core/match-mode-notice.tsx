import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Avertissement discret affiché sur un projet mode "match".
 *
 * Raison produit : en mode "match" les joueurs changent de côté entre jeux, et
 * la ré-identité joueur (side-swap) est gérée côté backend (branche
 * feat/match-mode) — PAS encore branchée sur ce front. Tant qu'elle n'est pas
 * intégrée, les stats par joueur (vitesse, fatigue) restent approximatives.
 *
 * Composant volontairement isolé + désactivable (prop `show`, défaut true) pour
 * qu'on puisse le retirer en un seul endroit quand feat/match-mode sera live.
 *
 * Bilingue FR/EN comme le reste de l'app.
 */
export function MatchModeNotice({
  className,
  show = true,
}: {
  className?: string;
  /** Passer `false` pour masquer (ex. après intégration feat/match-mode). */
  show?: boolean;
}) {
  if (!show) return null;

  return (
    <div
      role="note"
      className={cn(
        "flex items-start gap-3 rounded-xl border border-court-cyan/25 bg-court-cyan/5 px-4 py-3 text-sm text-muted-foreground",
        className,
      )}
    >
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-court-cyan" />
      <div className="space-y-0.5">
        <p className="font-medium text-foreground">
          Match — suivi des changements de côté en cours de développement.
        </p>
        <p>
          Les statistiques par joueur (vitesse, fatigue) peuvent être
          approximatives tant que la ré-identité joueur n&apos;est pas branchée.
        </p>
        <p className="text-xs italic opacity-70">
          Match — side-swap tracking is under development. Per-player stats
          (speed, fatigue) may be approximate until player re-identification is
          wired in.
        </p>
      </div>
    </div>
  );
}
