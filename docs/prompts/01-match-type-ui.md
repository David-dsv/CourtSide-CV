# PROMPT — Type entraînement/match à l'upload + projet tennis-full (Frontend)

## Contexte
Repo CourtSide-CV, frontend dans `web/` (Next.js 16 App Router, Tailwind v4,
shadcn base-ui `render=` pas `asChild`, pnpm). Lire `CLAUDE.md`, `PROJET.md`,
et `docs/pm.md` (workflow multi-sessions).

Décision produit : deux types d'analyse —
- **Entraînement** (`training`) : caméra fixe, un joueur qui ne change PAS de
  côté. Le verrou P1/P2 fixe actuel (`vision/player_track.py`) marche
  parfaitement. Cœur de cible CourtSide (amateur sur trépied).
- **Match** (`match`) : vraie vidéo de match, les joueurs CHANGENT de côté entre
  jeux. La logique de ré-identité side-swap est gérée par une AUTRE session
  backend (`feat/match-mode`, PAS ton sujet). Ton job = l'UI + le typage.

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-matchtype feat/match-type-ui
```
Branche : `feat/match-type-ui` (depuis `feat/accuracy-overhaul`, HEAD courant).
Un commit par étape, `git push -u origin feat/match-type-ui` après chaque.
`pnpm build` + `pnpm lint` doivent passer (0 erreur) à la fin.

## Objectif
Permettre à l'utilisateur de choisir explicitement le type (training/match) à
l'upload, stocker ce type sur le projet, l'afficher, et préparer un fixture
pour la vidéo tennis complète.

## Étapes

1. **Type** : ajoute à `web/src/lib/types.ts` :
   ```ts
   export type MatchType = "training" | "match";
   ```
   et un champ `matchType: MatchType;` sur `interface Project`.

2. **Upload** : `web/src/app/(app)/projects/new/page.tsx` — ajoute un
   **sélecteur explicite** (2 cartes / radio au moment de soumettre la vidéo) :
   « Entraînement » (défaut) vs « Match ». Sous-titre clair :
   *« Entraînement = caméra fixe, pas de changement de côté. Match = les joueurs
   changent de côté entre jeux (analyse plus longue). »*

3. **Fixtures** (`web/src/lib/mock/fixtures.ts`) :
   - marque `felix`, `tennis-demo`, `felix-sota` en `matchType: "training"`.
   - ajoute un 4e projet :
     ```ts
     id: "tennis-full",
     name: "Match démo — tennis complet",
     video: "tennis.mp4",
     matchType: "match",
     stats: tennisFullStats,   // frame_range [0,11760], fps 50
     ```
     Le `_stats.json` réel est `data/output/tennis_full_annotated_stats.json`
     (généré en parallèle). S'il n'existe pas encore, copie `raw/tennis.json`
     en placeholder et mets un `// TODO: stats réelles 235s`. La vidéo annotée
     réelle est `web/public/annotated/tennis-full_annotated.mp4`.

4. **Notice mode match** : crée un composant réutilisable
   `MatchModeNotice` (ex. `web/src/components/core/match-mode-notice.tsx`)
   affiché sur l'overview quand `project.matchType === "match"` :
   *« Match — suivi des changements de côté en cours de développement. Les
   stats par joueur (vitesse, fatigue) peuvent être approximatives tant que la
   ré-identité joueur n'est pas branchée. »* Composant dédié pour pouvoir le
   retirer proprement quand le backend `feat/match-mode` sera intégré.

## Conventions
- **frame-as-truth** respecté (tout cliquable → `setFrame`).
- Ne casse rien : highlights, replay, profil, momentum, heatmap = hors sujet.
- Bilingue FR/EN comme le reste du code.

## Validation
- `pnpm build` + `pnpm lint` (0 erreur).
- Vérifie visuellement (dev server) : le sélecteur à l'upload, la notice sur
  `/projects/tennis-full/overview`, l'absence de notice sur `/projects/felix`.

## CR à la fin (OBLIGATOIRE — renvoyé au chef de projet)
- `matchType` ajouté où (types, fixtures, upload, notice).
- Le nouveau projet `tennis-full` + sa source vidéo/stats.
- `pnpm build` + `lint` : résultats exacts.
- Props/composants exposés (`MatchModeNotice`) + TODO backend clairs.
- Captures mentales : ce qui rend bien vs fragile.
Pousse tout sur `feat/match-type-ui`.
