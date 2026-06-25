# PROMPT — Front : UI par-échange + ré-alignement vidéo coupée (Frontend)

> **Session S5.** Effort recommandé : **xhigh** (plusieurs composants React + nouvelle notion de timeline coupée à intégrer proprement ; build/lint + validation visuelle).

## Contexte
Repo CourtSide-CV, frontend dans `web/`. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, puis :
- `web/src/lib/types.ts` (`PipelineStats`, `Rally`, `Bounce`, `Shot`) — tu PEUX l'éditer **avec parcimonie** (c'est un point transverse ; signale toute modif dans le CR).
- `web/src/lib/analysis/highlights.ts` (`deriveRallies`, `deriveHighlights`).
- `web/src/components/court/court-minimap.tsx` (radar front — affiche TOUS les rebonds aujourd'hui, lignes 69–77, 197–209).
- `web/src/components/video/video-scrubber.tsx` + `video-frame-context.tsx` (lecture vidéo + mapping frame↔temps).
- Pages : `overview`, `bounces`, `shots`, `stats` sous `web/src/app/(app)/projects/[projectId]/`.

Tu possèdes **`web/src/**` en exclusivité**. Aucune autre session ne touche le front.

## Les problèmes côté front (alignés sur les corrections backend)
1. **#1 shot vs bounce** : dans l'UID front, les frappes et rebonds sont dans des sections séparées sans lien. À l'usage on les confond. → rends la distinction visuelle nette (icônes/couleurs cohérentes avec la vidéo) et, là où pertinent, montre le lien frappe→rebond résultant (`shotsWithResults` existe déjà dans highlights.ts).
2. **#3 stats par-échange** : aujourd'hui toutes les tables sont **cumulatives** (jamais groupées par rally). → ajoute une **vue/groupement par échange** (ÉCHANGE N, ses frappes/rebonds/outcome), cohérente avec le HUD vidéo. Les rallies sont déjà dispo (`stats.rallies` backend ou `deriveRallies` fallback).
3. **#4 radar** : `court-minimap.tsx` affiche **tous** les rebonds. → l'aligner sur la décision backend : afficher **uniquement le rebond courant** (le dernier `<= frame`), animé selon `frame` du `video-frame-context`. (Garde éventuellement une vue « tous les rebonds » SÉPARÉE sur la page bounces, mais le **radar live** ne montre qu'un rebond.)
4. **Vidéo coupée (nouveau, S2)** : la vidéo annotée sera **plus courte** que l'originale (frames hors-terrain coupées). Un sidecar `<video>_clipindex.json` (fourni par S2) mappe `annotated_frame → original_frame`. → le scrubber/contexte doit savoir lire ce mapping pour afficher un timecode/numéro de frame **cohérent avec l'original** (sinon « frame 1200 » dans la vidéo coupée ≠ frame 1200 réelle).

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-front feat/front-rallies-timeline
```
Branche : `feat/front-rallies-timeline` (depuis `feat/accuracy-overhaul`). Commit + push après chaque étape.

## Contrats que tu consommes (des autres sessions)
- **De S2** : format `_clipindex.json` =
  `{fps, original_total, annotated_total, kept_original_frames:[...], cut_spans:[[s,e]], start_frame}`.
  `kept_original_frames[k]` = index original de la k-ème frame annotée. Utilise-le pour `annotated→original`.
- Les champs **match-mode** (`match_mode`, `match.side_swaps`, `human_id` sur shots/rallies) existent déjà dans le vrai `_stats.json` (`raw/tennis-full.json`) mais ne sont **pas encore déclarés** dans `PipelineStats`. Tu peux les ajouter au type si ta vue par-échange les exploite (signale-le).

## Étapes
1. **Radar live = 1 rebond** : dans `court-minimap.tsx`, filtre les rebonds au **dernier `<= frame`** pour le rendu du marqueur live (anime via `frame`). Conserve la trajectoire/comète et les pucks. La liste exhaustive des rebonds reste sur la page `bounces`.
2. **Vue par-échange** : un composant « Échanges » (liste de rallies : ÉCHANGE N, n frappes, n rebonds, outcome coloré, durée). Clic → seek au `start_frame` du rally (via `video-frame-context`). Réutilise `deriveRallies`/`stats.rallies`.
3. **Shot↔bounce lisible** : icônes/couleurs distinctes et cohérentes (rebond = couleur depth ; frappe = couleur stroke), et un indicateur « cette frappe → ce rebond » via `shotsWithResults`.
4. **Timeline coupée** : charge `_clipindex.json` (s'il existe à côté de la vidéo annotée ; sinon fallback identité = pas de cut). Convertis annotated↔original pour l'affichage timecode/frame. Documente où.
5. Ajoute un mock `_clipindex.json` dans les fixtures pour tester la conversion même sans run réel (ex. identité + un cut synthétique).

## Validation + ré-itération (jusqu'à parfait)
- `pnpm lint` + `pnpm build` (depuis `web/`) — **zéro erreur**.
- **Validation visuelle** (dev server) : radar n'affiche qu'1 rebond live ; vue Échanges navigable ; shot/bounce non confondables ; timecode cohérent quand un `_clipindex.json` avec cut est présent.
- Ré-itère jusqu'à ce que tout soit cohérent et propre.

## CR à la fin (OBLIGATOIRE)
- Composants ajoutés/modifiés + props exposées (pour l'intégration).
- Toute modif de `web/src/lib/types.ts` (point transverse — détaille).
- Comment le radar live filtre à 1 rebond, comment la vue Échanges marche, comment la timeline coupée est ré-alignée.
- Le format `_clipindex.json` consommé (confirme la cohérence avec le contrat S2).
- Honnête : ce qui reste mock vs réel (le vrai clipindex viendra d'un run backend).
- Résultats pnpm lint + build.
Pousse tout sur `feat/front-rallies-timeline`. NE touche QUE `web/src/**`.
