# PROMPT — Cohérence depth ↔ position (single source of truth géométrique)

## Contexte
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et :
- `web/src/lib/court/projection.ts` (convention métrique + `pxToMeters`)
- `run_pipeline_8s.py` fonction `classify_bounce_depth` (~ligne 188, ~890)
- `web/src/components/court/court-minimap.tsx` + `placement-heatmap.tsx`

## Le problème (diagnostiqué)
Aujourd'hui la **couleur** du rebond (label `depth` = deep/mid/short) et sa
**position** sur la minimap (pixel projeté via homographie ou fallback) sont
calculées par **deux géométries différentes** :
- `depth` : `classify_bounce_depth` utilise sa propre géométrie ad-hoc
  (baselines à 18%/47%/82% de la frame, ratio dans la demi-court) — et
  **n'utilise pas l'homographie** même quand elle est disponible.
- position minimap : `pxToMeters` (H si passée, sinon fallback top/bot).

Mesuré sur les vrais fixtures : **aucune corrélation** entre `depth` et `y_px`.
Ex tennis : `short` à y=467 ET y=577, `mid` à y=709 ET y=321, `deep` à y=789.
Ex sota : des `short` de y=513 à y=605 partout.

→ Un point vert « court » peut s'afficher dans le fond du court. Visuellement
incohérent et malhonnête.

## Décision (ARRÊTÉE par le chef de projet)
**La position géométrique est l'unique source de vérité.** Le label `depth`
doit être **dérivé de la coordonnée métrique** projetée (Ym), pas calculé par
une géométrie séparée. Couleur et position seront TOUJOURS cohérentes.

## Convention métrique (source : projection.ts + ITF)
Origine au centre du court, `+Y` vers la far baseline. `HALF_LEN = 11.885 m`
(demi-court). Filet à `Ym = 0`. `|Ym| ∈ [0, 11.885]`.
ITF : service line à 6.40 m du filet.
→ Bandes par distance au filet `|Ym|` :
- `short` : `|Ym| < ~4.0` (proche du filet)
- `mid`   : `~4.0 ≤ |Ym| < ~7.0` (entre filet et fond)
- `deep`  : `|Ym| ≥ ~7.0` (proche du fond de court)
(les ~4.0/7.0 sont des constantes documentées dérivées des dims ITF, PAS du
magic tuning d'une vidéo. Documente le choix.)

## Worktree (OBLIGATOIRE — voir docs/pm.md)
```
git worktree add ../CourtSide-CV-depth feat/depth-position-coherence
```
Branche : `feat/depth-position-coherence` (depuis `feat/accuracy-overhaul`).
Commit/étape, push après chaque.

## Objectif — UNE seule source de vérité géométrique
Pipeline ET frontend doivent dériver `depth` de la position projetée (Ym).

## Étapes

### A. Pipeline (Python)
1. Passe `court_homography` à `classify_bounce_depth` (il est dispo à l'appel,
   `run_pipeline_8s.py:890`, juste non transmis — même bug pattern que la
   minimap). Nouvelle signature :
   `classify_bounce_depth(bounce_events, court_zones, frame_height, homography=None)`.
2. Quand `homography` est dispo : projette chaque `(x,y)` → `Ym` métrique (même
   `applyHomography` que `pxToMeters` côté TS — réutilise la logique de
   `vision/minimap.py` ou `models/court_detector.py`) et classe `depth` par
   bande `|Ym|` (ci-dessus). **Exact = true.**
3. Quand **pas** d'homographie (scalar/fallback) : garde le fallback
   géométrique actuel MAIS marque-le clairement, et s'assure qu'il reste dans
   les bornes (c'est inévitablement approximatif — le badge `~approx` l'admet
   déjà côté UI).

### B. Frontend (TypeScript)
1. Crée un helper **unique** `depthFromMeters(Ym)` dans
   `web/src/lib/court/projection.ts` (les mêmes bandes |Ym| que le pipeline).
2. `CourtMinimap` + `PlacementHeatmap` : la couleur/profondeur affichée d'un
   rebond vient de `depthFromMeters(Ym projeté)`, **pas** du `bounce.depth` du
   `_stats.json`. Comme ça, même sur un fixture legacy/scalar dont le depth
   pipeline est incohérent, l'affichage reste cohérent avec la position.
3. Garde `bounce.depth` du JSON disponible (pour les stats/listes), mais la
   **couleur sur la carte** = depth dérivé de la position. Documente ce choix.

### C. Fixtures
- Régénère les `_stats.json` felix/tennis/sota via le pipeline corrigé (short
  clips) et copie-les dans `web/src/lib/mock/raw/`. Vérifie qu'après correction,
  `depth` corrèle avec `y_px` (le test ci-dessous).

## Validation + ré-itération
- **Test de cohérence** (assert-based, `tests/test_depth_coherence.py`) : pour
  chaque rebond, le `depth` dérivé de `Ym` doit être dans la bonne bande. Sur
  felix/tennis/sota corrigés, plus aucun « short affiché au fond ».
- **Validation visuelle** (dev server, `/projects/<id>/minimap`) : à l'œil,
  couleur et position cohérentes sur les 3 projets.
- Ré-itère jusqu'à cohérence parfaite.

## CR à la fin (OBLIGATOIRE)
- La nouvelle source de vérité géométrique + les bornes |Ym| choisies.
- Côté pipeline : `classify_bounce_depth` + passage de `court_homography`.
- Côté frontend : `depthFromMeters` + où la couleur est désormais dérivée.
- Résultats du test de cohérence + validation visuelle (honnête : ce qui reste
  approximatif en mode scalar/fallback).
- py_compile + tests Python ; pnpm build + lint.
Pousse tout sur `feat/depth-position-coherence`. Ne casse pas highlights,
replay, match-type, match-mode (autres sessions).
