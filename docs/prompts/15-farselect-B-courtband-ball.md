# PROMPT — Far-select B : court-band géométrique + proximité balle (Backend) — SESSION 2/5

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente chaque hypothèse dans le CR, ne bloque jamais. Si tu
> manques d'idées → écris ton CR et arrête. Une question = run raté.

## Contexte — le diagnostic est DÉJÀ FAIT (mesuré)
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et **la mémoire** `courtside-far-player-selection-not-detection`.

**Le vrai problème (mesuré contre une GT humaine) :** le joueur far **EST détectable** (`yolov8m` conf=0.15 imgsz=1280 le trouve 8/8 plein cadre), mais `vision/pose.py` (détecteur conf=0.20 ligne ~164 ; `_pool_by_side` prend les plus gros/confiants au-dessus du filet) **sélectionne un spectateur/juge fixe en haut à droite**. Box "far" présente 100% mais sur le VRAI joueur 0.9% (2/225), distance moyenne 572px. → **bug de SÉLECTION + SEUIL, pas de détection/pose/re-train.**

## Ton angle (B) — gating géométrique de la zone de jeu + balle
**Distinct de la session A (qui mise sur le mouvement).** Toi, tu raisonnes GÉOMÉTRIE :
1. Dérive la **bande de court jouable** depuis la géométrie de la scène (les deux baselines / le court détecté / les zones de court si dispo, sinon la portée verticale des pieds des joueurs). Tout candidat au-dessus du filet mais **HORS de cette bande** = gradins/spectateur → rejeté.
2. Parmi les candidats DANS la bande, préfère celui le plus proche de l'**activité balle côté far** (la trajectoire balle indique où est le joueur far).
3. Baisse la conf far si nécessaire pour faire APPARAÎTRE le bon candidat (il existe à conf 0.15), mais c'est le gating géométrique qui le SÉLECTIONNE.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-farB feat/farselect-courtband-ball
```
Branche `feat/farselect-courtband-ball` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ Mémoire `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-farB/...`. **Touche UNIQUEMENT `vision/pose.py`** (+ un script de mesure jetable si besoin).

## Le benchmark (la GT humaine — commitée)
- `tests/fixtures/pose_gt/tennis_demo3.pose_gt.json` : POINT vérifié au centre du vrai joueur far par frame (225 frames).
- **Scorer** : `python tools/pose_gt/measure_far_coverage.py` → métrique cible **"far CORRECT (within 10%W)" %** (baseline 0.9% → le plus haut possible).
- Clip : `data/output/tennis_demo3.mp4`.

## La boucle (itère SANS plafond jusqu'à cible ou plateau prouvé)
mesure → change UNE chose dans `vision/pose.py` → re-mesure → garde si mieux, annule si pire. Logge le far-CORRECT % à chaque itération.

## Contraintes dures
- **Zéro hardcoding** : seuils = ratios de frame/fps/géométrie détectée, jamais des pixels de demo3. La bande de court se dérive de la scène, pas d'un nombre fixe. Anti-surapprentissage.
- **NE régresse PAS felix** ni aucun test (`test_bounce_regression` ≥0.72, `test_bounce_wasb_regression` ≥0.80, `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns`) — verts à chaque itération.

## Livrable — CR (OBLIGATOIRE)
`docs/research/farselect-courtband-ball-CR.md` : journal d'itérations (baseline 0.9% → chaque changement → far-CORRECT %), nombre final + distance moyenne, preuve felix intact, limites + risque surapprentissage, **l'idée unique la plus réutilisable** (pour l'intégrateur). Commit + push. VRAIS chiffres mesurés uniquement.
