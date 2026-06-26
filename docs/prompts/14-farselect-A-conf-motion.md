# PROMPT — Far-select A : conf far basse + rejet du spectateur statique (Backend) — SESSION 1/5

> **Session AUTONOME ULTRACODE.** Lance en `/ultracode`.
> ⛔ **NE POSE JAMAIS DE QUESTION à l'utilisateur. Aucun outil de clarification.**
> Décide seule, documente chaque hypothèse dans le CR, ne bloque jamais. Si tu
> manques d'idées → écris ton CR et arrête. Une question = run raté.

## Contexte — le diagnostic est DÉJÀ FAIT (mesuré)
Repo CourtSide-CV. Lire `CLAUDE.md`, `PROJET.md`, `docs/pm.md`, et **la mémoire** `courtside-far-player-selection-not-detection` (le diagnostic clé).

**Le vrai problème (mesuré contre une GT humaine) :** le joueur du fond (far/P1) **EST détectable** — `yolov8m` à `conf=0.15, imgsz=1280` le trouve **8/8** frames en plein cadre. MAIS le pipeline (`vision/pose.py` : détecteur à `conf=0.20` ligne ~164 ; `_pool_by_side` prend les plus gros/confiants au-dessus du filet) **sélectionne un spectateur/juge fixe en haut à droite**. Mesuré : box "far" présente **100%** des frames mais sur le VRAI joueur seulement **0.9% (2/225)**, distance moyenne **572px**. → **C'est un bug de SÉLECTION + SEUIL DE CONFIANCE, pas de détection/pose/re-train.**

## Ton angle (A) — conf far + rejet du spectateur par le MOUVEMENT
Le spectateur est **immobile** ; le joueur **bouge**. Un candidat dont la box bouge à peine sur N frames = spectateur → à rejeter. Combine :
1. Baisse la conf du détecteur de personnes pour la région FAR (le petit joueur a besoin de ~0.15) — mais sans noyer la scène de spectateurs (sois malin : conf basse SEULEMENT au-dessus du filet, ou re-détection ciblée).
2. Rejette les candidats STATIQUES (faible déplacement temporel de la box sur une fenêtre) → garde celui qui bouge comme un joueur.
3. Utilise la moitié-balle comme indice du côté far actif.

## Worktree (OBLIGATOIRE)
```
git worktree add ../CourtSide-CV-farA feat/farselect-conf-motion
```
Branche `feat/farselect-conf-motion` (depuis `feat/accuracy-overhaul`). Commit/push par itération.
⚠️ Mémoire `courtside-edit-targets-main-tree` : édite SOUS `/Users/vuong/Documents/CourtSide-CV-farA/...`. **Touche UNIQUEMENT `vision/pose.py`** (+ un script de mesure jetable si besoin).

## Le benchmark (la GT humaine — commitée)
- `tests/fixtures/pose_gt/tennis_demo3.pose_gt.json` : un POINT vérifié au centre du vrai joueur far par frame (225 frames).
- **Scorer** : `python tools/pose_gt/measure_far_coverage.py` → la métrique cible = **"far CORRECT (within 10%W)" %** (baseline 0.9% → le plus haut possible).
- Clip : `data/output/tennis_demo3.mp4`.

## La boucle (itère SANS plafond jusqu'à cible ou plateau prouvé)
mesure → change UNE chose dans `vision/pose.py` → re-mesure → garde si mieux, annule si pire. **Logge le far-CORRECT % à chaque itération.**

## Contraintes dures
- **Zéro hardcoding** (CLAUDE.md) : seuils = ratios de frame/fps/géométrie/mouvement, jamais des pixels de demo3. Anti-surapprentissage (1 seul clip).
- **NE régresse PAS felix** ni aucun test : `test_bounce_regression` (≥0.72), `test_bounce_wasb_regression` (≥0.80), `test_vx_veto`, `test_event_confusion_regression`, `test_sharp_turns` — verts à chaque itération.

## Livrable — CR (OBLIGATOIRE)
`docs/research/farselect-conf-motion-CR.md` : journal d'itérations (baseline 0.9% → chaque changement → far-CORRECT %), nombre final + distance moyenne, preuve felix intact, limites honnêtes + risque de surapprentissage, **l'idée unique la plus réutilisable** (pour l'intégrateur). Commit + push sur la branche. Le CR doit contenir les VRAIS chiffres mesurés, jamais des estimations.
